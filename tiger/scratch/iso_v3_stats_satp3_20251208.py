#!/usr/bin/env python3
import os
import time
import json
import yaml
import sqlite3
import argparse
import traceback
import tempfile

import numpy as np

from sotodlib import core
from sotodlib.core.flagman import has_all_cut, count_cuts
from sotodlib.preprocess import preprocess_util
from sotodlib.utils.procs_pool import get_exec_env


# ==================== Helpers ==================== #

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        'config_file_init',
        help="Preprocessing init configuration file"
    )
    parser.add_argument(
        'config_file_proc',
        help="Preprocessing proc configuration file"
    )
    parser.add_argument(
        '--nproc',
        type=int,
        default=16,
        help="Number of parallel processes to run on."
    )
    parser.add_argument(
        '--errlog-ext',
        default='iso_noise_check_err.txt',
        help="Error log file name."
    )
    parser.add_argument(
        '--savename',
        default='iso_cuts_check.npy',
        help="Base name for output (used to derive sqlite path)."
    )
    return parser


def sanitize_yaml_config(path, logger=None):
    """
    Create a sanitized temporary copy of a YAML config, commenting out
    suspicious top-level lines that don't contain ':' (like the stray
    absolute path line in your init config).
    """
    with open(path, "r") as f:
        lines = f.readlines()

    clean_lines = []
    for line in lines:
        stripped = line.lstrip()
        # Keep comments and blank lines
        if stripped.startswith("#") or stripped.strip() == "":
            clean_lines.append(line)
            continue

        # If it's a non-indented line with no ':', comment it out.
        if ":" not in line and not line.startswith(" "):
            clean_lines.append("# AUTO-COMMENTED: " + line)
        else:
            clean_lines.append(line)

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    tmp.writelines(clean_lines)
    tmp_path = tmp.name
    tmp.close()

    if logger is not None:
        logger.info(f"Using sanitized YAML config copy at {tmp_path}")

    return tmp_path


def safe_get_preprocess_context(config_file, logger=None):
    """
    Wrapper around preprocess_util.get_preprocess_context that
    auto-sanitizes slightly broken YAML configs.
    """
    try:
        return preprocess_util.get_preprocess_context(config_file)
    except yaml.YAMLError as e:
        if logger is not None:
            logger.info(
                f"YAML error for {config_file}: {e}. "
                "Attempting to sanitize config and retry."
            )
        tmp_path = sanitize_yaml_config(config_file, logger=logger)
        return preprocess_util.get_preprocess_context(tmp_path)


def has_axis_attr(axisman, name, logger=None):
    """
    Return True if axisman has attribute `name` without raising AttributeError.
    If missing, optionally log available fields.
    """
    try:
        getattr(axisman, name)
        return True
    except AttributeError:
        if logger is not None:
            fields = getattr(axisman, "_fields", {})
            logger.info(
                f"Missing preprocess layer: '{name}'. "
                f"Available fields: {list(fields.keys())}"
            )
        return False


def summarize_axisman(axisman, max_fields=None):
    """
    Build a nested dict summary of an AxisManager-like object:
    { field_name: { 'shape': ..., 'subfields': {...} } }.
    """
    summary = {}
    fields = getattr(axisman, "_fields", {})

    for name, obj in fields.items():
        entry = {}
        shape = getattr(obj, "shape", None)
        if shape is not None:
            try:
                entry["shape"] = tuple(shape)
            except TypeError:
                entry["shape"] = str(shape)
        else:
            entry["shape"] = None

        subfields = getattr(obj, "_fields", None)
        if subfields is not None:
            sub_summary = {}
            for sub_name, sub_obj in subfields.items():
                sub_shape = getattr(sub_obj, "shape", None)
                if sub_shape is not None:
                    try:
                        sub_summary[sub_name] = tuple(sub_shape)
                    except TypeError:
                        sub_summary[sub_name] = str(sub_shape)
                else:
                    sub_summary[sub_name] = None
            entry["subfields"] = sub_summary

        summary[name] = entry

    if max_fields is not None and len(summary) > max_fields:
        keys = list(summary.keys())[:max_fields]
        summary = {k: summary[k] for k in keys}
    return summary


def dump_preprocess_schema(init_preprocess, proc_preprocess, out_path, logger=None):
    """
    Summarize init/proc preprocess AxisManagers and write to JSON.
    """
    data = {
        "init_preprocess": summarize_axisman(init_preprocess),
        "proc_preprocess": summarize_axisman(proc_preprocess),
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)

    if logger is not None:
        logger.info(f"Wrote preprocess schema summary to {out_path}")


def add_scan_stats(x, field_name, prefix, keys, vals, logger=None):
    """
    Add PTP / STD / KURT / SKEW-like stats for an azss_stats* field on x (proc layer).

    - x: preprocess AxisManager (proc layer)
    - field_name: e.g. 'azss_statsT_left'
    - prefix: string for column name prefix, e.g. 'azss_T_left'
    - keys, vals: lists to append new entries to
    """
    if not has_axis_attr(x, field_name, logger=logger):
        keys.extend([
            f'{prefix}_ptp',
            f'{prefix}_std',
            f'{prefix}_kurt',
            f'{prefix}_skew',
        ])
        vals.extend([0, 0, 0, 0])
        return

    stats = getattr(x, field_name)

    # valid mask (if present)
    if hasattr(stats, 'valid'):
        m = has_all_cut(stats.valid)
    else:
        m = slice(None)

    # PTP
    if hasattr(stats, 'ptp'):
        y_ptp = stats.ptp[m] > 0.8
        val_ptp = np.sum(y_ptp, axis=-1)
    else:
        val_ptp = 0

    # STD
    if hasattr(stats, 'std'):
        std = stats.std[m]
        median_std = np.median(std, axis=1)[:, np.newaxis]
        y_std = std > median_std * 3.0
        val_std = np.sum(y_std, axis=-1)
    else:
        val_std = 0

    # Kurtosis
    if hasattr(stats, 'kurtosis'):
        y_kurt = np.abs(stats.kurtosis[m]) > 0.5
        val_kurt = np.sum(y_kurt, axis=-1)
    else:
        val_kurt = 0

    # Skew
    if hasattr(stats, 'skew'):
        y_skew = np.abs(stats.skew[m]) > 0.5
        val_skew = np.sum(y_skew, axis=-1)
    else:
        val_skew = 0

    keys.extend([
        f'{prefix}_ptp',
        f'{prefix}_std',
        f'{prefix}_kurt',
        f'{prefix}_skew',
    ])
    vals.extend([val_ptp, val_std, val_kurt, val_skew])


# ==================== Per-entry work ==================== #

def get_dict_entry(entry, config_file_init, config_file_proc):
    try:
        logger = preprocess_util.init_logger('subproc_logger')
        logger.info(f'Processing entry for {entry["dataset"]}')
        logger.info(f'Getting context for {entry["dataset"]}')

        # Init / proc contexts
        _, context_init = safe_get_preprocess_context(config_file_init, logger=logger)
        _, context_proc = safe_get_preprocess_context(config_file_proc, logger=logger)

        dets = {
            'wafer_slot': entry['dets:wafer_slot'],
            'wafer.bandpass': entry['dets:wafer.bandpass'],
        }

        # ---------- FIRST LAYER (INIT) ---------- #
        mdata_init = context_init.get_meta(entry['obs:obs_id'], dets=dets)
        del context_init
        x = mdata_init.preprocess

        keys = []
        vals = []

        # Basic counts
        keys.append('nsamps')
        vals.append(mdata_init.samps.count)
        keys.append('ndets')
        vals.append(mdata_init.dets.count)

        # Focal plane flags
        if has_axis_attr(x, 'fp_flags', logger=logger):
            m = has_all_cut(x.fp_flags.valid)
            keys.append('fp_cuts')
            vals.append(np.sum(has_all_cut(x.fp_flags.fp_nans)[m]))
        else:
            keys.append('fp_cuts')
            vals.append(0)

        # Trends
        if has_axis_attr(x, 'trends', logger=logger):
            m = has_all_cut(x.trends.valid)
            keys.append('trend_cuts')
            vals.append(np.sum(has_all_cut(x.trends.trend_flags)[m]))
        else:
            keys.append('trend_cuts')
            vals.append(0)

        # Turnarounds
        if has_axis_attr(x, 'turnaround_flags', logger=logger):
            m = has_all_cut(x.turnaround_flags.valid)
            keys.append('turnaround_nsamps')
            vals.append(np.sum([
                np.sum(np.ptp(r.ranges(), axis=1))
                for r, mask in zip(x.turnaround_flags.turnarounds.ranges, m) if mask
            ]))
        else:
            keys.append('turnaround_nsamps')
            vals.append(0)

        # Jumps slow
        if has_axis_attr(x, 'jumps_slow', logger=logger):
            m = has_all_cut(x.jumps_slow.valid)
            keys.append('jumps_slow_nsamps')
            vals.append(np.sum([
                np.sum(np.ptp(r.ranges(), axis=1))
                for r, mask in zip(x.jumps_slow.jump_flag.ranges, m) if mask
            ]))
            keys.append('jumps_slow_cuts')
            vals.append(np.sum(count_cuts(x.jumps_slow.jump_flag)[m] > 5))
        else:
            keys.extend(['jumps_slow_nsamps', 'jumps_slow_cuts'])
            vals.extend([0, 0])

        # Jumps 2π
        if has_axis_attr(x, 'jumps_2pi', logger=logger):
            m = has_all_cut(x.jumps_2pi.valid)
            keys.append('jumps_2pi_nsamps')
            vals.append(np.sum([
                np.sum(np.ptp(r.ranges(), axis=1))
                for r, mask in zip(x.jumps_2pi.jump_flag.ranges, m) if mask
            ]))
            keys.append('jumps_2pi_cuts')
            vals.append(np.sum(count_cuts(x.jumps_2pi.jump_flag)[m] > 20))
        else:
            keys.extend(['jumps_2pi_nsamps', 'jumps_2pi_cuts'])
            vals.extend([0, 0])

        # Glitches pre-HWPSS
        if has_axis_attr(x, 'glitches_pre_hwpss', logger=logger):
            g_pre = x.glitches_pre_hwpss
            m = has_all_cut(g_pre.valid)
            keys.append('glitch_pre_nsamps')
            vals.append(np.sum([
                np.sum(np.ptp(r.ranges(), axis=1))
                for r, mask in zip(g_pre.glitch_flags.ranges, m) if mask
            ]))
            keys.append('glitch_pre_cuts')
            vals.append(np.sum(count_cuts(g_pre.glitch_flags)[m] > 1000))
        else:
            keys.extend(['glitch_pre_nsamps', 'glitch_pre_cuts'])
            vals.extend([0, 0])

        # Glitches post-HWPSS
        if has_axis_attr(x, 'glitches_post_hwpss', logger=logger):
            g_post = x.glitches_post_hwpss
            m = has_all_cut(g_post.valid)
            keys.append('glitch_post_nsamps')
            vals.append(np.sum([
                np.sum(np.ptp(r.ranges(), axis=1))
                for r, mask in zip(g_post.glitch_flags.ranges, m) if mask
            ]))
            keys.append('glitch_post_cuts')
            vals.append(np.sum(count_cuts(g_post.glitch_flags)[m] > 100))
        else:
            keys.extend(['glitch_post_nsamps', 'glitch_post_cuts'])
            vals.extend([0, 0])

        # Det bias flags
        if has_axis_attr(x, 'det_bias_flags', logger=logger):
            m = has_all_cut(x.det_bias_flags.valid)
            keys.append('det_bias_cuts')
            vals.append(np.sum(has_all_cut(x.det_bias_flags.det_bias_flags)[m]))
        else:
            keys.append('det_bias_cuts')
            vals.append(0)

        # PTP flags
        if has_axis_attr(x, 'ptp_flags', logger=logger):
            m = has_all_cut(x.ptp_flags.valid)
            keys.append('ptp_cuts')
            vals.append(np.sum(has_all_cut(x.ptp_flags.ptp_flags)[m]))
        else:
            keys.append('ptp_cuts')
            vals.append(0)

        # Noise fits – noiseT, noiseQ, noiseU
        if has_axis_attr(x, 'noiseT', logger=logger):
            m = has_all_cut(x.noiseT.valid)
            keys.append('white_noise_cuts_T')
            vals.append(np.sum(
                ((x.noiseT.white_noise)[m] < 2e-6) |
                ((x.noiseT.white_noise)[m] > 80e-6)
            ))
        else:
            keys.append('white_noise_cuts_T')
            vals.append(0)

        if has_axis_attr(x, 'noiseQ', logger=logger):
            m = has_all_cut(x.noiseQ.valid)
            keys.append('white_noise_cuts_Q')
            vals.append(np.sum(
                ((x.noiseQ.white_noise)[m] < 2e-6) |
                ((x.noiseQ.white_noise)[m] > 80e-6)
            ))
        else:
            keys.append('white_noise_cuts_Q')
            vals.append(0)

        if has_axis_attr(x, 'noiseU', logger=logger):
            m = has_all_cut(x.noiseU.valid)
            keys.append('white_noise_cuts_U')
            vals.append(np.sum(
                ((x.noiseU.white_noise)[m] < 2e-6) |
                ((x.noiseU.white_noise)[m] > 80e-6)
            ))
        else:
            keys.append('white_noise_cuts_U')
            vals.append(0)

        # TOD stats T/Q/U (from init config)
        # T
        if has_axis_attr(x, 'tod_stats_T', logger=logger):
            m = has_all_cut(x.tod_stats_T.valid)
            noisy_subscan_indicator = np.zeros_like(x.tod_stats_T["std"][m], dtype=bool)
            keys.append('TOD_stats_T_ptp')
            y = x.tod_stats_T.ptp[m] > 0.8
            vals.append(np.sum(y, -1))
            keys.append('TOD_stats_T_std')
            median_std = np.median(x.tod_stats_T["std"][m], axis=1)[:, np.newaxis]
            y = x.tod_stats_T["std"][m] > median_std * 3.0
            vals.append(np.sum(y, -1))
            keys.append('TOD_stats_T_det_cut')
            vals.append(np.sum(
                np.sum(noisy_subscan_indicator, -1) >= mdata_init.subscans.count // 2
            ))
        else:
            keys.extend(['TOD_stats_T_ptp', 'TOD_stats_T_std', 'TOD_stats_T_det_cut'])
            vals.extend([0, 0, 0])

        # Q
        if has_axis_attr(x, 'tod_stats_Q', logger=logger):
            m = has_all_cut(x.tod_stats_Q.valid)
            noisy_subscan_indicator = np.zeros_like(x.tod_stats_Q["std"][m], dtype=bool)
            keys.append('TOD_stats_Q_ptp')
            y = x.tod_stats_Q.ptp[m] > 0.8
            vals.append(np.sum(y, -1))
            keys.append('TOD_stats_Q_std')
            median_std = np.median(x.tod_stats_Q["std"][m], axis=1)[:, np.newaxis]
            y = x.tod_stats_Q["std"][m] > median_std * 3.0
            vals.append(np.sum(y, -1))
            keys.append('TOD_stats_Q_kurt')
            y = (np.abs(x.tod_stats_Q['kurtosis']) > 0.5)
            vals.append(np.sum(y, -1))
            keys.append('TOD_stats_Q_skew')
            y = (np.abs(x.tod_stats_Q['skew']) > 0.5)
            vals.append(np.sum(y, -1))
            keys.append('TOD_stats_Q_det_cut')
            vals.append(np.sum(
                np.sum(noisy_subscan_indicator, -1) >= mdata_init.subscans.count // 2
            ))
        else:
            keys.extend([
                'TOD_stats_Q_ptp', 'TOD_stats_Q_std',
                'TOD_stats_Q_kurt', 'TOD_stats_Q_skew',
                'TOD_stats_Q_det_cut'
            ])
            vals.extend([0, 0, 0, 0, 0])

        # U
        if has_axis_attr(x, 'tod_stats_U', logger=logger):
            m = has_all_cut(x.tod_stats_U.valid)
            keys.append('TOD_stats_U_ptp')
            y = x.tod_stats_U.ptp[m] > 0.8
            vals.append(np.sum(y, -1))
            keys.append('TOD_stats_U_std')
            median_std = np.median(x.tod_stats_U["std"][m], axis=1)[:, np.newaxis]
            y = x.tod_stats_U["std"][m] > median_std * 3.0
            vals.append(np.sum(y, -1))
            keys.append('TOD_stats_U_kurt')
            y = (np.abs(x.tod_stats_U['kurtosis']) > 0.5)
            vals.append(np.sum(y, -1))
            keys.append('TOD_stats_U_skew')
            y = (np.abs(x.tod_stats_U['skew']) > 0.5)
            vals.append(np.sum(y, -1))
            keys.append('TOD_stats_U_det_cut')
            vals.append(np.sum(
                np.sum(noisy_subscan_indicator, -1) >= mdata_init.subscans.count // 2
            ))
        else:
            keys.extend([
                'TOD_stats_U_ptp', 'TOD_stats_U_std',
                'TOD_stats_U_kurt', 'TOD_stats_U_skew',
                'TOD_stats_U_det_cut'
            ])
            vals.extend([0, 0, 0, 0, 0])

        # Noisy subscans (init)
        if has_axis_attr(x, 'noisy_subscan_flags', logger=logger):
            m = has_all_cut(x.noisy_subscan_flags.valid)
            keys.append('noisy_subscans_nsamps')
            vals.append(np.sum([
                np.sum(np.ptp(r.ranges(), axis=1))
                for r, mask in zip(x.noisy_subscan_flags.valid_subscans.ranges, m) if mask
            ]))
        else:
            keys.append('noisy_subscans_nsamps')
            vals.append(0)

        # Noisy dets (init)
        if has_axis_attr(x, 'noisy_dets_flags', logger=logger):
            keys.append('noisy_subscans_cuts')
            vals.append(np.sum(~x.noisy_dets_flags.valid_dets))
        else:
            keys.append('noisy_subscans_cuts')
            vals.append(0)

        # Source / moon flags (init)
        if has_axis_attr(x, 'source_flags', logger=logger):
            m = has_all_cut(x.source_flags.valid)
            keys.append('source_flags_nsamps')
            vals.append(np.sum([
                np.sum(np.ptp(r.ranges(), axis=1))
                for r, mask in zip(x.source_flags.moon.ranges, m) if mask
            ]))
            keys.append('source_flags_cuts')
            vals.append(np.sum(has_all_cut(x.source_flags.moon)[m]))
        else:
            keys.extend(['source_flags_nsamps', 'source_flags_cuts'])
            vals.extend([0, 0])

        # ---------- SECOND LAYER (PROC) ---------- #
        mdata_proc = context_proc.get_meta(entry['obs:obs_id'], dets=dets)
        del context_proc
        x = mdata_proc.preprocess

        # Inv var flags (proc)
        if has_axis_attr(x, 'inv_var_flags', logger=logger):
            m = has_all_cut(x.inv_var_flags.valid)
            keys.append('inv_var_cuts')
            vals.append(np.sum(has_all_cut(x.inv_var_flags.inv_var_flags)[m]))
        else:
            keys.append('inv_var_cuts')
            vals.append(0)

        # Azimuth scan-speed stats (proc) – left/right, T/Q/U
        add_scan_stats(x, 'azss_statsT_left',
                       prefix='azss_T_left',
                       keys=keys, vals=vals, logger=logger)
        add_scan_stats(x, 'azss_statsT_right',
                       prefix='azss_T_right',
                       keys=keys, vals=vals, logger=logger)
        add_scan_stats(x, 'azss_statsQ_left',
                       prefix='azss_Q_left',
                       keys=keys, vals=vals, logger=logger)
        add_scan_stats(x, 'azss_statsQ_right',
                       prefix='azss_Q_right',
                       keys=keys, vals=vals, logger=logger)
        add_scan_stats(x, 'azss_statsU_left',
                       prefix='azss_U_left',
                       keys=keys, vals=vals, logger=logger)
        add_scan_stats(x, 'azss_statsU_right',
                       prefix='azss_U_right',
                       keys=keys, vals=vals, logger=logger)

        # Split flags → end_yield (proc)
        if has_axis_attr(x, 'split_flags', logger=logger):
            m = has_all_cut(x.split_flags.valid)
            keys.append('end_yield')
            vals.append(np.sum(m))
        else:
            keys.append('end_yield')
            vals.append(0)

        return (
            entry['obs:obs_id'],
            entry['dets:wafer_slot'],
            entry['dets:wafer.bandpass'],
            keys,
            vals,
        )

    except Exception as e:
        logger.info(f"Error in process for {entry['dataset']}")
        errmsg = f'{type(e)}: {e}'
        tb = ''.join(traceback.format_tb(e.__traceback__))
        return None, None, None, errmsg, tb


# ==================== Main driver ==================== #

def main(executor, as_completed_callable,
         config_file_init, config_file_proc,
         errlog_ext, savename, nproc):

    logger = preprocess_util.init_logger('main_proc')

    # Get proc configs to find archive location
    configs_proc, _ = safe_get_preprocess_context(config_file_proc, logger=logger)
    base_dir = os.path.dirname(configs_proc['archive']['index'])
    errlog = errlog_ext

    logger.info('connect to database')
    proc = core.metadata.ManifestDb(
        os.path.join(base_dir, 'process_archive.sqlite')
    )

    sqlite_path = savename.replace('.h5', '.sqlite')
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()

    run_list = proc.inspect()
    logger.info('run list created')

    # Find a first successful entry to define schema
    n_attempts = 0
    keys = None
    first_entry = None
    for candidate in run_list[::-14]:
        obsid, ws, band, keys, vals = get_dict_entry(
            entry=candidate,
            config_file_init=config_file_init,
            config_file_proc=config_file_proc
        )
        if obsid is not None:
            first_entry = candidate
            break
        n_attempts += 1
        logger.info(f"N_attempts = {n_attempts}")
        logger.info(f"error: {keys}, tb: {vals}")

    if first_entry is None or keys is None:
        logger.info("Could not find any successful entry to build schema from.")
        conn.close()
        return

    # Build table schema based on that first successful entry
    columns = ', '.join([f'"{k}" INTEGER' for k in keys])
    create_stmt = f'''
        CREATE TABLE IF NOT EXISTS results (
            obsid TEXT,
            ws TEXT,
            band TEXT,
            {columns},
            PRIMARY KEY (obsid, ws, band)
        )
    '''
    cur.execute(create_stmt)
    conn.commit()

    # Dump preprocess schema for that entry to JSON
    logger.info(f"Summarizing preprocess schema for {first_entry['obs:obs_id']} "
                f"{first_entry['dets:wafer_slot']} {first_entry['dets:wafer.bandpass']}")
    _, context_init_schema = safe_get_preprocess_context(config_file_init, logger=logger)
    _, context_proc_schema = safe_get_preprocess_context(config_file_proc, logger=logger)

    dets_schema = {
        'wafer_slot': first_entry['dets:wafer_slot'],
        'wafer.bandpass': first_entry['dets:wafer.bandpass'],
    }

    mdata_init_schema = context_init_schema.get_meta(first_entry['obs:obs_id'], dets=dets_schema)
    mdata_proc_schema = context_proc_schema.get_meta(first_entry['obs:obs_id'], dets=dets_schema)

    schema_out = sqlite_path.replace(".sqlite", "_preprocess_schema.json")
    dump_preprocess_schema(
        mdata_init_schema.preprocess,
        mdata_proc_schema.preprocess,
        schema_out,
        logger=logger
    )

    del proc
    logger.info('deleted database connection')

    n = 0
    ntot = len(run_list)

    logger.info(f"Writing to sqlite file at {sqlite_path}")
    futures = [
        executor.submit(
            get_dict_entry,
            entry=entry,
            config_file_init=config_file_init,
            config_file_proc=config_file_proc
        )
        for entry in run_list
    ]

    for future in as_completed_callable(futures):
        try:
            obsid, ws, band, keys, vals = future.result()
            logger.info(f'{n}/{ntot}: Unpacked future for {ws}, {band}')
        except Exception as e:
            logger.info(f'{n}/{ntot}: Future unpack error.')
            errmsg = f'{type(e)}: {e}'
            tb = ''.join(traceback.format_tb(e.__traceback__))
            with open(errlog, 'a') as f:
                f.write(f'\n{time.time()}, future.result() error\n{errmsg}\n{tb}\n')
            continue

        futures.remove(future)

        if obsid is None:
            logger.info('Writing error to log.')
            with open(errlog, 'a') as f:
                f.write(f'\n{time.time()}, error\n{keys}\n{vals}\n')
        else:
            try:
                col_names = ['obsid', 'ws', 'band'] + list(keys)
                placeholders = ','.join(['?'] * len(col_names))

                vals_to_store = [
                    (int(v) if isinstance(v, (np.integer, np.int64, np.int32)) else v)
                    for v in vals
                ]
                row_values = [obsid, ws, band] + vals_to_store
                insert_stmt = (
                    f'INSERT OR REPLACE INTO results '
                    f'({",".join(col_names)}) VALUES ({placeholders})'
                )
                cur.execute(insert_stmt, row_values)
                conn.commit()
                logger.info(f'{n}/{ntot}: Finished with {obsid} {ws} {band}.')
            except Exception as e:
                logger.info('Packaging and saving error.')
                errmsg = f'{type(e)}: {e}'
                tb = ''.join(traceback.format_tb(e.__traceback__))
                with open(errlog, 'a') as f:
                    f.write(f'\n{time.time()}, future.result() error\n{errmsg}\n{tb}\n')
                continue

        n += 1

    logger.info(f"All entries written to sqlite file at {sqlite_path}")
    conn.close()


# ==================== CLI entrypoint ==================== #

if __name__ == '__main__':
    args = get_parser().parse_args()
    rank, executor, as_completed_callable = get_exec_env(args.nproc)
    if rank == 0:
        main(
            executor=executor,
            as_completed_callable=as_completed_callable,
            **vars(args)
        )
