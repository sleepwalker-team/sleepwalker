#!/bin/env python3

import multiprocessing as mp
from sleepwalker.utils import logger, suppress_stdout_logging
from sleepwalker.core.signal import fix_edf_header
from sleepwalker.datasets.utils import get_edf_files_in_repo

def fix_edf_headers_no_out(e):
    with suppress_stdout_logging(logger):
        fix_edf_header(e)

edf_files = get_edf_files_in_repo("/raid/sleepwalker/ruhrlandklinik/raw", recursive=True)
print(f"Found {len(edf_files)} files")

num_workers = 16
 
logger.progress_start(len(edf_files), desc="Fixing files", leave=True)
if num_workers > 1:
    ctx = mp.get_context("fork")
    with ctx.Pool(num_workers) as pool:
        for _ in pool.imap_unordered(fix_edf_headers_no_out, edf_files):
            logger.progress_advance(1)
else:
    for e in edf_files:
        fix_edf_headers_no_out(e)
        logger.progress_advance(1)
logger.progress_close()

# for e in tqdm.tqdm(edf_files, desc="Fixing files"):
#     fix_edf_header(e,dry=False)