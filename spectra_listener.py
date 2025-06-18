import os
import time
import traceback
from datetime import datetime, timedelta, timezone

from api import SkyPortal
from execute_model import process_spectra

default_cache_name = 'spectra_listener_cache.txt'

def validate_monitor_spectra_args(
        client: SkyPortal,
        instrument_ids: list[int],
        lookback: int,
        interval: int,
        use_cache: bool,
        cache_dir: str,
):
    """
    Validate the arguments for the monitor_spectra function

    Parameters
    ----------
    client : SkyPortal
        SkyPortal API client
    instrument_ids : list[int]
        List of instrument IDs to monitor
    lookback : int
        Number of days to look back for new spectra
    interval : int
        Number of seconds to wait between queries
    use_cache : bool
        If True, cache processed spectra
    cache_dir : str
        Path to the cache directory
    function : callable
        Function to call on each new spectrum
    """
    if not isinstance(client, SkyPortal):
        raise ValueError('client must be a SkyPortal instance')

    if not isinstance(instrument_ids, list) or not all(isinstance(i, int) for i in instrument_ids):
        raise ValueError('instrument_ids must be a list of integers')

    if not isinstance(lookback, int) or lookback < 0:
        raise ValueError('lookback must be a non-negative integer')

    if not isinstance(interval, int) or interval < 0:
        raise ValueError('interval must be a non-negative integer')

    if use_cache:
        if not isinstance(cache_dir, str):
            raise ValueError('cache_dir must be a string if use_cache is True')

        if not os.path.exists(cache_dir):
            try:
                os.makedirs(cache_dir)
            except OSError as e:
                raise ValueError(f'Error creating cache directory: {e}')

    return True


def str_to_bool(value: bool):
    """
    Convert a bool-like argument to a boolean

    Parameters
    ----------
    value : bool, str, int
        Value argument

    Returns
    -------
    bool
        Value argument as a boolean
    """
    if value in ['True', 'true', 'T', 't', 'Yes', 'yes', 'Y', 'y', '1', 1, True]:
        return True
    else:
        return False


def _load_existing_cache(cache_dir: str, function_name: str):
    """
    Load the existing cache from disk

    Parameters
    ----------
    cache_dir : str
        Path to the cache directory
    function_name : str
        Name of the function called on the spectra

    Returns
    -------
    Set[int]
        Spectrum IDs that have already been processed by the function
    """
    cache_name = f'{function_name}_{default_cache_name}'
    try:
        with open(f'{cache_dir}/{cache_name}', 'r') as f:
            return set([int(line) for line in f.read().splitlines()])
    except FileNotFoundError:
        return set()


def _cache_spectra(id: int, cache_dir: str, function_name: str):
    """
    Add a spectrum ID to the cache on disk to avoid duplicates in the future

    Parameters
    ----------
    id : int
        Spectrum ID
    cache_dir : str
        Path to the cache directory
    function_name : str
        Name of the function called on the spectra
    """
    cache_name = f'{function_name}_{default_cache_name}'
    # if the id is not in the cache yet, add it
    already_processed = _load_existing_cache(cache_dir, function_name)
    if id not in already_processed:
        with open(f'{cache_dir}/{cache_name}', 'a') as f:
            f.write(f'{id}\n')


def _clear_cache(cache_dir: str, function_name: str):
    """
    Clear the cache on disk

    Parameters
    ----------
    cache_dir : str
        Path to the cache directory
    function_name : str
        Name of the function called on the spectra
    """
    cache_name = f'{function_name}_{default_cache_name}'
    with open(f'{cache_dir}/{cache_name}', 'w') as f:
        f.write('')


def monitor_spectra(
        client: SkyPortal,
        instrument_ids: list[int],
        lookback: int,
        interval: int,
        verbose: bool,
        use_cache: bool,
        cache_dir: str,
        clear_cache: bool,
        *args, **kwargs
):
    """
    Monitor SkyPortal for new spectra taken with the specified instruments
    and call the specified function on each new spectrum.
    Keep track of already processed spectra to avoid duplicates.

    Parameters
    ----------
    client : SkyPortal
        SkyPortal API client
    instrument_ids : list[int]
        List of instrument IDs to monitor
    lookback : int
        Number of days to look back for new spectra
    interval : int
        Number of seconds to wait between queries
    verbose : bool
        If True, print information at each step
    use_cache : bool
        If True, cache processed spectra
    cache_dir : str
        Path to the cache directory
    clear_cache : bool
        If True, clear the cache before starting
    args : list
        Positional arguments to pass to the function
    kwargs : dict
        Keyword arguments to pass to the function
    """
    verbose = str_to_bool(verbose)
    use_cache = str_to_bool(use_cache)
    clear_cache = str_to_bool(clear_cache)
    validate_monitor_spectra_args(client, instrument_ids, lookback, interval, use_cache, cache_dir)

    if use_cache:
        if clear_cache:
            _clear_cache(cache_dir, "process_spectra")
        already_processed = _load_existing_cache(cache_dir, "process_spectra")
    else:
        already_processed = set()

    while True:
        modified_before = datetime.now(timezone.utc)
        modified_after = modified_before - timedelta(days=1)
        status, data = client.get_spectra(
            instrument_ids=instrument_ids,
            modified_after=modified_after.isoformat().split('+')[0],
            modified_before=modified_before.isoformat().split('+')[0],
            minimal=True
        )
        if status != 200:
            print(f'Error fetching spectra: {data}')
            time.sleep(10)
            continue

        all_spectra = [s for s in data['data'] if s['id'] not in already_processed]
        if verbose and len(all_spectra) > 0:
            print(f'Found {len(all_spectra)} new spectra at {modified_before}')

        start = time.time()
        for s in all_spectra:
            try:
                print(f'New spectra: {s["id"]}')
                status, data = client.get_spectra(id=s['id'])
                if status != 200:
                    raise ValueError(f'Error fetching spectra {s["id"]}: {data}')
                process_spectra(data['data'])

                if use_cache:
                    _cache_spectra(s['id'], cache_dir, "process_spectra")
                already_processed.add(s['id'])
            except Exception as e:
                traceback.print_exc()
                print(f'Error processing spectra {s["id"]}: {e}')

        if verbose:
            if len(all_spectra) > 0:
                print(
                    f'Processed {len(already_processed)} spectra in {time.time() - start:.2f} seconds (sleeping for {interval} seconds)')
            else:
                print(
                    f'No new spectra found between {modified_after} and {modified_before} (sleeping for {interval} seconds)')

        time.sleep(interval)