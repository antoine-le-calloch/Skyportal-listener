import argparse

from api import SkyPortal
from spectra_listener import monitor_spectra

INSTANCE_URL = "https://fritz.science"
API_TOKEN = ""  # Replace with your actual API token
POLL_INTERVAL = 120  # Polling interval in seconds
LOOKBACK_DAYS = 1

seen_sources = set() # Track seen source ids to avoid duplication
header = {}

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "SkyPortal source listener\nExecutes an ML model on new spectra and prints the results.\n\n"
            "Example:\n"
            "  python skyportal_listener.py --token YOUR_API_TOKEN --lookback 2\n\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("--instance", type=str, default=INSTANCE_URL,
                        help="SkyPortal instance URL (default: %(default)s)")
    parser.add_argument("--token", type=str, help="API token")
    parser.add_argument("--interval", type=int, default=POLL_INTERVAL,
                        help="Polling interval in seconds (default: %(default)s)")
    parser.add_argument("--lookback", type=int, default=LOOKBACK_DAYS,
                        help="Number of days to look back for new spectra (default: %(default)s)")
    return parser.parse_args()


if __name__ == "__main__":
    # Override default values with command line arguments
    args = parse_args()
    API_TOKEN = args.token or API_TOKEN
    INSTANCE_URL = args.instance
    POLL_INTERVAL = args.interval
    LOOPBACK_DAYS = args.lookback

    if not API_TOKEN:
        print("API token is required. Please provide it using --token.")
        exit(1)
    else:
        print(f"Starting SkyPortal listener...")
    header = {"Authorization": f"token {API_TOKEN}"}

    client = SkyPortal(
        instance=INSTANCE_URL,
        port=443,
        token=API_TOKEN,
    )

    # start monitoring the spectra
    monitor_spectra(
        client,
        instrument_ids=[7, 9, 35, 2, 26, 3, 1117, 1108],
        # Ids correspond to: LRIS, KAST, SPRAT, SEDM, ALFOSC, DBSP, NGPS, GHTS TODO- add KCWI (1102) and Binospec (1076)
        lookback=LOOPBACK_DAYS,
        interval=POLL_INTERVAL,
        publish_to_skyportal=False,
        verbose=True,
        use_cache=True,
        clear_cache=False,
        cache_dir='cache'
    )


