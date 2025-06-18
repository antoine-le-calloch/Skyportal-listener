import requests
import argparse
import time
import numpy as np
import onnxruntime as ort

from datetime import datetime, timezone, timedelta
from scipy.interpolate import interp1d
from scipy.special import softmax

INSTANCE_URL = "https://fritz.science"
API_TOKEN = "YOUR_API_TOKEN"  # Replace with your actual API token
POLL_INTERVAL = 120  # Polling interval in seconds

seen_sources = set() # Track seen source ids to avoid duplication
header = {}

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "SkyPortal source listener\nExecutes an ML model on new spectra and prints the results.\n\n"
            "Example:\n"
            "  python skyportal_listener.py --token YOUR_API_TOKEN --start-time '2025-05-15T00:00:00Z'\n\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("--instance", type=str, default=INSTANCE_URL,
                        help="SkyPortal instance URL (default: %(default)s)")
    parser.add_argument("--token", type=str, help="API token")
    parser.add_argument("--interval", type=int, default=POLL_INTERVAL,
                        help="Polling interval in seconds (default: %(default)s)")
    parser.add_argument("--start-time", type=str,
                        help="Start UTC time for fetching new sources in ISO format, e.g. '2025-05-15T00:00:00Z' (default: 1 day ago)"
                        )
    return parser.parse_args()


def flux_zscore(spectra, wavelength_range=(3850, 8500), interp_length=4650):
    wavelengths = spectra[0]['wavelengths']
    fluxes = spectra[0]['fluxes']

    combined = np.stack((wavelengths, fluxes), axis=1)
    # mask = ~np.isfinite(combined).any(axis=1)
    # cleaned_data = combined[mask]

    wavelengths_cleaned = combined[:,0]
    fluxes_cleaned = combined[:,1]

    xs = np.linspace(wavelength_range[0], wavelength_range[1], interp_length)
    ys = interp1d(wavelengths_cleaned, fluxes_cleaned, kind='linear', bounds_error=False, 
                fill_value=(fluxes_cleaned[0], fluxes_cleaned[-1]))(xs)
        
    m, s = np.nanmean(ys), np.nanstd(ys)
    ys_norm = (ys - m) / s if s > 0 else np.zeros(interp_length, dtype=float)
    return ys_norm


def get_spectra(source_id):
    response = requests.get(f"{INSTANCE_URL}/api/sources/{source_id}/spectra", headers=header)
    response.raise_for_status()
    return response.json()['data']['spectra']


def get_new_sources(start_time, last_iteration):
    """Fetch new sources from SkyPortal."""
    since = last_iteration if last_iteration else start_time
    # Update last iteration before fetching new sources to avoid missing any sources
    last_iteration = datetime.now(timezone.utc)

    response = requests.get(f"{INSTANCE_URL}/api/sources",
        params={"savedAfter": since.isoformat(), "hasSpectrum": True},
        headers=header)
    response.raise_for_status()

    sources = response.json()["data"]["sources"]
    new_sources = []
    for source in sources:
        if source["id"] not in seen_sources:
            new_sources.append(source)
            seen_sources.add(source["id"])
    return new_sources, last_iteration


def process_source(source):
    spectra = get_spectra(source['id'])
    if not spectra:
        return

    input = flux_zscore(spectra)
    input_data = np.array(input.reshape(1, 1, len(input)).astype(np.float32))

    ort_session = ort.InferenceSession("SpectraCNN1D_4650.onnx")
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    ort_outputs = ort_session.run(None, ort_inputs)
    onnx_probs = ort_outputs[0]
    onnx_probs_scipy = softmax(onnx_probs[0])
    classes = ['AGN', 'Cataclysmic', 'SN II', 'SN IIP', 'SN IIb',
               'SN IIn', 'SN Ia', 'SN Ib', 'SN Ic', 'Tidal Disruption Event']

    output_dict = dict(zip(classes, onnx_probs_scipy.tolist()))
    print(output_dict)


def main_loop(start_time):
    last_iteration = None
    print("Starting SkyPortal source listener...")

    while True:
        try:
            new_sources, last_iteration = get_new_sources(start_time, last_iteration)
            for source in new_sources:
                try:
                    print(f"New source detected: {source['id']}")
                    process_source(source)
                    time.sleep(0.3)  # Sleep to avoid overwhelming the server
                except Exception as e:
                    print(f"Error processing source: {e}")
                    print("Waiting for 5 seconds and skipping to the next source...")
                    time.sleep(5)
            if new_sources:
                print("Listening for new sources...")
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    args = parse_args()

    # Override default values with command line arguments
    INSTANCE_URL = args.instance
    API_TOKEN = args.token or API_TOKEN
    POLL_INTERVAL = args.interval

    if not API_TOKEN:
        print("API token is required. Please provide it using --token.")
        exit(1)
    header = {"Authorization": f"token {API_TOKEN}"}

    if args.start_time:
        try:
            start_time = datetime.fromisoformat(args.start_time.replace("Z", "+00:00"))
        except ValueError:
            print("Invalid start time format. Use ISO format, e.g. '2025-10-20T00:00:00Z'")
            exit(1)
    else:
        start_time = datetime.now(timezone.utc) - timedelta(days=1)

    main_loop(start_time)