# SkyPortal Listener
SkyPortal Listener is a Python script designed to monitor new astronomical sources from the SkyPortal API, process their spectra using a machine learning model, and output classification results.


## Features
- Fetches new sources with spectra from SkyPortal.
- Processes spectra using a pre-trained ONNX model.
- Outputs classification probabilities.

## Requirements
- Python 3.8+
- Required Python libraries: requests, numpy, onnxruntime, scipy.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/antoine-le-calloch/Skyportal-listener.git
   cd Skyportal-listener
   ```
2. Install dependencies

## Usage
Run the script with the following command:
```bash
python skyportal_listener.py --token YOUR_API_TOKEN --start-time '2025-05-15T00:00:00Z'
```

### Arguments
- **`--instance`**: SkyPortal instance URL (default: https://fritz.science).
- **`--token`**: API token (required).
- **`--interval`**: Polling interval in seconds (default: 120).
- **`--start-time`**: Start time in ISO format, e.g. '2025-05-15T00:00:00Z' (default: 1 day ago)
