import numpy as np
import onnxruntime as ort

from scipy.interpolate import interp1d
from scipy.special import softmax

def flux_zscore(spectra, wavelength_range=(3850, 8500), interp_length=4650):
    try:
        wavelengths = np.array(spectra['wavelengths'], dtype=np.float64)
        fluxes = np.array(spectra['fluxes'], dtype=np.float64)
    except Exception as e:
        raise ValueError(f"Invalid input data in spectra: {e}")

    if len(wavelengths) != len(fluxes):
        raise ValueError("Mismatched lengths between wavelengths and fluxes")

    mask = np.isfinite(wavelengths) & np.isfinite(fluxes)
    if not np.any(mask):
        raise ValueError("No finite values in spectrum")

    wavelengths_cleaned = wavelengths[mask]
    fluxes_cleaned = fluxes[mask]

    if len(wavelengths_cleaned) < 2:
        raise ValueError("Too few data points after cleaning")

    xs = np.linspace(wavelength_range[0], wavelength_range[1], interp_length)
    ys = interp1d(wavelengths_cleaned, fluxes_cleaned, kind='linear', bounds_error=False,
                  fill_value=(fluxes_cleaned[0], fluxes_cleaned[-1]))(xs)

    m, s = np.nanmean(ys), np.nanstd(ys)
    ys_norm = (ys - m) / s if s > 0 else np.zeros(interp_length, dtype=float)
    return ys_norm

def process_spectra(spectra):
    input = flux_zscore(spectra)
    input_data = np.array(input.reshape(1, 1, len(input)).astype(np.float32))

    ort_session = ort.InferenceSession("SpectraCNN1D_4650.onnx")
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    ort_outputs = ort_session.run(None, ort_inputs)
    onnx_probs = ort_outputs[0]
    onnx_probs_scipy = softmax(onnx_probs[0])
    classes = ['AGN', 'Cataclysmic', 'II', 'IIP', 'IIb',
               'IIn', 'Ia', 'Ib', 'Ic', 'Tidal Disruption Event']

    output_dict = dict(zip(classes, onnx_probs_scipy.tolist()))
    return output_dict