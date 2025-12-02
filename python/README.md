# Cilia Beat Frequency (CBF), Python implementation

This folder contains a small collection of scripts for analyzing cilia motion from TIFF image sequences. **The benefit of using this over the Matlab implementation are the optimization widgets.** The main workflow is:

- Generating a binary mask to exclude background/empty regions (`binary_mask.py`).
- Running a Fourier-based cilia beat frequency analysis (`cbf_main.py`).
- Saving results as image overlays and histograms of cilia beat frequency

This README explains how to install dependencies and run the scripts.

## Requirements

Requirements can be found in the environment.yml file. If you use conda, feel free to do the following to install the environment:

```bash
conda env create -f environment.yml
conda activate cilia-beat-frequency
```

## Quick usage examples

Input: a directory containing TIFF files which will be naturally sorted (2D frames) or a directory holding a single 3D TIFF stack. If you have a single 3D TIFF stack, pass `--tiff_stack`. 


```bash
# Example with given frame_rate and already generated binary mask
python cbf_main.py /path/to/images_directory --frame_rate 100 --binary_mask /path/to/binary_mask.npy --output_path /path/to/outdir
# Example with tiff stack
python cbf_main.py /path/to/directory/with/single_3d_tiff --tiff_stack
```

Options:
- `--frame_rate` (float) — sampling rate (Hz).
- `--binary_mask` — path to a saved `.npy` file which must be same dimensions as image. If omitted, `cbf_main.py` will walk you through creating or selecting a mask interactively
- `--tiff_stack` — pass this if your images are in a single 3D TIFF file inside the given directory. Tiff stacks must be the only file in the directory (TODO: fix this)
- `--output_path` — destination directory for all saved results. If omitted, a default directory named **cilia_beat_frequency** one level up from the image directory will be used.

Options prompted via command-line:
- Clipping: if you wish to clip your time-series (ie. omit first frame because it is noise)
- Expected signal range: this algorithm relies on as expected range of frequencies you expect to find. For example, the study cited in the previous README.md used a max of 25 Hz as their literature review suggested that mice cilia beat frequencies would be below that value. This script would implement that by inputting "0 25".


## Output

- `significant_pixels_overlay{num_std}.tiff` — grayscale overlay with significant pixels colored green.
- `significant_pixel_freqs_overlay_cmap{num_std}.tiff` — colormap overlay showing dominant frequencies as a heatmap.
- `significant_pixel_freqs_cmap{num_std}.tiff` — colormap image containing only significant pixels.
- `colorscale{num_std}.tiff` — colorbar image used to interpret the colormap.
- `pixel_dominant_frequency{num_std}.npy` — raw NumPy array of per-pixel dominant frequency (NaN for non-significant pixels).
- `dominant_frequency_histogram{num_std}.png` and `dominant_frequency_density{num_std}` - dominant frequency histogram and density plot

## Widgets and notes on compatibility

This tool must be used in command line interface (CLI) with python, as these widgets were not developed to be used inside a .ipynb jupter notebook file.

## TODO

- Refactorization 
- Implement all options as CLI arguments or inputs (choose one)
- Output file with all saved parameters
