import os
import argparse
from preprocessing_utils import *
from visual_utils import *
import numpy as np
import time
from matplotlib.colors import Normalize
from matplotlib import cm
import matplotlib.colors
from scipy.ndimage import median_filter
from PIL import Image

def cbf_fft(image_series, frame_rate, signal_range, vis_image=None):
    """
    Computes Fourier transform of the image series and then extracts only necessary information for analysis.
    Args:
        image_series (numpy array): 3D array of shape (T, H, W) where T is the number of frames.
        frame_rate (float): Frame rate of the image series.
        expected_signal (tuple): Expected signal frequency range in Hz.
    Returns:
        tuple: (result_array, num_std)
            result_array: (H, W) array with significant pixels != 0. Significant pixels are filled with corresponding freqeuncy values.
            num_std: The value used for the significance threshold.
    """

    # Computer Fourier transform and power spectrum density
    fft = np.fft.fft(image_series-np.average(image_series, axis=0), axis=0)
    print(f"Successfully computed fft.")
    spectra = 2 * np.abs(fft)
    freq = np.fft.fftfreq(len(fft), d = 1/frame_rate)

    # Using expected signal frequency range, calculate noise_mean and noise_std
    signal_max = spectra[(freq > signal_range[0]) & (freq <= signal_range[1]), :, :].max(axis=0) # Stores the maximum amplitude for expected signal range
    frequency_max_index = spectra[(freq >= signal_range[0]) & (freq <= signal_range[1]), :, :].argmax(axis=0) # Stores the corresponding frequency for the maximum amplitude in the expected signal range
    frequency_max = freq[frequency_max_index] + signal_range[0]
    noise_max = spectra[(freq >= signal_range[1]) & (freq <= max(freq)), :, :].max(axis=0) # Stores the maximum amplitude for frequencies above the expected signal range
    noise_max = noise_max[noise_max != 0]  # Filter out all values of 0
    noise_max = noise_max.reshape(-1,1) # Reshape for next calculations
    noise_mean = np.mean(noise_max)
    noise_std = np.std(noise_max)

    # Test up to this point
    print(f"Non-zero values in binary mask: {np.count_nonzero(mask)}")
    print(f"Signal max shape: {signal_max.shape}, Noise max shape: {noise_max.shape}, Noise mean: {noise_mean}, Noise std: {noise_std}")
    # Widgetized num_std optimization:
        # display image: last image of image series
        # num_std, make a button to increase or decrease num_std by 0.25. Min 0 and max 10.
        # For all pixels in image, display pixel as green if the amplitude of the max frequency is > mean(noise_max) + num_std * std(noise_max)
    num_std = optimize_num_std_widget(image_series, signal_max, noise_mean, noise_std, vis_image=vis_image)

    result_array = np.where(signal_max > noise_mean + num_std * noise_std, signal_max, 0)
    # Apply 4x4 median filter to reduce noise
    result_array = median_filter(result_array, size=(4, 4))
    result_array = np.where(result_array > 0, frequency_max, np.nan)
    
    return result_array, num_std

def optimize_num_std_widget(image_series, signal_max, noise_mean, noise_std, vis_image=None):
    """
    Interactive widget to optimize num_std for significance thresholding.
    Shows the last image with green overlay for significant pixels.
    Returns the chosen num_std value.
    """
    if vis_image is None:
        vis_image = image_series[-1]
    num_std_init = 1.0

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.40)
    sig_mask = (signal_max > noise_mean + num_std_init * noise_std)
    rgb_img = np.stack([vis_image]*3, axis=-1)  # Turn display image into RGB
    rgb_img[sig_mask] = [0, 255, 0]  # Highlight significant pixels in green
    img_disp = ax.imshow(rgb_img)
    ax.set_title(f"num_std: {num_std_init:.2f}")
    ax.axis('off')

    fig.text(0.5, 0.05, f"Note: Pixels shown in green are significant at the selected num_std value.", ha='center', va='center', fontsize=10, color='red')

    axcolor = 'lightgoldenrodyellow'
    ax_slider = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
    slider = Slider(ax_slider, 'num_std', 0, 10, valinit=num_std_init, valstep=0.25)

    ax_finalize = plt.axes([0.7, 0.18, 0.2, 0.04])
    btn_finalize = Button(ax_finalize, 'Finalize')

    finalized = {'value': None}

    def update(val):
        num_std = slider.val
        sig_mask = (signal_max > noise_mean + num_std * noise_std)
        rgb_img = np.stack([vis_image]*3, axis=-1)  # Turn display image into RGB
        rgb_img[sig_mask] = [0, 255, 0]  # Highlight significant pixels in green
        img_disp.set_data(rgb_img)
        ax.set_title(f"num_std: {num_std:.2f}")
        fig.canvas.draw_idle()

    def finalize_clicked(event):
        finalized['value'] = slider.val
        plt.close(fig)

    slider.on_changed(update)
    btn_finalize.on_clicked(finalize_clicked)

    plt.show()

    while finalized['value'] is None:
        plt.pause(0.1)
    return finalized['value']

def prompt_signal_range(frame_rate):
    max_freq = frame_rate / 2
    print(f"\nEnter the expected signal frequency range in Hz (min and max, separated by a space).")
    print(f"Values must be between 0 and {max_freq:.2f} Hz to satisfy Nyquist sampling.")
    while True:
        try:
            user_input = input(f"Expected signal range (e.g., 5 35): ").strip()
            parts = user_input.split()
            if len(parts) != 2:
                print("Please enter two numbers separated by a space.")
                continue
            low, high = float(parts[0]), float(parts[1])
            if not (0 <= low < high <= max_freq):
                print(f"Values must satisfy 0 <= min < max <= {max_freq:.2f}. Try again.")
                continue
            return (low, high)
        except ValueError:
            print("Invalid input. Please enter two numeric values.")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run celia beat frequency analysis.")
    parser.add_argument("path", type=str, help="Path to the directory containing TIFF images.")
    parser.add_argument("--frame_rate", type=float, help="Frame rate for the output movie.")
    parser.add_argument("--binary_mask", type=str, help="Path to the binary mask file. If not provided, will walk user through to obtain a new mask.")
    parser.add_argument( "--output_path", type=str, default=None, help="Path to save the output movie. Defaults to one directory back from 'path' in a new directory called 'cilia_beat_frequency'.")
    parser.add_argument("--tiff_stack", action='store_true', help="Specify if the input is a single 3D TIFF file instead of a directory of 2D TIFF files. Note: an entire directory must be handed and this should be the only .tiff file in there.")

    args = parser.parse_args()

    # Validate path input
    if not os.path.isdir(args.path):
        raise ValueError(f"The specified path {args.path} is not a valid directory.")

    # Load in tiff image series
    print("Loading TIFF image series...")
    # If args.tiff_stack is True, load a single 3D TIFF file
    if args.tiff_stack:
        image_series = load_tiff_3d(os.path.join(args.path, [f for f in os.listdir(args.path) if f.endswith('.tiff') or f.endswith('.tif')][0]))
    # First, see if image_series is already defined (to avoid loading it twice)
    if 'image_series' not in locals():
        # If not defined, load the image series
        image_series = load_tiff_series(args.path)
        print(f"Loaded image series with shape: {image_series.shape}")

    # Handle multi-channel TIFFs (RGB, RGBA, etc.) by converting to grayscale
    # Weighted average for RGB to grayscale is found here: https://stackoverflow.com/questions/687261/converting-rgb-to-grayscale-intensity
    if image_series.ndim == 4:
        num_channels = image_series.shape[-1]
        if num_channels == 3:
            # RGB: Use weighted average for proper grayscale conversion
            image_series = (0.299 * image_series[..., 0] + 
                          0.587 * image_series[..., 1] + 
                          0.114 * image_series[..., 2]).astype(image_series.dtype)
            print(f"RGB image detected. Converted to grayscale using luminance weights. New shape: {image_series.shape}")
        elif num_channels == 4:
            # RGBA: Ignore alpha channel, use RGB with weighted average
            image_series = (0.299 * image_series[..., 0] + 
                          0.587 * image_series[..., 1] + 
                          0.114 * image_series[..., 2]).astype(image_series.dtype)
            print(f"RGBA image detected. Converted to grayscale (alpha channel ignored). New shape: {image_series.shape}")
        else:
            # Other multi-channel: Just take first channel and warn user
            image_series = image_series[..., 0]
            print(f"WARNING: {num_channels}-channel image detected. Using only first channel. New shape: {image_series.shape}")

    # Save first_img for visualization before applying mask
    first_img = image_series[1].copy()

    # Ask user if they want to clip their series
    clip_choice = input("\nDo you want to clip your time series? (y/n): ").strip().lower()
    if clip_choice == 'y':
        clip_indices = input("Enter two indices to clip to, separated by a comma (e.g., 0,50): ").strip()
        try:
            start_idx, end_idx = map(int, clip_indices.split(','))
            image_series = image_series[start_idx:end_idx]
            print(f"Time series clipped to indices [{start_idx}:{end_idx}].")
        except Exception as e:
            print(f"Invalid input for clipping indices: {e}")
    else:
        print("Skipping time series clipping.")

    # Load or generate binary mask
    if args.binary_mask is not None:
        if not os.path.isfile(args.binary_mask):
            raise ValueError(f"The specified binary mask {args.binary_mask} is not a valid file.")
        mask = np.load(args.binary_mask)
    else:
        # Search for .npy files in the directory
        npy_files = [f for f in os.listdir(args.path) if f.endswith('.npy')]
        selected_mask = None

        if npy_files:
            print("Found the following .npy files in the directory:")
            for idx, fname in enumerate(npy_files):
                print(f"{idx+1}: {fname}")
            print("Enter the number of the mask to use, or 0 to generate a new one:")
            while True:
                try:
                    choice = int(input("Your choice: ").strip())
                    if 0 <= choice <= len(npy_files):
                        break
                    else:
                        print("Invalid selection. Try again.")
                except ValueError:
                    print("Please enter a valid number.")
            if choice > 0:
                selected_mask = os.path.join(args.path, npy_files[choice-1])
                mask = np.load(selected_mask)
        if not selected_mask:
            # If args.tiff_stack is True, load a single 3D TIFF file
            if args.tiff_stack:
                if clip_choice == 'y':
                    image_series = load_tiff_3d(os.path.join(args.path, [f for f in os.listdir(args.path) if f.endswith('.tiff') or f.endswith('.tif')][0]))
                    image_series = image_series[start_idx:end_idx]
                else:
                    image_series = load_tiff_3d(os.path.join(args.path, [f for f in os.listdir(args.path) if f.endswith('.tiff') or f.endswith('.tif')][0]))
            else:
                if clip_choice == 'y':
                    image_series = load_tiff_series(args.path)
                    image_series = image_series[start_idx:end_idx]
                else:
                    image_series = load_tiff_series(args.path)
            print(f"Loaded image series with shape: {image_series.shape}")
            threshold, invert = visualize_binary_mask(image_series)
            mask = binary_mask(image_series, threshold, invert)

    # Automatically get frame_rate from image series name if not provided
    if args.frame_rate is None:
        args.frame_rate = get_frame_rate(args.path)
    # If the frame rate is not an integer, that might be strange, so let's ask the user
    if not isinstance(args.frame_rate, int):
        print(f"Frame rate is {args.frame_rate} Hz. Is this correct? (y/n)")
        response = input().strip().lower()
        if response != 'y':
            while True:
                try:
                    new_rate = float(input("Please enter the correct frame rate (Hz): ").strip())
                    args.frame_rate = new_rate
                    print(f"Frame rate set to {args.frame_rate} Hz.")
                    break
                except ValueError:
                    print("Invalid input. Please enter a numeric value for the frame rate.")

    # Set default output_path if not provided
    if args.output_path is None or args.output_path == "":
        parent_dir = os.path.abspath(os.path.join(args.path, os.pardir))
        output_dir = os.path.join(parent_dir, "cilia_beat_frequency")
        args.output_path = output_dir

    # Create the output directory if it does not exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Ask user for expected signal frequency range
    signal_range = prompt_signal_range(args.frame_rate)

    # Check with user on all parameters before loading data and running fourier transform
    print("\nSummary of parameters:")
    print(f"Image directory: {args.path}")
    print(f"Frame rate: {args.frame_rate} Hz")
    print(f"Signal frequency range: {signal_range[0]} - {signal_range[1]} Hz")
    print(f"Binary mask: {'Loaded from file' if args.binary_mask or selected_mask else 'Generated'}")
    print(f"Output path: {args.output_path}")
    if clip_choice == 'y':
        print(f"Time series clipped to indices: [{start_idx}:{end_idx}]")
    print("Proceed with these settings? (y/n)")
    proceed = input().strip().lower()
    if proceed != 'y':
        print("Aborting analysis.")
        exit(0)

    # Apply binary mask to image series
    print("Applying binary mask to image series...")
    #expanded_mask = np.broadcast_to(mask, image_series.shape)
    #image_series = np.ma.masked_where(expanded_mask == 0, image_series)
    image_series *= mask
    print("Binary mask applied.")

    # Clip time_series if requested
    if clip_choice == 'y':
        image_series = image_series[start_idx:end_idx]

    # Run fourier transform on the masked image series
    print("Running Fourier Transform on the masked image series...")
    start_time = time.time()
    result_array, num_std = cbf_fft(image_series, args.frame_rate, signal_range, vis_image=first_img)
    elapsed = time.time() - start_time
    print(f"Fourier Transform and num_std optimization completed in {elapsed:.2f} seconds.")

    # Saving results as images
    print(f"Saving results as images in output directory {args.output_path}...")

    # 1. Save overlay with significant pixels in color [0, 255, 0] as TIFF
    sig_mask = ~np.isnan(result_array)
    rgb_img = np.stack([first_img]*3, axis=-1).astype(np.uint8)
    rgb_img[sig_mask] = [0, 255, 0]
    Image.fromarray(rgb_img).save(os.path.join(args.output_path, "significant_pixels_overlay" + str(num_std) + ".tiff"))

    # 2. Save overlay with significant pixels as colormap as TIFF
    masked_freq = np.ma.masked_invalid(result_array)
    vmin, vmax = np.nanmin(result_array), np.nanmax(result_array)
    cmap = matplotlib.colormaps.get_cmap('plasma')
    normed_freq = (masked_freq - vmin) / (vmax - vmin)
    cmap_img = cmap(normed_freq.filled(0))[:, :, :3]  # RGB only
    cmap_img = (cmap_img * 255).astype(np.uint8)
    # Overlay on grayscale first_img
    gray_img = np.stack([first_img]*3, axis=-1).astype(np.uint8)
    overlay_img = gray_img.copy()
    overlay_img[sig_mask] = cmap_img[sig_mask]
    Image.fromarray(overlay_img).save(os.path.join(args.output_path, "significant_pixel_freqs_overlay_cmap" + str(num_std) + ".tiff"))

    # 3. Save image with only significant pixels as colormap (no background) as TIFF
    cmap_only = np.zeros_like(cmap_img)
    cmap_only[sig_mask] = cmap_img[sig_mask]
    Image.fromarray(cmap_only).save(os.path.join(args.output_path, "significant_pixel_freqs_cmap" + str(num_std) + ".tiff"))

    # 4. Save the colorscale as a separate image (vertical bar) with numerical ticks
    fig, ax = plt.subplots(figsize=(1, 6))
    fig.subplots_adjust(left=0.5, right=0.9, top=0.95, bottom=0.05)
    cmap = matplotlib.colormaps.get_cmap('plasma')
    norm = Normalize(vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax, orientation='vertical'
    )
    cbar.set_label('Dominant Frequency (Hz)')
    num_ticks = 8
    tick_locs = np.linspace(vmin, vmax, num_ticks)
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels([f"{tick:.2f}" for tick in tick_locs])
    dark_mode = True # Set to True if you want dark mode
    if dark_mode:
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        cbar.ax.tick_params(colors='white', labelsize=12)
        cbar.ax.xaxis.label.set_color('white')
        cbar.set_label('Dominant Frequency (Hz)', color='white')
    else:
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        cbar.ax.tick_params(colors='black', labelsize=12)
        cbar.ax.xaxis.label.set_color('black')
    # Make sure the axis is visible for ticks and labels
    ax.set_axis_on()
    ax.get_xaxis().set_visible(True)
    plt.savefig(os.path.join(args.output_path, "colorscale" + str(num_std) + ".tiff"), bbox_inches='tight', pad_inches=0.3, facecolor=fig.get_facecolor())
    plt.close()

    # 5. Save a numpy matrix of the pixel by dominant frequency as well as a corresponding histogram plot
    np.save(os.path.join(args.output_path, "pixel_dominant_frequency" + str(num_std) + ".npy"), result_array)
    plt.figure(figsize=(8, 6))
    plt.hist(result_array[~np.isnan(result_array)].flatten(), bins=50, color='blue', alpha=0.7)
    plt.xlabel('Dominant Frequency (Hz)')
    plt.ylabel('Number of Pixels')
    plt.title('Histogram of Dominant Frequencies')
    plt.savefig(os.path.join(args.output_path, "dominant_frequency_histogram" + str(num_std) + ".png"))
    plt.close()
    # And save a Gaussian density plot of this histogram
    from scipy.stats import gaussian_kde
    data = result_array[~np.isnan(result_array)].flatten()
    density = gaussian_kde(data)
    xs = np.linspace(np.min(data), np.max(data), 200)
    plt.figure(figsize=(8, 6))
    plt.plot(xs, density(xs), color='red')
    plt.fill_between(xs, density(xs), color='red', alpha=0.5)
    plt.xlabel('Dominant Frequency (Hz)')
    plt.ylabel('Density')
    plt.title('Gaussian Density Plot of Dominant Frequencies')
    plt.savefig(os.path.join(args.output_path, "dominant_frequency_density_plot" + str(num_std) + ".png"))
    plt.close()

    print(f"Finished saving results as images in output directory {args.output_path}...")

    # If all runs successfully, create a .txt file in the output directory with all parameters
    success_file = os.path.join(args.output_path, "success.txt")
    with open(success_file, "w") as f:
        f.write("Analysis completed successfully!\n\n")
        f.write("=== PARAMETERS ===\n")
        f.write(f"Image directory: {args.path}\n")
        f.write(f"TIFF stack mode: {args.tiff_stack}\n")
        f.write(f"Frame rate: {args.frame_rate} Hz\n")
        f.write(f"Signal frequency range: {signal_range[0]} - {signal_range[1]} Hz\n")
        f.write(f"Binary mask: {'Loaded from file' if args.binary_mask or selected_mask else 'Generated'}\n")
        if args.binary_mask:
            f.write(f"Binary mask path: {args.binary_mask}\n")
        elif selected_mask:
            f.write(f"Binary mask path: {selected_mask}\n")
        f.write(f"Output path: {args.output_path}\n")
        f.write(f"Image series shape: {image_series.shape}\n")
        if clip_choice == 'y':
            f.write(f"Time series clipped: Yes (indices {start_idx}:{end_idx})\n")
        else:
            f.write(f"Time series clipped: No\n")
        f.write(f"num_std threshold: {num_std}\n")
        f.write(f"Number of significant pixels: {np.count_nonzero(~np.isnan(result_array))}\n")
        f.write(f"Frequency range in results: {np.nanmin(result_array):.2f} - {np.nanmax(result_array):.2f} Hz\n")

# Example usage: python cbf_main.py /Users/josephbeller/Library/CloudStorage/Box-Box/Larina_team_folder/Joesph/Endometriosis_Organoids_data/070825die63p9_LSM02DC_4p4ms_2000x3000_0p5Vx0V_10ms_225khz_a/images
# Example with tiff stack: python cbf_main.py /path/to/directory/with/single_3d_tiff --tiff_stack
# Example with binary mask: python cbf_main.py /path/to/image_directory --binary_mask /path/to/mask.npy