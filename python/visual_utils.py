import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, RangeSlider, RectangleSelector  # Add RangeSlider and RectangleSelector
from preprocessing_utils import *

def visualize_binary_mask(image_series, brightness_gradient=False, notebook=False):
    """
    Visualize the binary mask for the average image in the series using matplotlib widgets.
    Args:
        image_series: 3D array of images
        brightness_gradient: Apply brightness gradient correction
        notebook: If True, use %matplotlib widget for Jupyter compatibility
    Returns:
        tuple: Finalized threshold value and invert flag.
    """
    if notebook:
        import matplotlib
        matplotlib.use('inline')
    
    avg_img = np.mean(image_series, axis=0)
    avg_img = avg_img.astype(np.uint8)
    if brightness_gradient:
        avg_img = apply_brightness_gradient_correction(avg_img)
    threshold_init = 0
    invert_init = False

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.40)
    mask = (avg_img < threshold_init) if not invert_init else (avg_img > threshold_init)
    rgb_img = np.stack([avg_img]*3, axis=-1)
    rgb_img[mask] = [255, 0, 0]
    img_disp = ax.imshow(rgb_img)
    ax.set_title(f"Threshold: {threshold_init} {'(Inverted)' if invert_init else ''}")
    ax.axis('off')

    # Add note below the image
    fig.text(0.5, 0.05, "Note: Pixels shown in red will be excluded from analysis.", ha='center', va='center', fontsize=10, color='red')

    axcolor = 'lightgoldenrodyellow'
    ax_thresh = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
    slider = Slider(ax_thresh, 'Threshold', 0, 255, valinit=threshold_init, valstep=1)

    ax_invert = plt.axes([0.25, 0.18, 0.15, 0.04])
    btn_invert = Button(ax_invert, 'Invert')

    ax_finalize = plt.axes([0.7, 0.18, 0.2, 0.04])
    btn_finalize = Button(ax_finalize, 'Finalize')

    finalized = {'value': None}
    invert_flag = [invert_init]

    def update(val):
        threshold = int(slider.val)
        invert = invert_flag[0]
        mask = (avg_img < threshold) if not invert else (avg_img > threshold)
        rgb_img = np.stack([avg_img]*3, axis=-1)
        rgb_img[mask] = [255, 0, 0]
        img_disp.set_data(rgb_img)
        ax.set_title(f"Threshold: {threshold} {'(Inverted)' if invert else ''}")
        fig.canvas.draw_idle()

    def invert_clicked(event):
        invert_flag[0] = not invert_flag[0]
        update(None)

    def finalize_clicked(event):
        finalized['value'] = (int(slider.val), invert_flag[0])
        plt.close(fig)

    slider.on_changed(update)
    btn_invert.on_clicked(invert_clicked)
    btn_finalize.on_clicked(finalize_clicked)

    plt.show()

    while finalized['value'] is None:
        plt.pause(0.1)
    return finalized['value']

