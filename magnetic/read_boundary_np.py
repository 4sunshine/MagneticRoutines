import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from prepro_wie.prepro_numba import prepro
    print("Numba version of preprocess_wie loaded")
except ImportError:
    print("Loading default wie_preprocessing algorithm")
    from prepro_wie.prepro import prepro


def get_skewed_diagonal(data):
    rows, cols = data.shape
    x = np.round(np.linspace(0, cols - 1, num=rows)).astype(int)  # Column indices
    y = np.arange(rows)  # Row indices (0 to 447)
    return data[y, x]  # Extract values along the path


def vertical_plot(b, id: int, out_dir: str = "images"):
    os.makedirs(out_dir, exist_ok=True)
    bv = np.vstack([b[1], b[2], b[0]])
    plt.figure(figsize=(6, 8), dpi=300, facecolor="black")  # figsize=(width, height) in inches
    plt.imshow(bv, origin='lower', cmap='nipy_spectral_r', vmin=-2000, vmax=2000)
    plt.grid(False)  # Disable grid
    plt.axis('off')  # Hide axes and ticks
    plt.savefig(os.path.join(out_dir, f"image_{id:03d}.png"))
    plt.close()


def read_np_folder(folder_path):
    """
    Iterates over all .npz files in the specified folder

    Args:
        folder_path (str or Path): Path to the folder containing .npz files.
    """
    folder = Path(folder_path)

    for i, npz_file in enumerate(folder.glob('*.npz')):
        data = np.load(npz_file.absolute())
        b_orig = data["B"]
        vertical_plot(b_orig, id=0, out_dir="images_denoise")
        b = np.transpose(b_orig, (0, 2, 1)).copy()
        _, nx, ny = b.shape
        nz = min(nx, ny)
        (b_ff_x, b_ff_y, b_ff_z), losses = prepro(b[0], b[1], b[2], 0.001, 0.01, nx, ny, nz)
        b_ff = np.stack([b_ff_x, b_ff_y, b_ff_z], axis=0).transpose((0, 2, 1))
        vertical_plot(b_ff, id=1, out_dir="images_denoise")
        # TODO: add saving Force-Free data
        break


    # video.release()


if __name__ == "__main__":
    read_np_folder(sys.argv[1])
