import os
import sys
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.widgets import Slider


class NoduleVisualizer:
    def __init__(self, nodule_patches_dir):
        self.nodule_dir = Path(nodule_patches_dir)
        self.nodules = self._find_all_nodules()
        self.current_nodule_idx = 0
        self.current_view = 'axial'  # axial, coronal, sagittal

    def _find_all_nodules(self):
        """Find all nodule patch files"""
        nodules = []

        for class_dir in sorted(os.listdir(self.nodule_dir)):
            class_path = self.nodule_dir / class_dir
            if not class_path.is_dir():
                continue

            for nifti_file in sorted(class_path.glob("*.nii.gz")):
                nodules.append({
                    'path': nifti_file,
                    'class': class_dir,
                    'name': nifti_file.stem
                })

        return nodules

    def visualize(self):
        """Start the visualization interface"""
        if not self.nodules:
            print("No nodule patches found!")
            return

        # Create the figure and axes
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.25)

        # Load the first nodule
        self._load_current_nodule()

        # Add slider for scrolling through slices
        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.slider = Slider(
            ax=ax_slider,
            label='Slice',
            valmin=0,
            valmax=self.volume.shape[self._get_slice_axis()] - 1,
            valinit=self.volume.shape[self._get_slice_axis()] // 2,
            valstep=1
        )
        self.slider.on_changed(self._update_slice)

        # Add buttons for navigation
        prev_button_ax = plt.axes([0.1, 0.05, 0.1, 0.04])
        self.prev_button = plt.Button(prev_button_ax, 'Previous')
        self.prev_button.on_clicked(self._prev_nodule)

        next_button_ax = plt.axes([0.8, 0.05, 0.1, 0.04])
        self.next_button = plt.Button(next_button_ax, 'Next')
        self.next_button.on_clicked(self._next_nodule)

        # Add view selection buttons
        axial_button_ax = plt.axes([0.3, 0.05, 0.1, 0.04])
        self.axial_button = plt.Button(axial_button_ax, 'Axial')
        self.axial_button.on_clicked(lambda event: self._change_view('axial'))

        coronal_button_ax = plt.axes([0.45, 0.05, 0.1, 0.04])
        self.coronal_button = plt.Button(coronal_button_ax, 'Coronal')
        self.coronal_button.on_clicked(lambda event: self._change_view('coronal'))

        sagittal_button_ax = plt.axes([0.6, 0.05, 0.1, 0.04])
        self.sagittal_button = plt.Button(sagittal_button_ax, 'Sagittal')
        self.sagittal_button.on_clicked(lambda event: self._change_view('sagittal'))

        # Initial display
        self._update_slice(self.volume.shape[self._get_slice_axis()] // 2)

        plt.show()

    def _load_current_nodule(self):
        """Load the current nodule's data"""
        if not self.nodules:
            return

        nodule_info = self.nodules[self.current_nodule_idx]
        image = sitk.ReadImage(str(nodule_info['path']))
        self.volume = sitk.GetArrayFromImage(image)
        self.spacing = image.GetSpacing()

        # Reset slider if it exists
        if hasattr(self, 'slider'):
            self.slider.valmax = self.volume.shape[self._get_slice_axis()] - 1
            self.slider.valinit = self.volume.shape[self._get_slice_axis()] // 2
            self.slider.set_val(self.volume.shape[self._get_slice_axis()] // 2)

    def _get_slice_axis(self):
        """Get the axis to slice along based on current view"""
        if self.current_view == 'axial':
            return 0  # First dimension in SimpleITK array is Z
        elif self.current_view == 'coronal':
            return 1  # Second dimension is Y
        else:  # sagittal
            return 2  # Third dimension is X

    def _update_slice(self, slice_idx):
        """Update the display to show the selected slice"""
        self.ax.clear()

        slice_idx = int(slice_idx)
        if self.current_view == 'axial':
            slice_data = self.volume[slice_idx, :, :]
        elif self.current_view == 'coronal':
            slice_data = self.volume[:, slice_idx, :]
        else:  # sagittal
            slice_data = self.volume[:, :, slice_idx]

        self.ax.imshow(slice_data, cmap='gray')

        # Add crosshair at center
        if self.current_view == 'axial':
            self.ax.axhline(slice_data.shape[0] // 2, color='r', alpha=0.3)
            self.ax.axvline(slice_data.shape[1] // 2, color='r', alpha=0.3)
        elif self.current_view == 'coronal':
            self.ax.axhline(slice_data.shape[0] // 2, color='r', alpha=0.3)
            self.ax.axvline(slice_data.shape[1] // 2, color='r', alpha=0.3)
        else:  # sagittal
            self.ax.axhline(slice_data.shape[0] // 2, color='r', alpha=0.3)
            self.ax.axvline(slice_data.shape[1] // 2, color='r', alpha=0.3)

        # Set title with information
        nodule_info = self.nodules[self.current_nodule_idx]
        class_names = {
            "0": "低分化",
            "1": "中分化",
            "2": "高分化",
            "3": "原位癌",
            "4": "微浸润"
        }
        class_name = class_names.get(nodule_info['class'], nodule_info['class'])

        title = f"Nodule: {nodule_info['name']} | Class: {class_name} ({nodule_info['class']}) | " \
                f"View: {self.current_view} | Slice: {slice_idx}/{self.volume.shape[self._get_slice_axis()] - 1}"
        self.ax.set_title(title)

        self.fig.canvas.draw_idle()

    def _next_nodule(self, event):
        """Move to the next nodule"""
        self.current_nodule_idx = (self.current_nodule_idx + 1) % len(self.nodules)
        self._load_current_nodule()
        self._update_slice(self.volume.shape[self._get_slice_axis()] // 2)

    def _prev_nodule(self, event):
        """Move to the previous nodule"""
        self.current_nodule_idx = (self.current_nodule_idx - 1) % len(self.nodules)
        self._load_current_nodule()
        self._update_slice(self.volume.shape[self._get_slice_axis()] // 2)

    def _change_view(self, view):
        """Change the current view orientation"""
        self.current_view = view
        # Update slider max value based on new view
        self.slider.valmax = self.volume.shape[self._get_slice_axis()] - 1
        self.slider.set_val(self.volume.shape[self._get_slice_axis()] // 2)
        self._update_slice(self.volume.shape[self._get_slice_axis()] // 2)


def plot_3d_views(nodule_path):
    """
    Plot orthogonal views of a single nodule
    """
    # Load the nodule
    image = sitk.ReadImage(str(nodule_path))
    volume = sitk.GetArrayFromImage(image)

    # Get the middle slice indices
    z_mid = volume.shape[0] // 2
    y_mid = volume.shape[1] // 2
    x_mid = volume.shape[2] // 2

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Axial view (top-down)
    axes[0].imshow(volume[z_mid, :, :], cmap='gray')
    axes[0].set_title(f'Axial View (z={z_mid})')
    axes[0].axhline(y_mid, color='r', alpha=0.3)
    axes[0].axvline(x_mid, color='r', alpha=0.3)

    # Coronal view (front)
    axes[1].imshow(volume[:, y_mid, :], cmap='gray')
    axes[1].set_title(f'Coronal View (y={y_mid})')
    axes[1].axhline(z_mid, color='r', alpha=0.3)
    axes[1].axvline(x_mid, color='r', alpha=0.3)

    # Sagittal view (side)
    axes[2].imshow(volume[:, :, x_mid], cmap='gray')
    axes[2].set_title(f'Sagittal View (x={x_mid})')
    axes[2].axhline(z_mid, color='r', alpha=0.3)
    axes[2].axvline(y_mid, color='r', alpha=0.3)

    plt.tight_layout()
    plt.show()


def batch_visualize_nodules(nodule_dir, num_samples=3):
    """
    Generate a summary visualization of multiple nodules
    """
    nodule_dir = Path(nodule_dir)

    # Find nodules by class
    nodules_by_class = {}

    for class_dir in sorted(os.listdir(nodule_dir)):
        class_path = nodule_dir / class_dir
        if not class_path.is_dir():
            continue

        nodule_files = list(class_path.glob("*.nii.gz"))
        if nodule_files:
            # Sample up to num_samples nodules per class
            samples = np.random.choice(nodule_files,
                                       size=min(num_samples, len(nodule_files)),
                                       replace=False)
            nodules_by_class[class_dir] = samples

    class_names = {
        "0": "低分化",
        "1": "中分化",
        "2": "高分化",
        "3": "原位癌",
        "4": "微浸润"
    }

    # Plot samples for each class
    rows = len(nodules_by_class)
    cols = num_samples
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    for i, (class_id, nodules) in enumerate(nodules_by_class.items()):
        class_name = class_names.get(class_id, class_id)

        for j, nodule_path in enumerate(nodules):
            # Load the nodule
            image = sitk.ReadImage(str(nodule_path))
            volume = sitk.GetArrayFromImage(image)

            # Get the middle axial slice
            z_mid = volume.shape[0] // 2

            # Plot
            if rows == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]

            ax.imshow(volume[z_mid, :, :], cmap='gray')
            ax.set_title(f"{nodule_path.stem}\nClass: {class_name}")
            ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    NODULE_DIR = r"G:\datasets\nodule_patches64"

    if len(sys.argv) > 1:
        # Check command line arguments
        if sys.argv[1] == "--batch":
            # Batch visualization
            num_samples = 3
            if len(sys.argv) > 2:
                try:
                    num_samples = int(sys.argv[2])
                except ValueError:
                    pass
            batch_visualize_nodules(NODULE_DIR, num_samples)
        elif os.path.isfile(sys.argv[1]):
            # Single nodule quick view
            plot_3d_views(sys.argv[1])
        else:
            print("Invalid arguments")
            print("Usage: python visualize_nodules.py [--batch [num_samples]]")
            print("   or: python visualize_nodules.py [path_to_nodule.nii.gz]")
    else:
        # Interactive visualization
        visualizer = NoduleVisualizer(NODULE_DIR)
        visualizer.visualize()