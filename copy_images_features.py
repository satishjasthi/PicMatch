import os
import random
import shutil
import glob

from tqdm import tqdm

def sample_images_and_features(image_folder, feature_folder, sample_size, dest_image_folder, dest_feature_folder):
    """
    Randomly samples a specified number of resized images along with their corresponding
    CLIP and caption features, and copies them to new folders.

    Args:
        image_folder (str): Path to the folder containing resized images.
        feature_folder (str): Path to the folder containing feature files.
        sample_size (int): Number of images to sample.
        dest_image_folder (str): Destination folder for sampled images.
        dest_feature_folder (str): Destination folder for sampled feature files.
    """

    # Ensure destination folders exist
    os.makedirs(dest_image_folder, exist_ok=True)
    os.makedirs(dest_feature_folder, exist_ok=True)

    # Get all resized image file names
    image_files = glob.glob(os.path.join(image_folder, "resized_*.jpg"))
    image_files.extend(glob.glob(os.path.join(image_folder, "resized_*.png")))
    image_files.extend(glob.glob(os.path.join(image_folder, "resized_*.jpeg")))

    # Check if there are enough images
    if len(image_files) < sample_size:
        raise ValueError("Not enough resized images in the source folder.")

    # Sample a subset of image files
    sampled_images = random.sample(image_files, sample_size)

    # Copy images and corresponding feature files
    for image_path in tqdm(sampled_images):
        image_name = os.path.basename(image_path)
        base_name, _ = os.path.splitext(image_name)

        # Construct paths for CLIP and caption feature files
        clip_feature_path = os.path.join(feature_folder, f"{base_name}_clip.npy")
        caption_feature_path = os.path.join(feature_folder, f"{base_name}_caption.npy")

        # Copy image file
        shutil.copy2(image_path, dest_image_folder)  # copy2 preserves metadata

        # Copy feature files (if they exist)
        if os.path.exists(clip_feature_path):
            shutil.copy2(clip_feature_path, dest_feature_folder)
        if os.path.exists(caption_feature_path):
            shutil.copy2(caption_feature_path, dest_feature_folder)

if __name__ == "__main__":
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent
    image_folder = str(PROJECT_ROOT / "data_old/images")
    feature_folder = str(PROJECT_ROOT / "data_old/features")
    sample_size = 1000
    dest_image_folder = str(PROJECT_ROOT / "data/images")
    dest_feature_folder = str(PROJECT_ROOT / "data/features")
    sample_images_and_features(image_folder, feature_folder, sample_size, dest_image_folder, dest_feature_folder)