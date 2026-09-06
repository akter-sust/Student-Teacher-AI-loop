import os
from PIL import Image

# Target maximum file size in bytes (2 MB)
MAX_SIZE_BYTES = 2 * 1024 * 1024


def compress_image(image_path, target_size=MAX_SIZE_BYTES):
    """Compresses an image to be under target_size (default 2MB)

    by reducing quality and downscaling if necessary.
    """
    file_size = os.path.getsize(image_path)
    if file_size <= target_size:
        print(
            f"Skipping {os.path.basename(image_path)}: already under 2MB ({file_size / 1e6:.2f} MB)"
        )
        return

    print(
        f"Processing {os.path.basename(image_path)}: initial size {file_size / 1e6:.2f} MB..."
    )

    with Image.open(image_path) as img:
        # Convert RGBA/P to RGB (required for JPG saving)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        # Step 1: Lower JPEG compression quality dynamically
        quality = 90
        temp_path = image_path + ".tmp.jpg"

        while quality >= 20:
            img.save(temp_path, "JPEG", quality=quality, optimize=True)
            if os.path.getsize(temp_path) <= target_size:
                os.replace(temp_path, image_path)
                final_size = os.path.getsize(image_path)
                print(
                    f"✓ Compressed {os.path.basename(image_path)} -> {final_size / 1e6:.2f} MB (Quality: {quality})"
                )
                return
            quality -= 10

        # Step 2: If still > 2MB after quality drop, downscale dimensions
        width, height = img.size
        scale = 0.8
        while os.path.getsize(temp_path) > target_size and scale > 0.1:
            new_size = (int(width * scale), int(height * scale))
            resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
            resized_img.save(temp_path, "JPEG", quality=75, optimize=True)
            scale -= 0.1

        if os.path.getsize(temp_path) <= target_size:
            os.replace(temp_path, image_path)
            final_size = os.path.getsize(image_path)
            print(
                f"✓ Resized & Compressed {os.path.basename(image_path)} -> {final_size / 1e6:.2f} MB"
            )
        else:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            print(f"⚠️ Could not reduce {os.path.basename(image_path)} below 2MB")


def process_folder(folder_path):
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(valid_extensions):
            full_path = os.path.join(folder_path, file_name)
            compress_image(full_path)


if __name__ == "__main__":
    # Change 'dataset' to your actual image folder path
    folder_dir = "./train-bubbling"
    process_folder(folder_dir)