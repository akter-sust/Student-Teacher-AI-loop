from PIL import Image
import random
import glob
import os

def rotate_keep_size_pil(image_path, output_path, angle):
    # Open the image
    img = Image.open(image_path)
    
    # Rotate around the center (expand=False keeps original canvas size)
    rotated_img = img.rotate(angle, expand=False, resample=Image.BICUBIC)
    
    # Save the result
    rotated_img.save(output_path)

if __name__ == "__main__":
    # Example usage (rotates 45 degrees counter-clockwise)
    dir_images = "./hard-negatives-bubbling"
    images = glob.glob(os.path.join(dir_images, "*.*"))
    for i, image_path in enumerate(images):
        print(f"\nProcessing {i+1}/{len(images)}: {image_path}...")
        for r in random.sample(range(0, 360, 5), k=22):  # Rotate in 45-degree increments
            print(f"..{r}", end="", flush=True)
            output_path = os.path.join(dir_images, f"rotated_{r}_" + os.path.basename(image_path))
            rotate_keep_size_pil(image_path, output_path, r)
