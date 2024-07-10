import cv2
import os

def LoadImages(Path):
    rgb_images = []
    gray_images = []
    
    for filename in os.listdir(Path):
        img_path = os.path.join(Path, filename)
        img_rgb = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        if img_rgb is not None:
            rgb_images.append(img_rgb)
            gray_images.append(img_gray)

    print(f"Total images loaded: {len(rgb_images)}")

    return rgb_images, gray_images