import cv2
import os

def check_images(source_folders):
    for folder in source_folders:
        print(f"Checking images in folder: {folder}")
        for filename in os.listdir(folder):
            if filename.endswith('.jpg'):
                path = os.path.join(folder, filename)
                image = cv2.imread(path)
                if image is None:
                    print(f"Failed to load: {path}")
def main():
    source_folders = ['xTV_Folder', 'yPhones_Folder', 'zLaptop_Folder']
    check_images(source_folders)

if __name__ == '__main__':
    main()