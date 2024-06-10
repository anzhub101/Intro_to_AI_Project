import cv2
import numpy as np
import os

def add_gaussian_noise(image):
    row, col, ch = image.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy_image = image + gauss
    return noisy_image

def augment_image(image_path, save_folder):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (512, 512))

    rotated = cv2.rotate(resized_image, cv2.ROTATE_90_CLOCKWISE)

    # Flip horizontally
    flipped = cv2.flip(resized_image, 1)

    # Add Gaussian noise
    noisy = add_gaussian_noise(resized_image.astype('float64'))

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(os.path.join(save_folder, f'{base_name}_original.jpg'), resized_image)
    cv2.imwrite(os.path.join(save_folder, f'{base_name}_rotated.jpg'), rotated)
    cv2.imwrite(os.path.join(save_folder, f'{base_name}_flipped.jpg'), flipped)
    cv2.imwrite(os.path.join(save_folder, f'{base_name}_noisy.jpg'), noisy)

def process_folders(source_folders, target_folders):
    for source_folder, target_folder in zip(source_folders, target_folders):
        # Process each image in source folder
        for image_file in os.listdir(source_folder):
            if image_file.endswith('.jpg'):
                image_path = os.path.join(source_folder, image_file)
                augment_image(image_path, target_folder)

def main():

    source_folders = ['xTV_Folder', 'yPhones_Folder', 'zLaptop_Folder']
    target_folders = ['New_TV', 'New_Phones', 'New_Laptop']

    process_folders(source_folders, target_folders)

if __name__ == '__main__':
    main()


