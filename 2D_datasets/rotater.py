from PIL import Image
import os

"""
This function takes the path of one image and then rotates it n-times by i-degree each time and then saves the
rotated images in passed save directory
"""
def rotate_and_save_image(image_path,angel, n, save_directory):
    # Open the image
    image = Image.open(image_path)
    
    # Create the directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Perform n rotations and save each image
    for i in range(n):
        # Rotate image by i degrees for each step
        rotated_image = image.rotate(angel * (i + 1), expand=True)
        
        # Save the rotated image with a unique name
        rotated_image.save(os.path.join(save_directory, f"rotated_image_{i+1}.png"))
        print(f"Saved rotated_image_{i+1}.png")


rotate_and_save_image('wood_chair/wood_chair.png',5, 32, 'wood_chair/by_5deg_32')

