"""Python code to filp an image"""

import cv2

def flip_and_save_image(input_image_path, output_image_path, flip_code):
    # Read the image from the input path
    image = cv2.imread(input_image_path)
    
    # Check if the image was successfully loaded
    if image is None:
        print(f"Error: Unable to load image at {input_image_path}")
        return
    
    # Flip the image
    flipped_image = cv2.flip(image, flip_code)
    
    # Save the flipped image to the output path
    cv2.imwrite(output_image_path, flipped_image)
    print(f"Flipped image saved to {output_image_path}")

# Example usage
input_image_path = 'Joconde.jpg'  # Replace with your input image path
output_image_path = 'Joconde_flip.jpg'  # Replace with your desired output image path
flip_code = 1  # 0 for vertical flip, 1 for horizontal flip, -1 for both

flip_and_save_image(input_image_path, output_image_path, flip_code)
