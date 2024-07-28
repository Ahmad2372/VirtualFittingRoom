from flask import Flask, render_template, request
import urllib.request
import cv2
import numpy as np
import os
from whitenoise import WhiteNoise

app = Flask(__name__)
app.wsgi_app = WhiteNoise(app.wsgi_app, root='static/')

'''
# URLs to images hosted on Shopify
woman_drawing_url = 'https://cdn.shopify.com/s/files/1/0649/2028/9453/files/woman_in_hijab_2.png?v=1721580794'  # Original image for shadows
mask_image_url = 'https://cdn.shopify.com/s/files/1/0649/2028/9453/files/woman_in_hijab_3.png?v=1721580794'      # Image with the mask
no_red_image_url = 'https://cdn.shopify.com/s/files/1/0649/2028/9453/files/woman_no_red.png?v=1721580795'        # Image without the red region
uploads_dir = os.path.join('static', 'uploads')
os.makedirs(uploads_dir, exist_ok=True)
'''
# URLs to images hosted on Shopify
woman_drawing_url = 'https://cdn.shopify.com/s/files/1/0649/2028/9453/files/Mod_original_contrast.png?v=1722169174'  # Original image for shadows
mask_image_url = 'https://cdn.shopify.com/s/files/1/0649/2028/9453/files/Mod_original_red.png?v=1722169172'      # Image with the mask
no_red_image_url = 'https://cdn.shopify.com/s/files/1/0649/2028/9453/files/Mod_original_cleared.png?v=1722169172'        # Image without the red region
uploads_dir = os.path.join('static', 'uploads')
os.makedirs(uploads_dir, exist_ok=True)

def download_image(url, save_path):
    urllib.request.urlretrieve(url, save_path)

def apply_texture_with_mask(original_image_path, mask_image_path, no_red_image_path, texture_image_path, output_path):
    # Load images
    original_image = cv2.imread(original_image_path, cv2.IMREAD_UNCHANGED)
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_UNCHANGED)
    no_red_image = cv2.imread(no_red_image_path, cv2.IMREAD_UNCHANGED)
    texture_image = cv2.imread(texture_image_path)

    if original_image is None:
        raise ValueError(f"Error loading original image from path: {original_image_path}")
    if mask_image is None:
        raise ValueError(f"Error loading mask image from path: {mask_image_path}")
    if no_red_image is None:
        raise ValueError(f"Error loading no-red image from path: {no_red_image_path}")
    if texture_image is None:
        raise ValueError(f"Error loading texture image from path: {texture_image_path}")

    # Ensure all images have the same dimensions
    dimensions = (original_image.shape[1], original_image.shape[0])  # width, height
    mask_image = cv2.resize(mask_image, dimensions)
    no_red_image = cv2.resize(no_red_image, dimensions)
    texture_image = cv2.resize(texture_image, dimensions)

    # Ensure the mask image has 4 channels (RGBA)
    if mask_image.shape[2] != 4:
        raise ValueError(f"Mask image must have 4 channels (RGBA), but it has {mask_image.shape[2]} channels")

    # Create a binary mask for the red color
    lower_red = np.array([0, 0, 255, 255])
    upper_red = np.array([0, 0, 255, 255])
    binary_mask = cv2.inRange(mask_image, lower_red, upper_red)

    # Convert the original image to grayscale to get luminance
    grayscale_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Normalize grayscale values to the range [0, 1]
    normalized_grayscale = grayscale_original / 255.0

    # Apply the texture to the no-red image based on the mask
    texture_applied_image = no_red_image.copy()
    for c in range(0, 3):
        texture_applied_image[:, :, c] = np.where(binary_mask == 255, texture_image[:, :, c], no_red_image[:, :, c])

    # Modulate the brightness of the applied texture based on the grayscale values
    result_image = texture_applied_image.copy()
    for c in range(0, 3):
        texture_channel = result_image[:, :, c].astype(float)
        adjusted_channel = texture_channel * normalized_grayscale  # Modulate brightness by the grayscale image
        adjusted_channel = np.clip(adjusted_channel, 0, 255).astype(np.uint8)
        result_image[:, :, c] = adjusted_channel

    # Save the result
    cv2.imwrite(output_path, result_image)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filename = file.filename.replace(' ', '_')
    file_path = os.path.join(uploads_dir, filename)
    file.save(file_path)

    # Download the static images
    original_image_path = os.path.join(uploads_dir, 'Mod_original.jpeg')
    mask_image_path = os.path.join(uploads_dir, 'Mod_original_red.png')
    no_red_image_path = os.path.join(uploads_dir, 'Mod_original_cleared.png')

    download_image(woman_drawing_url, original_image_path)
    download_image(mask_image_url, mask_image_path)
    download_image(no_red_image_url, no_red_image_path)

    result_image_path = os.path.join(uploads_dir, 'result_' + filename)
    apply_texture_with_mask(original_image_path, mask_image_path, no_red_image_path, file_path, result_image_path)

    return render_template('result.html', result_image=result_image_path)

if __name__ == '__main__':
    app.run(debug=True)
