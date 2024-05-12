# Doodle Recognition with MobileNet and OpenCV

This project implements doodle recognition using a MobileNet model trained on the Google QuickDraw dataset. It allows users to perform recognition on imported images.

Data found [here](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap;tab=objects?prefix=&forceOnObjectsSortingFiltering=false)

## Features

- Train a MobileNet model for doodle recognition using the QuickDraw dataset.
- Recognize human-drawn doodles from `.png` and `.jpg` images

## Dependencies

- Python 3.x
- TensorFlow
- NumPy
- OpenCV
- Pillow (optional, used for image processing)
- Matplotlib (optional, used for data visualization)

## Usage

1. Clone the repository:
`git clone https://github.com/AadiJo/Doodle-Recognition.git`

2. Navigate to the project directory:
`cd 'your-repo'`

3. Install the dependencies:
`pip install -r requirements.txt`

4. Run the main script:
`python gui.py`

5. Import image:
Click the icon in the top left to select an image. File format is 32 x 32, white drawing on black background

6. Detect image:
Click the detect button!
