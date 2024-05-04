# Doodle Recognition with MobileNet and OpenCV

This project implements doodle recognition using a MobileNet model trained on the Google QuickDraw dataset. It allows users to capture images from the webcam using OpenCV and perform doodle recognition on the captured images.

## Features

- Train a MobileNet model for doodle recognition using the QuickDraw dataset.
- Use the webcam to capture images for doodle recognition.
- Perform real-time doodle recognition on captured images.

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
Click the icon in the top left to select an image file
Format is 32 x 32, white drawing on black background

6. Detect image:
Click the detect button!