# Doodle Recognition with MobileNet and OpenCV

WORK IN PROGRESS

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
git clone https://github.com/AadiJo/Pictionary.git

2. Navigate to the project directory:
cd your-repo

3. Install the dependencies:
pip install -r requirements.txt

4. Run the training script to train the model:
python doodle_recognition.py


5. Once the model is trained, run the webcam doodle recognition script to use the webcam for doodle recognition:
python webcam_doodle_recognition.py

6. Follow the instructions on the console window:
   - Press 's' to take a snapshot from the webcam.
   - Press 'c' to perform doodle recognition on the captured snapshot.
   - Press 'q' to exit the program.
