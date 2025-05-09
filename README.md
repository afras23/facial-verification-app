# Facial Verification App

 This project is a real-time facial verification system built with Kivy and TensorFlow, utilising a Siamese Neural Network for secure and reliable identity verification via webcam. This app compares a live input image against pre-stored verification images to determine identity matches, making it useful for access control, secure authentication, or biometric experiments.

## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Tests](#tests)
- [Credits](#credits)
- [License](#license)

---

## Project Description

This application captures a user's webcam image and compares it to a set of stored
reference images using a pre-trained Siamese model. The model evaluates the
similarity between two faces using a custom L1 distance layer and makes a
verification decision based on defined thresholds.

The app is implemented using:
- **Kivy**: for real-time GUI and webcam interaction
- **TensorFlow/Keras**: for deep learning inference
- **OpenCV**: for image processing and capture
- **NumPy**: for tensor manipulation and aggregation

---

## Installation

### Requirements

- Python 3.6–3.10
- pip
- Virtual environment (recommended)

### Dependencies

```bash
pip install kivy tensorflow opencv-python numpy
```

### Clone the Repository

```bash
git clone https://github.com/afras23/facial-verification-app.git
cd facial-verification-app
```

### File Structure

```
faceverapp/
│
├── app/
├── application_data/
│   ├── input_image/
│   └── verification_images/
├── training_checkpoints/
├── siamesemodelv2.keras
├── faceid.py
├── layers.py
├── Facial_Verification_with_Siamese_Network.ipynb
└── README.md
```

---

## Usage

1. **Prepare reference images:**
   Add 1 or more JPEG images to `application_data/verification_images/`.

2. **Run the app:**

```bash
python faceid.py
```

3. **Verify your face:**
   - The app opens a window with a webcam feed.
   - Click the **Verify** button.
   - Your face is compared against the stored images.
   - Verification result: `Verified` or `Unverified`.

---

## Tests

### Manual Test Workflow

1. Add a valid image to `application_data/verification_images/`.
2. Run the app, ensure webcam captures and resizes correctly.
3. Test correct predictions with both matching and non-matching faces.

### Example Unit Test (in `test_layers.py`)

```python
import tensorflow as tf
from layers import L1Dist

def test_l1_distance():
    l1 = L1Dist()
    a = tf.constant([[1.0, 2.0, 3.0]])
    b = tf.constant([[2.0, 2.0, 4.0]])
    result = l1(a, b).numpy()
    assert all(result == [1.0, 0.0, 1.0]), f"Unexpected output: {result}"

if __name__ == "__main__":
    test_l1_distance()
    print("L1Dist test passed!")
```

### Run the Tests

```bash
python test_layers.py
```

---

## Credits

- **Author:** Anesah Fraser  
- Siamese network model architecture and application inspired by modern facial recognition systems, particularly the paper “Siamese Neural Networks for One-shot Image Recognition” by Koch et al., which guided the development of the model.


