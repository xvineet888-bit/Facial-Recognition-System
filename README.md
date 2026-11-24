# Face Identification System

A Python-based face identification project with a Flask web dashboard, image-based identification, and real-time webcam recognition.

**Developer:** Vineet Yadav  
**Language:** Python 3.10  
**Project Status:** Academic / Experimental Project

## Features

- Face detection using dlib and the `face_recognition` library.
- 128-dimensional face encoding generation.
- Face matching using Euclidean distance.
- Image-based face identification.
- Group image recognition.
- Real-time webcam recognition.
- Flask-based web dashboard.
- Recognition history and analytics.
- Person management.
- Model evaluation and comparison tools.
- Automated tests using pytest.

## Technology Stack

- Python 3.10
- Flask
- OpenCV
- dlib
- face_recognition
- NumPy
- scikit-learn
- Joblib
- Pandas
- Matplotlib
- SQLite
- HTML, CSS and JavaScript

## Project Setup

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd Facial-Recognition-System
```

### 2. Create a virtual environment

```bash
py -3.10 -m venv .venv
```

### 3. Activate it

**Windows:**

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** Installing dlib on Windows may require CMake and Microsoft Visual C++ Build Tools.

## Run the Web Application

Start the Flask dashboard:

```bash
python app.py
```

Open your browser:

```text
http://127.0.0.1:5000
```

## Dataset Setup

The project does not include personal face images or trained biometric data.

To use your own dataset, organize images by identity:

```text
data/
└── raw/
    ├── Person_1/
    │   ├── image1.jpg
    │   └── image2.jpg
    └── Person_2/
        ├── image1.jpg
        └── image2.jpg
```

Use clear, properly labelled images for better testing.

## Run Tests

```bash
python -m pytest -q
```

## Important Limitations

- Recognition depends on the quality of the input images and the available registered face data.
- Recognition accuracy has not been comprehensively evaluated on a large real-world dataset.
- The project does not include a law-enforcement or criminal-record database.
- It does not provide anti-spoofing or liveness detection.
- Results should be treated as experimental and should not be used as the sole basis for identifying a person in consequential situations.

## Privacy

This repository does not distribute personal face photographs, biometric encodings, local databases, or secret keys. Users should only process facial data with appropriate permission and safeguards.

## Credits

This project includes work adapted from the upstream project acknowledged in [CREDITS.md](CREDITS.md). Please refer to that file and the included LICENSE for attribution and licensing information.

---

**Developed by Vineet Yadav**