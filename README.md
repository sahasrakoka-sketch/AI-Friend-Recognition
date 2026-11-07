# 👥 AI Friend Recognition

A real-time face recognition system built using **Python and OpenCV** that detects multiple faces through a webcam and identifies registered people.

The project uses face detection and face feature extraction to recognize known individuals and label unrecognized faces as **Unknown**.

## ✨ Features

- **Real-Time Face Detection** – Detects faces using OpenCV YuNet.
- **Face Recognition** – Uses OpenCV SFace to extract facial embeddings and compare identities.
- **Multiple Face Recognition** – Identifies multiple people in a single camera frame.
- **Face Registration** – Registers people using images organized into individual folders.
- **Unknown Face Detection** – Displays `Unknown` when a detected face does not match a registered identity.
- **Webcam Integration** – Performs recognition using a live camera feed.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenCV | Image processing and computer vision |
| YuNet | Face detection |
| SFace | Face feature extraction and recognition |
| NumPy | Numerical operations |
| Pickle | Local storage of face embeddings |

## 📁 Project Structure

```text
AI-Friend-Recognition/
│
├── data/                          # Private face image datasets
│
├── models/
│   ├── face_detection_yunet_2023mar.onnx
│   ├── face_recognition_sface_2021dec.onnx
│   └── face_database.pkl           # Generated locally
│
├── camera_test.py
├── face_detection_test.py
├── predict_image.py
├── register_faces.py
├── live_recognition.py
│
├── .gitignore
└── README.md
```

**Note:** Private datasets and generated recognition data are intentionally excluded from this repository.

## ⚙️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/sahasrakoka-sketch/AI-Friend-Recognition.git
cd AI-Friend-Recognition
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate
```

### 3. Install dependencies

```bash
pip install opencv-contrib-python numpy
```

### 4. Download the required models

Download the official OpenCV Zoo YuNet and SFace ONNX models.

Place them inside the `models/` directory using these exact filenames:

- `face_detection_yunet_2023mar.onnx`
- `face_recognition_sface_2021dec.onnx`

### 5. Add face images

Create a separate folder for each person inside `data/`.

Example:

```text
data/
├── Person_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── image3.jpg
│
└── Person_2/
    ├── image1.jpg
    ├── image2.jpg
    └── image3.jpg
```

Use multiple clear images with different angles and lighting conditions for better recognition.

### 6. Register faces

Run:

```bash
python register_faces.py
```

This processes the images and generates the local face database:

```text
models/face_database.pkl
```

### 7. Start live recognition

```bash
python live_recognition.py
```

The webcam will open and attempt to identify registered people in real time.

## 🔄 How It Works

```text
Face Images
     |
     v
Face Registration
     |
     v
YuNet Face Detection
     |
     v
SFace Feature Extraction
     |
     v
Generate Face Embeddings
     |
     v
Compare with Registered Faces
     |
     v
Recognized Name / Unknown
```

## 🔒 Privacy & Data Protection

This project uses facial images and biometric representations, so privacy is important.

- Personal face datasets are excluded from Git.
- Generated face embeddings are stored locally.
- Teachable Machine model files and labels are excluded.
- No personal face images should be uploaded to the public repository.
- Only use face images with the individuals' informed consent.

## 🚀 Future Improvements

- Improve recognition accuracy under different lighting conditions.
- Add a user-friendly graphical interface.
- Build a face registration interface.
- Explore voice recognition integration.
- Add secure local storage for registered identities.
- Evaluate recognition performance using a proper test dataset.

## 👩‍💻 Author

**Koka Veda Sahasra**

B.Tech – Computer Science and Engineering  
Specialization: AI and Future Technologies  
SRM University-AP | Class of 2029

Interested in Artificial Intelligence, Machine Learning, Computer Vision and building practical AI applications.

---

⭐ If you find this project interesting, feel free to explore the repository!
