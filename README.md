🧑‍🤝‍🧑 Face Recognition System with InsightFace

This project implements a face recognition pipeline using InsightFace, OpenCV, and deep learning. It consists of two main modules:

Face Embedding Generation – processes a dataset of faces, applies augmentation, extracts embeddings with ArcFace, and visualizes clusters using t-SNE.

Face Recognition – loads stored embeddings, detects faces in new images, and performs recognition using cosine similarity.

⚙️ Installation

Clone the repository

git clone https://github.com/arijitb17/mini-project.git
cd mini-project


Create a virtual environment

python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows


Install dependencies

pip install -r requirements.txt

Requirements

Python 3.8+

OpenCV

NumPy

Matplotlib, Seaborn

scikit-learn

imgaug

Pillow

InsightFace

🚀 Usage
1. Generate Face Embeddings

Prepare your dataset under dataset/ with subfolders for each person.
Run:

python generate_embeddings.py


✅ Output:

face_embeddings.pkl → serialized embeddings dictionary

t-SNE visualization of face clusters

2. Run Face Recognition

Place test images inside test-images/ and run:

python recognize_faces.py


✅ Output:

Annotated images saved in output/

Console logs of detected identities with similarity scores

📊 Example Workflow

Train embeddings:

<img width="800" height="600" alt="graph" src="https://github.com/user-attachments/assets/e86c9f5f-3b33-42ad-b323-0746b439dc54" />

Run generate_embeddings.py → saves embeddings for Alice and Bob.

Test recognition:
Place group photo in test-images/.
Run recognize_faces.py → faces get detected, recognized, and saved in output/.

✨ Features

Augmentation: flipping, rotation, gamma contrast

Deep embeddings with ArcFace (buffalo_l model)

t-SNE visualization for clustering

Cosine similarity matching for recognition

Bounding box + label rendering on test images

Robust fallback if no faces detected in enhanced images

🔮 Future Improvements

Support for video streams / real-time recognition

Face enrollment via webcam

Web app integration (Flask / FastAPI)

More advanced similarity threshold tuning
