# mood-detector

🙂🙁😠 Person Mood Detector
A Streamlit proof-of-concept that detects faces in uploaded photos and classifies each person’s emotion using the FER library and MTCNN.

Demo only—no bias testing, no model governance.
For enterprise computer-vision pipelines, contact me.

🔍 What it does
Upload one or more images (.jpg, .jpeg, .png).

Detects faces using MTCNN.

Classifies emotion for each face (happy, sad, angry, surprised, neutral) via FER.

Annotates each face with a bounding box and the top emotion label.

Displays all annotated images and an interactive bar chart of emotion counts.

Downloadable: exports detailed results as emotion_detection_results.csv.

✨ Key Features
CPU-friendly: MTCNN + FER run locally without GPUs.

Session-cached model: loads detector once per session.

Interactive UI: instant feedback in Streamlit.

Downloadable output: CSV of per-face emotion scores and bounding boxes.

Single-file app: all logic in mood_detector_app.py.

🚀 Quick Start (Local)
bash
Copy
Edit
git clone https://github.com/THartyMBA/person-mood-detector.git
cd person-mood-detector
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run mood_detector_app.py
Open http://localhost:8501.

Upload your images and see mood annotations in real time.

☁️ Deploy on Streamlit Cloud
Push this repo (public or private) under THartyMBA to GitHub.

Go to streamlit.io/cloud → New app → select your repo & branch → Deploy.

Share your public URL—no secrets or API keys needed.

🛠️ Requirements
shell
Copy
Edit
streamlit>=1.32
pillow
numpy
pandas
fer
mtcnn
tensorflow
🗂️ Repo Structure
vbnet
Copy
Edit
person-mood-detector/
├─ mood_detector_app.py    ← single-file Streamlit app  
├─ requirements.txt  
└─ README.md               ← you’re reading it  
📜 License
CC0 1.0 – public-domain dedication. Attribution appreciated but not required.

🙏 Acknowledgements
Streamlit – rapid Python UIs

FER – facial emotion recognition

MTCNN – face detection

TensorFlow – underlying ML backend

Detect mood at a glance—enjoy! 🎉
