# mood_detector_app.py
"""
Person Mood Detector  🙂🙁😠
────────────────────────────────────────────────────────────────────────────
Upload one or more photos containing people. This POC:

1. Detects faces in each image.  
2. Classifies each face’s emotion (e.g., happy, sad, angry, surprised, neutral) using the FER library.  
3. Annotates each face with a bounding box and top emotion label.  
4. Displays all annotated images.  
5. Summarizes counts of each emotion across all images with an interactive bar chart.  
6. Lets you download the detailed results as a CSV.

*Demo only*—no production-grade bias testing or model governance.  
For enterprise CV pipelines, [contact me](https://drtomharty.com/bio).
"""

import io
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from fer import FER

# ────────────────────────────────────────────────────────────────────────────
# Model initialization (cached)
# ────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_detector():
    return FER(mtcnn=True)

detector = load_detector()

# ────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Person Mood Detector", layout="wide")
st.title("🙂🙁😠 Person Mood Detector")

st.info(
    "🔔 **Demo Notice**  \n"
    "This is a lightweight proof-of-concept using the FER library.  \n"
    "For production CV solutions (bias testing, scaling), [contact me](https://drtomharty.com/bio).",
    icon="💡"
)

uploaded_files = st.file_uploader(
    "Upload one or more images", type=["jpg","jpeg","png"], accept_multiple_files=True
)
if not uploaded_files:
    st.stop()

results = []
annotated_images = []

for file in uploaded_files:
    img = Image.open(file).convert("RGB")
    img_array = np.array(img)

    # detect emotions
    detections = detector.detect_emotions(img_array)

    draw = ImageDraw.Draw(img)
    for idx, face in enumerate(detections):
        x, y, w, h = face["box"]
        emotions = face["emotions"]
        # pick top emotion
        top_emotion, score = max(emotions.items(), key=lambda kv: kv[1])
        # annotate
        draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=2)
        draw.text((x, y - 10), f"{top_emotion} ({score:.2f})", fill="red")

        # save result entry
        results.append({
            "image": file.name,
            "face_index": idx,
            "x": x, "y": y, "w": w, "h": h,
            "emotion": top_emotion,
            **{e: emotions[e] for e in emotions}
        })

    annotated_images.append((file.name, img))

# ────────────────────────────────────────────────────────────────────────────
# Display annotated images
# ────────────────────────────────────────────────────────────────────────────
st.subheader("Annotated Images")
for name, img in annotated_images:
    st.image(img, caption=name, use_column_width=True)

# ────────────────────────────────────────────────────────────────────────────
# Display emotion summary
# ────────────────────────────────────────────────────────────────────────────
df = pd.DataFrame(results)
if df.empty:
    st.warning("No faces detected in uploaded images.")
    st.stop()

summary = df["emotion"].value_counts().reset_index()
summary.columns = ["Emotion", "Count"]

st.subheader("Emotion Counts")
st.bar_chart(summary.set_index("Emotion"))

# ────────────────────────────────────────────────────────────────────────────
# Download detailed results
# ────────────────────────────────────────────────────────────────────────────
csv = df.to_csv(index=False).encode()
st.download_button(
    "⬇️ Download detailed results as CSV",
    data=csv,
    file_name="emotion_detection_results.csv",
    mime="text/csv"
)
