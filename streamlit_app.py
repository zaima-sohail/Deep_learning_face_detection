import os
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import streamlit as st


@st.cache_resource(max_entries=1)
def load_detector():
    from mtcnn import MTCNN

    return MTCNN()


st.set_page_config(page_title="Deep-learning Face Detection", page_icon=":camera:")
st.title("Deep-learning Face Detection")
st.write("Upload an image to detect faces and facial landmarks using MTCNN.")

uploaded_image = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_image is not None:
    image_bytes = np.frombuffer(uploaded_image.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Could not read this image. Try uploading a valid image file.")
    else:
        with st.spinner("Detecting faces..."):
            from single_face_detection import annotate_faces

            annotated_image, faces = annotate_faces(image, load_detector())

        st.success(f"Detected {len(faces)} face(s).")
        original_column, result_column = st.columns(2)
        with original_column:
            st.subheader("Original image")
            st.image(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                width="stretch",
            )
        with result_column:
            st.subheader("Detection result")
            st.image(
                cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
                width="stretch",
            )

        encoded, result = cv2.imencode(".jpg", annotated_image)
        if not encoded:
            st.error("Could not encode the annotated image for download.")
        else:
            st.download_button(
                "Download annotated image",
                data=result.tobytes(),
                file_name=f"{Path(uploaded_image.name).stem}_detected.jpg",
                mime="image/jpeg",
            )
else:
    st.info("Upload a JPG, PNG, BMP, or WebP image to get started.")
