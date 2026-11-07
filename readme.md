# Deep-learning face detection

This project uses MTCNN, a deep-learning face detector, to find faces and facial
landmarks in an image.

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run detection on the included sample image:

```powershell
python single_face_detection.py
```

Launch the Streamlit image-upload app:

```powershell
python -m streamlit run streamlit_app.py
```

Upload an image in the browser to view detected faces and landmarks, then
download the annotated result.

Choose another image, save an annotated copy, or run without opening a window:

```powershell
python single_face_detection.py --image images\hollywood.jpg --output output.jpg --no-display
```