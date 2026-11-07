# Face Detection and Recognition using Deep Learning

A Python project that detects faces in images with **MTCNN** and recognizes people using deep learning face embeddings (**FaceNet / InceptionResnetV1**) with an SVM classifier.
<img width="1897" height="980" alt="image" src="https://github.com/user-attachments/assets/d2f80786-0c4e-47b3-acdf-077397d1e71f" />


## Features

- Face detection in images and live webcam video (MTCNN)
- Face recognition using pretrained deep learning embeddings
- Unknown-person rejection using a confidence threshold
- Works with small custom datasets (10-20 images per person)

## Project Structure

```
face detection/
├── images/                     # test images (e.g. vk.jpg)
├── dataset/                    # training data, one folder per person
│   ├── person_1/
│   └── person_2/
├── single_face_detection.py    # detect faces in one image (MTCNN + OpenCV)
├── face_recognition.py         # train, predict and webcam recognition
├── face_classifier.joblib      # saved model (created after training)
└── README.md
```

## Requirements

- Python 3.10 or 3.11 (recommended)
- Windows, Linux or macOS
- Webcam (optional, for live recognition)

## Installation

```
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

python -m pip install opencv-python mtcnn tensorflow
python -m pip install facenet-pytorch scikit-learn pillow tqdm joblib
```

If you see `No module named 'cv2'`, use `python -m pip install ...` so the packages install into the same Python that runs your script.

## Usage

### 1. Detect faces in an image

Place your image in the `images/` folder, then run:

```
python single_face_detection.py
```

A window opens with green boxes drawn around detected faces. Press any key to close it.

### 2. Prepare the dataset

Create one folder per person inside `dataset/` and add 10 or more clear photos of each person, with varied angles and lighting:

```
dataset/
├── alice/  1.jpg 2.jpg ...
└── bob/    1.jpg 2.jpg ...
```

### 3. Train the model

```
python face_recognition.py train --data dataset
```

This prints test accuracy and a classification report, and saves `face_classifier.joblib`.

### 4. Recognize a face

From an image:

```
python face_recognition.py predict --image images/test.jpg
```

From the webcam (press `q` to quit):

```
python face_recognition.py webcam
```

## How It Works

1. **Detection:** MTCNN finds and crops the face in each image.
2. **Embedding:** InceptionResnetV1 (pretrained on VGGFace2) converts each face into a 512-dimensional vector. Faces of the same person lie close together.
3. **Classification:** An SVM is trained on the embeddings to predict the person's name. If confidence is below `THRESHOLD` (default 0.60), the face is labelled "unknown".

## Datasets

- LFW (Labeled Faces in the Wild): http://vis-www.cs.umass.edu/lfw/
- CelebA: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- Olivetti Faces: built into scikit-learn (`fetch_olivetti_faces`)
- VGGFace2 and CASIA-WebFace: available through Kaggle mirrors
- Your own photos (with permission) also work well

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'cv2'` | `python -m pip install opencv-python` |
| `imshow` error / image is `None` | Check the image path, e.g. `images/vk.jpg` |
| `IndexError` on `faces[0]` | No face was detected; use a clearer image |
| TensorFlow install fails | Use Python 3.10 or 3.11 |

## Privacy Note

Face data is sensitive. Only use photos of people who have given their consent, and store them securely.

## Author

Zaima

## License

For educational use.
