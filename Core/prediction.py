#
# 생성된 모델을 불러와 예측하는 코드입니다.
#
#from tensorflow_docs.vis import embed
from tensorflow import keras
from imutils import paths

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os

# 하이퍼파라미터 정의
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

# 모델 불러오기
model = tf.keras.models.load_model('/home/jjaegii/django/DjangoFileUpload/Core/humanPoseModel.h5')

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

# 예측
def prepare_single_video(frames):
    feature_extractor = build_feature_extractor()
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(
                batch[None, j, :]
            )
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask

def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    #cap = cv2.VideoCapture("rtsp://jjaegii:12345@192.168.0.119:554/stream_ch00_0")
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

# This utility is for visualization.
# Referenced from:
# https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
def to_gif(images, path):
    converted_images = images.astype(np.uint8)
    imageio.mimsave("/home/jjaegii/django/DjangoFileUpload/media/Uploaded Files/" + path.replace(".mp4", ".gif"), converted_images, fps=10)
    # embed.embed_file("/home/jjaegii/django/DjangoFileUpload/media/Uploaded Files/" + path_to_gif + ".gif")

def sequence_prediction(path):
    train_df = pd.read_csv("/home/jjaegii/django/DjangoFileUpload/Core/train.csv")

    label_processor = keras.layers.StringLookup(
        num_oov_indices=0, vocabulary=np.unique(train_df["tag"])
    )
    class_vocab = label_processor.get_vocabulary()

    frames = load_video(os.path.join("/home/jjaegii/django/DjangoFileUpload/media/Uploaded Files", path))
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = model.predict([frame_features, frame_mask])[0]

    result = ""
    for i in np.argsort(probabilities)[::-1]:
        result += f"{class_vocab[i]}: {probabilities[i] * 100:5.2f}% "
        #print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    to_gif(frames, path)
    return result
    #return frames


#test_video = np.random.choice(test_df["video_name"].values.tolist())
#test_video = "v_ShavingBeard_g06_c01.avi"
#print(f"Test video path: {test_video}")
#test_frames = sequence_prediction(test_video)
#to_gif(test_frames[:MAX_SEQ_LENGTH])

# 입력받기
# predResult = sequence_prediction("v_CricketShot_g01_c01.avi")
# print(predResult)