from IPython.display import Audio, Video
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy import SpotifyException
import urllib
import librosa
from pydub import AudioSegment
import io
import xlrd
import warnings
from scipy.io.wavfile import write
import gc
import random
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPool2D, MaxPooling2D, Flatten, Input, BatchNormalization
from numpy import array
import importlib
import tensorflow_hub as hub
warnings.filterwarnings('ignore')
#import torch
#from diffusers import StableDiffusionPipeline

#model_id = 'OFA-Sys/small-stable-diffusion-v0'
#pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker = None, requires_safety_checker = False)
#pipe = pipe.to('cuda')

# emotion
# parameters
sr = 48000 # 採樣率(每秒採用的樣本數)
duration = 4 # 音頻持續時間
sample_sz = sr * duration

def gen_dataset(audio_url, x_audio, x_feature):
    global sr, duration, sample_sz

    raw_data, sr = librosa.load(audio_url, sr=sr, mono=True, offset=0.0) # raw_data=數據資料, mono=是否轉為單聲道, offset=從幾秒開始加載
    stride_duration = 1
    stride = int(sr * stride_duration)

    for i in range(0, len(raw_data), stride):
        split_audio = raw_data[i:i+sample_sz]
        if (len(split_audio) < sample_sz):
            break

        feature_mfcc = librosa.feature.mfcc(y=split_audio, sr=sr, hop_length=sr, n_mfcc=1)
        feature_mfcc = feature_mfcc.reshape((1, feature_mfcc.shape[1] * feature_mfcc.shape[0]))
        feature_rolloff = librosa.feature.spectral_rolloff(y=split_audio, sr=sr, hop_length=sr, roll_percent=0.85)
        feature_spectral_contrast = librosa.feature.spectral_contrast(y=split_audio, sr=sr, hop_length=sr, n_bands=1)
        feature_spectral_contrast = feature_spectral_contrast.reshape(
            (1, feature_spectral_contrast.shape[1] * feature_spectral_contrast.shape[0]))
        feature_rms = librosa.feature.rms(y=split_audio, hop_length=sr)
        feature_combine = np.hstack((feature_rms, feature_mfcc))
        feature_combine = np.hstack((feature_combine, feature_rolloff))
        feature_combine = np.hstack((feature_combine, feature_spectral_contrast))

        x_audio.append(split_audio)
        x_feature.append(feature_combine[0])

x_audio = []
x_feature = []

#audio_file_path = 'Music-emotion/music-emotion/song/Four Seasons_Summer 3rd movment/split_003.wav'
audio_file_path = 'Find a Way or Make One_0002.mp3'
gen_dataset(audio_file_path, x_audio, x_feature)

x_audio = np.array(x_audio)
x_feature = np.array(x_feature)

model_loss_path = 'model_loss1.keras'
model = tf.keras.models.load_model(model_loss_path)

pred_emotions = []
for idx in range(len(x_audio)):
    prediction = model.predict([x_audio[idx].reshape(1, 192000, 1), x_feature[idx].reshape(1, 25, 1)])
    pred_emotion = np.array(['happy', 'tensional', 'sad', 'peaceful'])[np.argmax(prediction, axis=1)]
    pred_emotion = str(pred_emotion)[2:-2]
    if pred_emotion == 'tensional':
        pred_emotion = 'sad'
    elif pred_emotion == 'peaceful':
        pred_emotion = 'happy'
    pred_emotions.append(pred_emotion)
pred_emo = max(pred_emotions)
print("emotion:", pred_emo)

# gerne
vggish = hub.load('https://www.kaggle.com/models/google/vggish/frameworks/TensorFlow2/variations/vggish/versions/1')

def extractFeatures(audioFile):
    try:
        # 載入音頻文件
        waveform, sr = librosa.load(audioFile)
        
        # 剪切沉默部分
        waveform, _ = librosa.effects.trim(waveform)
        
        # 使用 VGGish 提取特徵
        vggish_features = vggish(waveform).numpy()
        
        # 檢查特徵的形狀
        if len(vggish_features.shape) == 2:
            return vggish_features
        else:
            print('提取的特徵不是二維的。')
            return None
    except Exception as e:
        # 印出更詳細的錯誤信息
        print(f"提取特徵時發生錯誤: {e} \n檔案路徑: {audioFile}")
        return None
    
def predict_music_style_from_audio(audio_file):
    # 預處理音樂數據
    preprocessed_audio = extractFeatures(audio_file)
    
    preprocessed_audio = preprocessed_audio.reshape(1, preprocessed_audio.shape[0], preprocessed_audio.shape[1], 1)

    # 將音樂數據提供給模型進行預測
    prediction = model.predict(preprocessed_audio)
    
    # 返回預測結果
    return prediction

def cnn_melspect_2D(input_shape):
    """
    創建一個卷積神經網絡模型，用於處理2D Mel Spectrogram音頻特徵。

    Parameters:
    - input_shape (tuple): 輸入特徵的形狀，例如 (43, 128, 1)。

    Returns:
    - model (keras.Model): 創建的卷積神經網絡模型。
    """
    kernel_size = (3, 3)
    activation_func = 'relu'
    inputs = Input(shape=input_shape)

    # Convolutional block_1
    conv1 = Conv2D(32, kernel_size, activation=activation_func, padding='same')(inputs)
    pool1 = MaxPooling2D()(conv1)
    bn1 = BatchNormalization()(pool1)
    dropout1 = Dropout(0.2)(bn1)

    # Convolutional block_2
    conv2 = Conv2D(64, kernel_size, activation=activation_func, padding='same')(dropout1)
    pool2 = MaxPooling2D()(conv2)
    bn2 = BatchNormalization()(pool2)
    dropout2 = Dropout(0.2)(bn2)


    # Convolutional block_3
    conv1 = Conv2D(128, kernel_size, activation=activation_func, padding='same')(inputs)
    pool1 = MaxPooling2D()(conv1)
    bn1 = BatchNormalization()(pool1)
    dropout1 = Dropout(0.2)(bn1)

    # Flatten layer
    flat = Flatten()(dropout2)

    # Regular MLP
    dense1 = Dense(64, activation=activation_func)(flat)
    bn3 = BatchNormalization()(dense1)
    dropout3 = Dropout(0.2)(bn3)

    # Output layer
    output_layer = Dense(10, activation='softmax')(dropout3)

    model = Model(inputs=inputs, outputs=output_layer)
    return model

# 創建模型
input_shape = (43, 128, 1)  # 假設輸入形狀為 (43, 128, 1)
model = cnn_melspect_2D(input_shape)
model.load_weights('music_style_recognition_model.h5')

# 在應用程序中調用函數並顯示結果
predicted_style = predict_music_style_from_audio(audio_file_path)

# 假設 predicted_style 是從模型獲得的預測結果的向量
predicted_class_index = np.argmax(predicted_style)

class_labels = ['blues', 'classical', 'country', 'folk', 'hiphop', 'jazz', 'metal', 'opera', 'pop', 'rock']

# 使用提供的 class_labels 從中獲取對應的音樂風格類別
pred_style = class_labels[predicted_class_index]

# 打印預測的音樂風格類別
print("style:", pred_style)

prompt = f'A man who is {pred_emo} and {pred_style}.'
print(prompt)
#image = pipe(prompt).images[0]
#image.save('test.png')
