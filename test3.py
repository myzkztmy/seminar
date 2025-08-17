import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

data = "/Users/mizuy/OneDrive/ドキュメント/zemi/instruments"

woodwind_data = os.path.join(data, "woodwind")
strings_data = os.path.join(data, "strings")

file = []
label = []

print("管楽器: 読み込み中…")

for filename in os.listdir(woodwind_data):
  if filename.endswith('.mp3'):
    file_path = os.path.join(woodwind_data, filename)
    file.append(file_path)
    label.append('管楽器')

print("管楽器: 読み込み完了")

print("弦楽器: 読み込み中…")

for filename in os.listdir(strings_data):
  if filename.endswith('.mp3'):
    file_path = os.path.join(strings_data, filename)
    file.append(file_path)
    label.append('弦楽器')

print("弦楽器: 読み込み完了")

print(f"取得したデータ: {len(file)}個")

print("MFCC抽出中…")

def extract_features(path):
    y, sr = librosa.load(path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_ave = np.mean(mfcc.T, axis=0)
    return mfcc_ave

X = []
Y = []

for i, file_path in enumerate(file):
    features = extract_features(file_path)
    X.append(features)
    Y.append(label[i])

print("MFCC抽出完了")

X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

print("SVMモデルの学習中…")

model = svm.SVC(kernel='rbf', C=10, random_state=42)
model.fit(X_train, Y_train)

print("SVMモデルの学習完了")

Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)

print(f"精度: {accuracy * 100:.2f}%")

audio_path = "/Users/mizuy/OneDrive/ドキュメント/zemi/separated/htdemucs_6s/test_B/other.mp3"

print("楽器予測を開始")

try:
  audio_features = extract_features(audio_path)
  predicted_instrument = model.predict([audio_features])
except Exception as e:
  print(f"エラー: {e}")

print("終了")


print(f"予測された楽器: {predicted_instrument[0]}")