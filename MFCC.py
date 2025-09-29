import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

TENPO_MARKINGS = [
    (0, "Larghissimo", "非常に遅く"),
    (40, "Grave", "重々しく"),
    (41, "Largo", "ゆったりと"),                     
    (49, "Lento", "遅く"),
    (53, "Adagio", "ゆるやかに"),
    (57, "Andante", "歩くような速さで"),
    (73, "Moderato", "中くらいの速さで"),
    (97, "Allegretto", "やや速く、いくらか陽気に"),
    (109, "Allegro", "速く、陽気に"),
    (120, "Vivo", "生き生きと、活発に"),
    (133, "Vivace", "活発に、きびきびと"),
    (161, "Presto", "急速に"),
    (185, "Prestissimo", "きわめて急速に")
]

def extract_features(path, n_mfcc=40):
  y, sr = librosa.load(path, sr=None)
  mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
  mfcc_ave = np.mean(mfcc.T, axis=0)
  return mfcc_ave

def tempo(path):
  y, sr = librosa.load(path, sr=None)
  tempo_val, _ = librosa.beat.beat_track(y=y, sr=sr)
  for lower, marking, description in reversed(TENPO_MARKINGS):
    if tempo_val.item() >= lower:
      return tempo_val.item(), marking, description

def load_instrument_data(data):
     
  file_paths = []
  labels = []

  instrument_categories = {
    'strings': '弦楽器',
    'woodwind': '木管楽器', # 既存のwoodwindを木管楽器として扱う
    'brass': '金管楽器'      # 新しく追加する金管楽器データ
  }

  print("楽器データ読み込み開始")
  for folder, label_name in instrument_categories.items():
    folder = os.path.join(data, folder)
    
    print(f"{label_name}: 読み込み中…")
    for filename in os.listdir(folder):
      if filename.endswith('.mp3'):
        file_path = os.path.join(folder, filename)
        file_paths.append(file_path)
        labels.append(label_name)
    print(f"  {label_name}: 読み込み完了 ({len(os.listdir(folder))}個)")

  print(f"楽器データの読み込み完了 (計 {len(file_paths)}個)")

  X = []
  Y = []

  print("MFCC抽出中…")

  for i, file_path in enumerate(file_paths):
      features = extract_features(file_path)
      X.append(features)
      Y.append(labels[i])

  print("MFCC抽出完了")

  return np.array(X), np.array(Y)

def SVM_model(X, Y):
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state, stratify=Y)
  
  print("SVMモデルの学習中…")
  model = svm.SVC(kernel='rbf', C=10, random_state=42)
  model.fit(X_train, Y_train)
  print("SVMモデルの学習完了")

  Y_pred = model.predict(X_test)
  accuracy = accuracy_score(Y_test, Y_pred)
  print(f"精度: {accuracy * 100:.2f}%")

  return model, accuracy


def predict_instrument(model, audio_path):
  print(f"楽器予測を開始: {audio_path}")
  features = extract_features(audio_path)
  if features is not None:
    predicted_instrument = model.predict([features])
    print(f"予測された楽器: {predicted_instrument[0]}")
    return predicted_instrument[0]


if __name__ == "__main__":
  BASE_DIR = "/Users/mizuy/OneDrive/ドキュメント/seminar"
  INSTRUMENT_DATA = os.path.join(BASE_DIR, "instruments")
  SEPARATED_DATA = os.path.join(BASE_DIR, "separated", "htdemucs_6s", "test_A")

  drum = os.path.join(SEPARATED_DATA, "other.mp3")
  bpm = tempo(drum)

  print(f"bpm: {bpm[0]:.0f}")
  print(f"速度標語: {bpm[1]} ({bpm[2]}")