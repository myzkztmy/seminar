import librosa
import numpy as np
from sklearn import svm


def extract_features(path):
  y, sr = librosa.load(path, sr=None)
  mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
  mfcc_ave = np.mean(mfcc.T, axis=0)
  return mfcc_ave

print(extract_features("/Users/mizuy/OneDrive/ドキュメント/zemi/test.wav"))