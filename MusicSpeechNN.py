import os
import sys
import random
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from keras import layers
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras import optimizers

class Dataset:
    """
    This class organizes the GTZAN dataset into training and testing data
    """
    def __init__(self):
        self.train_data = [] 
        self.train_label = []

        self.test_data = []
        self.test_label = []

        speech_folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'music_speech/speech_wav')
        music_folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'music_speech/music_wav')

        speech_files = self.shuffle_audio_files(folder_path=speech_folder_path)
        music_files = self.shuffle_audio_files(folder_path=music_folder_path)

        file_num = 48
        count = 0
        for speech_file in speech_files:
            clip, sr = librosa.load(os.path.join(speech_folder_path, speech_file))
            if count < file_num:
                self.train_data.append(librosa.feature.melspectrogram(y=clip, sr=sr))
                self.train_label.append([1,0])
            else:
                self.test_data.append(librosa.feature.melspectrogram(y=clip, sr=sr))
                self.test_label.append([1,0])
            count += 1
        
        count = 0
        for music_file in music_files:
            clip, sr = librosa.load(os.path.join(music_folder_path, music_file))
            if count < file_num:
                self.train_data.append(librosa.feature.melspectrogram(y=clip, sr=sr))
                self.train_label.append([0,1])
            else:
                self.test_data.append(librosa.feature.melspectrogram(y=clip, sr=sr))
                self.test_label.append([0,1])
            count += 1

        self.train_data = np.array(self.train_data)
        self.train_label = np.array(self.train_label)

        self.test_data = np.array(self.test_data)
        self.test_label = np.array(self.test_label)
    
    def shuffle_audio_files(self, folder_path=None):
        audio_files = []
        if os.path.isdir(folder_path):
            for audio_file in os.listdir(folder_path):
                audio_files.append(audio_file)
        else:
            print ("Directory " + str(folder_path) + " does not exist")
            sys.exit(1)

        random.shuffle(audio_files)
        return audio_files


ds = Dataset()
print(ds.train_data.shape)
print(ds.train_label.shape)

opt = optimizers.Adam()
batch_size = 3
nb_epochs = 15

print('Build LSTM RNN model ...')
model = Sequential()
model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.00, return_sequences=True, input_shape=(ds.train_data.shape[1], ds.train_data.shape[2])))
model.add(LSTM(units=64, dropout=0.05, recurrent_dropout=0.00, return_sequences=False))
model.add(Dense(units=2, activation='softmax'))

print("\nCompiling ...")
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

print("\nTraining ...")
model.fit(x=ds.train_data, y=ds.train_label, batch_size=batch_size, epochs=nb_epochs)
            
print("\nTesting ...")
score, accuracy = model.evaluate(x=ds.test_data, y=ds.test_label, batch_size=batch_size, verbose=1)
print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)