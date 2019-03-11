import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from keras import layers
from keras.models import Sequential
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers import Dense
from keras import optimizers

class Dataset:
    def __init__(self):
        self.train_data = [] 
        self.train_label = []

        self.test_data = []
        self.test_label = []

        speech_folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'music_speech/speech_wav')
        music_folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'music_speech/music_wav')

        #speech_sum = sum([ord(c) for c in "Speech"])
        #music_sum = sum([ord(c) for c in "Music"])
        speech_sum = 0
        music_sum = 1

        file_num = 48
        count = 0
        for speech_file in os.listdir(speech_folder_path):
            clip, sr = librosa.load(os.path.join(speech_folder_path, speech_file))
            if count < file_num:
                self.train_data.append(librosa.feature.melspectrogram(y=clip, sr=sr))
                self.train_label.append([1,0])
            else:
                self.test_data.append(librosa.feature.melspectrogram(y=clip, sr=sr))
                self.test_label.append([1,0])
            count += 1
        
        count = 0
        for music_file in os.listdir(music_folder_path):
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

ds = Dataset()
print(ds.train_data.shape)
print(ds.train_label.shape)

#opt = optimizers.SGD(lr=0.5, momentum=0.5, decay=0.0, nesterov=False)
opt = optimizers.Adam()
batch_size = 1
nb_epochs = 50

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

#print(model.predict(x=ds.test_data, batch_size=1))
"""
def read_data(folder_path):
    #Reads .wav files from folder_path into a data array
    data = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        clip, sr = librosa.load(filepath)
        data.append(librosa.feature.melspectrogram(y=clip, sr=sr))
    return np.array(data)

print("\nLoading .wav files")
speech_folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'music_speech/speech_wav')
music_folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'music_speech/music_wav')

speech_data = read_data(speech_folder_path)
music_data = read_data(music_folder_path)
print("Finished Loading")
f_data = np.concatenate((speech_data, music_data))
print(f_data.shape)


#print(len(f_data[0]))
#print(len(f_data[-1]))

#print(f_data)
#print(len(f_data))


plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(f_data[-1], ref=np.max))
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.savefig("test_1.png")
"""

"""
opt = Adam()

print('Build LSTM RNN model ...')
model = Sequential()
model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=f_data[0].shape))
model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
model.add(Dense(units=2, activation='softmax'))


print("Compiling ...")
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

print("Training ...")
model.fit(genre_features.train_X, genre_features.train_Y, batch_size=batch_size, epochs=nb_epochs, callbacks= [TensorBoard(log_dir='./temp/log')])
"""