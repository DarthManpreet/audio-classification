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
        """
        Initalize the dataset into training and test data
        """
        #Training data and corresponding label
        self.train_data = [] 
        self.train_label = []

        #Test data and corresponding label
        self.test_data = []
        self.test_label = []

        #Folder paths
        speech_folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'music_speech/speech_wav')
        music_folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'music_speech/music_wav')

        #Hack to randomize the training data, just fetch the file names in random order
        speech_files = self.shuffle_audio_files(folder_path=speech_folder_path)
        music_files = self.shuffle_audio_files(folder_path=music_folder_path)

        #Number of files to add in the training data
        file_num = 50

        #Seperate speech audio into training and test data
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
        
        #Separate music audio into training and test data
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

        #Convert to NP arrays
        self.train_data = np.array(self.train_data)
        self.train_label = np.array(self.train_label)

        self.test_data = np.array(self.test_data)
        self.test_label = np.array(self.test_label)
    
    def shuffle_audio_files(self, folder_path=None):
        """
        This function takes the files in the folder path
        and returns a random order of files
        Parameters:
            1) folder_path = absolute path to get audio files
        Returns:
            audio_files = list containing randomized order of files in 
                          that directory
        """
        audio_files = []
        if os.path.isdir(folder_path):
            for audio_file in os.listdir(folder_path):
                audio_files.append(audio_file)
        else:
            print ("Directory " + str(folder_path) + " does not exist")
            sys.exit(1)

        random.shuffle(audio_files)
        return audio_files

#Initialize the dataset
ds = Dataset()
print(ds.train_data.shape)
print(ds.train_label.shape)

#Use Adam Optimzers
opt = optimizers.Adam()

#Configure Batch Size and Training Epochs
batch_size = 3
nb_epochs = 15

#Build NN
print('Build LSTM RNN model ...')
model = Sequential()
model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=(ds.train_data.shape[1], ds.train_data.shape[2])))
model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
model.add(Dense(units=2, activation='softmax'))

#Compile the NN
print("\nCompiling ...")
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

#Train the model
print("\nTraining ...")
model.fit(x=ds.train_data, y=ds.train_label, batch_size=batch_size, epochs=nb_epochs)
            
#Test the model
print("\nTesting ...")
score, accuracy = model.evaluate(x=ds.test_data, y=ds.test_label, batch_size=batch_size, verbose=1)
print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)