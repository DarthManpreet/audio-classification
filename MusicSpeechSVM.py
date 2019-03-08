import matplotlib.pyplot as plt
import numpy as np
import os
import librosa.display
import math
import pandas as pd
from sklearn import svm
from sklearn.model_selection import KFold

MINFOLDS = 2
MAXFOLDS = 10

def read_data(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        clip, sr = librosa.load(filepath)
        data.append(clip)
    return np.array(data)

def unison_shuffled_copies(a, b):
    """Taken from stack overflow. simultaneously permutes two numpy arrays"""
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def preprocess(data1, data2, train_to_test_ratio):
    data_indices = np.arange(0, len(data1))
    num_training_examples = int(math.ceil(len(data_indices) * train_to_test_ratio))
    training_indices = np.random.choice(data_indices, num_training_examples, replace=False)
    testing_indices = np.delete(data_indices, training_indices)
    np.random.shuffle(data1)
    np.random.shuffle(data2)
    training_data = np.concatenate((data1[training_indices], data2[training_indices]))
    testing_data = np.concatenate((data1[testing_indices], data2[testing_indices]))
    training_labels = np.append(np.full(num_training_examples, 0), np.full(num_training_examples, 1))
    testing_labels = np.append(np.full(len(data_indices) - num_training_examples, 0),
                               np.full(len(data_indices) - num_training_examples, 1))
    training_data, training_labels = unison_shuffled_copies(training_data, training_labels)
    testing_data, testing_labels = unison_shuffled_copies(testing_data, testing_labels)
    return training_data, testing_data, training_labels, testing_labels

def spectrogram(signal):
    sample_rate = 22050
    S = librosa.feature.melspectrogram(y=signal, sr = sample_rate)
    print np.shape(np.array(S))
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S,ref = np.max),
                             y_axis = 'mel', fmax = 8000,
                             x_axis = 'time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()

# Constructs confusion matrix with actual answers and
# predicted answers, which are provided as input
def confusion_matrix(ans, predicts):
    confusion_answer = pd.Series(ans, name='Answer')
    confusion_predicted = pd.Series(predicts, name='Predicted')
    return pd.crosstab(confusion_answer, confusion_predicted)

# Calculates Accuracy of prediction against ground truth labels
def prediction_stats(ans, predicts):
    total_correct = 0
    for i in range(len(predicts)):
        if ans[i] == predicts[i]:
            total_correct = total_correct + 1
    return total_correct / float(len(predicts))


print("\nReading Information")
speech_folder_path = '/Users/Asher/Downloads/musicspeech/speechwav'
music_folder_path = '/Users/Asher/Downloads/musicspeech/musicwav'
speech_data = read_data(speech_folder_path)
music_data = read_data(music_folder_path)
print("Finished Reading")
data = np.concatenate((speech_data, music_data))
labels = np.append(np.full(len(speech_data), 0),
                   np.full(len(music_data), 1))

time = []
accuracy = []
cm = []
for i in range(MINFOLDS, MAXFOLDS+1, 2):
    print("\n\nNUM SPLITS : " + str(i) + "\n")
    kf = KFold(n_splits=i)
    episode_accuracy = np.full((kf.get_n_splits(data)), 0, dtype=float)
    episode_cm = np.full((kf.get_n_splits(data),2,2), 0, dtype=float)
    data, labels = unison_shuffled_copies(data, labels)
    j = 0
    for train_index, test_index in kf.split(data):
        train_data, test_data = data[train_index], data[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]
        clf = svm.SVC(gamma='scale')
        print "\tFitting"
        clf.fit(train_data, train_labels)
        print "\tPredicting"
        test_predictions = clf.predict(test_data)
        current_accuracy = prediction_stats(test_labels, test_predictions)
        current_cm = confusion_matrix(test_labels, test_predictions)
        episode_accuracy[j] = current_accuracy
        episode_cm[j] = current_cm
        print("\tCurrent Accuracy: " + str(episode_accuracy[j]))
        j = j + 1
    print("\nAverage Accuracy: " + str(np.mean(np.array(episode_accuracy))))
    print("Average Confusion Matrix:")
    print(str(np.mean(np.array(episode_cm), axis=0)))
    accuracy.append(np.mean(np.array(episode_accuracy)))
    cm.append(np.mean(np.array(episode_cm), axis=0))
    time.append(i)

best_accuracy = np.argmax(np.array(accuracy))
print("\n\n CONCLUSION\n")
print("The best accuracy is " + str(np.array(accuracy)[best_accuracy]))
print("It has confusion matrix")
print(np.array(cm)[best_accuracy])
print("And it occurred when the K-Folds split was "
       + str(2 + 2*best_accuracy))
plt.plot(time, accuracy)
plt.show()

# spectrogram(speech_data[0])
# print np.shape(speech_data[0])

# time = []
# accuracy = []
# cm = []
# for i in range(50, 91, 5):
#     episode_accuracy = np.full((NUMEPOCHS), 0, dtype=float)
#     episode_cm = np.full((NUMEPOCHS,2,2), 0, dtype=float)
#     for j in range(NUMEPOCHS):
#         train_data, test_data, train_labels, test_labels = preprocess(speech_data, music_data, i/float(100))
#         clf = svm.SVC(gamma='scale')
#         print "fitting"
#         clf.fit(train_data, train_labels)
#         print "predicting"
#         test_predictions = clf.predict(test_data)
#         current_accuracy = prediction_stats(test_labels, test_predictions)
#         current_cm = confusion_matrix(test_labels, test_predictions)
#         episode_accuracy[j] = current_accuracy
#         episode_cm[j] = current_cm
#         print("Current Accuracy: ", episode_accuracy[j])
#         print("Current Confusion Matrix: ", episode_cm[j])
#     print np.mean(np.array(episode_cm), axis=0)
#     print np.mean(np.array(episode_accuracy))
#     accuracy.append(np.mean(np.array(episode_accuracy)))
#     time.append(i)
#     cm.append(np.mean(np.array(episode_cm), axis=0))
#
# best_accuracy = np.argmax(np.array(accuracy))
# print("The best accuracy is " + str(np.array(accuracy)[best_accuracy]))
# print("It has confusion matrix")
# print(np.array(cm)[best_accuracy])
# print("And it occurred when the training to test ratio was "
#        + str(50 + 5*best_accuracy) + "%")
# plt.plot(time, accuracy)
# plt.show()






