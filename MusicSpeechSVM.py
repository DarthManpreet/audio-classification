import matplotlib.pyplot as plt
import numpy as np
import os
import librosa.display
import math
import pandas as pd
import random
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier

MINFOLDS = 6
MAXFOLDS = 8
FOLDSTEP = 1
SAMPLERATE = 22050

def read_data(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        clip, sr = librosa.load(filepath)
        data.append(clip)
    return np.array(data)

def freq_conversion(time_data, sample_rate):
    freq_data = []
    for i in range(len(time_data)):
        freq_data.append(
                         np.array(
                                  librosa.feature.melspectrogram(y=time_data[i], sr=sample_rate)).flatten())
    return np.array(freq_data)

def unison_shuffled_copies(a, b, c):
    """Taken from stack overflow. simultaneously permutes two numpy arrays"""
    assert len(a) == len(b) and len(a) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]

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

# Calculates Accuracy of prediction against ground truth labels
def voting_prediction_stats(ans, t1, f1, t2, f2):
    print(str(ans) + "Ground Truths")
    print(str(t1) + "Time Linear")
    print(str(f1) + "Freq Linear")
    print(str(t2) + "Time Gaussian")
    print(str(f2) + "Freq Guassian")
    final_predictions = t1 + f1 + t2 + f2
    for i in range(len(t1)):
        if final_predictions[i] > 2:
            final_predictions[i] = 1
        elif final_predictions[i] < 2:
            final_predictions[i] = 0
        else:
            final_predictions[i] = random.randint(0, 1)
    total_correct = 0
    for i in range(len(final_predictions)):
        if ans[i] == final_predictions[i]:
            total_correct = total_correct + 1
    print(str(np.array(final_predictions)) + "Final Predictions")
    return total_correct / float(len(final_predictions)), final_predictions

print("\nReading Information")
speech_folder_path = '/Users/Asher/Downloads/musicspeech/speechwav'
music_folder_path = '/Users/Asher/Downloads/musicspeech/musicwav'
speech_data = read_data(speech_folder_path)
music_data = read_data(music_folder_path)
print("Finished Reading")
t_data = np.concatenate((speech_data, music_data))
print("Converting to Frequency Data")
f_data = freq_conversion(t_data, SAMPLERATE)
print("Finished Conversion ")
labels = np.append(np.full(len(speech_data), 0),
                   np.full(len(music_data), 1))

#####################################
# AdaBoosted Support Vector Machine #
#####################################

time = []
accuracy = []
cm = []
for i in range(MINFOLDS, MAXFOLDS+1, FOLDSTEP):
    print("\n\nNUM SPLITS : " + str(i) + "\n")
    kf = KFold(n_splits=i)
    episode_accuracy = np.full((kf.get_n_splits(f_data)), 0, dtype=float)
    episode_cm = np.full((kf.get_n_splits(f_data),2,2), 0, dtype=float)
    _, f_data, labels = unison_shuffled_copies(t_data, f_data, labels)
    j = 0
    for train_index, test_index in kf.split(f_data):
        train_f_data, test_f_data = f_data[train_index], f_data[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]
        flin_clf = AdaBoostClassifier(svm.LinearSVC(), n_estimators=500, learning_rate=1.0, algorithm='SAMME')
        print "\tFitting"
        flin_clf.fit(train_f_data, train_labels)
        print "\tPredicting"
        flin_test_predictions = flin_clf.predict(test_f_data)
        current_accuracy = prediction_stats(test_labels, flin_test_predictions)
        current_cm = confusion_matrix(test_labels, flin_test_predictions)
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
      + str(MINFOLDS + FOLDSTEP*best_accuracy))
plt.plot(time, accuracy)
plt.show()


######################
# Mixture of Experts #
######################

time = []
accuracy = []
cm = []
for i in range(MINFOLDS, MAXFOLDS+1, FOLDSTEP):
    print("\n\nNUM SPLITS : " + str(i) + "\n")
    kf = KFold(n_splits=i)
    episode_accuracy = np.full((kf.get_n_splits(f_data)), 0, dtype=float)
    episode_cm = np.full((kf.get_n_splits(f_data),2,2), 0, dtype=float)
    t_data, f_data, labels = unison_shuffled_copies(t_data, f_data, labels)
    j = 0
    for train_index, test_index in kf.split(f_data):
        train_t_data, test_t_data = t_data[train_index], t_data[test_index]
        train_f_data, test_f_data = f_data[train_index], f_data[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]
        tlin_clf = svm.SVC(gamma='scale', kernel='linear')
        flin_clf = svm.SVC(gamma='scale', kernel='linear')
        trbf_clf = svm.SVC(gamma='scale', kernel='rbf')
        frbf_clf = svm.SVC(gamma='scale', kernel='rbf')
        print "\tFitting"
        tlin_clf.fit(train_t_data, train_labels)
        flin_clf.fit(train_f_data, train_labels)
        trbf_clf.fit(train_t_data, train_labels)
        frbf_clf.fit(train_f_data, train_labels)
        print "\tPredicting"
        tlin_test_predictions = tlin_clf.predict(test_t_data)
        flin_test_predictions = flin_clf.predict(test_f_data)
        trbf_test_predictions = trbf_clf.predict(test_t_data)
        frbf_test_predictions = frbf_clf.predict(test_f_data)
        current_accuracy, comb_predictions = voting_prediction_stats(test_labels,
                                                                     tlin_test_predictions,
                                                                     flin_test_predictions,
                                                                     trbf_test_predictions,
                                                                     frbf_test_predictions)
                                                                     current_cm = confusion_matrix(test_labels, comb_predictions)
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
      + str(MINFOLDS + FOLDSTEP*best_accuracy))
plt.plot(time, accuracy)
plt.show()
