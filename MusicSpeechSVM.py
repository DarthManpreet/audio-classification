import matplotlib.pyplot as plt
import numpy as np
import os
import librosa.display
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
    """Reads .wav files from folder_path into a data array"""
    data = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        clip, sr = librosa.load(filepath)
        data.append(clip)
    return np.array(data)

def freq_conversion(time_data, sample_rate):
    """Converts time series data into a spectrogram of frequencies.
    Specifically, the Mel-Spectrogram"""
    freq_data = []
    for i in range(len(time_data)):
        freq_data.append(
            np.array(
                librosa.feature.melspectrogram(y=time_data[i], sr=sample_rate))
            .flatten())
    return np.array(freq_data)

def unison_shuffled_copies(a, b, c):
    """Taken from stack overflow. Identically permutes three arrays"""
    assert len(a) == len(b) and len(a) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]

def confusion_matrix(ans, predicts):
    """Constructs confusion matrix with actual answers and
    predicted answers, which are provided as input"""
    confusion_answer = pd.Series(ans, name='Answer')
    confusion_predicted = pd.Series(predicts, name='Predicted')
    return pd.crosstab(confusion_answer, confusion_predicted)

def prediction_stats(ans, predicts):
    """Calculates Accuracy of prediction against ground truth labels"""
    total_correct = 0
    for i in range(len(predicts)):
        if ans[i] == predicts[i]:
            total_correct = total_correct + 1
    return total_correct / float(len(predicts))

def voting_prediction_stats(ans, t1, f1, t2, f2):
    """Calculates accuracy of prediction against ground truth labels.
    The prediction is generated by majority vote among 4 experts.
    If there's a tie then a random prediction is guessed"""
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

# Read all .wav files, convert them to frequency domain, and then
# generate ground truth labels for data.
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
# This algorithm uses a support vector machine trained on frequency data
# to make its predictions. A linear kernel is used for the SVM, and it is
# AdaBoosted to increase its performance by around 3%.
print("\n--------------------")
print("ADABOOST")
print("--------------------")

# Arrays for printing out average accuracy and confusion matrices
# over the course of an epoch
time = []
accuracy = []
cm = []

# Perform K-Folds Cross-Validation on the dataset, where 'K' is
# controlled by the range of the 'for' loop
for i in range(MINFOLDS, MAXFOLDS+1, FOLDSTEP):
    # Prepare for training epoch. Set the number of folds, zero out
    # epoch arrays for accuracy and confusion matrices, and shuffle
    # time and frequency data. Time data is shuffled here even though
    # it's not used to allow for the data to be lined up with the freq
    # data for the following 'Mixture of Experts' algorithm.
    print("\n\nNUM FOLDS : " + str(i) + "\n")
    kf = KFold(n_splits=i)
    episode_accuracy = np.full((kf.get_n_splits(f_data)), 0, dtype=float)
    episode_cm = np.full((kf.get_n_splits(f_data),2,2), 0, dtype=float)
    t_data, f_data, labels = unison_shuffled_copies(t_data, f_data, labels)
    j = 0
    for train_index, test_index in kf.split(f_data):
        # Pull out the training and test data specified by the indices
        # generated by K-Folds
        train_f_data, test_f_data = f_data[train_index], f_data[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        # Create, fit, and predict with an AdaBoosted Support Vector Machine
        # which uses a linear kernel.
        flin_clf = AdaBoostClassifier(svm.LinearSVC(),
                        n_estimators=500, learning_rate=1.0, algorithm='SAMME')
        print("\tFitting")
        flin_clf.fit(train_f_data, train_labels)
        print("\tPredicting")
        flin_test_predictions = flin_clf.predict(test_f_data)

        # Check accuracy and confusion matrix of the model and store for future
        # analysis
        current_accuracy = prediction_stats(test_labels, flin_test_predictions)
        current_cm = confusion_matrix(test_labels, flin_test_predictions)
        episode_accuracy[j] = current_accuracy
        episode_cm[j] = current_cm
        print("\tCurrent Accuracy: " + str(episode_accuracy[j]))
        j = j + 1

    # Compute average performance over the K-Folding epoch.
    print("\nAverage Accuracy: " + str(np.mean(np.array(episode_accuracy))))
    print("Average Confusion Matrix:")
    print(str(np.mean(np.array(episode_cm), axis=0)))
    accuracy.append(np.mean(np.array(episode_accuracy)))
    cm.append(np.mean(np.array(episode_cm), axis=0))
    time.append(i)

# Print out results of training and testing
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
# The same procedure will be done as above, only four SVM's will
# trained and used for prediction. In this case, the majority vote
# of the models will be the prediction, with a tie decided by a
# coin flip. The SVM's used are trained on either time or frequency
# data, and use either a linear or gaussian kernel. For more detail
# on the algorithm, look to the AdaBoost algorithm above.
print("\n--------------------")
print("MIXTURE OF EXPERTS")
print("--------------------")

time = []
accuracy = []
cm = []
for i in range(MINFOLDS, MAXFOLDS+1, FOLDSTEP):
    print("\n\nNUM FOLDS : " + str(i) + "\n")
    kf = KFold(n_splits=i)
    episode_accuracy = np.full((kf.get_n_splits(f_data)), 0, dtype=float)
    episode_cm = np.full((kf.get_n_splits(f_data),2,2), 0, dtype=float)
    t_data, f_data, labels = unison_shuffled_copies(t_data, f_data, labels)
    j = 0
    for train_index, test_index in kf.split(f_data):
        train_t_data, test_t_data = t_data[train_index], t_data[test_index]
        train_f_data, test_f_data = f_data[train_index], f_data[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        # Here is where the four models are fitted and used for prediction.
        tlin_clf = svm.SVC(gamma='scale', kernel='linear')
        flin_clf = svm.SVC(gamma='scale', kernel='linear')
        trbf_clf = svm.SVC(gamma='scale', kernel='rbf')
        frbf_clf = svm.SVC(gamma='scale', kernel='rbf')
        print("\tFitting")
        tlin_clf.fit(train_t_data, train_labels)
        flin_clf.fit(train_f_data, train_labels)
        trbf_clf.fit(train_t_data, train_labels)
        frbf_clf.fit(train_f_data, train_labels)
        print("\tPredicting")
        tlin_test_predictions = tlin_clf.predict(test_t_data)
        flin_test_predictions = flin_clf.predict(test_f_data)
        trbf_test_predictions = trbf_clf.predict(test_t_data)
        frbf_test_predictions = frbf_clf.predict(test_f_data)

        # The models then vote on the test data, with ties being broken by
        # a coin flip. The combined final prediction is returned for use
        # when creating the confusion matrix.
        current_accuracy, comb_predictions = voting_prediction_stats(test_labels,
                                                tlin_test_predictions, flin_test_predictions,
                                                trbf_test_predictions, frbf_test_predictions)
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
