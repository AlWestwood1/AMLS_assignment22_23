from os.path import join
from pathlib import Path
ROOT = Path(__file__).parent #Root directory is the folder this file is placed in
import pandas as pd
from importlib.machinery import SourceFileLoader
somemodule = SourceFileLoader('lab2_landmarks', join(ROOT, "lab2_landmarks.py")).load_module() #Importing feature extraction program
import lab2_landmarks as l2
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif




def get_data():
    
    tr_X, tr_y = l2.extract_features_labels_train() #Extract features from training data
    tr_Y = np.array([tr_y, -(tr_y - 1)]).T
    print("Training features extracted")
    
    te_X, te_y = l2.extract_features_labels_test() #Extract features from test data
    te_Y = np.array([te_y, -(te_y - 1)]).T
    print("Testing features extracted")

    return tr_X, tr_Y, te_X, te_Y

def feature_selection(tr_X2, tr_Y2, te_X2):

    #Select the best K features using the ANOVA F-value
    sel = SelectKBest(f_classif, k=62)

    #Fit to training data and return
    selected_features = sel.fit(tr_X2, tr_Y2)
    indices_selected = selected_features.get_support(indices=True)
    columns_selected = [tr_X2.columns[i] for i in indices_selected]

    tr_Xsel = tr_X2[columns_selected]
    te_Xsel = te_X2[columns_selected]

    return tr_Xsel, te_Xsel


def log_reg(xTrain, yTrain, xTest, yTest):
    classifier = LogisticRegression(max_iter = 10000)
    classifier.fit(xTrain, yTrain)

    pred = classifier.predict(xTest)

    accuracy = accuracy_score(yTest, pred)
    print("Accuracy of logistic regression:", accuracy)
    return accuracy

def random_forest(xTrain, yTrain, xTest, yTest):
    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(xTrain, yTrain)

    pred = classifier.predict(xTest)

    accuracy = accuracy_score(yTest, pred)
    print("Accuracy of random forest:", accuracy)
    return accuracy

def find_best_k(xTrain, yTrain, xTest, yTest):
    k_range = range(1, 70)
    k_scores = []

    for k in k_range:
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(xTrain, yTrain)

        pred = classifier.predict(xTest)

        score = accuracy_score(yTest, pred)
        k_scores.append(score)
    
    best_k = k_scores.index(max(k_scores)) + 1
    print(best_k)
    return best_k


def knn(xTrain, yTrain, xTest, yTest):
    classifier = KNeighborsClassifier(n_neighbors = 11)
    classifier.fit(xTrain, yTrain)

    pred = classifier.predict(xTest)

    accuracy = accuracy_score(yTest, pred)
    print("Accuracy of KNN:", accuracy)

def img_SVM(training_images, training_labels, test_images, test_labels):

    #Create SVM classifier and fit to training data
    classifier = svm.SVC(kernel='linear')
    classifier.fit(training_images, training_labels)

    #Find predictions for test images and calculate accuracy
    pred = classifier.predict(test_images)
    #print(pred)
    accuracy = accuracy_score(test_labels, pred)

    print("Accuracy of SVM:", accuracy)

    return accuracy


### MAIN PROGRAM ###
tr_X, tr_Y, te_X, te_Y= get_data() #get data

#print(tr_X.shape)
#print(te_X.shape)

#Reshape data into correct diminsions (panda dataframe used to make feature selection easier)
tr_X2 = pd.DataFrame(tr_X.reshape((4795, 68*2)))
tr_Y2 = list(zip(*tr_Y))[0]
te_X2 = pd.DataFrame(te_X.reshape((969, 68*2)))
te_Y2 = list(zip(*te_Y))[0]

#Remove unnecessary features
tr_Xsel, te_Xsel = feature_selection(tr_X2, tr_Y2, te_X2)

print(tr_Xsel.shape)

#Find accuracy of each algorithm
#best_k = find_best_k(tr_Xsel, tr_Y2, te_Xsel, te_Y2)
log_reg_acc = log_reg(tr_Xsel, tr_Y2, te_Xsel, te_Y2)
random_forrest_acc = random_forest(tr_Xsel, tr_Y2, te_Xsel, te_Y2)
knn_acc = knn(tr_Xsel, tr_Y2, te_Xsel, te_Y2)
SVM_acc=img_SVM(tr_Xsel, tr_Y2, te_Xsel, te_Y2)