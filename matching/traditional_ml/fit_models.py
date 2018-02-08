from sys import argv, platform
import numpy as np
from datetime import date


# Algorithms
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Sklearn utils
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import RandomOverSampler

import autosklearn.classification
import sklearn.model_selection


if platform=="darwin":
    argv = [None, "auto", "optn", "none", "scaler"]

algo = argv[1]
envtype = argv[2]
add = argv[3]
preprocess = argv[4]

X = np.load("data/X_{}.npy".format(envtype))
Y = np.load("data/Y_{}.npy".format(envtype))


if add == "node2vec":
    E = np.load("data/E_{}.npy".format(envtype))
    XX = np.hstack([X, E])
elif add == "networkx":
    G = np.load("data/G_{}.npy".format(envtype))
    XX = np.hstack([X, G])
elif add == "both": 
    G = np.load("data/G_{}.npy".format(envtype))
    E = np.load("data/E_{}.npy".format(envtype))
    XX = np.hstack([X, G, E])
elif add == "none":
    XX = X


if preprocess == "pca":    
    scaler = PCA(whiten=True, n_components=5)

elif preprocess == "scaler":
    scaler = StandardScaler()

else:
    ValueError("Unknown preprocessing option")

    
X_train, X_test, y_train, y_test = train_test_split(XX, Y)

ros = RandomOverSampler()

#%%


algorithms = {
    "lr": LogisticRegressionCV(n_jobs=-1, penalty="l2",
                              solver="saga", verbose=True),
    "svc": SVC(C=10.0, kernel="rbf", gamma="auto",
               verbose=True),
    "rf": RandomForestClassifier(n_estimators=5000, n_jobs=-1),      
    "mlp": MLPClassifier(hidden_layer_sizes=(100, 100)),
    "grb": GradientBoostingClassifier(n_estimators=1000),
    "auto": autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=10*3600)
    }

# Resampling instead of using "class_weight" produces better results (empiricaly)
X_train, y_train = ros.fit_sample(X_train, y_train)

pipe = Pipeline(steps = [("scaler", scaler),
                         ("algo", algorithms[algo])])
    
pipe.fit(X_train, y_train)

#%%
joblib.dump(pipe, "{}_{}_{}_{}.pkl".format(envtype, algo, add, preprocess))

yhat = pipe.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, yhat).ravel()
print("TP/(TP+FN):",tp/(tp+fn), "TN/(TN+FP):", tn/(tn+fp))


if platform == "linux":
    with open("results/confusion_matrices2.txt", "a") as f:
        f.write("{} {} {} {} {} {} {} {} {}\n"\
                .format(date.today(), envtype, algo, add, preprocess, tp, tn, fp, fn))
    

#%%
