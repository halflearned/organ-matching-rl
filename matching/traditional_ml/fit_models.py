from sys import argv, platform
import numpy as np
from datetime import date

# Algorithms
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neurasdasdasdl_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Sklearn utils
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.cs import confusion_matrix

from imblearn.over_sampling import RandomOverSampler

if platform=="darwin":
    argv = [None, "grb", "optn", "both"]

algo = argv[1]
envtype = argv[2]
add = argv[3]

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

    
X_train, X_test, y_train, y_test = train_test_split(XX, Y)

scaler = PCA(whiten=True, n_components=5)#StandardScaler() 
ros = RandomOverSampler()

#%%


algorithms = {
    "lr": LogisticRegressionCV(n_jobs=-1, penalty="l2",
                              solver="saga", verbose=True),
    "svc": SVC(C=10.0, kernel="rbf", gamma="auto",
               verbose=True),
    "rf": RandomForestClassifier(n_estimators=200, n_jobs=-1),      
    "mlp": MLPClassifier(hidden_layer_sizes=(100, 100)),
    "grb": GradientBoostingClassifier(n_estimators=1000)
    }

# Resampling instead of using "class_weight" produces better results (empiricaly)
X_train, y_train = ros.fit_sample(X_train, y_train)    

pipe = Pipeline(steps = [("scaler", scaler),
                         ("algo", algorithms[algo])])
    
pipe.fit(X_train, y_train)

#%%
joblib.dump(pipe, "{}_{}_{}.pkl".format(envtype, algo, add))

yhat = pipe.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, yhat).ravel()
print("TP/(TP+FN):",tp/(tp+fn), "TN:", tn/(tn+fp))


with open("results/confusion_matrices.txt", "a") as f:
    f.write("{} {} {} {} {} {} {} {}\n"\
            .format(date.today(), envtype, algo, add, tp, tn, fp, fn))


