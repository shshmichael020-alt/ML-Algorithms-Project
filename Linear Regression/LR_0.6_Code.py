import numpy as np
import pandas as pd

path = "MachineLearningCSV/MachineLearningCVE"

df1 = pd.read_csv(
    path + "/Tuesday-WorkingHours.pcap_ISCX.csv",
    low_memory=False
)

df2 = pd.read_csv(
    path + "/Wednesday-workingHours.pcap_ISCX.csv",
    low_memory=False
)

df3 = pd.read_csv(
    path + "/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    low_memory=False
)

dataset = pd.concat([df1, df2, df3], ignore_index=True)

print("Dataset loaded successfully!")
print("Shape:", dataset.shape)

dataset.columns = dataset.columns.str.strip()

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

X = X.apply(pd.to_numeric, errors='coerce')

X.replace([np.inf, -np.inf], np.nan, inplace=True)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)

from sklearn.preprocessing import LabelEncoder

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.6,
    random_state=0,
    stratify=y
)
# ==============================
# SGD CLASSIFIER
# ==============================

from sklearn.linear_model import SGDClassifier

classifier = SGDClassifier(
    random_state=0,
    max_iter=1000,
    tol=1e-3
)

classifier.fit(X_train, y_train)

y_pred_sgd = classifier.predict(X_test)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# ==============================
# EVALUATION METRICS
# ==============================

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


# ==============================
# SGD CLASSIFIER RESULTS
# ==============================

print("\n==============================")
print("SGD CLASSIFIER RESULTS")
print("==============================")

cm = confusion_matrix(y_test, y_pred_sgd)

print("\nConfusion Matrix:")
print(cm)

print("\nAccuracy:",
      accuracy_score(y_test, y_pred_sgd))

print("\nPrecision:",
      precision_score(y_test, y_pred_sgd, average='weighted'))

print("\nRecall:",
      recall_score(y_test, y_pred_sgd, average='weighted'))

print("\nF1 Score:",
      f1_score(y_test, y_pred_sgd, average='weighted'))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_sgd))


