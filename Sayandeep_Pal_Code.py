#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


training_data = pd.read_csv(r"C:\Users\sayandeep\Downloads\training_data.csv")


training_data_class = pd.read_csv(r"C:\Users\sayandeep\Downloads\training_data_targets.csv", header=None, names=['Class'])

# Count the occurrences of each class
class_counts = training_data_class['Class'].value_counts()


plt.bar(class_counts.index, class_counts.values, color=['blue', 'green'])
plt.title('Number of Patients and Healthy People')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


# In[2]:


nan_values = training_data.isna().sum()

print("NaN Values in Training Data:")
print(nan_values)


# In[3]:


duplicates = training_data.duplicated().sum()

print("Duplicate Rows in Training Data:")
print(duplicates)


# In[4]:


display(training_data)


# In[5]:


# Calculate the minimum and maximum values for each feature
min_vals = training_data.min()
max_vals = training_data.max()

# Perform Min-Max scaling to normalize the data
normalized_data = (training_data - min_vals) / (max_vals - min_vals)


print("Normalized Training Data:")
display(normalized_data)


# In[6]:


variances = normalized_data.var()

print("Variances of Each Feature:")
print(variances)


# In[7]:


variance_threshold = 0.015

# Calculate the variance of each feature
variances = normalized_data.var()

# Filter out features below the variance threshold
selected_features =normalized_data.loc[:, variances >= variance_threshold]


num_features = selected_features.shape[1]
print(f"Number of selected features: {num_features}")


# In[8]:


# Find the correlation between different features
correlation_matrix = selected_features.corr()


print("Correlation Matrix:")
display(correlation_matrix)


# In[9]:


import seaborn as sns





# Find the correlation between different features
correlation_matrix = selected_features.corr()


plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm",linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[10]:


correlation_matrix = selected_features.corr()


correlation_threshold = 0.8

# Find highly correlated features and drop the one with lower variance
selected_features_new = selected_features.copy()
dropped_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            feature1 = correlation_matrix.columns[i]
            feature2 = correlation_matrix.columns[j]
            if feature1 not in dropped_features and feature2 not in dropped_features:
                if selected_features[feature1].var() > selected_features[feature2].var():
                    selected_features_new = selected_features_new.drop(columns=[feature2])
                    dropped_features.add(feature2)
                else:
                    selected_features_new = selected_features_new.drop(columns=[feature1])
                    dropped_features.add(feature1)


num_features = selected_features_new.shape[1]
print(f"Number of selected features: {num_features}")


# In[11]:


display(selected_features_new)


# In[12]:


from sklearn.decomposition import PCA

#  Taking 15 PCs
pca = PCA(n_components=15) 
pca_result = pca.fit_transform(selected_features_new)


pca_dataframe = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
print("PCA Result:")
display(pca_dataframe)


# In[13]:


# Obtain the explained variance ratio and the total variance explained
explained_variance_ratio = pca.explained_variance_ratio_
total_variance_explained = sum(explained_variance_ratio)


print("Explained Variance Ratio:\n")
print(explained_variance_ratio)
print("\n")
print(f"Total Variance Explained: {total_variance_explained}")







# In[14]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
# Convert the class labels to a 1D array
training_data_class = training_data_class.values.ravel()
display(training_data_class)


# In[15]:


test_data = pd.read_csv(r"C:\Users\sayandeep\Downloads\test_data.csv")


# In[16]:


nan_values = test_data.isna().sum()

print("NaN Values in Training Data:")
print(nan_values)


# In[17]:


duplicates = test_data.duplicated().sum()

print("Duplicate Rows in Training Data:")
print(duplicates)


# In[18]:


display(test_data)


# In[19]:


min_vals = test_data.min()
max_vals = test_data.max()

# Perform Min-Max scaling to normalize the data
test_data = (test_data - min_vals) / (max_vals - min_vals)


print("Normalized Test Data:")
display(test_data)


# In[20]:


variances = test_data.var()

print("Variances of Each Feature:")
print(variances)


# In[21]:


variance_threshold = 0.015

# Calculate the variance of each feature
variances = test_data.var()

# Filter out features below the variance threshold
test_data =test_data.loc[:, variances >= variance_threshold]


num_features = test_data.shape[1]
print(f"Number of selected features: {num_features}")


# In[22]:


correlation_matrix = test_data.corr()


print("Correlation Matrix:")
display(correlation_matrix)


# In[23]:


correlation_matrix = test_data.corr()


correlation_threshold = 0.8

# Find highly correlated features and drop the one with lower variance
test_data = test_data.copy()
dropped_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            feature1 = correlation_matrix.columns[i]
            feature2 = correlation_matrix.columns[j]
            if feature1 not in dropped_features and feature2 not in dropped_features:
                if test_data[feature1].var() > test_data[feature2].var():
                    test_data = test_data.drop(columns=[feature2])
                    dropped_features.add(feature2)
                else:
                    test_data = test_data.drop(columns=[feature1])
                    dropped_features.add(feature1)


num_features = test_data.shape[1]
print(f"Number of selected features: {num_features}")


# In[24]:


display(test_data)


# In[25]:


from sklearn.decomposition import PCA

# Same Number of PCs as in training data
pca = PCA(n_components=15) 
pca_result = pca.fit_transform(test_data)


test_data = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
print("PCA Result:")
display(test_data)


# In[26]:


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

rf_params = {'max_depth': None, 'n_estimators': 100, 'criterion': 'entropy'}
rf_classifier = RandomForestClassifier(**rf_params)


predicted_labels = cross_val_predict(rf_classifier, pca_dataframe, training_data_class, cv=10)


class_report = classification_report(training_data_class, predicted_labels)


def specificity_score(training_data_class, predicted_labels):
    tn, fp, fn, tp = confusion_matrix(training_data_class, predicted_labels).ravel()
    specificity = tn / (tn + fp)
    return specificity

# Calculate the specificity for each fold
specificities = []
for i in range(10):
    start_index = i * len(training_data_class) // 10
    end_index = (i + 1) * len(training_data_class) // 10
    current_specificity = specificity_score(training_data_class[start_index:end_index], predicted_labels[start_index:end_index])
    specificities.append(current_specificity)


average_specificity = np.mean(specificities)
print("Classification Report:")
print(class_report)
print(f"Average Specificity: {average_specificity}")

accuracy = accuracy_score(training_data_class, predicted_labels)
recall = recall_score(training_data_class, predicted_labels, average='macro')
precision = precision_score(training_data_class, predicted_labels, average='macro')
f1 = f1_score(training_data_class, predicted_labels, average='macro')

print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"F1 Score: {f1}")


# In[27]:


conf_matrix = confusion_matrix(training_data_class, predicted_labels)


class_names = ['Healthy', 'Patient']
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()


# In[28]:


from sklearn.model_selection import GridSearchCV

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20,30,40],
    'criterion': ['gini', 'entropy','log_loss'],
    'max_features':['sqrt', 'log2', None]
}

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=10, scoring='f1_macro')

# Fit the GridSearchCV object to the data
grid_search.fit(pca_dataframe, training_data_class)

rf_pred=grid_search.predict(test_data)
print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)
print("Predictions:",rf_pred)


# In[29]:


from sklearn.tree import DecisionTreeClassifier


dt_params = {'max_depth': None, 'max_features':'sqrt', 'criterion': 'entropy','ccp_alpha':0.01}
dt_classifier = DecisionTreeClassifier(**dt_params)


predicted_labels_dt = cross_val_predict(dt_classifier, pca_dataframe, training_data_class, cv=10)


class_report_dt = classification_report(training_data_class, predicted_labels_dt)

def specificity_score(training_data_class, predicted_labels_dt):
    tn, fp, fn, tp = confusion_matrix(training_data_class, predicted_labels_dt).ravel()
    specificity = tn / (tn + fp)
    return specificity


specificities = []
for i in range(10):
    start_index = i * len(training_data_class) // 10
    end_index = (i + 1) * len(training_data_class) // 10
    current_specificity = specificity_score(training_data_class[start_index:end_index], predicted_labels_dt[start_index:end_index])
    specificities.append(current_specificity)


average_specificity = np.mean(specificities)

print("Decision Tree Classification Report:")
print(class_report_dt)
print(f"Average Specificity: {average_specificity}")

accuracy_dt = accuracy_score(training_data_class, predicted_labels_dt)
recall_dt = recall_score(training_data_class, predicted_labels_dt, average='macro')
precision_dt = precision_score(training_data_class, predicted_labels_dt, average='macro')
f1_dt = f1_score(training_data_class, predicted_labels_dt, average='macro')

print(f"Decision Tree Accuracy: {accuracy_dt}")
print(f"Decision Tree Recall: {recall_dt}")
print(f"Decision Tree Precision: {precision_dt}")
print(f"Decision Tree F1 Score: {f1_dt}")


conf_matrix_dt = confusion_matrix(training_data_class, predicted_labels_dt)
print("Decision Tree Confusion Matrix:")

class_names = ['Healthy', 'Patient']

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Decision Tree Confusion Matrix')
plt.show()


# In[30]:


from sklearn.model_selection import GridSearchCV


param_grid = {
    'ccp_alpha': [0.01,0.05,0.1],
    'max_depth': [None, 10, 20,30,40],
    'criterion': ['gini', 'entropy','log_loss'],
    'max_features':['sqrt', 'log2', None]
}


grid_search_dt = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_grid, cv=10, scoring='f1_macro')


grid_search_dt.fit(pca_dataframe, training_data_class)

dt_pred=grid_search_dt.predict(test_data)
print("Best Parameters: ", grid_search_dt.best_params_)
print("Best Score: ", grid_search_dt.best_score_)
print("Predictions:",dt_pred)


# In[31]:


from sklearn.svm import SVC


svm_params = {'C': 1.0, 'kernel': 'rbf'}
svm_classifier = SVC(**svm_params)


predicted_labels_svm = cross_val_predict(svm_classifier, pca_dataframe, training_data_class, cv=10)


class_report_svm = classification_report(training_data_class, predicted_labels_svm)

def specificity_score(training_data_class, predicted_labels_svm):
    tn, fp, fn, tp = confusion_matrix(training_data_class, predicted_labels_svm).ravel()
    specificity = tn / (tn + fp)
    return specificity


specificities = []
for i in range(10):
    start_index = i * len(training_data_class) // 10
    end_index = (i + 1) * len(training_data_class) // 10
    current_specificity = specificity_score(training_data_class[start_index:end_index], predicted_labels_svm[start_index:end_index])
    specificities.append(current_specificity)


average_specificity = np.mean(specificities)

print(f"Average Specificity: {average_specificity}")

print("SVM Classification Report:")
print(class_report_svm)


accuracy_svm = accuracy_score(training_data_class, predicted_labels_svm)
recall_svm = recall_score(training_data_class, predicted_labels_svm, average='macro')
precision_svm = precision_score(training_data_class, predicted_labels_svm, average='macro')
f1_svm = f1_score(training_data_class, predicted_labels_svm, average='macro')

print(f"SVM Accuracy: {accuracy_svm}")
print(f"SVM Recall: {recall_svm}")
print(f"SVM Precision: {precision_svm}")
print(f"SVM F1 Score: {f1_svm}")


conf_matrix_svm = confusion_matrix(training_data_class, predicted_labels_svm)
print("SVM Confusion Matrix:")

class_names = ['Healthy', 'Patient']

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('SVM Confusion Matrix')
plt.show()


# In[32]:


from sklearn.model_selection import GridSearchCV


param_grid_svm = {'C': [0.1, 1, 10,100], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}


grid_search_svm = GridSearchCV(estimator=SVC(), param_grid=param_grid_svm, cv=10, scoring='f1_macro')


grid_search_svm.fit(pca_dataframe, training_data_class)

svm_pred=grid_search_svm.predict(test_data)
print("Best Parameters for SVM: ", grid_search_svm.best_params_)
print("Best Score for SVM: ", grid_search_svm.best_score_)
print("Predictions:",svm_pred)


# In[33]:


from sklearn.linear_model import LogisticRegression


logreg_params = {'penalty':'l1', 'solver': 'liblinear'}
logreg_classifier = LogisticRegression(**logreg_params)


predicted_labels_logreg = cross_val_predict(logreg_classifier, pca_dataframe, training_data_class, cv=10)


class_report_logreg = classification_report(training_data_class, predicted_labels_logreg)

def specificity_score(training_data_class, predicted_labels_logreg):
    tn, fp, fn, tp = confusion_matrix(training_data_class, predicted_labels_logreg).ravel()
    specificity = tn / (tn + fp)
    return specificity


specificities = []
for i in range(10):
    start_index = i * len(training_data_class) // 10
    end_index = (i + 1) * len(training_data_class) // 10
    current_specificity = specificity_score(training_data_class[start_index:end_index], predicted_labels_logreg[start_index:end_index])
    specificities.append(current_specificity)


average_specificity = np.mean(specificities)

print(f"Average Specificity: {average_specificity}")

print("Logistic Regression Classification Report:")
print(class_report_logreg)


accuracy_logreg = accuracy_score(training_data_class, predicted_labels_logreg)
recall_logreg = recall_score(training_data_class, predicted_labels_logreg, average='macro')
precision_logreg = precision_score(training_data_class, predicted_labels_logreg, average='macro')
f1_logreg = f1_score(training_data_class, predicted_labels_logreg, average='macro')

print(f"Logistic Regression Accuracy: {accuracy_logreg}")
print(f"Logistic Regression Recall: {recall_logreg}")
print(f"Logistic Regression Precision: {precision_logreg}")
print(f"Logistic Regression F1 Score: {f1_logreg}")


conf_matrix_logreg = confusion_matrix(training_data_class, predicted_labels_logreg)
print("Logistic Regression Confusion Matrix:")
class_names = ['Healthy', 'Patient']


plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_logreg, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Logistic Regression Confusion Matrix')
plt.show()


# In[34]:


from sklearn.model_selection import GridSearchCV


param_grid_logreg = {'penalty': ['l1','l2'], 'solver': ['liblinear', 'lbfgs','newton-cg', 'newton-cholesky', 'sag', 'saga'],'max_iter': [100, 200, 300]}


grid_search_logreg = GridSearchCV(estimator=LogisticRegression(), param_grid=param_grid_logreg, cv=10, scoring='f1_macro')


grid_search_logreg.fit(pca_dataframe, training_data_class)

logreg_pred=grid_search_logreg.predict(test_data)
print("Best Parameters for Logistic Regression: ", grid_search_logreg.best_params_)
print("Best Score for Logistic Regression: ", grid_search_logreg.best_score_)
print("Predictions:",logreg_pred)


# In[35]:


from sklearn.neighbors import KNeighborsClassifier


knn_params = {'n_neighbors': 5, 'weights': 'uniform', 'metric': 'minkowski','algorithm': 'auto'}
knn_classifier = KNeighborsClassifier(**knn_params)


predicted_labels_knn = cross_val_predict(knn_classifier,pca_dataframe, training_data_class, cv=10)


class_report_knn = classification_report(training_data_class, predicted_labels_knn)

def specificity_score(training_data_class, predicted_labels_knn):
    tn, fp, fn, tp = confusion_matrix(training_data_class, predicted_labels_knn).ravel()
    specificity = tn / (tn + fp)
    return specificity

specificities_knn = []
for i in range(10):
    start_index = i * len(training_data_class) // 10
    end_index = (i + 1) * len(training_data_class) // 10
    current_specificity = specificity_score(training_data_class[start_index:end_index], predicted_labels_knn[start_index:end_index])
    specificities_knn.append(current_specificity)


average_specificity_knn = np.mean(specificities_knn)

print(f"Average Specificity: {average_specificity_knn}")

print("K Nearest Neighbors Classification Report:")
print(class_report_knn)


accuracy_knn = accuracy_score(training_data_class, predicted_labels_knn)
recall_knn = recall_score(training_data_class, predicted_labels_knn, average='macro')
precision_knn = precision_score(training_data_class, predicted_labels_knn, average='macro')
f1_knn = f1_score(training_data_class, predicted_labels_knn, average='macro')

print(f"K Nearest Neighbors Accuracy: {accuracy_knn}")
print(f"K Nearest Neighbors Recall: {recall_knn}")
print(f"K Nearest Neighbors Precision: {precision_knn}")
print(f"K Nearest Neighbors F1 Score: {f1_knn}")


conf_matrix_knn = confusion_matrix(training_data_class, predicted_labels_knn)
print("K Nearest Neighbors Confusion Matrix:")
class_names = ['Healthy', 'Patient']

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('K Nearest Neighbors Confusion Matrix')
plt.show()


# In[36]:


from sklearn.model_selection import GridSearchCV



param_grid_knn = {'n_neighbors': [3,4,5,6,7,8,9,10], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}


grid_search_knn = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid_knn, cv=10, scoring='f1_macro')


grid_search_knn.fit(pca_dataframe, training_data_class)

knn_pred=grid_search_knn.predict(test_data)
print("Best Parameters for K Nearest Neighbors: ", grid_search_knn.best_params_)
print("Best Score for K Nearest Neighbors: ", grid_search_knn.best_score_)
print("Predictions:",knn_pred)


# In[37]:


from sklearn.naive_bayes import GaussianNB


naive_bayes_params = {}  # No parameters to be tuned for Naive Bayes


naive_bayes_classifier = GaussianNB(**naive_bayes_params)


predicted_labels_nb = cross_val_predict(naive_bayes_classifier, pca_dataframe, training_data_class, cv=10)


class_report_nb = classification_report(training_data_class, predicted_labels_nb)

def specificity_score(training_data_class, predicted_labels_nb ):
    tn, fp, fn, tp = confusion_matrix(training_data_class,predicted_labels_nb ).ravel()
    specificity = tn / (tn + fp)
    return specificity

specificities_nb = []
for i in range(10):
    start_index = i * len(training_data_class) // 10
    end_index = (i + 1) * len(training_data_class) // 10
    current_specificity = specificity_score(training_data_class[start_index:end_index], predicted_labels_nb [start_index:end_index])
    specificities_nb.append(current_specificity)


average_specificity_nb = np.mean(specificities_nb)

print(f"Average Specificity: {average_specificity_nb}")

accuracy_nb = accuracy_score(training_data_class, predicted_labels_nb)
recall_nb = recall_score(training_data_class, predicted_labels_nb, average='macro')
precision_nb = precision_score(training_data_class, predicted_labels_nb, average='macro')
f1_nb = f1_score(training_data_class, predicted_labels_nb, average='macro')


print("Naive Bayes Classification Report:")
print(class_report_nb)
print(f"Naive Bayes Accuracy: {accuracy_nb}")
print(f"Naive Bayes Recall: {recall_nb}")
print(f"Naive Bayes Precision: {precision_nb}")
print(f"Naive Bayes F1 Score: {f1_nb}")


conf_matrix_nb = confusion_matrix(training_data_class, predicted_labels_nb)
print("Naive Bayes Confusion Matrix:")
class_names = ['Healthy', 'Patient']


plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_nb, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Naive Bayes Confusion Matrix')
plt.show()


# We see that the Logistic Regression Classifier is performing the best with specificity of 93% and F1 score of 88%. We also used GridSearch Algorithm to find the best parameters for training the Logistic Regression Classifier and hence use this to predict the target variable of the test dataset.We also see that it has significantly the least number of misclassifications among all the other classifiers.

# In[50]:


logreg_params = {'max_iter': 100, 'penalty': 'l1', 'solver': 'saga'}
logreg_classifier = LogisticRegression(**logreg_params)

logreg_classifier.fit(pca_dataframe, training_data_class)
predicted_test_labels = logreg_classifier.predict(test_data)


with open('predicted_labels.txt', 'w') as f:
    for label in predicted_test_labels:
        f.write("%s\n" % label)

print("Predicted labels saved to predicted_labels.txt")

