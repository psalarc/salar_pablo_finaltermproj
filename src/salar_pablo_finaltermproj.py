#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # CS634 Final Project

# Name: Pablo Salar Carrera <br>
# UCID: ps2255 <br>
# Instructor: Dr. Yasser <br>
# Class: CS634

# In[ ]:





# In[ ]:





# ## Tutorial:

# Please find below the code and each step to be able to work on it. <br

# In[ ]:





# In[ ]:





# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import brier_score_loss
from sklearn.metrics import auc
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE



# Dataset link:<br>
# https://www.kaggle.com/datasets/iamsouravbanerjee/heart-attack-prediction-dataset?resource=download

# In[3]:


dataset = pd.read_csv('salar_pablo_finaltermproj.csv', index_col = 0)


# In[4]:


dataset.head()


# In[5]:


dataset.shape


# In[6]:


dataset.dtypes


# Let's check for missing values and duplicates in the dataset.

# In[7]:


dataset.isna().sum()


# In[8]:


dataset.duplicated().sum()


# There are no missing values or duplicates, therefore we can continue analyzing the data.

# Let's see how the numerical columns are distributed

# In[9]:


dataset.describe()


# In[10]:


for column in dataset.columns:
    unique_values = dataset[column].unique()
    print(f"Unique values in '{column}': {unique_values}")
    print()


# In[11]:


for column in dataset.columns:
    print(f"Value counts for '{column}':")
    print(dataset[column].value_counts())
    print()


# We group the numerical and categorical columns for future preprocessing and analysis

# In[12]:


numerical_columns = dataset.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = dataset.select_dtypes(include=['object']).columns.tolist()

print(len(numerical_columns), "Numerical Columns: ", numerical_columns)
print(len(categorical_columns), "Categorical Columns: ", categorical_columns)


# In[13]:


for col in numerical_columns:
    print(f"\nDistribution of 'Heart Attack Risk' by {col}:")
    print(pd.crosstab(dataset[col], dataset["Heart Attack Risk"]))


# In[14]:


for col in categorical_columns:
    print(f"\nDistribution of 'Heart Attack Risk' by {col}:")
    print(pd.crosstab(dataset[col], dataset["Heart Attack Risk"]))


# In[15]:


target_column = 'Heart Attack Risk'


# In[16]:


target_counts = dataset[target_column].value_counts()
print("Counts of each target category:\n", target_counts)


# We check how the target data is distributed. There are 5624 cases in which there is no presence of heart attack risk, while 3139 cases where there exists a risk. <br>

# In[17]:


plt.figure(figsize=(6, 6))
dataset[target_column].value_counts().plot.pie(autopct='%1.1f%%', startangle=140)
plt.title(f"Proportion of {target_column}")
plt.ylabel("")  # Hide the y-label
plt.show()


# The proportion of the target variable, Heart Attack Risk, is 35.8% of presence of heart attack risk and 64.2% of no presence of heart attack risk. 

# In[18]:


# Plotting histograms for a few key numerical features
fig, axs = plt.subplots(3, 3, figsize=(15, 12))
sbn.histplot(dataset['Age'], kde=True, ax=axs[0, 0]).set(title="Age Distribution")
sbn.histplot(dataset['Cholesterol'], kde=True, ax=axs[0, 1]).set(title="Cholesterol Distribution")
sbn.histplot(dataset['Heart Rate'], kde=True, ax=axs[0, 2]).set(title="Heart Rate Distribution")
sbn.histplot(dataset['BMI'], kde=True, ax=axs[1, 0]).set(title="BMI Distribution")
sbn.histplot(dataset['Triglycerides'], kde=True, ax=axs[1, 1]).set(title="Triglycerides Distribution")
sbn.histplot(dataset['Exercise Hours Per Week'], kde=True, ax=axs[1, 2]).set(title="Exercise Hours Distribution")
sbn.histplot(dataset['Sedentary Hours Per Day'], kde=True, ax=axs[2, 0]).set(title="Sedentary Hours Distribution")
sbn.histplot(dataset['Physical Activity Days Per Week'], kde=True, ax=axs[2, 1]).set(title="Physical Activity Days")
sbn.histplot(dataset['Sleep Hours Per Day'], kde=True, ax=axs[2, 2]).set(title="Sleep Hours Distribution")
plt.tight_layout()
plt.show()


# In[19]:


# Correlation heatmap
numerical_data = dataset.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(14, 10))
sbn.heatmap(numerical_data.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()


# We check for any correlations between the numerical data, but we do not find any highly correlated features. There is only a slightly positive correlation between the smoking and the age of the patients of the dataset. 

# In[20]:


# Plotting Exercise Hours and Sedentary Hours against Heart Attack Risk
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

sbn.boxplot(x='Heart Attack Risk', y='Exercise Hours Per Week', data=dataset, ax=axs[0])
axs[0].set_title('Exercise Hours Per Week by Heart Attack Risk')

sbn.boxplot(x='Heart Attack Risk', y='Sedentary Hours Per Day', data=dataset, ax=axs[1])
axs[1].set_title('Sedentary Hours Per Day by Heart Attack Risk')

plt.tight_layout()
plt.show()


# Define the function to calculate the metrics using the formulas from the slides. 

# In[21]:


def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics based on the confusion matrix.

    Parameters:
        y_true (list or array): True labels.
        y_pred (list or array): Predicted labels.

    Returns:
        pd.DataFrame: DataFrame containing the metrics.
    """
    # Get confusion matrix values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate metrics manually
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) != 0 else 0
    tss = (tp / (tp + fn) if (tp + fn) != 0 else 0) - fpr
    hss = (2 * (tp * tn - fp * fn)) / (
        ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    ) if ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)) != 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    error_rate = 1 - accuracy
    balanced_acc = (recall + specificity) / 2


    # Store the metrics
    metrics = {
        "FPR" : fpr,
        "FNR" : fnr,
        "TSS" : tss,
        "HSS" : hss,
        "Accuracy" : accuracy,
        "Precision" : precision,
        "Recall/Sensitivity" : recall,
        "Specificity" : specificity,
        "F1 Measure" : f1,
        "Error Rate" : error_rate,
        "Balanced Accuracy" : balanced_acc
    }
    return metrics


# We use LabelEncoder to convert the object variables into numbers so our models can use the data. 

# In[22]:


# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Apply Label Encoding to each categorical column
for col in categorical_columns:
    dataset[col] = label_encoder.fit_transform(dataset[col])


# In[23]:


dataset.head()


# For a better use of the models, we standarize the numerical features using RobustScaler()

# In[24]:


# Standardize numerical features
scaler = RobustScaler()
dataset[numerical_columns] = scaler.fit_transform(dataset[numerical_columns])


# In[25]:


dataset.head()


# In[26]:


# Splitting features and target
X = dataset.drop(columns=['Heart Attack Risk'])
y = dataset['Heart Attack Risk']


# In[27]:


# Define the 10-fold cross-validator
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


# In[28]:


randomforest = RandomForestClassifier(random_state=42)
KNN = KNeighborsClassifier()


# In[29]:


# Define parameter grids for hyperparameter tuning
KNN_param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

randomforest_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30]
}


# In[31]:


# Perform grid search for KNN
KNN_grid_search = GridSearchCV(KNN, KNN_param_grid, cv=5, verbose=1, n_jobs=-1)
KNN_grid_search.fit(X, y)
best_KNN = KNN_grid_search.best_estimator_


# In[32]:


best_KNN


# In[33]:


# Perform grid search for Random Forest
randomforest_grid_search = GridSearchCV(randomforest, randomforest_param_grid, cv=5, verbose=1, n_jobs=-1)
randomforest_grid_search.fit(X, y)
best_randomforest = randomforest_grid_search.best_estimator_


# In[34]:


best_randomforest


# In[35]:


all_metrics = []


# In[36]:


# Loop through each fold
for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
    print(f"----- Metrics for all Algorithms in Iteration {fold + 1} -----")
    
    # Split data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # ---- Random Forest ----
    best_randomforest.fit(X_train, y_train)
    y_pred_randomforest = best_randomforest.predict(X_test)
    randomforest_metrics = calculate_metrics(y_test, y_pred_randomforest)
    
    # ---- KNN ----
    best_KNN.fit(X_train, y_train)
    y_pred_KNN = best_KNN.predict(X_test)
    KNN_metrics = calculate_metrics(y_test, y_pred_KNN)
    
    # ---- LSTM ----
    # Reshape for LSTM input (samples, timesteps, features)
    X_train_lstm = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test_lstm = X_test.values.reshape(X_test.shape[0], 1, X_test.shape[1])
    
    # Convert target to categorical for LSTM
    y_train_lstm = to_categorical(y_train, num_classes=2)
    y_test_lstm = to_categorical(y_test, num_classes=2)
    
    # Define LSTM model
    lstm_model = Sequential([
        Input(shape=(1, X_train.shape[1])),
        LSTM(64, activation='relu'),
        Dropout(0.2),
        Dense(2, activation='softmax')
    ])
    lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=16, verbose=0)
    
    y_pred_lstm = np.argmax(lstm_model.predict(X_test_lstm), axis=1)
    
    # Compute LSTM metrics
    lstm_metrics = calculate_metrics(y_test, y_pred_lstm)
    
    # ---- Combine Metrics ----
    all_metrics_fold = {
        'KNN': KNN_metrics,
        'Random Forest': randomforest_metrics,
        'LSTM': lstm_metrics
    }
    
    # Store metrics for the current fold
    all_metrics.append(all_metrics_fold)

    # ---- Print Metrics for All Algorithms in Current Iteration ----
    iteration_metrics = pd.DataFrame({
        'Metric': list(KNN_metrics.keys()),
        'KNN': list(KNN_metrics.values()),
        'Random Forest': list(randomforest_metrics.values()),
        'LSTM': list(lstm_metrics.values())
    })
    print(iteration_metrics.to_string(index=False))


# In[37]:


def plot_individual_roc_curve(y_test, y_pred_probs, model_name):
    """
    Plot the ROC curve for an individual model.

    Parameters:
        y_test (array-like): True labels for the test set.
        y_pred_probs (array-like): Predicted probabilities for the positive class.
        model_name (str): Name of the model.
    """
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    # Calculate AUC
    roc_auc = auc(fpr, tpr)

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})', color='blue', lw=2)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line (random classifier)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title(f'ROC Curve - {model_name}', fontsize=16)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# In[38]:


# ---- KNN ----
y_pred_KNN_probs = best_KNN.predict_proba(X_test)[:, 1]  # Probability for class 1
plot_individual_roc_curve(y_test, y_pred_KNN_probs, 'KNN')

# ---- Random Forest ----
y_pred_randomforest_probs = best_randomforest.predict_proba(X_test)[:, 1]  # Probability for class 1
plot_individual_roc_curve(y_test, y_pred_randomforest_probs, 'Random Forest')

# ---- LSTM ----
y_pred_lstm_probs = lstm_model.predict(X_test_lstm)[:, 1]  # Probability for class 1
plot_individual_roc_curve(y_test, y_pred_lstm_probs, 'LSTM')


# In[39]:


# Function to plot ROC curve and AUC
def plot_roc_curve(y_test, y_pred_probs, model_name):
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    # Calculate AUC
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

# Initialize plot
plt.figure(figsize=(10, 8))

# ---- KNN ----
y_pred_knn_probs = best_KNN.predict_proba(X_test)[:, 1]  # Probability for class 1
plot_roc_curve(y_test, y_pred_knn_probs, 'KNN')

# ---- Random Forest ----
y_pred_rf_probs = best_randomforest.predict_proba(X_test)[:, 1]  # Probability for class 1
plot_roc_curve(y_test, y_pred_rf_probs, 'Random Forest')

# ---- LSTM ----
y_pred_lstm_probs = lstm_model.predict(X_test_lstm)[:, 1]  # Probability for class 1
plot_roc_curve(y_test, y_pred_lstm_probs, 'LSTM')

# Customize plot
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # Diagonal line (random classifier)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve (AUC Comparison)')
plt.legend(loc='lower right')
plt.grid(True)

# Show the plot
plt.show()


# In[40]:


# Initialize an empty dictionary to store average metrics for each model
average_metrics = {'Metric': []}

# Gather metrics for each model across folds
for model_name in ['KNN', 'Random Forest', 'LSTM']:
    # Extract all metric values for the current model from the folds
    metrics_by_model = {metric: [] for metric in all_metrics[0][model_name]}
    for fold_metrics in all_metrics:
        for metric, value in fold_metrics[model_name].items():
            metrics_by_model[metric].append(value)
    
    # Compute average metrics for the current model
    average_metrics['Metric'] = list(metrics_by_model.keys())
    average_metrics[model_name] = [sum(values) / len(values) for values in metrics_by_model.values()]

# Convert the average metrics dictionary to a DataFrame
avg_performance_df = pd.DataFrame(average_metrics)

# Print the DataFrame
print("Average Performance Across All Folds:")
print(avg_performance_df.round(decimals=2))

# Create a bar plot for the metrics comparison
plt.figure(figsize=(12, 8))
sbn.set_theme(style="whitegrid")

# Melt the DataFrame for plotting
melted_df = avg_performance_df.melt(id_vars='Metric', var_name='Model', value_name='Value')

# Plot the data using seaborn
sbn.barplot(data=melted_df, x='Metric', y='Value', hue='Model', palette='viridis')

# Customize the plot
plt.title('Average Metrics Comparison Across Models', fontsize=16)
plt.xlabel('Metrics', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.legend(title='Model', fontsize=10)
plt.tight_layout()

# Show the plot
plt.show()


# In this experiment, we can observe that the models do not achieve a high accuracy. In general terms, the models predict pretty well the cases where there is no risk of a heart attack (the negative cases), but struggle to predict correctly the positive cases. This is pretty bad because a good model in the healthcare industry should predict and be focused when there is a risk of heart attack to try to prevent it.

# In[41]:


# Random Forest Feature Selection
importances = best_randomforest.feature_importances_
feature_names = X.columns

# Create a DataFrame for better visualization
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 8))
sbn.barplot(x='Importance', y='Feature', data=feature_importance_df, palette="viridis")
plt.title('Feature Importance from Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()


# We check for feature importance and we reduce the size of features in the training data, focusing on the most important ones.

# In[42]:


# Select top 10 features
top_10_features = feature_importance_df.head(10)['Feature'].tolist()

# Update dataset with only the top 10 features
X_top_10 = X[top_10_features]

# Print selected features
print("Top 10 Features Selected:")
print(top_10_features)


# As well we use SMOTE to balance the data since there is some class imbalance. 

# In[43]:


# Initialize SMOTE
smote = SMOTE(random_state=42)

# Apply SMOTE to the reduced dataset (X_top_10)
X_smote, y_smote = smote.fit_resample(X_top_10, y)

# Display the new class distribution
from collections import Counter
print(f"Class distribution after SMOTE: {Counter(y_smote)}")


# In[44]:


# Plot class distribution before and after SMOTE
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].bar(Counter(y).keys(), Counter(y).values(), color=['blue', 'orange'])
ax[0].set_title('Before SMOTE')
ax[1].bar(Counter(y_smote).keys(), Counter(y_smote).values(), color=['blue', 'orange'])
ax[1].set_title('After SMOTE')
plt.show()


# In[45]:


# Perform cross-validation with top 10 features
for fold, (train_index, test_index) in enumerate(kf.split(X_top_10, y)):
    print(f"----- Metrics for all Algorithms in Iteration {fold + 1} -----")
    
    # Split data
    X_train, X_test = X_top_10.iloc[train_index], X_top_10.iloc[test_index]
    y_train, y_test = y_smote.iloc[train_index], y_smote.iloc[test_index]
    
    # ---- Random Forest ----
    best_randomforest.fit(X_train, y_train)
    y_pred_randomforest = best_randomforest.predict(X_test)
    randomforest_metrics = calculate_metrics(y_test, y_pred_randomforest)
    
    # ---- KNN ----
    best_KNN.fit(X_train, y_train)
    y_pred_KNN = best_KNN.predict(X_test)
    KNN_metrics = calculate_metrics(y_test, y_pred_KNN)
    
   
    # ---- LSTM ----
    X_train_lstm = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test_lstm = X_test.values.reshape(X_test.shape[0], 1, X_test.shape[1])
    y_train_lstm = to_categorical(y_train, num_classes=2)
    y_test_lstm = to_categorical(y_test, num_classes=2)
    
    lstm_model = Sequential([
        Input(shape=(1, X_train.shape[1])),
        LSTM(64, activation='relu'),
        Dropout(0.2),
        Dense(2, activation='softmax')
    ])
    lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=16, verbose=0)
    
    y_pred_lstm = np.argmax(lstm_model.predict(X_test_lstm), axis=1)
    lstm_metrics = calculate_metrics(y_test, y_pred_lstm)
    
    # ---- Combine Metrics ----
    all_metrics_fold = {
        'KNN': KNN_metrics,
        'Random Forest': randomforest_metrics,
        'LSTM': lstm_metrics
    }
    
    # Store metrics for the current fold
    all_metrics.append(all_metrics_fold)

    # ---- Print Metrics for All Algorithms in Current Iteration ----
    iteration_metrics = pd.DataFrame({
        'Metric': list(KNN_metrics.keys()),
        'KNN': list(KNN_metrics.values()),
        'Random Forest': list(randomforest_metrics.values()),
        'LSTM': list(lstm_metrics.values())
    })
    print(iteration_metrics.to_string(index=False))


# In[48]:


# ---- KNN ----
y_pred_KNN_probs = best_KNN.predict_proba(X_test)[:, 1]  # Probability for class 1
plot_individual_roc_curve(y_test, y_pred_KNN_probs, 'KNN')

# ---- Random Forest ----
y_pred_randomforest_probs = best_randomforest.predict_proba(X_test)[:, 1]  # Probability for class 1
plot_individual_roc_curve(y_test, y_pred_randomforest_probs, 'Random Forest')

# ---- LSTM ----
y_pred_lstm_probs = lstm_model.predict(X_test_lstm)[:, 1]  # Probability for class 1
plot_individual_roc_curve(y_test, y_pred_lstm_probs, 'LSTM')


# In[47]:


# Initialize plot
plt.figure(figsize=(10, 8))

# ---- KNN ----
y_pred_knn_probs = best_KNN.predict_proba(X_test)[:, 1]  # Probability for class 1
plot_roc_curve(y_test, y_pred_knn_probs, 'KNN')

# ---- Random Forest ----
y_pred_rf_probs = best_randomforest.predict_proba(X_test)[:, 1]  # Probability for class 1
plot_roc_curve(y_test, y_pred_rf_probs, 'Random Forest')

# ---- LSTM ----
y_pred_lstm_probs = lstm_model.predict(X_test_lstm)[:, 1]  # Probability for class 1
plot_roc_curve(y_test, y_pred_lstm_probs, 'LSTM')

# Customize plot
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # Diagonal line (random classifier)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve (AUC Comparison)')
plt.legend(loc='lower right')
plt.grid(True)

# Show the plot
plt.show()


# In[46]:


# Initialize an empty dictionary to store average metrics for each model
average_metrics = {'Metric': []}

# Gather metrics for each model across folds
for model_name in ['KNN', 'Random Forest', 'LSTM']:
    # Extract all metric values for the current model from the folds
    metrics_by_model = {metric: [] for metric in all_metrics[0][model_name]}
    for fold_metrics in all_metrics:
        for metric, value in fold_metrics[model_name].items():
            metrics_by_model[metric].append(value)
    
    # Compute average metrics for the current model
    average_metrics['Metric'] = list(metrics_by_model.keys())
    average_metrics[model_name] = [sum(values) / len(values) for values in metrics_by_model.values()]

# Convert the average metrics dictionary to a DataFrame
avg_performance_df = pd.DataFrame(average_metrics)

# Print the DataFrame
print("Average Performance Across All Folds:")
print(avg_performance_df.round(decimals=2))

# Create a bar plot for the metrics comparison
plt.figure(figsize=(12, 8))
sbn.set_theme(style="whitegrid")

# Melt the DataFrame for plotting
melted_df = avg_performance_df.melt(id_vars='Metric', var_name='Model', value_name='Value')

# Plot the data using seaborn
sbn.barplot(data=melted_df, x='Metric', y='Value', hue='Model', palette='viridis')

# Customize the plot
plt.title('Average Metrics Comparison Across Models', fontsize=16)
plt.xlabel('Metrics', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.legend(title='Model', fontsize=10)
plt.tight_layout()

# Show the plot
plt.show()


# In conclusion, even though we balanced the data and reduce the dimension of the training features, the models did not improve their performance. Therefore, for the future we probably should work a bit more on the data preprocessing to see if we can get more relevant data.

# ## Side Notes
# To be able to run this code make sure to have installed the following libraries: pandas, numpy, seaborn, matplotlib, sklearn, tensorflow and imbalanced-learn. <br>
# To install the libraries write the following in the terminal: pip install (name of the library). <br>
# Another way to install the libraries is to write in a cell of the jupiter notebook: !pip install (name of the library). <br>
# Moreover, it is important to have the required csv file, jupiter notebook, and python file in the same folder in order to run the codes.

# https://github.com/psalarc/salar_pablo_finaltermproj
