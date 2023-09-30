#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import optuna


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 5000)

plt.style.use('fivethirtyeight')

df = pd.read_excel('train_data.xlsx')
test_df = pd.read_excel('test_data.xlsx')

columns = ['Email ID', 'Ticket Type', 'CGPA',
           'Speaking Skills','ML Knowledge', 'Placement Status']

train_df = df[columns]
test_df = test_df[columns]

col_rename_mapper = {
    'Email ID': 'email', 'Ticket Type' : 'ticket_type',
    'CGPA':'cgpa','Speaking Skills':'speaking_skills',
    'ML Knowledge':'ml_knowledge', 'Placement Status':'placement_status'
}

train_df = train_df.rename(col_rename_mapper, axis=1)
test_df = test_df.rename(col_rename_mapper, axis=1)

train_df.placement_status.fillna('uncertain', inplace=True)




def clean_and_format_text(text):
    
    text = text.lower()
    
    text = text.replace('.', '')
    text = text.replace(',', '')
    text = text.replace('-', '')
    text = text.replace('[', '')
    text = text.replace(']', '')
    text = text.replace('&', '_')
    text = text.replace('?', '')
    text = text.replace(' ', '_')
    text = text.replace(':', '')
    text = text.replace('__', '_')
    text = text.strip('_')

    return text

train_df.ticket_type = train_df.ticket_type.apply(clean_and_format_text)
test_df.ticket_type = test_df.ticket_type.apply(clean_and_format_text)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from category_encoders import OrdinalEncoder

label_encoder_1 = LabelEncoder()
label_encoder_2 = OrdinalEncoder()



train_df.email = label_encoder_2.fit_transform(train_df.email)
test_df.email = label_encoder_2.transform(test_df.email) 


train_df.placement_status = label_encoder_1.fit_transform(train_df.placement_status)

train_dummies = pd.get_dummies(train_df.ticket_type)
test_dummies = pd.get_dummies(test_df.ticket_type)

train_df = pd.concat([train_df, train_dummies], axis=1)
test_df = pd.concat([test_df, test_dummies], axis=1)

train_df.drop('ticket_type', axis=1, inplace=True)
test_df.drop('ticket_type', axis=1, inplace=True)

new_column1 = np.zeros(len(test_df))
new_column2 = np.zeros(len(test_df))

test_df.insert(5, 'art_of_resume_building', new_column1)
test_df.insert(6, 'artificial_intelligence', new_column2)


target = 'placement_status'

X, y = train_df.drop(['placement_status'], axis=1), train_df.placement_status


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


params = {'max_depth': 74, 'subsample': 0.690961899344375, 'colsample_bytree': 0.26307203741471596, 
          'colsample_bylevel': 0.6323577392545147, 'min_child_weight': 7, 'reg_lambda': 0.12420600128963495, 
          'reg_alpha': 0.041915238518139705, 'n_estimators': 635, 'learning_rate': 0.036533167160145506}

xgb = XGBClassifier(**params)

xgb.fit(X_train, y_train)


def evaluate_classifier(X_train, X_test, y_train, y_test, model):
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    return accuracy, precision, recall, f1

accuracy, precision, recall, f1 = evaluate_classifier(X_train, X_test, y_train, y_test, xgb)

print(f"{'Metric':<15}{'Value':<10}")
print('-' * 25)
print(f"{'Accuracy':<15}{accuracy:<10.4f}")
print(f"{'Precision':<15}{precision:<10.4f}")
print(f"{'Recall':<15}{recall:<10.4f}")
print(f"{'F1 Score':<15}{f1:<10.4f}")



# def objective(trial):
#     params = {
#         'max_depth': trial.suggest_int('max_depth', 5, 500),
#         'subsample': trial.suggest_float('subsample', 0.1, 1),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
#         'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.01, 1.0),
#         'min_child_weight': trial.suggest_int('min_child_weight', 0.5, 30),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0, log=True),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
#         'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
#     }

#     model = XGBClassifier(**params)
    
#     model.fit(X_train, y_train)
    
#     test_scores = model.score(X_test, y_test.ravel())
    
#     return test_scores

# progress_bar = tqdm(total=100, desc='Optimizing Hyperparameters', dynamic_ncols=True)

# def callback(study, trial):
#     progress_bar.n = len(study.trials)
#     progress_bar.update(1)

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=100, callbacks=[callback])

# progress_bar.close()

# best_params = study.best_params
# best_score = study.best_value

# print("Best Hyperparameters for XGBClassifier:")
# print(best_params)
# print("Best R2 Score: {:.3f}".format(best_score))


independent_vars, target = test_df.drop(['placement_status'], axis=1), test_df.placement_status
test_df.placement_status = xgb.predict(test_df.drop(['placement_status'], axis=1))
test_df.placement_status.value_counts()
test_df.placement_status.to_excel('predicted_test_data.xlsx', index=False)





