import time
from IPython.display import clear_output
import numpy    as np
import pandas   as pd
import seaborn  as sb
import matplotlib.pyplot as plt
import sklearn  as skl
import math

from sklearn import pipeline      # Pipeline
from sklearn import preprocessing # OrdinalEncoder, LabelEncoder
from sklearn import impute
from sklearn import compose
from sklearn import model_selection # train_test_split
from sklearn import metrics         # accuracy_score, balanced_accuracy_score, plot_confusion_matrix
from sklearn import set_config

set_config(display='diagram') # Useful for display the pipeline


def data_enhancement_1(data):
    
    #Copy the initial dataframe to a dummy dataframe
    generated_data = data.copy()

    #Calculate the std for each parameter
    trestbps_std = generated_data['trestbps'].std()
    chol_std = generated_data['chol'].std()
    thalach_std = generated_data['thalach'].std()

    for i in range(data.shape[0]):
        if np.random.randint(2) == 1:
            generated_data['trestbps'].values[i] += trestbps_std/10
        else:
            generated_data['trestbps'].values[i] -= trestbps_std/10

        if np.random.randint(2) == 1:
            generated_data['chol'].values[i] += chol_std/10
        else:
            generated_data['chol'].values[i] -= chol_std/10

        if np.random.randint(2) == 1:
            generated_data['thalach'].values[i] += thalach_std/10
        else:
            generated_data['thalach'].values[i] -= thalach_std/10
    
    return generated_data

df = pd.read_csv('data/data.csv')
df.drop(['slope','thal', 'ca'], axis=1, inplace=True)
df.replace('?', -99999, inplace=True)

df = df.astype(
        {'trestbps': 'int64',
        'chol': 'int64',
        'fbs': 'int64',
        'restecg': 'int64',
        'thalach': 'int64',
        'exang': 'int64',
        }
        )

df.rename(columns={df.columns[-1]: 'target'}, inplace=True)

cat_vars = ['sex', 'cp', 'fbs', 'restecg', 'exang']
num_disc_vars = ['age']
num_cont_vars = ['chol', 'trestbps', 'thalach']
num_float_vars = ['oldpeak']

num_preproc_1 = pipeline.Pipeline(steps=[('num_disc_imputer', impute.SimpleImputer(missing_values=-99999, strategy='median'))])
num_preproc_2 = pipeline.Pipeline(steps=[('num_cont_imputer', impute.SimpleImputer(missing_values=-99999, strategy='mean'))])
num_preproc_3 = pipeline.Pipeline(steps=[('num_cont_scaler', preprocessing.StandardScaler()), ('num_float_imputer', impute.SimpleImputer(missing_values=-99999, strategy='constant', fill_value=-1))])

cat_preproc =  pipeline.Pipeline(steps=[('cat_imputer', impute.SimpleImputer(missing_values=-99999, strategy='constant', fill_value=-1)),
                                        ('encoder', preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value = -1))])

tree_prepro = compose.ColumnTransformer(transformers=[
    ('num1', num_preproc_1, num_disc_vars),
    ('num2', num_preproc_2, num_cont_vars),
    ('num3', num_preproc_3, num_float_vars),
    ], remainder='drop') 

from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.ensemble      import ExtraTreesClassifier
from sklearn.ensemble      import AdaBoostClassifier
from sklearn.ensemble      import GradientBoostingClassifier
from sklearn.experimental  import enable_hist_gradient_boosting # Necesary for HistGradientBoostingClassifier
from sklearn.ensemble      import HistGradientBoostingClassifier
from xgboost               import XGBClassifier
from lightgbm              import LGBMClassifier
from catboost              import CatBoostClassifier
from sklearn.svm           import SVC

tree_classifiers = {
  "Decision Tree": DecisionTreeClassifier(),
  "Extra Trees":   ExtraTreesClassifier(n_estimators=100),
  "Random Forest": RandomForestClassifier(n_estimators=100),
  "AdaBoost":      AdaBoostClassifier(n_estimators=100),
  "Skl GBM":       GradientBoostingClassifier(n_estimators=100),
  "Skl HistGBM":   HistGradientBoostingClassifier(max_iter=100),
  "XGBoost":       XGBClassifier(n_estimators=100),
  "LightGBM":      LGBMClassifier(n_estimators=100),
  "CatBoost":      CatBoostClassifier(n_estimators=100),
  "SVM":           SVC(kernel='linear')
}

tree_classifiers = {name: pipeline.make_pipeline(tree_prepro, model) for name, model in tree_classifiers.items()}

X = df.drop('target',axis=1)
y = df['target']

X_train, x_test, Y_train, y_test = model_selection.train_test_split(
    X, y,
    test_size=0.1,
    stratify = y,   # ALWAYS RECOMMENDED FOR BETTER VALIDATION
    random_state=42  # Recommended for reproducibility
)

gen_data = data_enhancement_1(X_train)

extra_sample = gen_data.sample(math.floor(gen_data.shape[0] * 30 / 100))
x_train_enhanced = pd.concat([X_train, extra_sample.iloc[:,:-1]])
y_train_enhanced = pd.concat([Y_train, extra_sample.iloc[:,-1]])

x_train, x_val, y_train, y_val = model_selection.train_test_split(
    X_train, Y_train,
    test_size=0.2,
    stratify = Y_train,   # ALWAYS RECOMMENDED FOR BETTER VALIDATION
    random_state=42  # Recommended for reproducibility
)

results = pd.DataFrame({'Model': [], 'Heart Att. Acc.': [], 'Healthy Acc.': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})

for model_name, model in tree_classifiers.items():

    start_time = time.time()
    model.fit(x_train, y_train)
    total_time = time.time() - start_time
        
    pred = model.predict(x_val)
    
    results = results.append({"Model":    model_name,
                              "Accuracy": metrics.accuracy_score(y_val, pred)*100,
                              "Bal Acc.": metrics.balanced_accuracy_score(y_val, pred)*100,
                              "Heart Att. Acc.": metrics.recall_score(y_val,pred)*100,
                              "Healthy Acc.": metrics.precision_score(y_val, pred)*100,
                              "Time":     total_time},
                              ignore_index=True)


results_ord = results.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
results_ord.index += 1 
results_ord.style.bar(subset=['Accuracy', 'Bal Acc.'], vmin=0, vmax=100, color='#5fba7d')

print(results_ord)

best_model = tree_classifiers[results_ord.iloc[0].Model]
best_model.fit(X_train,Y_train)

test_pred = best_model.predict(x_test)

print("Heart Accuracy:", metrics.recall_score(y_test, test_pred))
print("Healthy Accuracy:", metrics.precision_score(y_test, test_pred))
print("Overall Accuracy:", metrics.accuracy_score(y_test, test_pred))