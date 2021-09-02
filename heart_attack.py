#Maths
import numpy as np
import pandas as pd
import math
import time

#Graphs and Plots
import sweetviz as vs
import PIL as Image
import matplotlib.pyplot as plt
import seaborn as sns

#Data Processing
from sklearn import pipeline
from sklearn import preprocessing #OrdinalEncoder and OHE
from sklearn import impute
from sklearn import compose
from sklearn import model_selection #train_test_split
from sklearn import metrics #accuracy score, balanced_accuracy_score, confusion_matrix

# Data Information

'''

1. age (#)
2. sex : 1= Male, 0= Female (Binary)
3. (cp)chest pain type (4 values -Ordinal):Value 1: typical angina ,Value 2: atypical angina, Value 3: non-anginal pain , Value 4: asymptomatic
4. (trestbps) resting blood pressure (#)
5. (chol) serum cholesterol in mg/dl (#)
6. (fbs)fasting blood sugar > 120 mg/dl(Binary)(1 = true; 0 = false)
7. (restecg) resting electrocardiography results(values 0,1,2)
8. (thalach) maximum heart rate achieved (#)
9. (exang) exercise induced angina (binary) (1 = yes; 0 = no)
10. (oldpeak) = ST depression induced by exercise relative to rest (#)
11. (slope) of the peak exercise ST segment (Ordinal) (Value 1: up sloping , Value 2: flat , Value 3: down sloping )
12. (ca) number of major vessels (0–3, Ordinal) colored by fluoroscopy
13. (thal) maximum heart rate achieved — (Ordinal): 3 = normal; 6 = fixed defect; 7 = reversible defect
target = 0 not suffering // 1 suffering


'''

# STEP 1-1
# Read Data
data = pd.read_csv("data/heart.csv")
#data.rename(columns={'num       ':'target'},inplace=True)



num_vars = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak'] #with target?
cat_vars = [ 'sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall' ]

#print(data.dtypes)
data.columns


#print(preprocessing_tree)

# STEP 2
# 2.1 Row Data Enhancement
# Several functions are created to enhance the rows by different categories

#num_vars = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak'] #with target?
#cat_vars = [ 'sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall' ]

# FUNCTION 1 : Row Enhancement based on slope
def data_enhancement_slope_based(data):
    copy_data = data.copy()
    gen_data = pd.DataFrame()

    for slp_i in copy_data['slp'].unique():
        slp_data = copy_data[copy_data['slp']==slp_i]
        trtbps_std = slp_data['trtbps'].std()
        chol_std = slp_data['chol'].std()
        thalachh_std = slp_data['thalachh'].std()
        oldpeak_std = slp_data['oldpeak'].std()

        for i in range(slp_data.shape[0]):
            
            if np.random.randint(2) == 1:
                slp_data['trtbps'].values[i] += trtbps_std
            else:
                slp_data['trtbps'].values[i] -= trtbps_std
                
            if np.random.randint(2) == 1:
                slp_data['chol'].values[i] += chol_std
            else:
                slp_data['chol'].values[i] -= chol_std
                
            if np.random.randint(2) == 1:
                slp_data['thalachh'].values[i] += thalachh_std
            else:
                slp_data['thalachh'].values[i] -= thalachh_std
                
            if np.random.randint(2) == 1:
                slp_data['oldpeak'].values[i] += oldpeak_std
            else:
                slp_data['oldpeak'].values[i] -= oldpeak_std
            
        gen_data = pd.concat([gen_data, slp_data])
            
    return gen_data

#Function 2 : Row Enhancement based on cp then slope
def data_enhancement_cp_slope_based(data):
    copy_data = data.copy()
    gen_data = pd.DataFrame()

    for cp_i in copy_data['cp'].unique():
        cp_data = copy_data[copy_data['cp']==cp_i]
        
        for slp_i in copy_data['slp'].unique():
            slp_data = cp_data[cp_data['slp']==slp_i]
            trtbps_std = slp_data['trtbps'].std()
            chol_std = slp_data['chol'].std()
            thalachh_std = slp_data['thalachh'].std()
            oldpeak_std = slp_data['oldpeak'].std()

            for i in range(slp_data.shape[0]):
            
                if np.random.randint(2) == 1:
                    slp_data['trtbps'].values[i] += trtbps_std
                else:
                    slp_data['trtbps'].values[i] -= trtbps_std
                
                if np.random.randint(2) == 1:
                    slp_data['chol'].values[i] += chol_std
                else:
                    slp_data['chol'].values[i] -= chol_std
                
                if np.random.randint(2) == 1:
                    slp_data['thalachh'].values[i] += thalachh_std
                else:
                    slp_data['thalachh'].values[i] -= thalachh_std
                
                if np.random.randint(2) == 1:
                    slp_data['oldpeak'].values[i] += oldpeak_std
                else:
                    slp_data['oldpeak'].values[i] -= oldpeak_std
            
            gen_data = pd.concat([gen_data, slp_data])

    return gen_data

#2.2 Column Data Enhancement
data['chol_age'] = data['chol']/data['age']
data['target']=data['output']
data.drop(['output'], axis=1, inplace=True)

# STEP 3
# Train Test Split & Concatenation of Raw Data & Enhanced Data

# 3.1 Train Test Split

#Concatenation
gen_data = data_enhancement_cp_slope_based(data)
extra_sample = gen_data.sample(math.floor(gen_data.shape[0]/4))
enhanced_data = pd.concat([data, extra_sample])

x = enhanced_data.iloc[:,:-1]
y = enhanced_data.iloc[:,-1]

#Spliting
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=0)

# Redefine the cat_vars and num_vars based on Column Data Enhancement
num_vars = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak'] #with target?
cat_vars = [ 'sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall' ]

cat_pipe = pipeline.Pipeline(steps= [
    ('ordinal', preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value',
    unknown_value = np.nan))
    ])

num_pipe = pipeline.Pipeline(steps=[
    ('imputer', impute.SimpleImputer(strategy='mean')),
    #('scalar', preprocessing.QuantileTransformer(n_quantiles=200, output_distribution='normal', random_state=10))
])

preprocessing_tree = compose.ColumnTransformer(transformers=[
    ('num', num_pipe, num_vars),
    ('cat', cat_pipe, cat_vars),
], remainder='drop') #Drop other vars not in num_vars or cat_vars

# STEP 4
# Use Pipeline and Train Model

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
}

tree_classifiers = {name: pipeline.make_pipeline(preprocessing_tree, model) for name, model in tree_classifiers.items()}

#Results

results = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})

for model_name, model in tree_classifiers.items():

    start_time = time.time()
    model.fit(x_train, y_train)
    total_time = time.time() - start_time
        
    pred = model.predict(x_test)
    
    results = results.append({"Model":    model_name,
                              "Accuracy": metrics.accuracy_score(y_test, pred)*100,
                              "Bal Acc.": metrics.balanced_accuracy_score(y_test, pred)*100,
                              "Time":     total_time},
                              ignore_index=True)

results_ord = results.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
results_ord.index += 1 
results_ord.style.bar(subset=['Accuracy', 'Bal Acc.'], vmin=0, vmax=100, color='#5fba7d')

confusion_matrix = metrics.confusion_matrix(y_test,pred)

print(f'The Best ML Model is {results_ord.iloc[0].Model} with an accuracy of {results_ord.iloc[0].Accuracy}')


