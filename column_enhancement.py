import math
import numpy as np
import pandas as pd

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


#num_vars = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak'] #with target?
#cat_vars = [ 'sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall' ]

def chol_age(data):
    dfe = data.copy()
    dfe['chol_age'] = dfe['chol']/dfe['age']

    return dfe

data = pd.read_csv("data/heart.csv")


def heart_defect(data):

    dfe = data.copy()
    defect = []
    
    #Check every row for defects
    for index, x in dfe.iterrows():
        if x['cp'] == 1 or x['cp']==2:
            if x['thalachh'] > 150:
                defect.append(1)
            else:
                defect.append(2)

        else:
            if x['thalachh'] > 150:
                defect.append(3)
            else:
                defect.append(4)
    
    dfe['defect'] = pd.DataFrame(defect)

    return dfe

def add_columns(df):
    
    dfe = df.copy()
    
    dfe = chol_age(dfe)
    dfe = heart_defect(dfe)

    rename_target = lambda x: x
    dfe['target']=dfe['output'].apply(rename_target)
    dfe.drop(['output'], axis=1, inplace=True)

    return dfe




