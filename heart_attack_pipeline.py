import pandas as pd
from time import time

#Data Processing
from sklearn import pipeline
from sklearn import preprocessing #OrdinalEncoder and OHE
from sklearn import impute
from sklearn import compose
from sklearn.model_selection import train_test_split #train_test_split
from sklearn import metrics #accuracy score, balanced_accuracy_score, confusion_matrix
import pickle as pk

import column_enhancement as ce
import row_enhancement as re




def training_pipeline(dataframe, enhance_features=True, enhance_data=True, percent = 25, save_model=False) -> float:
    
    df = dataframe.copy()

    if enhance_features:
        df = ce.add_columns(df)

        num_vars = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak', 'chol_age']
        cat_vars = [ 'sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall', 'defect' ]
    
    if not enhance_features:
        num_vars = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
        cat_vars = [ 'sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall']
    
    if enhance_data:
        df = re.data_enhancement_cp_slope_based(df, percent)
        # df.drop(['target'], axis = 1, inplace=True)

    #Pre-Processing Pipelines
    cat_pipe = pipeline.Pipeline(steps= [
    ('ordinal', preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value = -1))
    ])

    num_pipe = pipeline.Pipeline(steps=[
    ('imputer', impute.SimpleImputer(strategy='mean')),
    ])

    tree_preprocessing = compose.ColumnTransformer(transformers=[
            ('num', num_pipe, num_vars),
            ('cat', cat_pipe, cat_vars),
    ], remainder='drop') #Drop other vars not in num_vars or cat_vars

    #Train Test Split
    x, y = df.iloc[:, :-1], df.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                           test_size=0.2, 
                                                           random_state=0)

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
    "XGBoost":       XGBClassifier(n_estimators=100, use_label_encoder=False),
    "LightGBM":      LGBMClassifier(n_estimators=100),
    "CatBoost":      CatBoostClassifier(n_estimators=100, verbose=False)
    }

    # Training Procedure

    results = pd.DataFrame({'Model': [], 'Heart Att. Acc.': [], 'Healthy Acc.': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})

    for model_name, model in tree_classifiers.items():
        
        pipe = pipeline.Pipeline([
            ('preprocessor', tree_preprocessing),
            ('classifier', model)
        ])

        start_time = time()

        pipe.fit(x_train, y_train)
        y_pred = pipe.predict(x_test)

        total_time = time() - start_time

        results = results.append({
                              "Model":    model_name,
                              "Accuracy": metrics.accuracy_score(y_test, y_pred)*100,
                              "Bal Acc.": metrics.balanced_accuracy_score(y_test, y_pred)*100,
                              "Time":     total_time
                              },
                              ignore_index=True)       
    
    results_ord = results.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
    results_ord.index += 1 
    results_ord.style.bar(subset=['Accuracy', 'Bal Acc.'], vmin=0, vmax=100, color='#5fba7d')

    #Determine the best model
    best_model = tree_classifiers[results_ord.iloc[0].Model]
    best_model.fit(x_train, y_train)

    #Saving the Model
    if save_model:
              model_directory = './model/optimal_model.pkl'
              with open(model_directory, 'wb') as file:
                     pk.dump(best_model, file)
    

    return #metrics.accuracy_score(y_test, pred)*100,


'''
TO BE ADDED ON LATER VERSION
ADDING PREDICTION PIPELINE FOR SPECIFIC EXAMPLE

def prediction_pipeline(dataframe, enhance_features=True, enhance_data=True, percent = 25, save_model=False) -> float:
    
    df = dataframe.copy()

    if enhance_features:
        df = ce.add_columns(df)

        rename_target = lambda x: x
        df['target']=dfe['output'].apply(rename_target)
        df.drop(['output'], axis=1, inplace=True)

        num_vars = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak', 'chol_age']
        cat_vars = [ 'sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall', 'defect' ]
    
    if not enhance_features:
        num_vars = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
        cat_vars = [ 'sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall']
    
    if enhance_data:
        df = re.data_enhancement_cp_slope_based(df, percent)
        # df.drop(['target'], axis = 1, inplace=True)

    #Pre-Processing Pipelines
    cat_pipe = pipeline.Pipeline(steps= [
    ('ordinal', preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value = -1))
    ])

    num_pipe = pipeline.Pipeline(steps=[
    ('imputer', impute.SimpleImputer(strategy='mean')),
    ])

    tree_preprocessing = compose.ColumnTransformer(transformers=[
            ('num', num_pipe, num_vars),
            ('cat', cat_pipe, cat_vars),
    ], remainder='drop') #Drop other vars not in num_vars or cat_vars

    #Train Test Split
    x, y = df.iloc[:, :-1], df.iloc[:, -1]

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
    "Decision Tree": DecisionTreeRegressor(),
    "Extra Trees": ExtraTreesRegressor(),
    "Random Forest": RandomForestRegressor(),
    "AdaBoost": AdaBoostRegressor(),
    "Skl GBM": GradientBoostingRegressor(),
    "Skl HistGBM": HistGradientBoostingRegressor(),
    "XGBoost": XGBRegressor(use_label_encoder=False), #To remove the output on terminal
    "LightGBM": LGBMRegressor(),
    "CatBoost": CatBoostRegressor(verbose=False), #To remove the output on terminal
    }

    # Training Procedure

    for model_name, model in tree_classifiers.items():
        
        pipe = pipeline.Pipeline([
            ('preprocessor', tree_preprocessing),
            ('classifier', model)
        ])

        start_time = time()

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        total_time = time() - start_time

        results = results.append({
                              "Model":    model_name,
                              "Accuracy": metrics.accuracy_score(y_test, pred)*100,
                              "Bal Acc.": metrics.balanced_accuracy_score(y_test, pred)*100,
                              "Time":     total_tim
                              },
                              ignore_index=True)       
    
    results_ord = results.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
    results_ord.index += 1 
    results_ord.style.bar(subset=['Accuracy', 'Bal Acc.'], vmin=0, vmax=100, color='#5fba7d')

    best_model = tree_classifiers[results_ord.iloc[0].Model]

    if save_model:
              model_directory = './model/optimal_model.pkl'
              with open(model_directory, 'wb') as file:
                     pk.dump(best_model, file)
    
    return metrics.accuracy_score(y_test, pred)*100,

'''



    