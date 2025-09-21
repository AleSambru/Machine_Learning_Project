# importing relevant libraries
from math import isnan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# sklearn
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score



### == Function implementation of function used in notebook 

def upload_results(label, coeff_df, train_metrics_df, test_metrics_df, dic):

    dic[label] = {
        'coeff': coeff_df,
        'metrics': {
            'train': train_metrics_df,
            'test': test_metrics_df
        }
    }

    return dic


def split_train_target(df, Y_labels = ['T_degC', 'Salnty'], drop_labels =['T_degC', 'Salnty'], test_size = 0.2):
    # we define the design matrix
    
    y = df[Y_labels]
    X = df.drop(columns=drop_labels)
    feature_cols = X.columns
    
    # train/test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
        
    return y_train, X_train, y_test, X_test,feature_cols

def define_features(df, Y_labels = ['T_degC', 'Salnty'], drop_labels =['T_degC', 'Salnty'], test_train_size = 0.2, test_val_size = 0.25):
    # we define the design matrix
    
    y = df[Y_labels]
    X = df.drop(columns=drop_labels)
    feature_cols = X.columns
    
    # train/test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_train_size, random_state=0)
    
    # split train in train/val (es. 25% del train → 20% totale)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_val_size, random_state=0)
    
    return y_train, X_train, y_test, X_test, y_val, X_val,feature_cols


# ============================
# ==== Regression methods ====
# ============================

def OLS(df, Y_labels=['T_degC', 'Salnty'], drop_labels=['T_degC', 'Salnty'], 
        test_train_size=0.2, test_val_size=0.25, val=True):
    
    # Split features/target
    if val:  # original dataset is split in 3 datasets
        y_train, X_train, y_test, X_test, y_val, X_val, feature_cols = define_features(
            df, Y_labels=Y_labels, drop_labels=drop_labels, 
            test_train_size=test_train_size, test_val_size=test_val_size)
    else:  # original dataset is split in 2 datasets: Train and test
        y_train, X_train, y_test, X_test, feature_cols = split_train_target(
            df, Y_labels=Y_labels, drop_labels=drop_labels, test_size=test_train_size)
    
    # == Rescale design matrix
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    
    # == Train the model
    ols = LinearRegression().fit(X_train_scaled, y_train)
    
    # == Extract coefficients + intercept
    coef_df = pd.DataFrame(ols.coef_.T, index=feature_cols, columns=Y_labels)
    intercept_df = pd.DataFrame([ols.intercept_], index=["Intercept"], columns=Y_labels)
    coef_df = pd.concat([coef_df, intercept_df])
    
    # == Predictions
    y_train_pred  = ols.predict(X_train_scaled)
    y_test_pred = ols.predict(X_test_scaled)
    
    if val:  # original dataset is split in 3 datasets
        X_val_scaled = scaler.transform(X_val)  # use transform, not fit_transform
        y_val_pred  = ols.predict(X_val_scaled)
        return coef_df, y_train, y_train_pred, y_test, y_test_pred, y_val, y_val_pred
    else:  # original dataset is split in 2 datasets
        return coef_df, y_train, y_train_pred, y_test, y_test_pred

def MSE_train_size(df , Y_labels=['T_degC', 'Salnty'], drop_labels=['T_degC', 'Salnty'], N = 50, min_train_size = 20, perc = 0.9):

    # total number of salmples 
    total_samples = len(df)
    max_train_size = int(perc * total_samples)

    # definition of the train size
    train_sizes = np.unique(np.geomspace(min_train_size, max_train_size, N).astype(int))

    # list to save data
    train_errors = []
    test_errors = []

    # features and labels
    y = df[Y_labels]
    X = df.drop(columns=drop_labels)
    feature_cols = X.columns
    
    
    for size in train_sizes:
        test_fraction = 1 - (size / total_samples)
        # split 
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_fraction, random_state=42
        )

        # Standardization
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled  = scaler_X.transform(X_test)

        # OLS
        ols = LinearRegression().fit(X_train_scaled, y_train)

        # Predictions
        y_train_pred = ols.predict(X_train_scaled)
        y_test_pred  = ols.predict(X_test_scaled)

        # MSE
        train_errors.append([ size, mean_squared_error(y_train, y_train_pred)])
        test_errors.append([ size, mean_squared_error(y_test, y_test_pred)])
        
        test_errors_df = pd.DataFrame(test_errors, columns = ['train_size', 'MSE'])
        train_errors_df = pd.DataFrame(train_errors, columns = ['train_size', 'MSE'])
        
        
        
    return test_errors_df, train_errors_df

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    metrics = [["MSE", mse], 
               ["MAE", mae], 
               ["R2", r2]]
    
    metrics_df = pd.DataFrame(metrics) 
    df = metrics_df.set_index(0).T
    return df


# Definiamo una funzione per scatter plot Predetto vs Reale
def plot_pred_vs_true(y_true, y_pred, label, ax, color = 'blue'):
    ax.scatter(y_true, y_pred, alpha=0.5, color  = color, label = label, ec = 'white' )
    ax.plot([y_true.min(), y_true.max()],[y_true.min(), y_true.max()],'r--', lw=2, ms = 10)
    
    return
    
    


# Funzione per calcolare MSE medio da CV
def compute_cv_mse(alphas, X_train, y_train, cv):
    mse_list = []
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        scores = cross_val_score(ridge, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
        mse_list.append(-np.mean(scores))
    return np.array(mse_list)
    

    
def best_shrinkage(df , Y_labels = ['T_degC', 'Salnty'], drop_labels =['T_degC', 'Salnty'], min1 = -1, max1 = 1 , min2 = -1, max2 = -0.5, cv = 5):
    
    y_train, X_train, y_test, X_test, y_val, X_val,feature_cols = define_features(df = df, Y_labels = Y_labels, drop_labels =drop_labels)

    # Definiamo due range di alphas
    alphas_range1 = np.logspace(min1,max1, 50)  
    alphas_range2 = np.logspace(min2,max2, 50)    


    # Calcoliamo i due andamenti
    mse_cv_1 = compute_cv_mse(alphas_range1, X_train, y_train, cv)
    mse_cv_2 = compute_cv_mse(alphas_range2, X_train, y_train, cv)

    # Troviamo i best alpha separatamente
    best_alpha_1 = alphas_range1[np.argmin(mse_cv_1)]
    best_alpha_2 = alphas_range2[np.argmin(mse_cv_2)]

    # Grafici affiancati
    fig, axes = plt.subplots(1, 2, figsize=(14,6), sharey=False)

    # Primo range
    axes[0].semilogx(alphas_range1, mse_cv_1, marker='o', color='blue')
    axes[0].axvline(best_alpha_1, color='red', linestyle='--',
                    label=f'Min MSE at α={best_alpha_1:.2f}')
    axes[0].set_xlabel('Lambda (alpha)')
    axes[0].set_ylabel('MSE medio K-fold CV')
    axes[0].legend()
    axes[0].grid(True)

    # Secondo range
    axes[1].semilogx(alphas_range2, mse_cv_2, marker='o', color='blue')
    axes[1].axvline(best_alpha_2, color='red', linestyle='--',
                    label=f'Min MSE at α={best_alpha_2:.2f}')
    axes[1].set_xlabel('Lambda (alpha)')
    axes[1].tick_params(axis='x', rotation=45, which='both')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()



    print(f'Miglior alpha (range max) : {best_alpha_1}')
    print(f'Miglior alpha (range min) : {best_alpha_2}')
    return best_alpha_1, best_alpha_2


def Ridge_Regression(X_train, y_train, X_test, y_test, alpha):
    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Ridge regression
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_train = ridge_model.predict(X_train_scaled)
    y_pred_test  = ridge_model.predict(X_test_scaled)

    return X_train_scaled, X_test_scaled, y_pred_train, y_pred_test, y_train, y_test, scaler, ridge_model


def Lasso_MultiOutputRegressor(X_train, y_train, X_test, y_test, feature_cols, Y_labels, alpha=0.01):
    # Standardizzazione
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    
    # Lasso multitarget
    lasso = MultiOutputRegressor(Lasso(alpha=alpha, max_iter=50000))
    lasso.fit(X_train_scaled, y_train)
    
    # Costruzione DataFrame coefficienti
    coef_dict = {}
    intercept_dict = {}
    
    for i, target in enumerate(Y_labels):
        coef_dict[target] = lasso.estimators_[i].coef_
        intercept_dict[target] = lasso.estimators_[i].intercept_
    
    coef_df = pd.DataFrame(coef_dict, index=feature_cols)
    coef_df.loc['Intercept'] = intercept_dict
    
    # Predizioni
    y_train_pred = lasso.predict(X_train_scaled)
    y_test_pred  = lasso.predict(X_test_scaled)
    
    return coef_df, y_train, y_train_pred, y_test, y_test_pred


'''def Lasso_regression(df=chifis_df, Y_labels=['T_degC', 'Salnty'], drop_labels=['T_degC', 'Salnty'], alpha=0.01):
    # Split features/target
    y_train, X_train, y_val, X_val, y_test, X_test, feature_cols = define_features(df, Y_labels, drop_labels)
    
    # Standardizzazione
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)
    
    # Lasso regression
    lasso = Lasso(alpha=alpha).fit(X_train_scaled, y_train)
    
    # Coefficienti
    coef_df = pd.DataFrame(lasso.coef_.reshape(-1, len(Y_labels)), index=feature_cols, columns=Y_labels)
    coef_df.loc['Intercept'] = lasso.intercept_
    
    # Predizioni
    y_train_pred = lasso.predict(X_train_scaled)
    y_val_pred   = lasso.predict(X_val_scaled)
    y_test_pred  = lasso.predict(X_test_scaled)
    
    return coef_df, y_train, y_train_pred, y_test, y_test_pred, y_val, y_val_pred
'''
'''def LASSO_Regression(X_train, y_train, X_test, y_test, alpha):
    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Ridge regression
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_train = lasso_model.predict(X_train_scaled)
    y_pred_test  = lasso_model.predict(X_test_scaled)

    return X_train_scaled, X_test_scaled, y_pred_train, y_pred_test, y_train, y_test, scaler, lasso_model'''

'''
def Ridge_regression(df, Y_labels=['T_degC', 'Salnty'], drop_labels=['T_degC', 'Salnty'], alpha=1.0, test_train_size=0.2, test_val_size=0.25, val=True):
    
    # Split features/target
    if val:  # dataset split in 3
        y_train, X_train, y_test, X_test, y_val, X_val, feature_cols = define_features(
            df, Y_labels=Y_labels, drop_labels=drop_labels, 
            test_train_size=test_train_size, test_val_size=test_val_size)
    else:  # dataset split in 2
        y_train, X_train, y_test, X_test, feature_cols = split_train_target(
            df, Y_labels=Y_labels, drop_labels=drop_labels, test_size=test_train_size)

    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    
    # Ridge regression
    ridge = Ridge(alpha=alpha)
    if y_train.shape[1] == 1:
        ridge.fit(X_train_scaled, y_train.values.ravel())
    else:
        ridge.fit(X_train_scaled, y_train)
    
    # Coefficienti
    coef_df = pd.DataFrame(ridge.coef_.T, index=feature_cols, columns=Y_labels)
    coef_df.loc['Intercept'] = ridge.intercept_
    
    # Predizioni
    y_train_pred = ridge.predict(X_train_scaled)
    y_test_pred  = ridge.predict(X_test_scaled)
    
    if val:  # dataset split in 3
        X_val_scaled = scaler.transform(X_val)
        y_val_pred  = ridge.predict(X_val_scaled)
        return coef_df, y_train, y_train_pred, y_test, y_test_pred, y_val, y_val_pred
    else:  # dataset split in 2
        return coef_df, y_train, y_train_pred, y_test, y_test_pred'''
    
    

    



'''def Lasso_regression(df=chifis_df, Y_labels=['T_degC', 'Salnty'], drop_labels=['T_degC', 'Salnty'], alpha=0.01):
    # Split features/target
    y_train, X_train, y_val, X_val, y_test, X_test, feature_cols = define_features(df, Y_labels, drop_labels)
    
    # Standardizzazione
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)
    
    # Lasso regression
    lasso = Lasso(alpha=alpha).fit(X_train_scaled, y_train)
    
    # Coefficienti
    coef_df = pd.DataFrame(lasso.coef_.reshape(-1, len(Y_labels)), index=feature_cols, columns=Y_labels)
    coef_df.loc['Intercept'] = lasso.intercept_
    
    # Predizioni
    y_train_pred = lasso.predict(X_train_scaled)
    y_val_pred   = lasso.predict(X_val_scaled)
    y_test_pred  = lasso.predict(X_test_scaled)
    
    return coef_df, y_train, y_train_pred, y_test, y_test_pred, y_val, y_val_pred'''