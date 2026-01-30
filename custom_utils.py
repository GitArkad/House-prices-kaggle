import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import optuna
import shap
import sklearn

from scipy import stats
from scipy.stats import ttest_ind

# Модели
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# Подготовка данных и Pipeline
import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, RobustScaler, 
    MinMaxScaler, OrdinalEncoder, TargetEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin

# Валидация и метрики
from sklearn.model_selection import (
    train_test_split, GridSearchCV, 
    RandomizedSearchCV, cross_val_score, KFold
)
from sklearn.metrics import (
    accuracy_score, roc_auc_score, 
    root_mean_squared_log_error
)
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import f_regression, chi2

# Настройки
warnings.filterwarnings('ignore')
sklearn.set_config(transform_output="pandas")


# -----ФУНКЦИИ--------

# печать числовых признаков
def plot_numer(dfp):    
    cols = dfp.select_dtypes(include='number').columns
    n_cols = 3
    n_rows = (len(cols) - 1) // n_cols + 1

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten() # Превращаем матрицу осей в плоский список

    for i, col in enumerate(cols):
        sns.histplot(dfp[col], ax=axes[i], kde=True)
        axes[i].set_title(f'Распределение {col}')

    # Удаляем пустые графики, если колонок меньше, чем ячеек в сетке
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# печать категориальных признаков
def plot_cat(dfp):
    cols = dfp.select_dtypes(include='object').columns
    n_cols = 3
    n_rows = (len(cols) - 1) // n_cols + 1

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(18, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        # Используем countplot для категориальных данных
        sns.countplot(data=dfp, x=col, ax=axes[i], palette='viridis', hue=col, legend=False)
        
        axes[i].set_title(f'Частота: {col}', fontsize=12)
        axes[i].tick_params(axis='x', rotation=45) # Поворачиваем текст, чтобы не накладывался
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Количество')

    # Удаляем пустые ячейки
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# --- Классы ---

class HousePricesSmartImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stats_ = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.stats_ = {}
        
        if 'LotFrontage' in X.columns and 'Neighborhood' in X.columns:
            self.stats_['lot_medians'] = X.groupby('Neighborhood')['LotFrontage'].median().to_dict()
            self.stats_['lot_overall'] = X['LotFrontage'].median()

        if 'MSZoning' in X.columns and 'MSSubClass' in X.columns:
            mode_series = X.groupby('MSSubClass')['MSZoning'].agg(
                lambda x: x.mode().iat[0] if not x.mode().empty else np.nan
            )
            self.stats_['zoning_modes'] = mode_series.to_dict()
            overall_zoning_mode = X['MSZoning'].mode()
            self.stats_['zoning_overall'] = overall_zoning_mode.iat[0] if not overall_zoning_mode.empty else None
            
        return self

    def transform(self, X):
        X = X.copy()
        if 'LotFrontage' in X.columns:
            X['LotFrontage'] = X['LotFrontage'].fillna(X['Neighborhood'].map(self.stats_.get('lot_medians', {})))
            X['LotFrontage'] = X['LotFrontage'].fillna(self.stats_.get('lot_overall'))
        if 'MSZoning' in X.columns:
            X['MSZoning'] = X['MSZoning'].fillna(X['MSSubClass'].map(self.stats_.get('zoning_modes', {})))
            X['MSZoning'] = X['MSZoning'].fillna(self.stats_.get('zoning_overall'))
        if 'GarageYrBlt' in X.columns and 'YearBuilt' in X.columns:
            X['GarageYrBlt'] = X['GarageYrBlt'].fillna(X['YearBuilt'])
        if 'Functional' in X.columns:
            X['Functional'] = X['Functional'].fillna('Typ')
        return X

class HousePricesFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
        X['HouseAge'] = X['YrSold'] - X['YearBuilt']
        X['YearsSinceRemodel'] = X['YrSold'] - X['YearRemodAdd']
        X['IsNew'] = (X['YearBuilt'] == X['YrSold']).astype(int)
        return X

    def get_feature_names_out(self, input_features=None):
        new_cols = ['TotalSF', 'YearsSinceRemodel', 'HouseAge', 'IsNew']
        return list(input_features) + new_cols

class CatBoostTags:
    @property
    def estimator_type(self): 
        return "regressor"
    def __getattr__(self, name): 
        return None
