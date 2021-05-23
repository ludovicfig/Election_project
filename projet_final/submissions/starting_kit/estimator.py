import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from xgboost.sklearn import  XGBRegressor
from sklearn.impute import SimpleImputer

def class_features_transformation(data, col):
    return pd.concat([data.drop(col, axis=1), pd.get_dummies(data[col])], axis=1)


def CODGEO_to_departement(string):
    """Retourne le numéro du département du code postal donné en entrée.

Entrée
------
string : chaîne de caractère donnant le code postal de la commune

Sortie
------
numéro du département (int, à part pour la corse)
"""
    if '2A' in string:
        return '2A'
    if '2B' in string:
        return '2B'
    else:
        return int(string[:2].lstrip('0'))  #on prend les deux premiers chiffres du code postal


def to_num(x):
    """Retourne le numéro du département en entier si c'en est un.

Entrée
------
string : chaîne de caractère donnant le code postal de la commune

Sortie
------
numéro du département (int, à part pour la corse ou autre Z..)
"""
    if type(x) == str:
        for char in x:
            if char not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                return(x)
            else:
                return int(x)
        else:
            return x

def _preprocessor(X):

    df_election = X

    col_nombre = ['Nb' in c for c in df_election.columns] 
    total = df_election['Nb Femme'] + df_election['Nb Homme'] 
    for c in df_election.columns[col_nombre]:
        df_election[c] = df_election[c] / total
    df_election = df_election.replace([np.inf, -np.inf], 0)
    Categorical_columns = [
                      'Orientation Economique',
                      'SEG Croissance POP',
                      'Urbanité Ruralité',
                      'Dynamique Démographique BV',
                      'Environnement Démographique',
                      'Fidélité',
                      'SYN MEDICAL',
                      'Seg Dyn Entre',
                      'SEG Environnement Démographique Obsolète',
                      'Seg Cap Fiscale',
                      'DYN SetC',
                      'CP' ]
    for col in Categorical_columns:
        df_election = class_features_transformation(df_election,col)
    
    return df_election


def get_estimator():

    drop_cols = ['CODGEO', 'LIBGEO', 'REG', 'DEP', 'Code Nuance',
                 'Code du département']
    base_cols = [
        'Orientation Economique', 'SEG Croissance POP', 'Urbanité Ruralité',
        'Dynamique Démographique BV', 'Environnement Démographique',
        'Fidélité', 'SYN MEDICAL', 'Seg Dyn Entre',
        'SEG Environnement Démographique Obsolète', 'Seg Cap Fiscale',
        'DYN SetC', 'CP', 'MED14', 'Nb Femme', 'Nb Homme'
        ]
    
    base_transformer = FunctionTransformer(
        _preprocessor, validate=False
    )

    base_transformer = make_pipeline(
        base_transformer, SimpleImputer(strategy='most_frequent')
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('base', base_transformer, base_cols),
            ('drop cols', 'drop', drop_cols),
        ], remainder='passthrough')  # remainder='drop' or 'passthrough'

    regressor = XGBRegressor()

    pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('Regressor', regressor)
    ])

    return pipeline