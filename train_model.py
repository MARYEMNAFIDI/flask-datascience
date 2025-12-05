import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

ROOT = os.path.dirname(__file__)
DATA_PATH = os.path.join(ROOT, '..', 'data', 'housing_data.csv')
MODEL_OUT = os.path.join(ROOT, 'housing_model.pkl')

# Normalized column names mapping (original CSV has spaces / mixed case)
COL_RENAMES = {
    'new_price': 'price',
    'chambres': 'chambres',
    'salles de bains': 'salles_de_bains',
    'surface': 'surface',
    'ascenseur': 'ascenseur',
    'floor': 'floor',
    'terrasse': 'terrasse',
    'parking': 'parking',
    'Type': 'type_propriete',
    'City': 'city',
    'Nighberd': 'quartier',
    'address': 'address',
    'desc': 'desc'
}

def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset introuvable: {path}. Placez `housing_data.csv` dans le dossier data/")
    df = pd.read_csv(path)
    # Normalize column names
    df = df.rename(columns=lambda c: c.strip())
    df = df.rename(columns=COL_RENAMES)
    return df

def coerce_binary(df, col):
    # Map Yes/No, Yes/No-like values to 1/0; also handle 'Yes'/'No' and 'Yes,' etc.
    if col not in df.columns:
        return df
    df[col] = df[col].astype(str).str.strip().str.lower().map(lambda v: 1 if v in ['yes','y','oui','o','true','1'] else 0)
    return df

def preprocess_and_train(df, filter_city='Rabat'):
    # Force training only on the requested city (Rabat)
    if 'city' in df.columns:
        df_city = df[df['city'].str.strip().str.lower() == filter_city.lower()].copy()
        print(f'Filtering data to city={filter_city}, rows={df_city.shape[0]}')
    else:
        raise ValueError('Colonne `city` introuvable dans le dataset; impossible de filtrer sur Rabat.')

    if df_city.shape[0] < 20:
        raise ValueError('Trop peu de lignes pour entraîner (besoin d\'au moins 20 filtrées pour Rabat).')

    # Ensure target exists
    if 'price' not in df_city.columns:
        raise ValueError('Colonne cible `price` introuvable (attendu `new_price` dans le CSV).')

    # Select candidate features present in dataset
    # include quartier if present
    candidate_features = ['surface', 'chambres', 'salles_de_bains', 'floor', 'ascenseur', 'terrasse', 'parking', 'type_propriete', 'quartier']
    features = [c for c in candidate_features if c in df_city.columns]

    X = df_city[features].copy()
    y = df_city['price']

    # Coerce binary flags
    for col in ['ascenseur', 'terrasse', 'parking']:
        if col in X.columns:
            X = coerce_binary(X, col)

    # Numeric columns
    num_cols = [c for c in ['surface', 'chambres', 'salles_de_bains', 'floor'] if c in X.columns]
    cat_cols = [c for c in features if c not in num_cols]

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ], remainder='drop')

    from sklearn.ensemble import RandomForestRegressor

    # Use RandomForest with limited depth to reduce overfitting
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f'Training on {X_train.shape[0]} samples with features: {features}')
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f'Validation R^2: {score:.3f}')
    return model, score, features

if __name__ == '__main__':
    print('Loading data from', DATA_PATH)
    df = load_data()
    model, score, features = preprocess_and_train(df, filter_city='Casablanca')
    joblib.dump({'model': model, 'features': features}, MODEL_OUT)
    print('Model saved to', MODEL_OUT)
    print(f'Training done. R^2 = {score:.3f}')

