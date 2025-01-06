import joblib

import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def vectorize_flows(df, categorical_cols, numeric_cols, label_col=None, scaler_path=None, one_hot_encoder_path=None, is_test=False):
    """
    Transforme les flux en vecteurs de caractéristiques numériques à partir d'un DataFrame directement.

    :param df: DataFrame contenant les flux.
    :param categorical_cols: Colonnes catégoriques à encoder.
    :param numeric_cols: Colonnes numériques à normaliser.
    :param label_col: Nom de la colonne contenant les labels.
    :param scaler_path: Chemin de l'Objet de normalisation des données.
    :param one_hot_encoder_path: Chemin de l'Objet d'encodage one-hot.
    :param is_test: Indique si les données sont pour l'entraînement ou le test.
    :return: DataFrame contenant les colonnes numériques normalisées et les colonnes catégoriques enc
    """

    # Vérification des colonnes manquantes
    missing_cols = [col for col in numeric_cols + categorical_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns {missing_cols} not in index")



    # Transformation des ip en classes
    df['src_ip'] = df['src_ip'].apply(ip_to_class)
    df['dst_ip'] = df['dst_ip'].apply(ip_to_class)

    # Séparer les labels si disponibles
    y = df[label_col].values if label_col and label_col in df.columns else None

    # Convertir toutes les colonnes catégoriques en chaîne de caractères
    df[categorical_cols] = df[categorical_cols].astype(str)

    categorical_data = df[categorical_cols]
    numeric_data = df[numeric_cols]




    # Initialisation des objets pour l'entraînement si nécessaire
    if not is_test: # le training
        scaler = StandardScaler()
        one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        categorical_encoded = one_hot_encoder.fit_transform(categorical_data)
        numeric_normalized = scaler.fit_transform(numeric_data)
    else:
        # Charger les objets de normalisation et d'encodage
        scaler = joblib.load(scaler_path)
        one_hot_encoder = joblib.load(one_hot_encoder_path)
        categorical_encoded = one_hot_encoder.transform(categorical_data)
        numeric_normalized = scaler.transform(numeric_data)

    # Créer un DataFrame pour les colonnes catégoriques encodées
    categorical_columns = one_hot_encoder.get_feature_names_out(categorical_cols)
    categorical_df = pd.DataFrame(categorical_encoded, columns=categorical_columns, index=df.index)

    # Gestion des colonnes numériques
    numeric_df = pd.DataFrame(numeric_normalized, columns=numeric_cols, index=df.index)

    # re-Concaténer les colonnes catégoriques et numériques
    x = pd.concat([numeric_df, categorical_df], axis=1)

    # enregister les objets de normalisation et d'encodage
    if not is_test:
        joblib.dump(scaler, scaler_path)
        joblib.dump(one_hot_encoder, one_hot_encoder_path)

    # Ajouter les labels si disponibles
    if y is not None:
        x['label'] = y

    return x


def ip_to_class(ip:str)->str:
    """
    Convertit une adresse IP en classe d'adresse.
    """
    ip = ip.split(".")
    if ip[0] == "10":
        return "A"
    elif ip[0] == "172" and 16 <= int(ip[1]) <= 31:
        return "B"
    elif ip[0] == "192" and ip[1] == "168":
        return "C"
    else:
        return "D"