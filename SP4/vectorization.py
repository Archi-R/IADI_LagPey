import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tools import *

def vectorize_flows(
    df,
    categorical_cols,
    numeric_cols,
    label_col=None,
    scaler_path=None,
    one_hot_encoder_path=None,
    is_test=False
):
    """
    Vectorise les flux réseau (train ou test) :
      - Convertit certaines colonnes IP (src_ip, dst_ip) si nécessaire.
      - Transforme les colonnes catégorielles (OneHotEncoder).
      - Normalise les colonnes numériques (StandardScaler).
      - Gère les labels si 'label_col' est indiqué.

    Paramètres :
      :param df: DataFrame contenant les flux.
      :param categorical_cols: Liste des colonnes catégorielles à encoder.
      :param numeric_cols: Liste des colonnes numériques à normaliser.
      :param label_col: Nom de la colonne contenant les labels (facultatif).
      :param scaler_path: Chemin du StandardScaler sauvegardé (ou à sauvegarder).
      :param one_hot_encoder_path: Chemin du OneHotEncoder sauvegardé (ou à sauvegarder).
      :param is_test: Booléen, False pour l'entraînement (fit), True pour le test (transform).

    Retourne :
      Un DataFrame contenant :
        - Les colonnes numériques normalisées
        - Les colonnes catégorielles encodées en one-hot
        - Optionnellement la colonne 'label' (si label_col existe dans df)
    """
    df = clean_df(df)

    # 1) Vérifier la présence des colonnes nécessaires
    required_cols = numeric_cols + categorical_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Colonnes manquantes dans le DataFrame : {missing_cols}")

    # 2) Conversion IP en classes

    df['src_ip'] = df['src_ip'].apply(ip_to_class)
    df['dst_ip'] = df['dst_ip'].apply(ip_to_class)

    # 3) Extraction éventuelle du label
    y = df[label_col].values if label_col and label_col in df.columns else None

    # 4) Forcer les colonnes catégorielles en 'string' pour éviter les soucis dtype
    df[categorical_cols] = df[categorical_cols].astype(str)

    # 5) Séparer les features catégorielles / numériques
    categorical_data = df[categorical_cols]
    numeric_data = df[numeric_cols]

    if not is_test:
        # === PHASE D'ENTRAÎNEMENT ===
        # Initialiser le scaler et le OHE avec handle_unknown='ignore'
        scaler = StandardScaler()
        one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        # Fit + transform sur l'entraînement
        categorical_encoded = one_hot_encoder.fit_transform(categorical_data)
        numeric_normalized = scaler.fit_transform(numeric_data)

        # Sauvegarder les objets entraînés
        joblib.dump(scaler, scaler_path)
        joblib.dump(one_hot_encoder, one_hot_encoder_path)
        print(one_hot_encoder_path)
        print(scaler_path)
    else:
        # === PHASE DE TEST ===
        # Charger les objets de normalisation et d'encodage
        scaler = joblib.load(scaler_path)
        one_hot_encoder = joblib.load(one_hot_encoder_path)

        # Transform uniquement (pas de fit)
        categorical_encoded = one_hot_encoder.transform(categorical_data)
        numeric_normalized = scaler.transform(numeric_data)



    # 6) Recréer un DataFrame pour les données catégorielles encodées
    #    On récupère les noms de features générés par le OHE.
    categorical_columns = one_hot_encoder.get_feature_names_out(categorical_cols)
    categorical_df = pd.DataFrame(
        categorical_encoded,
        columns=categorical_columns,
        index=df.index
    )

    # Idem pour les colonnes numériques
    numeric_df = pd.DataFrame(
        numeric_normalized,
        columns=numeric_cols,
        index=df.index
    )

    # 7) Concaténer le tout
    x = pd.concat([numeric_df, categorical_df], axis=1)

    # 8) Réintégrer la colonne 'label' si elle existe
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