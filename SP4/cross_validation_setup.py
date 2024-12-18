import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold


def prepare_cross_validation_data(csv_path: str, apps_list, label_col='label'):
    """
    Prépare les données pour la cross-validation selon la Q3.
    - Filtre par nom d'application
    - Split en train/test (80/20) stratifié pour chaque application
    - Concatène tous les sets de test pour un X_test, y_test global
    - Concatène tous les sets de train pour un X_train, y_train global
    - Met en place une stratified 5-fold cross-validation sur le train global
    """

    df = pd.read_csv(csv_path)
    # Assure-toi que la colonne application_name et le label_col existent
    # Assure-toi que df contient déjà des features vectorisées (X) et le label (y)

    # Séparation des features et du label
    y = df[label_col].values
    X = df.drop(columns=[label_col])

    # Stockage des parties train/test
    X_train_all = []
    y_train_all = []
    X_test_all = []
    y_test_all = []

    for app in apps_list:
        # Filtrer sur l'application
        subset = df[df['application_name'] == app]

        if subset.empty:
            continue

        y_sub = subset[label_col].values
        X_sub = subset.drop(columns=[label_col])

        # Split train/test 80/20 stratifié
        X_tr, X_te, y_tr, y_te = train_test_split(X_sub, y_sub, test_size=0.2,
                                                  stratify=y_sub, random_state=42)

        X_train_all.append(X_tr)
        y_train_all.append(y_tr)
        X_test_all.append(X_te)
        y_test_all.append(y_te)

    # Concaténer tous les sets
    X_train = pd.concat(X_train_all, ignore_index=True)
    y_train = pd.Series([item for arr in y_train_all for item in arr])

    X_test = pd.concat(X_test_all, ignore_index=True)
    y_test = pd.Series([item for arr in y_test_all for item in arr])

    # Mise en place d'une stratified 5-fold CV sur le train
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Exemple d'utilisation : obtention des indices
    # (plus tard, tu utiliseras ces indices pour entraîner et valider)
    fold_indices = []
    for train_index, val_index in skf.split(X_train, y_train):
        fold_indices.append((train_index, val_index))

    return X_train, y_train, X_test, y_test, fold_indices
