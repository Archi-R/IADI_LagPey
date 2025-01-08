import os
import numpy as np
import pandas as pd
from joblib import load
from vectorization import vectorize_flows

def evaluate_flows(test_csv_path,
                   train_vectorized_dir,
                   models_dir,
                   app_names,
                   output_file,
                   categorical_cols,
                   numeric_cols):
    """
    Évalue chaque flux d'un fichier CSV de test,
    applique la vectorisation et le modèle correspondant,
    puis génère un fichier de sortie (label, proba) dans l'ordre original.

    Args:
        test_csv_path (str): Chemin vers le fichier CSV contenant les flux à évaluer.
        train_vectorized_dir (str): Dossier contenant les scalers et one-hot encoders pour chaque application.
        models_dir (str): Dossier contenant les modèles entraînés pour chaque application.
        app_names (list): Liste des noms d'applications acceptés.
        output_file (str): Chemin pour le fichier de sortie contenant les prédictions.
        vectorize_func (callable): Fonction de vectorisation (ex. `vectorize_flows_test`).
        categorical_cols (list): Colonnes catégorielles telles qu'au moment du train.
        numeric_cols (list): Colonnes numériques telles qu'au moment du train.
    """

    # 1) Charger le fichier de test entier
    test_df = pd.read_csv(test_csv_path, on_bad_lines='skip')

    # DataFrame final qui contiendra : index original, label prédit, proba
    result_records = []

    # 2) Pour chaque application connue, on filtre
    for app_name in app_names:
        # Sous-ensemble pour cette application
        subset_df = test_df[test_df["application_name"] == app_name].copy()
        if subset_df.empty:
            continue  # Aucune ligne pour cette application

        # 3) Charger les objets (scaler, OHE, modèle) entraînés pour cette application
        app_vectorized_dir = os.path.join(train_vectorized_dir, app_name)
        scaler_path = os.path.join(app_vectorized_dir, 'scaler.joblib')
        encoder_path = os.path.join(app_vectorized_dir, 'ohe.joblib')
        model_path = os.path.join(models_dir, app_name, f"model_{app_name}.joblib")

        try:
            model = load(model_path)
            # 4) Vectoriser l’ensemble des lignes du subset pour cette application
            X = vectorize_flows(
                df=subset_df,
                categorical_cols=categorical_cols,
                numeric_cols=numeric_cols,
                label_col=None,  # On n'a pas besoin de label ici
                scaler_path=scaler_path,
                one_hot_encoder_path=encoder_path,
                is_test = True
            )

            # B0 bricolage, jpp il est 1h du mat
            missing_features = {"dst2src_bytes.1", "dst2src_packets.1"}
            for col in missing_features:
                if col not in X.columns:
                    X[col] = 0  # on crée la colonne en la mettant à 0
            train_features = model.feature_names_in_
            # B1) Ajouter les colonnes manquantes (celles que le modèle a vu au fit mais qui sont absentes dans X)
            missing_features = [col for col in train_features if col not in X.columns]
            for col in missing_features:
                X[col] = 0  # on met du 0 par défaut

            # B2) Supprimer les colonnes en trop (celles qui n’étaient pas connues du modèle)
            extra_features = [col for col in X.columns if col not in train_features]
            if extra_features:
                X.drop(columns=extra_features, inplace=True)

            # B3) Réordonner X pour qu’il soit dans le même ordre que train_features
            X = X[train_features]

            # 5) Obtenir les prédictions
            preds = model.predict(X)
            if len(model.classes_) == 1:
                probas = np.zeros(len(X))  # ou np.ones(len(X)) selon la classe
            else:
                # Cas normal : deux classes
                probas = model.predict_proba(X)[:, 1]

            # 6) Stoker les résultats
            #    On sauvegarde (index original, label prédit, proba)
            for original_idx, label_pred, proba_pred in zip(subset_df.index, preds, probas):
                result_records.append((original_idx, int(label_pred), float(proba_pred)))

        except Exception as e:
            raise e
            print(f"Erreur lors du traitement de l'application '{app_name}': {e}")
            # Optionnellement, on pourrait ignorer ou lever une exception ici
            # On peut aussi affecter des prédictions par défaut
            # pass

    # 7) Remettre le tout dans l'ordre original du CSV
    #    On trie sur l’index pour retrouver l’ordre d’apparition dans le test_df.
    result_df = pd.DataFrame(result_records, columns=["original_idx", "label", "proba"])
    result_df.sort_values(by="original_idx", inplace=True)

    # 8) Sauvegarder dans un CSV final (seulement label et proba, ou plus si besoin)
    result_df[["label", "proba"]].to_csv(output_file, index=False)
    print(f"Résultats sauvegardés dans : {output_file}")





def OLD_evaluate_flows(test_csv_path, train_vectorized_dir, models_dir, app_names, output_file):
    """
    Évalue chaque flux d'un fichier CSV de test, le vectorise, applique le modèle correspondant et génère un fichier de sortie.

    Args:
        test_csv_path (str): Chemin vers le fichier CSV contenant les flux à évaluer.
        train_vectorized_dir (str): Dossier contenant les scalers et one-hot encoders pour chaque application.
        models_dir (str): Dossier contenant les modèles entraînés pour chaque application.
        app_names (list): Liste des noms d'applications acceptés.
        output_file (str): Chemin pour le fichier de sortie contenant les prédictions.
    """
    # Charger le fichier de test
    test_df = pd.read_csv(test_csv_path)

    # Liste pour stocker les résultats finaux
    results = []

    for _, row in test_df.iterrows():
        app_name = row.get('application_name')  # Récupérer l'application via 'application_name'

        if app_name not in app_names:
            continue  # Ignorer les flux hors des app_names

        try:
            # Charger les objets pour l'application
            app_vectorized_dir = os.path.join(train_vectorized_dir, app_name)
            scaler_path = os.path.join(app_vectorized_dir, 'scaler.joblib')
            encoder_path = os.path.join(app_vectorized_dir, 'ohe.joblib')
            model_path = os.path.join(models_dir, app_name, f"model_{app_name}.joblib")

            scaler = load(scaler_path)
            one_hot_encoder = load(encoder_path)
            model = load(model_path)

            # Préparer la ligne pour vectorisation
            test_row = pd.DataFrame([row])  # Convertir la ligne en DataFrame
            train_features = list(one_hot_encoder.get_feature_names_out()) + list(scaler.feature_names_in_)

            # Vectorisation des colonnes catégoriques
            categorical_cols = one_hot_encoder.get_feature_names_out()
            missing_categorical_cols = [col for col in categorical_cols if col not in test_row.columns]
            for col in missing_categorical_cols:
                test_row[col] = 0  # Ajouter les colonnes manquantes
            categorical_data = one_hot_encoder.transform(test_row[categorical_cols])

            # Vectorisation des colonnes numériques
            numeric_cols = scaler.feature_names_in_
            missing_numeric_cols = [col for col in numeric_cols if col not in test_row.columns]
            for col in missing_numeric_cols:
                test_row[col] = 0  # Ajouter les colonnes manquantes
            numeric_data = scaler.transform(test_row[numeric_cols])

            # Concaténation des données vectorisées
            vectorized_row = pd.DataFrame(
                data=np.hstack([numeric_data, categorical_data]),
                columns=numeric_cols + list(categorical_cols)
            )

            # Aligner les colonnes sur les features du modèle entraîné
            missing_features = [col for col in train_features if col not in vectorized_row.columns]
            for col in missing_features:
                vectorized_row[col] = 0  # Ajouter les colonnes manquantes
            extra_features = [col for col in vectorized_row.columns if col not in train_features]
            vectorized_row = vectorized_row.drop(columns=extra_features)

            # Prédire avec le modèle
            prediction = model.predict(vectorized_row)
            proba = model.predict_proba(vectorized_row)[:, 1]  # Probabilité pour la classe 1

            # Ajouter le résultat
            results.append((int(prediction[0]), float(proba[0])))

        except Exception as e:
            print(f"Erreur avec {app_name} : {e}")
            raise e
            results.append((0, 0.0))  # En cas d'erreur, on attribue '0' et une proba de 0

    # Sauvegarder les résultats dans un fichier CSV
    with open(output_file, 'w') as f:
        for label, proba in results:
            f.write(f"{label},{proba:.6f}\n")

    print(f"Résultats sauvegardés dans : {output_file}")
