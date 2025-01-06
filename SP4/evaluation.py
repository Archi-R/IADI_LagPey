import os
import numpy as np
import pandas as pd
from joblib import load


def evaluate_flows(test_csv_path, train_vectorized_dir, models_dir, app_names, output_file):
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
            results.append((0, 0.0))  # En cas d'erreur, on attribue '0' et une proba de 0

    # Sauvegarder les résultats dans un fichier CSV
    with open(output_file, 'w') as f:
        for label, proba in results:
            f.write(f"{label},{proba:.6f}\n")

    print(f"Résultats sauvegardés dans : {output_file}")
