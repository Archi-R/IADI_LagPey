import numpy as np

def match_and_predict_flowfile(
    flow_file_path,          # "FLOW_FILE_1.csv"
    flows_folder,            # "dataset_test/2.fan/"
    output_file,             # ex: "FLOW_FILE_1_labeled.csv"
    categorical_cols,
    numeric_cols,
    train_vectorized_dir,    # dossier contenant scalers/encoders
    models_dir,              # dossier contenant les modèles
    app_names
):
    """
    Lis un flow_file (FLOW_FILE_1.csv), fait un merge avec
    les CSV du dossier 2.fan, applique la pipeline de séparation
    + vectorisation + prédiction, puis réinjecte les résultats
    (lab, proba_1) dans le flow_file initial.

    On suppose que:
      - La séparation (étape 4) se base sur "application_name".
      - On a un fallback "unknown" si la 'application_name' est manquante.
    """

    # 1) Lecture du flow_file (celui où 'lab' est parfois '?')
    print("[INFO] Lecture de", flow_file_path)
    flow_file_df = pd.read_csv(flow_file_path)

    # 2) Concaténer tous les CSV du dossier flows_folder
    #    On suppose qu'ils partagent le même format de colonnes.
    print("[INFO] Concaténation des CSV de", flows_folder)
    big_flows_df = pd.DataFrame()
    for csv_name in os.listdir(flows_folder):
        if csv_name.endswith(".csv"):
            temp_path = os.path.join(flows_folder, csv_name)
            temp_df = pd.read_csv(temp_path)
            big_flows_df = pd.concat([big_flows_df, temp_df], ignore_index=True)

    # 3) Merge (ou join) sur les colonnes clés
    #    => adaptables selon l'entête que tu utilises vraiment
    merge_keys = [
        "first_seen_ms", "last_seen_ms",
        "src_ip", "src_port", "dst_ip", "dst_port",
        "protocol"
    ]
    # On s'assure que ces colonnes existent dans les 2 DataFrames
    for mk in merge_keys:
        if mk not in flow_file_df.columns:
            flow_file_df[mk] = np.nan
        if mk not in big_flows_df.columns:
            big_flows_df[mk] = np.nan

    print("[INFO] Fusion des données (merge) sur les clefs :", merge_keys)
    merged_df = flow_file_df.merge(
        big_flows_df,
        how="left",    # ou "inner" si on veut uniquement les matches
        on=merge_keys
    )

    # 4) Gérer l'application_name manquante => "unknown"
    if "application_name" not in merged_df.columns:
        merged_df["application_name"] = "unknown"
    else:
        merged_df["application_name"] = merged_df["application_name"].fillna("unknown")

    # 5) Prédiction pour chaque application
    result_records = []

    # On ajoute la classe "unknown" pour tout flux non matché
    full_app_list = list(app_names) + ["unknown"]
    # Optionnel : on retire les doublons si "unknown" est déjà dans app_names
    full_app_list = list(set(full_app_list))

    for app_name in full_app_list:
        subset_app = merged_df[merged_df["application_name"] == app_name].copy()
        if subset_app.empty:
            continue

        # Indices
        subset_idx = subset_app.index

        # Fichier(s) model + scaler + encoder
        # => S'il n'y en a pas pour "unknown", on met par défaut
        app_dir = os.path.join(train_vectorized_dir, app_name)
        model_path = os.path.join(models_dir, app_name, f"model_{app_name}.joblib")
        scaler_path = os.path.join(app_dir, "scaler.joblib")
        encoder_path = os.path.join(app_dir, "ohe.joblib")

        if not os.path.exists(model_path):
            # pas de modèle => on met un label et proba par défaut
            for idx in subset_idx:
                # -1 => inconnu, 0.0 => proba nulle, ou comme tu veux
                result_records.append((idx, -1, 0.0))
            continue

        # 6) Vectorisation : is_test=True
        try:
            X = vectorize_flows(
                df=subset_app,
                categorical_cols=categorical_cols,
                numeric_cols=numeric_cols,
                label_col=None,
                scaler_path=scaler_path,
                one_hot_encoder_path=encoder_path,
                is_test=True
            )
        except Exception as vec_e:
            print("[ERROR] Vectorization failed for", app_name, ":", vec_e)
            # On met un label par défaut
            for idx in subset_idx:
                result_records.append((idx, -2, 0.0))
            continue

        # B0) On aligne X sur le modèle
        try:
            model = load(model_path)
            train_features = model.feature_names_in_
            # Ajouter colonnes manquantes
            missing_features = [col for col in train_features if col not in X.columns]
            for col in missing_features:
                X[col] = 0
            # Retirer colonnes en trop
            extra_features = [col for col in X.columns if col not in train_features]
            if extra_features:
                X.drop(columns=extra_features, inplace=True)
            # Réordonner
            X = X[train_features]

            # 7) Prédiction
            preds = model.predict(X)
            if len(model.classes_) == 1:
                # Modèle entraîné sur une seule classe
                # => predict_proba => shape (N,1)
                # Choix : forcer la proba = 0 ou 1
                # (Ajuste selon la classe = 0 ou 1)
                probas = np.zeros(len(X))
            else:
                probas = model.predict_proba(X)[:, 1]

            # 8) Stockage
            for row_idx, label_pred, proba_pred in zip(subset_idx, preds, probas):
                result_records.append((row_idx, int(label_pred), float(proba_pred)))

        except Exception as model_e:
            print("[ERROR] Prediction failed for", app_name, ":", model_e)
            for idx in subset_idx:
                result_records.append((idx, -3, 0.0))

    # 9) Rassembler tous les résultats dans l'ordre du merged_df
    results_df = pd.DataFrame(result_records, columns=["idx", "label_pred", "proba_1"])
    results_df.set_index("idx", inplace=True)

    merged_df = merged_df.join(results_df, how="left")

    # 10) Revenir au flow_file_df (même nombre de lignes, même ordre),
    #     on recopie label_pred et proba_1
    flow_file_df = flow_file_df.join(
        merged_df[["label_pred", "proba_1"]],
        how="left"
    )

    # On complète 'lab' uniquement si c'était '?'
    if "lab" not in flow_file_df.columns:
        flow_file_df["lab"] = flow_file_df["label_pred"]
    else:
        flow_file_df["lab"] = flow_file_df.apply(
            lambda row: row["label_pred"]
            if (str(row["lab"]) == "?" and not pd.isna(row["label_pred"]))
            else row["lab"],
            axis=1
        )

    # 11) Sauvegarde finale
    flow_file_df.rename(columns={"proba_1": "proba_suspicious"}, inplace=True)
    flow_file_df.to_csv(output_file, index=False)
    print(f"[DONE] Fichier de sortie généré : {output_file}")