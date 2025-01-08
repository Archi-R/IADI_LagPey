import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve
from joblib import dump, load
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def train(model, vectorized_df, label_col='label', save_path='trained_model.joblib'):
    """
    Prépare les données pour la cross-validation.
    Sauvegarde le modèle après l'entraînement.
    """
    # Séparation des features et du label
    y = vectorized_df[label_col].values
    X = vectorized_df.drop(columns=[label_col])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Entraîner le modèle
        model.fit(X_train, y_train)

    # Sauvegarde du modèle
    dump(model, save_path)

def train_knn(vectorized_df, save_path, label_col='label'):
    """
    Entraîne un classificateur k-NN sur les données vectorisées en utilisant une recherche d'hyperparamètres.
    - Utilise une validation croisée avec KFold.
    - Effectue une Grid Search pour trouver la meilleure valeur de k.
    - Sauvegarde le modèle avec les meilleurs hyperparamètres.
    """
    knn = KNeighborsClassifier()

    param_grid = {
        'n_neighbors': [9, 5],
        'weights': ['uniform'],
        'metric': ['euclidean']
    }

    if label_col not in vectorized_df.columns:
        raise KeyError(f"Column '{label_col}' not found in the dataset.")

    y = vectorized_df[label_col].values
    X = vectorized_df.drop(columns=[label_col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    split_save_path = os.path.join(save_path, 'train_test_split_knn.joblib')
    dump((X_train, X_test, y_train, y_test), split_save_path)
    print(f"Ensembles train/test sauvegardés à : {split_save_path}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid,
                               cv=kf, scoring='accuracy', n_jobs=2, verbose=0)

    try:
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        print(f"Meilleurs hyperparamètres k-NN : {best_params}")
        print(f"Score de validation croisée : {best_score:.4f}")

        best_model = grid_search.best_estimator_

        evaluate(best_model, X_test, y_test, save_path)

        return best_model, best_params, best_score

    except Exception as e:
        print(f"Erreur lors de l'entraînement k-NN : {e}")
        raise e

def train_naive_bayes(vectorized_df, save_path, label_col='label'):
    if label_col not in vectorized_df.columns:
        raise KeyError(f"Column '{label_col}' not found in the dataset.")

    y = vectorized_df[label_col].values
    X = vectorized_df.drop(columns=[label_col])

    if X.isnull().any().any():
        raise ValueError("Données manquantes détectées.")

    if (X < 0).any().any():
        # transformation des données pour éliminer les valeurs négatives
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Tailles des ensembles : X_train = {X_train.shape}, y_train = {y_train.shape}")

    dump((X_train, X_test, y_train, y_test), os.path.join(save_path, 'train_test_split_nb.joblib'))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {'alpha': [0.5, 1.0, 2.0]}
    grid_search = GridSearchCV(estimator=MultinomialNB(), param_grid=param_grid,
                               cv=kf, scoring='accuracy', n_jobs=2, verbose=0)

    grid_search.fit(X_train, y_train)  # Assurez-vous d'utiliser y_train ici

    best_model = grid_search.best_estimator_

    evaluate(best_model, X_test, y_test, save_path)
    print(f"Meilleurs hyperparamètres : {grid_search.best_params_}")
    return best_model, grid_search.best_params_, grid_search.best_score_



def train_rf(vectorized_df, save_path, label_col='label'):
    """
    Entraîne une Random Forest sur les données vectorisées en utilisant une recherche d'hyperparamètres.
    - Utilise une validation croisée avec KFold.
    - Effectue une Grid Search pour trouver les meilleurs hyperparamètres.
    - Sauvegarde le modèle avec les meilleurs hyperparamètres.
    - Sauvegarde les ensembles d'entraînement et de test.
    """

    # Initialisation du modèle Random Forest
    rf = RandomForestClassifier(random_state=42)

    # Paramètres pour la recherche d'hyperparamètres
    param_grid = {
        'n_estimators': [50, 100],  # Réduit le nombre d'arbres
        'max_depth': [10, 20],  # Réduit la profondeur maximale
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True]
    }

    # Vérifier la présence de la colonne label
    if label_col not in vectorized_df.columns:
        raise KeyError(f"Column '{label_col}' not found in the dataset.")

    # Séparation des features et des labels
    y = vectorized_df[label_col].values
    X = vectorized_df.drop(columns=[label_col])

    # Split en train/test et sauvegarde
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    split_save_path = os.path.join(save_path, 'train_test_split.joblib')
    dump((X_train, X_test, y_train, y_test), split_save_path)
    print(f"Ensembles train/test sauvegardés à : {split_save_path}")

    # Configuration de la validation croisée
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Grid Search avec validation croisée
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=kf, scoring='accuracy', n_jobs=2, verbose=0)

    try:
        # Exécution de la recherche
        grid_search.fit(X_train, y_train)

        # Meilleurs paramètres et score
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        print(f"Meilleurs hyperparamètres : {best_params}")
        print(f"Score de validation croisée : {best_score:.4f}")

        # Sauvegarde du meilleur modèle
        best_model = grid_search.best_estimator_

        # Évaluer le modèle
        try:
            evaluate(best_model, X_test, y_test, save_path)
        except Exception as e:
            print(f"Erreur lors de l'évaluation : {e}")

        return best_model, best_params, best_score

    except Exception as e:
        print(f"Erreur lors de l'entraînement : {e}")
        raise e

def evaluate(model, X_test, y_test, save_path):
    """
    Évalue le modèle sur plusieurs métriques et génère une courbe ROC.

    Args:
        model: Modèle entraîné.
        X_test: Données de test (features).
        y_test: Données de test (labels).
        save_path: Chemin où sauvegarder les résultats et la courbe ROC.
    """
    # Prédictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    # Calcul des métriques
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
    }

    # Calcul du ROC AUC si applicable
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)

        # Générer la courbe ROC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure()
        plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.4f})'.format(metrics['roc_auc']))
        plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        os.makedirs(os.path.join(save_path, 'perf'), exist_ok=True)
        roc_path = os.path.join(save_path, 'perf', 'roc.png')
        plt.savefig(roc_path)
        plt.close()
        print(f"Courbe ROC sauvegardée à : {roc_path}")

    # Sauvegarder les métriques dans un CSV
    os.makedirs(os.path.join(save_path, 'perf'), exist_ok=True)
    metrics_path = os.path.join(save_path, 'perf', 'perf.csv')
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Métriques sauvegardées à : {metrics_path}")

def evaluate_saved(path):
    """
    Charge les ensembles et le modèle sauvegardés, puis évalue le modèle.

    Args:
        path (str): Chemin vers le dossier contenant le modèle et les ensembles sauvegardés.
    """
    # Charger les ensembles train/test
    split_save_path = os.path.join(path, 'train_test_split.joblib')
    X_train, X_test, y_train, y_test = load(split_save_path)
    print(f"Ensembles train/test chargés depuis : {split_save_path}")

    # Charger le modèle nomme model_{app_name}.joblib
    model_path = os.path.join(path, f"model_{os.path.basename(path)}.joblib")
    model = load(model_path)
    print(f"Modèle chargé depuis : {model_path}")

    # Évaluer le modèle
    evaluate(model, X_test, y_test, path)