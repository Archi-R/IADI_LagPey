import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from joblib import dump


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


def train_rf(rf, vectorized_df, save_path, best_params=None, label_col='label'):
    """
    Entraîne une Random Forest sur les données vectorisées en utilisant une recherche d'hyperparamètres.
    - Utilise une validation croisée avec KFold.
    - Effectue une Grid Search pour trouver les meilleurs hyperparamètres.
    - Sauvegarde le modèle avec les meilleurs hyperparamètres.
    """

    # Paramètres pour la recherche d'hyperparamètres : initial ou raffiné
    param_grid = None
    if best_params is None:
        # Définir une recherche initiale complète pour le premier fichier
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
    else:
        param_grid = refine_param_grid(best_params)

    # Séparation des features et du label
    y = vectorized_df[label_col].values
    X = vectorized_df.drop(columns=[label_col])

    # Configuration de la validation croisée
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Si aucun modèle n'est fourni, entraîner un modèle par défaut
    if rf is None:
        rf = RandomForestClassifier()
        rf.fit(X, y)
        return rf, None, None

    # Grid Search avec validation croisée
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=kf, scoring='accuracy', n_jobs=-1, verbose=2)

    # Exécution de la recherche
    print("Recherche des meilleurs hyperparamètres...")
    grid_search.fit(X, y)

    # Meilleurs paramètres et score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    # print(f"Meilleurs hyperparamètres : {best_params}")
    # print(f"Score de validation croisée : {best_score}")

    # Sauvegarde du meilleur modèle
    best_model = grid_search.best_estimator_
    dump(best_model, save_path)
    print(f"Meilleur modèle sauvegardé à : {save_path}")

    return best_model, best_params, best_score


# Affiner autour des meilleurs paramètres pour les fichiers suivants
def refine_param_grid(best_params):
    """
    Crée une grille affinée basée sur les meilleurs paramètres.
    """
    refined_grid = {
        'n_estimators': [
            max(50, best_params.get('n_estimators', 100) - 50),
            best_params.get('n_estimators', 100),
            best_params.get('n_estimators', 100) + 50
        ],
        'max_depth': [
            None if best_params.get('max_depth') is None else max(1, best_params['max_depth'] - 10),
            best_params.get('max_depth', None),
            None if best_params.get('max_depth') is None else best_params['max_depth'] + 10
        ],
        'min_samples_split': [
            max(2, best_params.get('min_samples_split', 2) - 2),
            best_params.get('min_samples_split', 2),
            best_params.get('min_samples_split', 2) + 2
        ],
        'min_samples_leaf': [
            max(1, best_params.get('min_samples_leaf', 1) - 1),
            best_params.get('min_samples_leaf', 1),
            best_params.get('min_samples_leaf', 1) + 1
        ],
        'bootstrap': [best_params.get('bootstrap', True)]
    }
    return refined_grid