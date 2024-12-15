import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def knn_classification(csv_path: str, n_splits: int = 5, k_range: range = range(1, 21)):
    """
    Entraîne un modèle k-NN sur les flux réseau et effectue une validation croisée pour évaluer le meilleur k.

    Args:
        csv_path (str): Chemin vers le fichier CSV contenant les flux.
        n_splits (int): Nombre de splits pour la validation croisée.
        k_range (range): Plage de valeurs de k à tester pour k-NN.

    Returns:
        dict: Meilleure précision pour chaque valeur de k et le meilleur k.
    """
    # Charger les données
    data = pd.read_csv(csv_path)

    # Vérifier que les colonnes nécessaires sont présentes
    if 'application_name' not in data.columns or 'label' not in data.columns:
        raise ValueError("Les colonnes 'application_name' ou 'label' sont absentes du fichier CSV.")

    # Sélectionner les caractéristiques et l'étiquette
    features = data.drop(columns=['application_name', 'label'])
    labels = data['label']

    # Initialiser StratifiedKFold pour la validation croisée
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Dictionnaire pour stocker les scores pour chaque k
    scores = {}

    # Tester différentes valeurs de k
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)

        # Calculer la validation croisée (précision moyenne pour chaque k)
        cross_val_score_results = cross_val_score(knn, features, labels, cv=kf, scoring='accuracy')
        scores[k] = cross_val_score_results.mean()

        print(f"Précision pour k={k}: {scores[k]:.4f}")

    # Trouver le meilleur k (celui avec la meilleure précision moyenne)
    best_k = max(scores, key=scores.get)
    print(f"Meilleur k : {best_k} avec une précision moyenne de {scores[best_k]:.4f}")

    # Visualiser les performances pour chaque k
    plt.plot(k_range, list(scores.values()))
    plt.xlabel('Valeur de k')
    plt.ylabel('Précision moyenne')
    plt.title('Évaluation du modèle k-NN pour différentes valeurs de k')
    plt.show()

    return best_k, scores


csv_path = ""
best_k, scores = knn_classification(csv_path)
