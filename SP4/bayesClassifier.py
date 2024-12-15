import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt


def naive_bayes_classification(csv_path: str, n_splits: int = 5, alpha_range: range = [0.1, 1, 10]):
    """
    Entraîne un modèle Naive Bayes sur les flux réseau et effectue une validation croisée pour évaluer le meilleur alpha.

    Args:
        csv_path (str): Chemin vers le fichier CSV contenant les flux.
        n_splits (int): Nombre de splits pour la validation croisée.
        alpha_range (range): Plage des valeurs d'alpha (lissage de Laplace) à tester pour Naive Bayes.

    Returns:
        dict: Meilleure précision pour chaque valeur d'alpha et le meilleur alpha.
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

    # Dictionnaire pour stocker les scores pour chaque alpha
    scores = {}

    # Tester différentes valeurs de alpha
    for alpha in alpha_range:
        nb = MultinomialNB(alpha=alpha)

        # Calculer la validation croisée (précision moyenne pour chaque alpha)
        cross_val_score_results = cross_val_score(nb, features, labels, cv=kf, scoring='accuracy')
        scores[alpha] = cross_val_score_results.mean()

        print(f"Précision pour alpha={alpha}: {scores[alpha]:.4f}")

    # Trouver le meilleur alpha (celui avec la meilleure précision moyenne)
    best_alpha = max(scores, key=scores.get)
    print(f"Meilleur alpha : {best_alpha} avec une précision moyenne de {scores[best_alpha]:.4f}")

    # Visualiser les performances pour chaque alpha
    plt.plot(alpha_range, list(scores.values()))
    plt.xlabel('Valeur de alpha')
    plt.ylabel('Précision moyenne')
    plt.title('Évaluation du modèle Naive Bayes pour différentes valeurs de alpha')
    plt.show()

    return best_alpha, scores


# Exemple d'utilisation
csv_path = "flows_with_labels.csv"  # Remplacez par le chemin réel de votre fichier CSV
best_alpha, scores = naive_bayes_classification(csv_path)
