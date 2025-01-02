import pandas as pd
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier


def train(vectorized_df, label_col='label'):
    """
    Prépare les données pour la cross-validation selon la Q3.
    - le df est déjà vectorisé, et filtré pour chaque application
    - Split en train/test (80/20)
    - Met en place une stratified 5-fold cross-validation sur le train global
    """

    # Séparation des features et du label
    y = vectorized_df[label_col].values
    X = vectorized_df.drop(columns=[label_col])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # compter le nombre de 1 dans le train et le test
        print(f"Fold {i + 1} - Train: {sum(y_train)}")
        print(f"Fold {i + 1} - Test: {sum(y_test)}")

        knn = KNeighborsClassifier()

        knn.fit(X_train, y_train)

        # print accuracy, recall, precision
        print(f"Fold {i + 1} - Accuracy: {knn.score(X_test, y_test)}")
        print(f"Fold {i + 1} - Recall: {recall_score(y_test, knn.predict(X_test))}")
        print(f"Fold {i + 1} - Precision: {precision_score(y_test, knn.predict(X_test))}")
