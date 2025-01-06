

from pcapLoader import csv_to_reader
import pandas as pd
########### tool temporaire
def quels_champs_sont_constants(csv_path: str)->dict:
    # on charge le fichier
    reader = csv_to_reader(csv_path)
    # on récupère les noms des champs
    fields = reader[0].keys()
    # on recupère le nombre de lignes
    nb_lignes = len(reader)
    # on crée un dictionnaire pour stocker les valeurs des champs
    values = {}
    # on initialise le dictionnaire avec des ensembles vides
    for field in fields:
        values[field] = set()
    # on parcourt les lignes
    for row in reader:
        # on parcourt les champs
        for field in fields:
            # on ajoute la valeur du champ à l'ensemble correspondant
            values[field].add(row[field])

    constants = {}
    faibles = {}
    moyens = {}

    # on parcourt les champs
    for field in fields:
        if len(values[field]) == 1:
            constants[field] = values[field]
        elif len(values[field]) < 10:
            faibles[field] = values[field]
        elif len(values[field]) < 50:
            moyens[field] = values[field]

    print("Champs constants : ")
    for field in constants:
        print(f"\n\t{field}, valeurs : ")
        for value in constants[field]:
            print("\t\t"+value)
    print("Champs faibles : ")
    for field in faibles:
        print(f"\n\t{field}, valeurs : ")
        for value in faibles[field]:
            print("\t\t"+value)
    print("Champs moyens : ")
    for field in moyens:
        print(f"\n\t{field}, valeurs : ")
        for value in moyens[field]:
            print("\t\t"+value)

# quels_champs_sont_constants("C:\\Projets_GIT_C\\ENSIBS\\ia_detection\\IADI_LagPey\\pcap_folder\\dataset_train\\csv\\trace_a_10.csv")

def valeurs_uniques(csv_path: str, cols_to_look: list, unique_values: dict[str, set]) -> dict[str, set]:
    """
    Ajoute au dictionnaire d'entrée toutes les nouvelles valeurs uniques trouvées dans les colonnes spécifiées.

    Args:
        csv_path (str): Chemin vers le fichier CSV à analyser.
        cols_to_look (list): Liste des colonnes à examiner.
        unique_values (dict[str, set]): Dictionnaire contenant les ensembles de valeurs uniques déjà collectées.

    Returns:
        dict[str, set]: Dictionnaire mis à jour avec les nouvelles valeurs uniques trouvées.
    """
    # Charger le fichier CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Erreur lors du chargement du fichier CSV : {e}")

    # Initialiser les clés manquantes dans le dictionnaire
    for col in cols_to_look:
        if col not in unique_values:
            unique_values[col] = set()

    # Parcourir les colonnes spécifiées et mettre à jour les ensembles
    for col in cols_to_look:
        if col in df.columns:
            new_values = set(df[col].dropna().unique())  # Éliminer les doublons et les valeurs NaN
            unique_values[col].update(new_values)
        else:
            print(f"Attention : la colonne '{col}' n'existe pas dans le fichier {csv_path}.")

    return unique_values


def fix_ligne10000(csv_path: str)->str:
    import csv
    # réécrire tout le csv sauf la ligne 10000
    # créer un nouveau fichier
    new_csv_path = csv_path.replace(".csv", "_fix10000.csv")
    # on charge le fichier
    reader = csv_to_reader(csv_path)
    # parcour les lignes
    with open(new_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=reader[0].keys())
        writer.writeheader()
        for i in range(len(reader)):
            if i != 10000:
                writer.writerow(reader[i])
            else:
                print("Ligne 10000")
                print(reader[i])
    # fermer
    f.close()
    # retourner le nouveau fichier




    # # on charge le fichier
    # reader = csv_to_reader(csv_path)
    # # on récupère les noms des champs
    # #aller à la ligne 10000
    # fields:list = reader[0].keys()
    #
    # new_raw = reader[10000].copy()
    # i=0
    # for field in fields:
    #     if field not in ["id", "expiration_id", "fan_in", "fan_out", "label"]:
    #         new_raw[field] = reader[10000][field[i-2]
    #

    return new_csv_path

# fix_ligne10000("C:\\Projets_GIT_C\\ENSIBS\\ia_detection\\IADI_LagPey\\pcap_folder\\dataset_train\\csv\\all_data_with_fan_labeled.csv")

def subset_divizor(df, list_of_values, field):
    """
    Divise un dataframe en sous-dataframes selon les valeurs d'un champ
    :param df:
    :param list_of_values:
    :param field:
    :return: dict de sous-dataframes
    """
    dict_apps = {}
    for app in list_of_values:
        dict_apps[app] = df[(df[field] == app) | (df[field].str.contains(app))]

    return dict_apps


def json_set_int_encoder(obj):
    from numpy import int64

    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, int64):
        return int(obj)
    else:
        return obj


from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import os

def get_scaler(scaler_path: str)->StandardScaler:
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = StandardScaler()
        joblib.dump(scaler, scaler_path)

    return scaler

def get_onehotencoder(onehot_path: str)->StandardScaler:
    if os.path.exists(onehot_path):
        onehot = joblib.load(onehot_path)
    else:
        onehot = OneHotEncoder()
        joblib.dump(onehot, onehot_path)

    return onehot


def nettoyeur(folder_path, log_file):
    """
    Nettoie les fichiers d'un dossier récursivement, sans supprimer le dossier lui-même
    supprime aussi les dossier enfants et leurs fichiers
    :param folder_path: chemin du dossier
    :return: None
    """
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            try:
                os.remove(os.path.join(root, file))
            except Exception as e:
                date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                txt_error = f"{date},nettoyeur,{os.path.join(root, file)},{e}\n"
                print(txt_error)

        for dir in dirs:
            try:
                nettoyeur(os.path.join(root, dir), log_file)
                os.rmdir(os.path.join(root, dir))
            except Exception as e:
                date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                txt_error = f"{date},nettoyeur,{os.path.join(root, dir)},{e}\n"
                print(txt_error)