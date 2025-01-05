import json
import os
from datetime import datetime
import sys

import joblib

sys.path.append('/home/logstudent/IADI_LagPey')

# import joblib
# import pandas as pd
# from pandas.core.dtypes.cast import ensure_dtype_can_hold_na
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier

from tools import *
from SP4.pcapLoader import *
from SP4.labeling import *
from SP4.vectorization import *
from labeling import label_flows
from vectorization import vectorize_flows
from cross_validation_setup import train_rf, evaluate_saved

pcap_dir = "../dataset_train/pcap"
csv_pur_dir = "../dataset_train/csv/1.pur"
csv_fan_dir = "../dataset_train/csv/2.fan"
csv_labeled_dir = "../dataset_train/csv/3.labeled"
csv_sep_protocol_dir = "../dataset_train/csv/4.sep_protocol"
csv_vectorized_dir = "../dataset_train/csv/5.vectorized"
train_gt_path = "../dataset_train/TRAIN.gt.csv"
unique_values_path = "../dataset_train/unique_values.json"
models_path = "../models/"




start_time = datetime.now()
last_etape_end_time = datetime.now()

# create or clear if exists the log file
log_file = open("log_file.csv", "w")
log_file.write("date,etape,fichier,erreur\n")


time_window = 60  # pour fan_in/fan_out

def pipeline(limit, skip_phase, stop_at_phase, is_test=False):

    global last_etape_end_time, pcap_dir, csv_pur_dir, csv_fan_dir, csv_labeled_dir, csv_sep_protocol_dir, csv_vectorized_dir, train_gt_path, unique_values_path, models_path

    if is_test:
        for v in [pcap_dir, csv_pur_dir, csv_fan_dir, csv_labeled_dir, csv_sep_protocol_dir, csv_vectorized_dir, train_gt_path, unique_values_path, models_path]:
            v.replace("dataset_train", "dataset_test")

    if skip_phase < 1 <= stop_at_phase:
        etape_1_transformation(limit)
        print("Temps étape : ", datetime.now() - last_etape_end_time)
        last_etape_end_time = datetime.now()

    if skip_phase < 2 <= stop_at_phase:
        etape_2_fan()
        print("Temps étape : ", datetime.now() - last_etape_end_time)
        print("Temps total : ", datetime.now() - start_time)
    if skip_phase < 3 <= stop_at_phase:
        etape_3_label()
        print("Temps étape : ", datetime.now() - last_etape_end_time)
        print("Temps total : ", datetime.now() - start_time)
    if skip_phase < 4 <= stop_at_phase:
        etape_4_separation()
        print("Temps étape : ", datetime.now() - last_etape_end_time)
        print("Temps total : ", datetime.now() - start_time)
    if skip_phase < 5 <= stop_at_phase:
        etape_5_vectorisation()
        print("Temps étape : ", datetime.now() - last_etape_end_time)
        print("Temps total : ", datetime.now() - start_time)
    if skip_phase < 6 <= stop_at_phase:
        etape_6_entrainement()
        print("Temps étape : ", datetime.now() - last_etape_end_time)
        print("Temps total : ", datetime.now() - start_time)

    log_file.close()
    print("Temps d'exécution : ", datetime.now() - start_time)


def nettoyeur(folder_path, log_file):
    """
    Nettoie les fichiers d'un dossier récursivement, sans supprimer le dossier lui-même
    supprime aussi les dossier enfants et leurs fichiers
    :param folder_path: chemin du dossier
    :return: None
    """
    return
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            try:
                os.remove(os.path.join(root, file))
            except Exception as e:
                date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                txt_error = f"{date},nettoyeur,{os.path.join(root, file)},{e}\n"
                print(txt_error)
                log_file.write(txt_error)
        for dir in dirs:
            try:
                nettoyeur(os.path.join(root, dir), log_file)
                os.rmdir(os.path.join(root, dir))
            except Exception as e:
                date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                txt_error = f"{date},nettoyeur,{os.path.join(root, dir)},{e}\n"
                print(txt_error)
                log_file.write(txt_error)

################################################################

# Variables à ajuster



def etape_1_transformation(limit):
    print("1. Transformation des pcap en csv")
    etape = 1
    pcap_files = [f for f in os.listdir(pcap_dir) if f.endswith(".pcap")]
    csv_files = []
    i = 0
    for pcap in pcap_files:
        if i >= limit: break
        try:
            pcap_path = os.path.join(pcap_dir, pcap)
            csv_path = pcap_to_csv(pcap_path, csv_pur_dir, cleaning=True)
            csv_files.append(csv_path)
        except Exception as e:
            print("exception")
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"{date},{etape},{pcap},{e}\n")
        i += 1

def etape_2_fan():
    print("2. Enrichissement avec fan_in/fan_out")
    etape = 2
    csv_files = [os.path.join(csv_pur_dir, f) for f in os.listdir(csv_pur_dir) if f.endswith(".csv")]
    enriched_csv_files = []
    for csv_file in csv_files:
        try:
            enriched_csv = add_fan_features(csv_file, csv_fan_dir, time_window=time_window)
            enriched_csv_files.append(enriched_csv)
        except Exception as e:
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            txt_error = f"{date},{etape},{csv_file},{e}\n"
            print(txt_error)
            log_file.write(txt_error)

    nettoyeur(csv_pur_dir, log_file)

def etape_3_label():
    print("3. Labeling des flux avec TRAIN.gt.csv")
    etape = 3
    enriched_csv_files = [os.path.join(csv_fan_dir, f) for f in os.listdir(csv_fan_dir) if f.endswith(".csv")]
    labeled_csvs = []
    for enriched_csv in enriched_csv_files:
        try:
            labeled_csv = label_flows(enriched_csv, csv_labeled_dir, train_gt_path)
            labeled_csvs.append(labeled_csv)
        except Exception as e:
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            txt_error = f"{date},{etape},{enriched_csv},{e}\n"
            print(txt_error)
            log_file.write(txt_error)

    nettoyeur(csv_fan_dir, log_file)

def etape_4_separation():
    """
        Sépare les fichiers CSV étiquetés en sous-ensembles basés sur le champ 'application_name',
        et stocke chaque sous-ensemble dans un sous-dossier nommé selon l'application.
        """
    print("4. Séparation en sous-ensembles")
    etape = 4
    labeled_csvs = [os.path.join(csv_labeled_dir, f) for f in os.listdir(csv_labeled_dir) if f.endswith(".csv")]
    apps_sous_ensembles = ["HTTP", "IMAP", "DNS", "SMTP", "ICMP", "SSH", "FTP"]
    separated_csvs = []

    # Vérifier s'il existe des fichiers étiquetés
    if not labeled_csvs:
        print(f"Aucun fichier trouvé dans {csv_labeled_dir}.")
        return

    for labeled_csv in labeled_csvs:
        try:
            df = pd.read_csv(labeled_csv)

            # Diviser en sous-ensembles par application_name
            dict_sub_df = subset_divizor(df, apps_sous_ensembles, 'application_name')

            filename = os.path.basename(labeled_csv).split(".")[0]

            for app_name, sub_df in dict_sub_df.items():
                try:
                    # Créer un sous-dossier pour l'application
                    app_dir = os.path.join(csv_sep_protocol_dir, app_name)
                    os.makedirs(app_dir, exist_ok=True)

                    # Sauvegarder le fichier dans le sous-dossier
                    sub_csv_path = os.path.join(app_dir, f"{filename}_{app_name}.csv")
                    sub_df.to_csv(sub_csv_path, index=False)
                    separated_csvs.append(sub_csv_path)

                except Exception as e:
                    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    txt_error = f"{date},{etape},{app_name},{e}\n"
                    print(txt_error)
                    log_file.write(txt_error)

        except Exception as e:
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            txt_error = f"{date},{etape},{labeled_csv},{e}\n"
            print(txt_error)
            log_file.write(txt_error)

    nettoyeur(csv_labeled_dir, log_file)

def etape_5_vectorisation():
    print("5. Vectorisation des flux")

    separated_csvs = []
    for root, _, files in os.walk(csv_sep_protocol_dir):
        for f in files:
            if f.endswith(".csv"):
                separated_csvs.append(os.path.join(root, f))

    etape = 5
    vectorized_csvs = []
    print ("5.0. transformation des IPs en classes")
    for file_path in separated_csvs:
        try:
            # remplacement direct dans le fichier
            df = pd.read_csv(file_path)
            df['src_ip'] = df['src_ip'].apply(ip_to_class)
            df['dst_ip'] = df['dst_ip'].apply(ip_to_class)
            df.to_csv(file_path, index=False)
        except Exception as e:
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            txt_error = f"{date},{etape},{file_path},{e}\n"
            print(txt_error)
            log_file.write(txt_error)

    categorical_cols = [
        'protocol',
        'src_ip',
        'dst_ip',
        'src_port',
        'dst_port',
        'application_name',
        'application_category_name',
        'requested_server_name'
    ]

    numeric_cols = [
        'bidirectional_packets',  # Paquets bidirectionnels
        'bidirectional_bytes',  # Octets bidirectionnels
        'fan_in',  # Nombre d'adresses connectées vers cette IP
        'fan_out',  # Nombre d'adresses connectées depuis cette IP
        'bidirectional_duration_ms',  # Durée bidirectionnelle
        'src2dst_duration_ms',  # Durée source -> destination
        'src2dst_packets',  # Paquets source -> destination
        'dst2src_packets',  # Paquets destination -> source
        'src2dst_bytes',  # Octets source -> destination
        'dst2src_bytes',  # Octets destination -> source
        'dst2src_duration_ms',  # Durée destination -> source
        'dst2src_packets',  # Paquets destination -> source
        'dst2src_bytes',  # Octets destination -> source
        'bidirectional_mean_ps',  # Moyenne de paquets par seconde bidirectionnels
        'bidirectional_max_ps',  # Maximum de paquets par seconde bidirectionnels
        'src2dst_mean_ps',  # Moyenne de paquets par seconde source -> destination
        'src2dst_max_ps',  # Maximum de paquets par seconde source -> destination
        'dst2src_mean_ps',  # Moyenne de paquets par seconde destination -> source
        'dst2src_max_ps',  # Maximum de paquets par seconde destination -> source
    ]
    print("5.1. valeurs pour chaque colonne")

    unique_values = {}
    # si le fichier existe, on le charge
    if os.path.exists(unique_values_path):
        with open(unique_values_path, 'r') as f:
            unique_values = json.load(f)
            # transformation des list en set pour le traitement
            for key in unique_values:
                unique_values[key] = set(unique_values[key])
    else:
        for file_path in separated_csvs:
            try:
                unique_values = valeurs_uniques(file_path, categorical_cols, unique_values)
                # enregistrement des valeurs uniques
                # transformation des set en list pour le json
                with open(unique_values_path, 'w') as f:
                    json.dump(unique_values, f, default=json_set_int_encoder)
            except Exception as e:
                # delete the file
                os.remove(unique_values_path)
                date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                txt_error = f"{date},{etape},{file_path},{e}\n"
                print(txt_error)
                log_file.write(txt_error)



    print("5.2. vectorisation")
    for file_path in separated_csvs:
        try:
            df = pd.read_csv(file_path)
            vectorized_df = vectorize_flows(df
                                            , categorical_cols=categorical_cols
                                            , numeric_cols=numeric_cols
                                            , label_col='label'
                                            , unique_values=unique_values
                                            )
            vectorized_csv_path = file_path.replace("4.sep_protocol", "5.vectorized")
            # si le dossier n'existe pas, on le crée
            if not os.path.exists(os.path.dirname(vectorized_csv_path)):
                os.makedirs(os.path.dirname(vectorized_csv_path))
            vectorized_df.to_csv(vectorized_csv_path, index=False)
            vectorized_csvs.append(vectorized_csv_path)

        except Exception as e:
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            txt_error = f"{date},{etape},{file_path},{e}\n"
            print(txt_error)
            log_file.write(txt_error)

    nettoyeur(csv_sep_protocol_dir, log_file)

def etape_6_entrainement():

    print("6. Entrainement et sauvegarde du modèle")
    etape = 6
    for app_name in ["HTTP", "IMAP", "DNS", "SMTP", "ICMP", "SSH", "FTP"]:
        try:
            save_path = os.path.join(models_path, app_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # dossier de l'app qu'on regarde
            vectorized_app_folder = os.path.join(csv_vectorized_dir, app_name)
            # liste des fichiers csv de l'app
            vectorized_csvs = [os.path.join(vectorized_app_folder, f) for f in os.listdir(vectorized_app_folder) if f.endswith(".csv")]
            # Concaténer les fichiers en un seul DataFrame
            # attention c'est gros comme Melissandre
            dataset = pd.concat([pd.read_csv(file) for file in vectorized_csvs])
            print(app_name)
            rf, best_params, best_score = train_rf(dataset, save_path)

            # enregistrement du modèle
            model_path = os.path.join(save_path, f"model_{app_name}.joblib")
            joblib.dump(rf, model_path)
            print("\n\n")

        except Exception as e:
            raise e
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            txt_error = f"{date},{etape},{app_name},{e}\n"
            print(txt_error)
            log_file.write(txt_error)

    nettoyeur(csv_vectorized_dir, log_file)

if __name__ == '__main__':

    pipeline(54, 4, 6, is_test=False)

    pipeline(54, 0, 5, is_test=True)
