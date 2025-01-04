import json
import os
from datetime import datetime

import joblib
import pandas as pd
from pandas.core.dtypes.cast import ensure_dtype_can_hold_na
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from tools import *
from SP4.pcapLoader import *
from SP4.labeling import *
from SP4.vectorization import *
from labeling import label_flows
from vectorization import vectorize_flows
from cross_validation_setup import train, train_rf

pcap_dir = "..\\pcap_folder\\dataset\\pcap"
csv_pur_dir = "..\\pcap_folder\\dataset\\csv\\1.pur"
csv_fan_dir = "..\\pcap_folder\\dataset\\csv\\2.fan"
csv_labeled_dir = "..\\pcap_folder\\dataset\\csv\\3.labeled"
csv_sep_protocol_dir = "..\\pcap_folder\\dataset\\csv\\4.sep_protocol"
csv_vectorized_dir = "..\\pcap_folder\\dataset\\csv\\5.vectorized"

start_time = datetime.now()
last_etape_end_time = datetime.now()

# create or clear if exists the log file
log_file = open("log_file.csv", "w")
log_file.write("date,etape,fichier,erreur\n")

train_gt_path = "../pcap_folder/dataset/TRAIN.gt.csv"
time_window = 60  # pour fan_in/fan_out

def pipeline(limit, skip_phase):

    global last_etape_end_time

    if skip_phase < 1:
        etape_1_transformation(limit)
        print("Temps étape : ", datetime.now() - last_etape_end_time)
        last_etape_end_time = datetime.now()

    if skip_phase < 2:
        etape_2_fan()
        print("Temps étape : ", datetime.now() - last_etape_end_time)
        print("Temps total : ", datetime.now() - start_time)
    if skip_phase < 3:
        etape_3_label()
        print("Temps étape : ", datetime.now() - last_etape_end_time)
        print("Temps total : ", datetime.now() - start_time)
    if skip_phase < 4:
        etape_4_separation()
        print("Temps étape : ", datetime.now() - last_etape_end_time)
        print("Temps total : ", datetime.now() - start_time)
    if skip_phase < 5:
        etape_5_vectorisation()
        print("Temps étape : ", datetime.now() - last_etape_end_time)
        print("Temps total : ", datetime.now() - start_time)
    if skip_phase < 6:
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
    print("4. Séparation en sous-ensembles")
    etape = 4
    labeled_csvs = [os.path.join(csv_labeled_dir, f) for f in os.listdir(csv_labeled_dir) if f.endswith(".csv")]
    apps_sous_ensembles = ["HTTP", "IMAP", "DNS", "SMTP", "ICMP", "SSH", "FTP"]
    separated_csvs = []
    if not labeled_csvs:
        labeled_csvs = [os.path.join(csv_labeled_dir, f) for f in os.listdir(csv_labeled_dir) if f.endswith(".csv")]
    for labeled_csv in labeled_csvs:
        try:
            df = pd.read_csv(labeled_csv)
            dict_sub_df = subset_divizor(df, apps_sous_ensembles, 'application_name')

            filename = os.path.basename(labeled_csv).split(".")[0]

            for app_name in dict_sub_df:
                try:
                    sub_df = dict_sub_df[app_name]
                    sub_csv_path = os.path.join(csv_sep_protocol_dir, f"{filename}_{app_name}.csv")
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

    separated_csvs = [os.path.join(csv_sep_protocol_dir, f) for f in os.listdir(csv_sep_protocol_dir) if f.endswith(".csv")]

    etape = 5
    vectorized_csvs = []
    categorical_cols = [
        'protocol',
        'src_ip',
        'dst_ip',
        'application_name',
        'application_category_name',
        'user_agent',
        'content_type',
    ]

    numeric_cols = [
        'bidirectional_packets',  # Paquets bidirectionnels
        'bidirectional_bytes',  # Octets bidirectionnels
        'fan_in',  # Nombre d'adresses connectées vers cette IP
        'fan_out',  # Nombre d'adresses connectées depuis cette IP
        'bidirectional_duration_ms',  # Durée bidirectionnelle
        'src2dst_duration_ms',  # Durée source -> destination
        'src2dst_packets',  # Paquets source -> destination
        'src2dst_bytes',  # Octets source -> destination
        'dst2src_duration_ms',  # Durée destination -> source
        'dst2src_packets',  # Paquets destination -> source
        'dst2src_bytes',  # Octets destination -> source
        'bidirectional_min_ps',  # Minimum de paquets par seconde bidirectionnels
        'bidirectional_mean_ps',  # Moyenne de paquets par seconde bidirectionnels
        'bidirectional_stddev_ps',  # Écart-type de paquets par seconde bidirectionnels
        'bidirectional_max_ps',  # Maximum de paquets par seconde bidirectionnels
        'src2dst_min_ps',  # Minimum de paquets par seconde source -> destination
        'src2dst_mean_ps',  # Moyenne de paquets par seconde source -> destination
        'src2dst_stddev_ps',  # Écart-type de paquets par seconde source -> destination
        'src2dst_max_ps',  # Maximum de paquets par seconde source -> destination
        'dst2src_min_ps',  # Minimum de paquets par seconde destination -> source
        'dst2src_mean_ps',  # Moyenne de paquets par seconde destination -> source
        'dst2src_stddev_ps',  # Écart-type de paquets par seconde destination -> source
        'dst2src_max_ps',  # Maximum de paquets par seconde destination -> source
        'bidirectional_syn_packets',  # Paquets SYN bidirectionnels
        'bidirectional_ack_packets',  # Paquets ACK bidirectionnels
        'bidirectional_psh_packets',  # Paquets PSH bidirectionnels
        'bidirectional_rst_packets',  # Paquets RST bidirectionnels
        'bidirectional_fin_packets',  # Paquets FIN bidirectionnels
    ]
    print("5.1. valeurs pour chaque colonne")
    unique_values_path = "..\\pcap_folder\\dataset\\unique_values.json"
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
                unique_values_path = "..\\pcap_folder\\dataset\\unique_values.json"
                # transformation des set en list pour le json
                with open(unique_values_path, 'w') as f:
                    json.dump(unique_values, f, default=json_set_int_encoder)
            except Exception as e:
                # delete the file
                os.remove(unique_values_path)
                raise e
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
    model_path = "..\\pcap_folder\\model.joblib"
    rf = None
    best_params = None
    vectorized_csvs = [os.path.join(csv_vectorized_dir, f) for f in os.listdir(csv_vectorized_dir) if f.endswith(".csv")]

    for file_path in vectorized_csvs:
        try:
            vectorized_df = pd.read_csv(file_path)
            rf, best_params, best_score = train_rf(rf, vectorized_df, model_path, best_params=best_params)
        except Exception as e:
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            txt_error = f"{date},{etape},{file_path},{e}\n"
            print(txt_error)
            log_file.write(txt_error)

    nettoyeur(csv_vectorized_dir, log_file)

if __name__ == '__main__':

    pipeline(15, 4)