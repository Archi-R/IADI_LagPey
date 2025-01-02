import os
import pandas as pd
from pandas.core.dtypes.cast import ensure_dtype_can_hold_na

from tools import *
from SP4.pcapLoader import *
from SP4.labeling import *
from SP4.vectorization import *
from labeling import label_flows
from vectorization import vectorize_flows
from cross_validation_setup import train


# pcap_folder = "../pcap_folder/dataset"

# # Creation des fichier csv
# pcap_to_csv(pcap_path= pcap_folder)
#
# # Ajout des fan in & fan out
# csv_folder = "../pcap_folder/csv"
# i = 0
# for cv_file in csv_folder:
#     i += 1
#     add_fan_features(cv_file)
#
#     # limiter la casse pour le test
#     if i >= 1:
#         break


################################################################
# Variables à ajuster
pcap_dir = "..\\pcap_folder\\dataset\\pcap"
csv_dir = "..\\pcap_folder\\dataset\\csv"
train_gt_path = "../pcap_folder/dataset/TRAIN.gt.csv"
time_window = 60  # pour fan_in/fan_out

LIMIT = 1  # Limite pour le nombre de fichiers à traiter

# Liste des pcap
if __name__ == '__main__':
    if(0):
        pcap_files = [f for f in os.listdir(pcap_dir) if f.endswith(".pcap")]

        print("1. Transformation des pcap en csv")
        csv_files = []
        i = 0
        for pcap in pcap_files:
            if i >= LIMIT: break
            pcap_path = os.path.join(pcap_dir, pcap)
            csv_path = pcap_to_csv(pcap_path)
            # csv_path = "..\\pcap_folder\\dataset\\csv\\trace_a_1.csv.temp"
            csv_path = csv_cleaner(csv_path)
            i += 1
        csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith(".csv")]

        print("2. Enrichissement avec fan_in/fan_out")
        # On peut éventuellement concaténer tous les CSV en un seul avant fan_in/out
        # Supposons qu'on fasse fan_in/out sur chaque CSV individuellement (adapter si besoin)
        enriched_csv_files = []
        for csv_file in csv_files:
            enriched_csv = add_fan_features(csv_file, time_window=time_window)
            enriched_csv_files.append(enriched_csv)

        print("2.5 Concaténer tous les CSV enrichis en un seul dataset global")

        all_data = pd.concat([pd.read_csv(f) for f in enriched_csv_files], ignore_index=True)
        global_csv = os.path.join(csv_dir,"all_data_with_fan.csv")
        all_data.to_csv(global_csv, index=False)


        print("3. Labeling des flux avec TRAIN.gt.csv")
        labeled_csv = label_flows(global_csv, train_gt_path)

    labeled_csv = "..\\pcap_folder\\dataset\\csv\\all_data_with_fan_labeled_fix10000.csv"

    # loading the labeled csv
    df = pd.read_csv(labeled_csv)

    print("4. Séparation en sous-ensembles")
    apps_sous_ensembles = ["HTTP", "IMAP", "DNS", "SMTP", "ICMP", "SSH", "FTP"]

    dict_sub_df = subset_divizor(df, apps_sous_ensembles, 'application_name')

    print("5. Vectorisation des flux")
    categorical_cols = ['protocol', 'src_ip', 'dst_ip']
    numeric_cols = [
        'bidirectional_packets', 'bidirectional_bytes', 'fan_in', 'fan_out',
        'bidirectional_duration_ms',
        'src_port', 'dst_port'
        #'src_to_dst_packets', 'src_to_dst_bytes', 'dst_to_src_packets', 'dst_to_src_bytes',
        #'src_to_dst_duration_ms', 'dst_to_src_duration_ms',
        # put other shits

    ]

    protocol_values = valeurs_uniques(labeled_csv, 'protocol')

    for app_name in dict_sub_df:
        print(f"5[{app_name}] Vectorisation du sous-ensemble")
        vectorized_df = vectorize_flows(dict_sub_df[app_name]
                                        , categorical_cols=categorical_cols
                                        , numeric_cols=numeric_cols
                                        , label_col='label'
                                        , protocol_list=protocol_values
                                        )

        # dict_sub_vectorized_df[app_name] = vectorized_df # ne sert pas à grand chose

        vectorized_csv_path = os.path.join(pcap_dir, f"vectorized_labeled_dataset_{app_name}.csv")
        vectorized_df.to_csv(vectorized_csv_path, index=False)

        print(f"6[{app_name}] Entrainement et évaluation")

        train(vectorized_df, label_col='label')

    print("Préparation terminée.")
        # print("X_train shape:", X_train.shape)
        # print("y_train shape:", y_train.shape)
        # print("X_test shape:", X_test.shape)
        # print("y_test shape:", y_test.shape)
        # print("Fold indices:", fold_indices)

    print("Fin du traitement.")
