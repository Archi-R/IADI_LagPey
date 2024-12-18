import os
import pandas as pd
from pandas.core.dtypes.cast import ensure_dtype_can_hold_na

from tools import *
from SP4.pcapLoader import *
from SP4.labeling import *
from SP4.vectorization import *
from labeling import label_flows
from vectorization import vectorize_flows
from cross_validation_setup import prepare_cross_validation_data


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
train_gt_path = "..\\pcap_folder\\dataset\\TRAIN.gt"
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


        print("3. Labeling des flux avec TRAIN.gt")
        labeled_csv = label_flows(global_csv, train_gt_path)

    labeled_csv = "..\\pcap_folder\\dataset\\csv\\all_data_with_fan_labeled_fix10000.csv"

    print("4.Préparation pour la cross-validation ")
    # final_df = pd.DataFrame(X)
    # final_df['label'] = y
    # #final_dataset_csv = os.path.join(pcap_dir, "final_dataset.csv")
    # final_df.to_csv(labeled_csv, index=False)

    X_train, y_train, X_test, y_test, fold_indices = prepare_cross_validation_data(labeled_csv,
                                                                                   apps_list=["HTTP", "IMAP", "DNS", "SMTP", "ICMP", "SSH", "FTP"],
                                                                                   label_col='label')

    print("Préparation terminée.")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    print("Fold indices:", fold_indices)

    print("5. Vectorisation")




    # Ici, tu dois avoir déjà défini protocol_one_hot_vector, apps_one_hot_vector, ip_split,
    # id,expiration_id,src_ip,src_mac,src_oui,src_port,dst_ip,dst_mac,dst_oui,dst_port,
    # protocol,ip_version,vlan_id,tunnel_id,
    # bidirectional_first_seen_ms,bidirectional_last_seen_ms,bidirectional_duration_ms,bidirectional_packets,bidirectional_bytes,
    # src2dst_first_seen_ms,src2dst_last_seen_ms,src2dst_duration_ms,src2dst_packets,src2dst_bytes,dst2src_first_seen_ms,dst2src_last_seen_ms,dst2src_duration_ms,dst2src_packets,dst2src_bytes,bidirectional_min_ps,bidirectional_mean_ps,bidirectional_stddev_ps,bidirectional_max_ps,src2dst_min_ps,src2dst_mean_ps,src2dst_stddev_ps,src2dst_max_ps,dst2src_min_ps,dst2src_mean_ps,dst2src_stddev_ps,dst2src_max_ps,bidirectional_min_piat_ms,bidirectional_mean_piat_ms,bidirectional_stddev_piat_ms,bidirectional_max_piat_ms,src2dst_min_piat_ms,src2dst_mean_piat_ms,src2dst_stddev_piat_ms,src2dst_max_piat_ms,dst2src_min_piat_ms,dst2src_mean_piat_ms,dst2src_stddev_piat_ms,dst2src_max_piat_ms,bidirectional_syn_packets,bidirectional_cwr_packets,bidirectional_ece_packets,bidirectional_urg_packets,bidirectional_ack_packets,bidirectional_psh_packets,bidirectional_rst_packets,bidirectional_fin_packets,src2dst_syn_packets,src2dst_cwr_packets,src2dst_ece_packets,src2dst_urg_packets,src2dst_ack_packets,src2dst_psh_packets,src2dst_rst_packets,src2dst_fin_packets,dst2src_syn_packets,dst2src_cwr_packets,dst2src_ece_packets,dst2src_urg_packets,dst2src_ack_packets,dst2src_psh_packets,dst2src_rst_packets,dst2src_fin_packets,application_name,application_category_name,application_is_guessed,application_confidence,requested_server_name,client_fingerprint,server_fingerprint,user_agent,content_type,fan_out,fan_in,fan_out,fan_in

    # Recuperer toutes la valeurs de protocol possibles
    protocol_values = valeurs_uniques(labeled_csv, 'protocol')

    # Recuperer toutes la valeurs de application_name possibles
    application_name_values = valeurs_uniques(labeled_csv, 'application_name')

    categorical_cols = ['protocol', 'application_name', 'src_ip', 'dst_ip']
    numeric_cols = [
        'bidirectional_packets', 'bidirectional_bytes', 'fan_in', 'fan_out',
        'bidirectional_duration_ms', 'bidirectional_first_seen_ms',
        'bidirectional_last_seen_ms'
    ]

    # TODO : voir ce qon fait de ça :
    df_train = X_train.copy()
    df_train[label_col] = y_train

    df_test = X_test.copy()
    df_test[label_col] = y_test

    # TODO refactoriser vectorize_flows pour qu'il prenne en entrée un DataFrame, ou (x, y,...)
    X, y = vectorize_flows( df_test
                           , categorical_cols=categorical_cols
                           , numeric_cols=numeric_cols
                           , label_col='label'
                           , apps_list=application_name_values
                           , protocol_list=protocol_values
                           )

