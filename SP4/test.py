import json
import os
from datetime import datetime
import sys
from evaluation import evaluate_flows


import joblib

sys.path.append('/home/logstudent/IADI_LagPey')

from tools import *
from SP4.pcapLoader import *
from SP4.labeling import *
from SP4.vectorization import *
from labeling import label_flows
from vectorization import vectorize_flows
from cross_validation_setup import train_rf, train_naive_bayes, train_knn


def pipeline(limit, start_at_phase, stop_at_phase, is_test=False, model_type='rf'):
    # create or clear if exists the log file
    log_file = open("log_file.csv", "w")
    log_file.write("date,etape,fichier,erreur\n")

    time_window = 60

    pcap_dir = None
    csv_pur_dir = None
    csv_fan_dir = None
    csv_labeled_dir = None
    csv_sep_protocol_dir = None
    csv_vectorized_dir = None
    train_gt_path = "../dataset_train/TRAIN.gt.csv"
    models_path = "../models/"
    scaler_path = "../dataset_train/sacaler.joblib"
    onehotencoder_path = "../dataset_train/ohe.joblib"
    if is_test:
        pcap_dir = "../dataset_test/pcap"
        csv_pur_dir = "../dataset_test/csv/1.pur"
        csv_fan_dir = "../dataset_test/csv/2.fan"
        csv_labeled_dir = None
        csv_sep_protocol_dir = "../dataset_test/csv/4.sep_protocol"
        csv_vectorized_dir = "../dataset_test/csv/5.vectorized"
    else: # train
        pcap_dir = "../dataset_train/pcap"
        csv_pur_dir = "../dataset_train/csv/1.pur"
        csv_fan_dir = "../dataset_train/csv/2.fan"
        csv_labeled_dir = "../dataset_train/csv/3.labeled"
        csv_sep_protocol_dir = "../dataset_train/csv/4.sep_protocol"
        csv_vectorized_dir = "../dataset_train/csv/5.vectorized"

    start_time = datetime.now()
    last_etape_end_time = datetime.now()

    if start_at_phase <= 1 <= stop_at_phase:
        etape_1_transformation(limit, pcap_dir, csv_pur_dir)
        print("Temps étape : ", datetime.now() - last_etape_end_time)
        last_etape_end_time = datetime.now()

    if start_at_phase <= 2 <= stop_at_phase:
        etape_2_fan(csv_pur_dir, csv_fan_dir, time_window)
        print("Temps étape : ", datetime.now() - last_etape_end_time)
        print("Temps total : ", datetime.now() - start_time)
    if (start_at_phase <= 3 <= stop_at_phase) and not is_test:
        etape_3_label(csv_fan_dir, csv_labeled_dir, train_gt_path)
        print("Temps étape : ", datetime.now() - last_etape_end_time)
        print("Temps total : ", datetime.now() - start_time)
    if start_at_phase <= 4 <= stop_at_phase:
        if not is_test:
            etape_4_separation(csv_labeled_dir, csv_sep_protocol_dir, is_test)
        else:
            etape_4_separation(csv_fan_dir, csv_sep_protocol_dir, is_test)
        print("Temps étape : ", datetime.now() - last_etape_end_time)
        print("Temps total : ", datetime.now() - start_time)
    if start_at_phase <= 5 <= stop_at_phase:
        etape_5_vectorisation(csv_sep_protocol_dir, csv_vectorized_dir, is_test=is_test)
        print("Temps étape : ", datetime.now() - last_etape_end_time)
        print("Temps total : ", datetime.now() - start_time)
    if (start_at_phase <= 6 <= stop_at_phase) and not is_test:
        etape_6_entrainement(csv_vectorized_dir, models_path, model_type)
        print("Temps étape : ", datetime.now() - last_etape_end_time)
        print("Temps total : ", datetime.now() - start_time)
    if (start_at_phase <= 7 <= stop_at_phase) and not is_test:
        etape_7_entrainement(csv_vectorized_dir, models_path)
        print("Temps étape : ", datetime.now() - last_etape_end_time)
        print("Temps total : ", datetime.now() - start_time)

    log_file.close()
    print("Temps d'exécution : ", datetime.now() - start_time)


def etape_1_transformation(limit, pcap_dir, csv_pur_dir):
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
        i += 1


def etape_2_fan(csv_pur_dir, csv_fan_dir, time_window):
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


def etape_3_label(csv_fan_dir, csv_labeled_dir, train_gt_path):
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


def etape_4_separation(from_dir, csv_sep_protocol_dir, is_test=False):
    """
        Sépare les fichiers CSV étiquetés en sous-ensembles basés sur le champ 'application_name',
        et stocke chaque sous-ensemble dans un sous-dossier nommé selon l'application.
        :param
            from_dir: dossier contenant les fichiers CSV étiquetés OU les fichiers CSV à évaluer
            to_dir: dossier de destination où stocker les sous-ensembles
        """
    print("4. Séparation en sous-ensembles")
    etape = 4
    labeled_csvs = [os.path.join(from_dir, f) for f in os.listdir(from_dir) if f.endswith(".csv")]

    apps_sous_ensembles = get_app_list()
    separated_csvs = []

    # Vérifier s'il existe des fichiers étiquetés
    if not labeled_csvs:
        print(f"Aucun fichier trouvé dans {from_dir}.")
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


        except Exception as e:
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            txt_error = f"{date},{etape},{labeled_csv},{e}\n"
            print(txt_error)


def etape_5_vectorisation(csv_sep_protocol_dir, csv_vectorized_dir, is_test=False):
    print("5. Vectorisation des flux")

    etape = 5

    for app_name in get_app_list():
        try:
            separated_csvs = [os.path.join(csv_sep_protocol_dir, app_name, f) for f in
                              os.listdir(os.path.join(csv_sep_protocol_dir, app_name)) if f.endswith(".csv")]
            dataset = pd.concat([pd.read_csv(f, on_bad_lines='skip') for f in separated_csvs], ignore_index=True)

            if is_test: # test
                label_col = None
                trained_scaler_ohe_dir = "../dataset_train/csv/5.vectorized"

                scaler_path = os.path.join(trained_scaler_ohe_dir, app_name, "scaler.joblib")
                ohe_path = os.path.join(trained_scaler_ohe_dir, app_name, "ohe.joblib")


            else: # train
                label_col = 'label'
                scaler_path = os.path.join(csv_vectorized_dir, app_name, "scaler.joblib")
                ohe_path = os.path.join(csv_vectorized_dir, app_name, "ohe.joblib")

            vectorized_df = vectorize_flows(
                dataset,
                get_categorical_cols(),
                get_numeric_cols(),
                label_col=label_col,
                scaler_path=scaler_path,
                one_hot_encoder_path=ohe_path,
                is_test=is_test
            )



            # enregister le fichier vectorisé dans 5.vectorized/app_name/app_name_vectorized.csv
            vectorized_csv_path = os.path.join(csv_vectorized_dir, app_name, f"{app_name}_vectorized.csv")
            vectorized_df.to_csv(vectorized_csv_path, index=False)

        except Exception as e:
            raise e
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            txt_error = f"{date},{etape},{app_name},{e}\n"
            print(txt_error)


def etape_6_entrainement(csv_vectorized_dir, models_path, model_type):
    print("6. Entrainement et sauvegarde du modèle " + model_type)
    etape = 6

    for app_name in get_app_list():
        try:
            save_path = os.path.join(models_path,model_type, app_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            dataset = pd.read_csv(os.path.join(csv_vectorized_dir, app_name, f"{app_name}_vectorized.csv"))
            print(app_name)

            if(model_type == 'rf'):
                model, best_params, best_score = train_rf(dataset, save_path)
            elif(model_type == 'nb'):
                model, best_params, best_score = train_naive_bayes(dataset, save_path)
            elif(model_type == 'knn'):
                model, best_params, best_score = train_knn(dataset, save_path)

            # enregistrement du modèle
            model_path = os.path.join(save_path, f"model_{app_name}.joblib")
            # enregistrement des best_params en csv
            best_params_path = os.path.join(save_path, f"{model_type}best_params_{app_name}.csv")
            with open(best_params_path, 'w') as f:
                for key in best_params.keys():
                    f.write("%s,%s\n" % (key, best_params[key]))
            joblib.dump(model, model_path)
            print("\n\n")

        except Exception as e:
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            txt_error = f"{date},{etape},{app_name},{e}\n"
            print(txt_error)

def etape_7_entrainement(csv_vectorized_dir, models_path):
    print("7. Entrainement et sauvegarde de tout les modèles")
    models = ["rf", "nb", "knn"]
    for model in models:
        etape_6_entrainement(csv_vectorized_dir, models_path, model)

if __name__ == '__main__':
    pipeline(54, 6, 6, is_test=False, model_type='nb')

    #pipeline(28, 0, 2, is_test=True)

    # evaluate_flows("../dataset_test/csv/2.fan/trace_b_21.csv",
    #                "../dataset_train/csv/5.vectorized",
    #                "../models",
    #                get_app_list(),
    #                "../Challenge1_rendu.csv",
    #                get_categorical_cols(),
    #                get_numeric_cols()
    #               )

    # OLD_evaluate_flows(
    #     test_csv_path="../dataset_test/csv/2.fan/trace_b_21.csv",
    #     train_vectorized_dir="../dataset_train/csv/5.vectorized",
    #     models_dir="../models",
    #     app_names=["HTTP", "IMAP", "DNS", "SMTP", "ICMP", "SSH", "FTP"],
    #     output_file="../Challenge1_rendu.csv"
    # )
