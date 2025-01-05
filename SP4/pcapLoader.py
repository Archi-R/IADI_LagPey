import nfstream
import csv
import os
import pandas as pd
############
def pcap_to_csv(pcap_path:str, dest_folder:str, cleaning=False)->str:
    # read pcap file with nfstream
    # and write the flows to a CSV file
    # id,expiration_id,src_ip,src_mac,src_oui,src_port,dst_ip,dst_mac,dst_oui,dst_port,protocol,ip_version,vlan_id,tunnel_id,bidirectional_first_seen_ms,bidirectional_last_seen_ms,bidirectional_duration_ms,bidirectional_packets,bidirectional_bytes,src2dst_first_seen_ms,src2dst_last_seen_ms,src2dst_duration_ms,src2dst_packets,src2dst_bytes,dst2src_first_seen_ms,dst2src_last_seen_ms,dst2src_duration_ms,dst2src_packets,dst2src_bytes,application_name,application_category_name,application_is_guessed,application_confidence,requested_server_name,client_fingerprint,server_fingerprint,user_agent,content_type
    csv_name = os.path.join(dest_folder, os.path.basename(pcap_path).replace(".pcap", ".csv.temp"))
    nfstream.NFStreamer(
        source=pcap_path,
        decode_tunnels=True,
        idle_timeout=60,
        active_timeout=120,
        statistical_analysis=True
    ).to_csv(csv_name)
    if cleaning:
        csv_name = csv_cleaner(csv_name)
    else:
        csv_name = csv_name.replace(".temp", "")
    # suppression du fichier temporaire
    return csv_name


def csv_cleaner(csv_path_in: str):
    """
    Nettoie un fichier CSV en supprimant les lign
    """
    csv_path_out = csv_path_in.replace(".temp", "")
    reader = csv_to_reader(csv_path_in)
    with open(csv_path_out, 'w+', encoding='utf-8', newline='') as outfile:

        writer = csv.writer(outfile)

        list_of_fields = reader[0].keys()
        writer.writerow(list_of_fields)

        r = 0
        for row in reader:
            r += 1
            row_has_error = False
            for field in list_of_fields:
                #### RÉPARATION DE LIGNE INCOMPLETE
                if row[field] == '':
                    row[field] = '0'

                # si ce n'est pas un nombre
                # elif not row[field].isnumeric():
                #     # si il manque un guillemet à la fin
                #     if row[field][-1] != "'":
                #         row[field] = row[field] + "'"
                #     # si il manque un guillemet au début
                #     if row[field][0] != "'":
                #         row[field] = "'" + row[field]


                ### VERIFICATION DU NOMBRE DE CHAMPS NONE
                if row[field] is None:
                    none_fields = []
                    for _field in list_of_fields:
                        if row[_field] is None:
                            none_fields.append(_field)

                    if len(none_fields) >= len(row)/3: # si plus d'un tiers des champs sont vides
                        # on considère que la ligne est pourrie
                        row_has_error = True
                        # on arrête de chercher pour sortir plus vite
                        break # break de : "for field in list_of_fields:"
                    else:
                        # on remplace les champs vides par des 0, car la ligne présente juste quelques valeurs vides
                        for none_field in none_fields:
                            row[none_field] = '0'

            if not row_has_error:
                writer.writerow([row[field] for field in list_of_fields])
    # close the files
    outfile.close()
    # remove the temp file
    # os.remove(csv_path_in)
    return csv_path_out

def csv_to_reader(csv_path: str)->list:
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = list(csv.DictReader(f))
            return reader
    except UnicodeDecodeError:
        # If UTF-8 encoding fails, try with 'latin-1'
        print("------   Erreur d'encodage UTF-8, essai avec latin-1  ------")
        with open(csv_path, 'r', encoding='latin-1') as f:
            reader = list(csv.DictReader(f))
            return reader
    except Exception as e:
        print(f"Une erreur s'est produite lors de la lecture du CSV : {e}")
        return []


def get_values(reader: list, key: str)->list:
    values = []
    for row in reader:
        values.append(row[key])
    return values




def add_fan_features(csv_path: str, destination,time_window: int = 60) -> str:
    """
    Ajoute les colonnes fan-in et fan-out à un fichier CSV existant.

    Args :
        csv_path (str): Chemin vers le fichier CSV existant.
        time_window (int): Taille de la fenêtre temporelle en secondes (par défaut 60).

    Returns :
        str : Chemin vers le fichier enrichi avec fan-in et fan-out.
    """
    output_csv = os.path.join(destination, os.path.basename(csv_path))

    # Charger le fichier CSV en DataFrame
    df = pd.read_csv(csv_path)

    # Convertir le temps de la fenêtre en millisecondes
    time_window_ms = time_window * 1000

    # Trier les données par temps pour faciliter le traitement
    df = df.sort_values(by='bidirectional_first_seen_ms')

    # Créer des colonnes pour fan-out et fan-in
    df['fan_out'] = 0
    df['fan_in'] = 0

    # Fenêtre glissante pour calculer les métriques
    for idx, row in df.iterrows():
        current_time = row['bidirectional_first_seen_ms']
        src_ip = row['src_ip']
        dst_ip = row['dst_ip']

        # Filtrer les lignes dans la fenêtre temporelle
        time_window_df = df[
            (df['bidirectional_first_seen_ms'] >= current_time - time_window_ms) &
            (df['bidirectional_first_seen_ms'] <= current_time + time_window_ms)
        ]

        # Calculer fan-out et fan-in
        fan_out_set = time_window_df[time_window_df['src_ip'] == src_ip]['dst_ip'].unique()
        fan_in_set = time_window_df[time_window_df['dst_ip'] == dst_ip]['src_ip'].unique()

        df.at[idx, 'fan_out'] = len(fan_out_set)
        df.at[idx, 'fan_in'] = len(fan_in_set)

    # Sauvegarder le fichier enrichi
    df.to_csv(output_csv, index=False)

    return output_csv
