import nfstream
import csv
import os
############
def pcap_to_csv(pcap_path:str):
    # read pcap file with nfstream
    # and write the flows to a CSV file
    # id,expiration_id,src_ip,src_mac,src_oui,src_port,dst_ip,dst_mac,dst_oui,dst_port,protocol,ip_version,vlan_id,tunnel_id,bidirectional_first_seen_ms,bidirectional_last_seen_ms,bidirectional_duration_ms,bidirectional_packets,bidirectional_bytes,src2dst_first_seen_ms,src2dst_last_seen_ms,src2dst_duration_ms,src2dst_packets,src2dst_bytes,dst2src_first_seen_ms,dst2src_last_seen_ms,dst2src_duration_ms,dst2src_packets,dst2src_bytes,application_name,application_category_name,application_is_guessed,application_confidence,requested_server_name,client_fingerprint,server_fingerprint,user_agent,content_type
    csv_name = pcap_path.replace("\\pcap\\","\\csv\\").replace(".pcap", ".csv.temp")
    nfstream.NFStreamer(
        source=pcap_path,
        decode_tunnels=True,
        idle_timeout=60,
        active_timeout=120,
        statistical_analysis=True
    ).to_csv(csv_name)
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


def add_fan_features(csv_path: str, time_window: int = 60) -> str:
    """
    Ajoute les colonnes fan-in et fan-out à un fichier CSV existant.

    Pour chaque ligne, parcourt les lignes environnantes dans une fenêtre de temps T,
    et compte les adresses IP distinctes pour calculer fan-in et fan-out.

    Args :
        csv_path (str): Chemin vers le fichier CSV existant.
        time_window (int): Taille de la fenêtre temporelle en secondes (par défaut 60).

    Returns :
        str : Chemin vers le fichier enrichi avec fan-in et fan-out.
    """
    output_csv = csv_path.replace(".csv", "_with_fan.csv")
    reader = csv_to_reader(csv_path)  # Charge le fichier CSV comme une liste de dictionnaires

    # Convertir le temps de la fenêtre en millisecondes
    time_window_ms = time_window * 1000

    # Parcourir chaque ligne pour calculer fan-in et fan-out
    for i, current_row in enumerate(reader):
        src_ip = current_row['src_ip']
        dst_ip = current_row['dst_ip']
        current_time = int(current_row['bidirectional_first_seen_ms'])

        fan_out_set = set()  # Stocke les IP cibles distinctes pour src_ip
        fan_in_set = set()  # Stocke les IP sources distinctes pour dst_ip

        # Parcourir les lignes environnantes dans la fenêtre temporelle
        j =0
        for neighbor_row in reader:
            j = j+1
            if neighbor_row['bidirectional_first_seen_ms'] is None:
                print(j)
            neighbor_time = int(neighbor_row['bidirectional_first_seen_ms'])

            # Vérifie si la ligne est dans la fenêtre temporelle
            if abs(current_time - neighbor_time) <= time_window_ms:
                # Ajoute les adresses IP correspondantes
                if neighbor_row['src_ip'] == src_ip:
                    fan_out_set.add(neighbor_row['dst_ip'])
                if neighbor_row['dst_ip'] == dst_ip:
                    fan_in_set.add(neighbor_row['src_ip'])

        # Calcul des tailles des ensembles (nombre d'adresses distinctes)
        current_row['fan_out'] = len(fan_out_set)
        current_row['fan_in'] = len(fan_in_set)

    # Écriture du fichier CSV enrichi
    try:
        with open(output_csv, 'w+', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(reader[0].keys()) + ['fan_out', 'fan_in'])
            writer.writeheader()
            writer.writerows(reader)
        print(f"Fichier enrichi avec fan-in et fan-out : {output_csv}")
    except Exception as e:
        print(f"Une erreur s'est produite lors de l'écriture du CSV : {e}")

    return output_csv




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

# quels_champs_sont_constants("C:\\Projets_GIT_C\\ENSIBS\\ia_detection\\IADI_LagPey\\pcap_folder\\dataset\\csv\\trace_a_10.csv")