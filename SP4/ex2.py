from SP4 import pcapLoader
import pandas as pd

def q2(csv_path: str):

    # get all the csv files
    set_of_ips = set()
    reader = pcapLoader.csv_to_reader(csv_path)
    for row in reader:
        set_of_ips.add(row['src_ip'])
        set_of_ips.add(row['dst_ip'])

        # Charger le fichier CSV dans un DataFrame
    df = pd.read_csv(csv_path)

    # Convertir les timestamps en format datetime
    df['first_seen'] = pd.to_datetime(df['bidirectional_first_seen_ms'], unit='ms')
    df['last_seen'] = pd.to_datetime(df['bidirectional_last_seen_ms'], unit='ms')

    # On utilise le timestamp de 'first_seen' comme point de départ pour l'intervalle
    df.set_index('first_seen', inplace=True)

    # construction de la structure de données de résultat
    # {ips:[
    #   ipi: {
    #       intervalles: [
    #       {nb_distinct_connected_ips: int, cumulated_payload: int}, # intervalle 1
    #       {nb_distinct_connected_ips: int, cumulated_payload: int}, # intervalle 2
    #       ...
    #       ]
    #   }
    # ]}
    ip_data = {}
    i = 0
    for ipi in set_of_ips:
        ip_data[ipi] = {
            "intervalles": []
        }
        borne_basse = df.index[0]
        while borne_basse < df.index[-1]:
            borne_haute = borne_basse + pd.Timedelta(minutes=5)
            interval_data = df[(df.index >= borne_basse) & (df.index < borne_haute)]
            connected_ips = set()
            payload_size = 0
            for index, row in interval_data.iterrows():
                # On vérifie si l'ip est connectée à ipi
                if row['src_ip'] == ipi or row['dst_ip'] == ipi:
                    connected_ips.add(row['src_ip'])
                    connected_ips.add(row['dst_ip'])
                    payload_size += int(row['bidirectional_bytes']) + int(row['src2dst_bytes']) + int(row['dst2src_bytes'])
            borne_basse = borne_haute

            ip_data[ipi]["intervalles"].append({
                "nb_distinct_connected_ips": len(connected_ips),
                "cumulated_payload": payload_size,
                "timestamp": borne_haute  # Enregistrer la borne supérieure comme timestamp
            })
        i += 1

    return ip_data