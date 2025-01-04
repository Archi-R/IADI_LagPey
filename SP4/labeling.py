import csv
import os

import pandas as pd

def load_ground_truth(gt_path: str)->tuple[list[dict], set[str], set[str]]:
    """
    Charge la ground truth dans une structure
    Format attendu: first_timestamp, last_timestamp, ip_src, ip_dst, port_src, port_dst, protocol
    :param gt_path:
    :return: list[dict] : la gt
    set[str] : les ip sources
    set[str] : les ip destinations
    """

    gt_data = []
    ip_sources = set()
    ip_destinations = set()
    with open(gt_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            gt_data.append({
                'start': float(row['first_timestamp_ms']),
                'end': float(row['last_timestamp_ms']),
                'src_ip': row['src_ip'],
                'dst_ip': row['dst_ip'],
                'src_port': row['src_port'],
                'dst_port': row['dst_port'],
                'protocol': row['protocol']
            })
            ip_sources.add(row['src_ip'])
            ip_destinations.add(row['dst_ip'])
    return gt_data, ip_sources, ip_destinations

def label_flows(csv_path: str, destination: str, gt_path: str) -> str:
    """
        Ajoute une colonne 'label' aux flows en fonction de la ground truth.
        """
    # Charger les données
    data = pd.read_csv(csv_path)
    gt_data = pd.read_csv(gt_path)

    # Filtrer les GT pertinents par IP
    gt_dict = {}
    for _, row in gt_data.iterrows():
        key = (row['src_ip'], row['dst_ip'])
        if key not in gt_dict:
            gt_dict[key] = []
        gt_dict[key].append(row)

    # Ajouter une colonne 'label' avec des valeurs par défaut à 0
    data['label'] = 0

    # Vectorisation avec pandas
    def check_match(row):
        key = (row['src_ip'], row['dst_ip'])
        if key in gt_dict:
            for gt in gt_dict[key]:
                if (row['src_port'] == gt['src_port'] and
                        row['dst_port'] == gt['dst_port'] and
                        row['protocol'] == gt['protocol'] and (
                        # Comparaison bidirectionnelle
                        (gt['first_timestamp_ms'] <= row['bidirectional_first_seen_ms'] <= gt['last_timestamp_ms'] or
                         gt['first_timestamp_ms'] <= row['bidirectional_last_seen_ms'] <= gt['last_timestamp_ms']) or
                        # Comparaison directionnelle source vers destination
                        (gt['first_timestamp_ms'] <= row['src2dst_first_seen_ms'] <= gt['last_timestamp_ms'] or
                         gt['first_timestamp_ms'] <= row['src2dst_last_seen_ms'] <= gt['last_timestamp_ms']) or
                        # Comparaison directionnelle destination vers source
                        (gt['first_timestamp_ms'] <= row['dst2src_first_seen_ms'] <= gt['last_timestamp_ms'] or
                         gt['first_timestamp_ms'] <= row['dst2src_last_seen_ms'] <= gt['last_timestamp_ms'])
                        )):
                    return 1
        return 0

    # Appliquer la fonction de matching
    data['label'] = data.apply(check_match, axis=1)

    # Sauvegarder le fichier avec les labels
    output_csv = os.path.join(destination, os.path.basename(csv_path))
    data.to_csv(output_csv, index=False)
    return output_csv