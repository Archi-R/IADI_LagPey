import csv

def load_ground_truth(gt_path: str)->list[dict]:
    """
    Charge la ground truth dans une structure
    Format attendu: first_timestamp, last_timestamp, ip_src, ip_dst, port_src, port_dst, protocol
    :param gt_path:
    :return: list[dict]
    """

    gt_data = []
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
    return gt_data

def label_flows(csv_path: str, gt_path: str) -> str:
    """
    Ajoute une colonne 'label' aux flows en fonction de la ground truth.
    Label = 1 si le flux recoupe un flux GT (même 5-tuple et plage temps qui se chevauchent), sinon 0.
    1 = attaque, 0 = normal
    """
    output_csv = csv_path.replace(".csv", "_labeled.csv")
    # Charger data et gt
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    gt_data = load_ground_truth(gt_path)

    # Création d'un index par 5-tuple pour accélérer la recherche
    # Clé : (src_ip, dst_ip, src_port, dst_port, protocol)
    gt_index = {}
    for g in gt_data:
        key = (g['src_ip'], g['dst_ip'], g['src_port'], g['dst_port'], g['protocol'])
        if key not in gt_index:
            gt_index[key] = []
        gt_index[key].append(g)

    # Associer les labels
    for row in data:
        key = (row['src_ip'], row['dst_ip'], row['src_port'], row['dst_port'], row['protocol'])
        row['label'] = 0
        if key in gt_index:
            flow_start = float(row['bidirectional_first_seen_ms'])/1000.0
            flow_end = float(row['bidirectional_last_seen_ms'])/1000.0

            # Vérifier si le flux intersecte une période de la ground truth
            for g in gt_index[key]:
                # Chevauchement temporel ?
                if not (flow_end < g['start'] or flow_start > g['end']):
                    row['label'] = 1
                    break

    # Écriture du CSV labellisé
    fieldnames = list(data[0].keys())
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"Fichier labellisé: {output_csv}")
    return output_csv