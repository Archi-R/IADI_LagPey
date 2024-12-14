import os
from dotenv import load_dotenv
import pandas as pd
from elasticsearch import Elasticsearch, helpers
from elasticsearch.helpers import scan

def create_or_get_es():
    # Initialiser la connexion à Elasticsearch
    load_dotenv()
    return Elasticsearch(
        "https://localhost:9200",
        basic_auth=("elastic", os.getenv("ELASTIC_PASSWORD")),
        verify_certs=False
    )

def clean_data(document):
    for key in ['bidirectional_bytes', 'src2dst_bytes', 'dst2src_bytes', 'bidirectional_packets', 'src2dst_packets', 'dst2src_packets']:
        if key in document and (document[key] is None or pd.isna(document[key])):
            document[key] = 0  # Remplacer NaN par 0

    # Convertir les valeurs 0 et 1 en booléens pour application_is_guessed
    if 'application_is_guessed' in document:
        # Si la valeur est 1, c'est True, sinon c'est False
        document['application_is_guessed'] = document['application_is_guessed'] == 1

    return document

def create_or_get_index(index_name):
    client = create_or_get_es()
    if not client.indices.exists(index=index_name):
        index_body = {
            "mappings": {
                "properties": {
                    "id": {"type": "integer"},
                    "expiration_id": {"type": "integer"},
                    "src_ip": {"type": "ip"},
                    "src_mac": {"type": "keyword"},
                    "src_oui": {"type": "keyword"},
                    "src_port": {"type": "integer"},
                    "dst_ip": {"type": "ip"},
                    "dst_mac": {"type": "keyword"},
                    "dst_oui": {"type": "keyword"},
                    "dst_port": {"type": "integer"},
                    "protocol": {"type": "integer"},
                    "ip_version": {"type": "integer"},
                    "vlan_id": {"type": "integer"},
                    "tunnel_id": {"type": "integer"},
                    "bidirectional_first_seen_ms": {"type": "date", "format": "epoch_millis"},
                    "bidirectional_last_seen_ms": {"type": "date", "format": "epoch_millis"},
                    "bidirectional_duration_ms": {"type": "long"},
                    "bidirectional_packets": {"type": "integer"},
                    "bidirectional_bytes": {"type": "long"},
                    "src2dst_first_seen_ms": {"type": "date", "format": "epoch_millis"},
                    "src2dst_last_seen_ms": {"type": "date", "format": "epoch_millis"},
                    "src2dst_duration_ms": {"type": "long"},
                    "src2dst_packets": {"type": "integer"},
                    "src2dst_bytes": {"type": "long"},
                    "dst2src_first_seen_ms": {"type": "date", "format": "epoch_millis"},
                    "dst2src_last_seen_ms": {"type": "date", "format": "epoch_millis"},
                    "dst2src_duration_ms": {"type": "long"},
                    "dst2src_packets": {"type": "integer"},
                    "dst2src_bytes": {"type": "long"},
                    "application_name": {"type": "keyword"},
                    "application_category_name": {"type": "keyword"},
                    "application_is_guessed": {"type": "boolean"},
                    "application_confidence": {"type": "integer"},
                    "requested_server_name": {"type": "keyword"},
                    "client_fingerprint": {"type": "keyword"},
                    "server_fingerprint": {"type": "keyword"},
                    "user_agent": {"type": "keyword"},
                    "content_type": {"type": "keyword"}
                }
            }
        }
        client.indices.create(index=index_name, body=index_body)

    return client.indices.get(index=index_name)


# Prepare a generator to yield documents in the bulk API format
def generate_data(df, index_name):
    for _, row in df.iterrows():
        document = {
            "_index": index_name,
            "_source": {
                "id": row['id'],
                "expiration_id": row['expiration_id'],
                "src_ip": row['src_ip'],
                "src_mac": row['src_mac'],
                "src_oui": row['src_oui'],
                "src_port": row['src_port'],
                "dst_ip": row['dst_ip'],
                "dst_mac": row['dst_mac'],
                "dst_oui": row['dst_oui'],
                "dst_port": row['dst_port'],
                "protocol": row['protocol'],
                "ip_version": row['ip_version'],
                "vlan_id": row['vlan_id'],
                "tunnel_id": row['tunnel_id'],
                "bidirectional_first_seen_ms": row['bidirectional_first_seen_ms'],
                "bidirectional_last_seen_ms": row['bidirectional_last_seen_ms'],
                "bidirectional_duration_ms": row['bidirectional_duration_ms'],
                "bidirectional_packets": row['bidirectional_packets'],
                "bidirectional_bytes": row['bidirectional_bytes'],
                "src2dst_first_seen_ms": row['src2dst_first_seen_ms'],
                "src2dst_last_seen_ms": row['src2dst_last_seen_ms'],
                "src2dst_duration_ms": row['src2dst_duration_ms'],
                "src2dst_packets": row['src2dst_packets'],
                "src2dst_bytes": row['src2dst_bytes'],
                "dst2src_first_seen_ms": row['dst2src_first_seen_ms'],
                "dst2src_last_seen_ms": row['dst2src_last_seen_ms'],
                "dst2src_duration_ms": row['dst2src_duration_ms'],
                "dst2src_packets": row['dst2src_packets'],
                "dst2src_bytes": row['dst2src_bytes'],
                "application_name": row['application_name'],
                "application_category_name": row['application_category_name'],
                "application_is_guessed": row['application_is_guessed'],
                "application_confidence": row['application_confidence'],
                "requested_server_name": row['requested_server_name'],
                "client_fingerprint": row['client_fingerprint'],
                "server_fingerprint": row['server_fingerprint'],
                "user_agent": row['user_agent'],
                "content_type": row['content_type']
            }
        }
        cleaned_document = clean_data(document)
        yield cleaned_document


# Load the CSV data into a Pandas DataFrame
def csv_to_df(_csv_file):
    df = pd.read_csv(_csv_file)
    df = df.astype(str).map(lambda x: "NaN" if x == "" or pd.isna(x) else x)
    return df


# # Use the bulk helper to index the data and capture failed documents
def indexer(df, index_name):
    client = create_or_get_es()
    try:
        # Nettoyer le DataFrame pour remplacer les valeurs NaN
        df.fillna(0, inplace=True)  # Remplacer NaN par 0
        # Après avoir chargé le DataFrame
        df['application_is_guessed'] = df['application_is_guessed'].astype(bool)

        success, failed = helpers.bulk(client, generate_data(df, index_name), stats_only=False, raise_on_error=False)
        if failed:
            print(f"{failed} documents failed to index.")
    except helpers.BulkIndexError as e:
        print(f"Error during indexing: {e}")
        for error in e.errors:
            print(error)
