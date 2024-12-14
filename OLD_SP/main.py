from elastic_transport.client_utils import client_meta_version
from elasticsearch import Elasticsearch
import es_module
import pandas as pd
import warnings
import os
from api_elks import *
# Ignorer les avertissements
from urllib3.exceptions import InsecureRequestWarning, SecurityWarning
warnings.simplefilter('ignore', InsecureRequestWarning)
warnings.simplefilter('ignore', SecurityWarning)


if __name__ == '__main__':
    # csv_file_path = "csv_files/trace-24-02-01-00-00-01-1706742001.csv"
    csv_folder = "csv_files"
    index_name = "pcap-flows"
    # create the index
    client = es_module.create_or_get_es()
    client.indices.delete(index='pcap-flows', ignore_unavailable=True)  # Supprime l'index existant pour refaire depuis le début
    es_module.create_or_get_index(index_name)

    all_protocols = {}
    all_apps = {}

    # read the csv files
    for csv_file in os.listdir(csv_folder):
        csv_file_path = os.path.join(csv_folder, csv_file)
        df = pd.read_csv(csv_file_path)
        # bulk insert the data
        es_module.indexer(df, index_name)
        # get the list of all the (distinct) protocols contained in elastic search with the index pcap-flows
        #protocols = get_disctinct_protocols("pcap-flows")
        # add the protocols to the all_protocols dictionary
        applications = get_disctinct_applications("pcap-flows")
        # for protocol in protocols:
        #     if protocol['key'] in all_protocols:
        #         all_protocols[protocol['key']] += protocol['doc_count']
        #     else:
        #         all_protocols[protocol['key']] = protocol['doc_count']
        for application in applications:
            if application in all_apps:
                all_apps[application] += 1
            else:
                all_apps[application] = 1


    for protocol in all_protocols:
        print(f"{protocol}: {all_protocols[protocol]}")
    print("Applications:")
    for app in all_apps:
        print(f"{app}: {all_apps[app]}")
    # get the list of all the (distinct) applications contained in elastic search with the index pcap-flows
    #print(es_module.get_Disctinct_Applications(index_name))

