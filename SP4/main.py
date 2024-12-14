from nfstream import NFStreamer

# Chemin vers le dossier des pcap
pcap_folder = "../pcap_folder"
old_pcap_folder = pcap_folder+"/old_pcap_files"

test_pcap_file = old_pcap_folder+"/trace-24-02-01-00-00-01-1706742001.pcap" # CHANGE


streamer = NFStreamer(source=test_pcap_file,
                       idle_timeout=60,
                       active_timeout=120,
                       statistical_analysis=True)

flows = streamer.to_dataframe()
print(flows.head())
