import pandas as pd
from vectorisator import flow_to_vector

# add the traffic load (in number of packets) that is exchanged with the
# ip source during a time window of size T centered in the middle of the
# flow time span.
# 2
# = ∑
# pour tous les flows dont l'heure de début est comprise entre t et t+T autour de l'heure de débur de F
#     on regarde si l'IP source est la même que celle de F
#         si oui, on ajoute le nombre de packets échangés à la liste
def get_traffic_load_src_ip(src_ip, F: pd.Series, T: int) -> int:
    t = F['bidirectional_first_seen_ms']
    if F['bidirectional_first_seen_ms'] >= t and F['bidirectional_first_seen_ms'] <= t + T:
        if F['src_ip'] == src_ip:
            return F['bidirectional_packets']


# • add the traffic load (in number of packets) that is exchanged with the
# ip destination during a time window of size T centered in the middle
# of the flow time span.
# 2
# = ∑
# pour tous les flows dont l'heure de début est comprise entre t et t+T autour de l'heure de débur de F
#     on regarde si l'IP destination est la même que celle de F
#         si oui, on ajoute le nombre de packets échangés
def get_traffic_load_dst_ip(dst_ip, F: pd.Series, T: int) -> int:
    t = F['bidirectional_first_seen_ms']
    if F['bidirectional_first_seen_ms'] >= t and F['bidirectional_first_seen_ms'] <= t + T:
        if F['dst_ip'] == dst_ip:
            return F['bidirectional_packets']


# • add the interconnection degree (number of existing links) of the ip
# source during a time window of size T centered in the middle of the
# flow time span.
# 2
# = ∑
# pour tous les flows dont l'heure de début est comprise entre t et t+T autour de l'heure de débur de F
#     on regarde si l'IP source est la même que celle de F
#         si oui, on ajoute le nombre
def get_interconnection_degree_src_ip(src_ip, F: pd.Series, T: int) -> int:
    t = F['bidirectional_first_seen_ms']
    if F['bidirectional_first_seen_ms'] >= t and F['bidirectional_first_seen_ms'] <= t + T:
        if F['src_ip'] == src_ip:
            return 1
        else:
            return 0


# • add the interconnection degree (number of existing links) of the ip
# destination during a time window of size T centered in the middle of
# the flow time span.
# 2
# = ∑
# pour tous les flows dont l'heure de début est comprise entre t et t+T autour de l'heure de débur de F
#    on regarde si l'IP destination est la même que celle de F
#        si oui, on ajoute le nombre
#       sinon, on ajoute 0
def get_interconnection_degree_dst_ip(dst_ip, F: pd.Series, T: int) -> int:
    t = F['bidirectional_first_seen_ms']
    if t <= F['bidirectional_first_seen_ms'] <= t + T:
        if F['dst_ip'] == dst_ip:
            return 1
        else:
            return 0


def data_enrichment_to_vector(flows: pd.DataFrame, t: int) -> list[int]:
    for F in flows:
        src_ip = F['src_ip']
        dst_ip = F['dst_ip']

        traffic_load_src_ip = 0
        traffic_load_dst_ip = 0
        interconnection_degree_src_ip = 0
        interconnection_degree_dst_ip = 0

        for flow in flows:
            traffic_load_src_ip += get_traffic_load_src_ip(src_ip, flow, t)
            traffic_load_dst_ip += get_traffic_load_dst_ip(dst_ip, flow, t)
            interconnection_degree_src_ip += get_interconnection_degree_src_ip(src_ip, flow, t)
            interconnection_degree_dst_ip += get_interconnection_degree_dst_ip(dst_ip, flow, t)

        #enrichissement
        F['traffic_load_src_ip'] = traffic_load_src_ip
        F['traffic_load_dst_ip'] = traffic_load_dst_ip
        F['interconnection_degree_src_ip'] = interconnection_degree_src_ip
        F['interconnection_degree_dst_ip'] = interconnection_degree_dst_ip

        # Vectorisation du flow
        vectorized_flow = flow_to_vector(F)


    return vectorized_flow