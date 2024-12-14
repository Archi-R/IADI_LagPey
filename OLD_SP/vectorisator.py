import cityhash
import data_enrichment

def name_to_int(name: str) -> int:
    return cityhash.CityHash64(name)

def protocol_one_hot_vector(protocol:int)->list[int]:
    protocol_list = [1,2,6,17,58,-1]
    if protocol not in protocol_list:
        print ("Unknown protocol")
        protocol = -1
    return [1 if i==protocol else 0 for i in protocol_list]

def apps_one_hot_vector(app:str)->list[int]:
    apps_list = ['TLS.Google','DNS','NetBIOS','NTP','TLS','LDAP','DNS.Microsoft','DNS.Google','HTTP','Unknown']
    if app not in apps_list:
        print ("Invalid application")
        return [0 for i in apps_list]
    return [1 if i==app else 0 for i in apps_list]

def ip_split(ip:str)->list[int]:
    return [int(i) for i in ip.split('.')]


def flow_to_vector(flow:dict)->list[int]:
    vectorized_flow = []

    for value in flow.values():
        if type(value) == int:
            vectorized_flow.append(value)
        elif value:
            vectorized_flow.append(1)
        elif not value or value is None:
            vectorized_flow.append(0)

    # Ajout des index categoriels
    vectorized_flow.append(protocol_one_hot_vector(flow['protocol']))
    vectorized_flow.append(apps_one_hot_vector(flow['application_name']))
    vectorized_flow.append(ip_split(flow['src_ip']))
    vectorized_flow.append(ip_split(flow['dst_ip']))

    return vectorized_flow