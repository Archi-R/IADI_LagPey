import pandas as pd
from sklearn.preprocessing import StandardScaler
import cityhash

def vectorize_flows(df, categorical_cols=None, numeric_cols=None, label_col='label', apps_list=None, protocol_list=None):
    """
    Transforme les flux en vecteurs de caractéristiques numériques à partir d'un DataFrame directement.
    """
    if categorical_cols is None:
        categorical_cols = ['protocol', 'application_name']

    if numeric_cols is None:
        numeric_cols = [
            'bidirectional_packets', 'bidirectional_bytes', 'fan_in', 'fan_out',
            'bidirectional_duration_ms', 'bidirectional_first_seen_ms',
            'bidirectional_last_seen_ms'
        ]

    # Check if the required columns exist
    missing_cols = [col for col in numeric_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns {missing_cols} not in index")

    y = df[label_col].values
    x = df[categorical_cols + numeric_cols].copy()

    for col in categorical_cols:
        if col == 'protocol':
            x['protocol'] = x['protocol'].apply(generic_one_hottizator, possible_values_set=protocol_list)
        elif col == 'application_name':
            x['application_name'] = x['application_name'].apply(generic_one_hottizator, possible_values_set=apps_list)
        elif col in ['src_ip', 'dst_ip']:
            x[col] = x[col].apply(ip_split)

    if 'protocol' in x.columns:
        protocol_expanded = pd.DataFrame(x['protocol'].tolist(), index=x.index)
        protocol_expanded.columns = [f'protocol_{i}' for i in range(protocol_expanded.shape[1])]
        x.drop(columns=['protocol'], inplace=True)
        x = pd.concat([x, protocol_expanded], axis=1)

    if 'application_name' in x.columns:
        apps_expanded = pd.DataFrame(x['application_name'].tolist(), index=x.index)
        apps_expanded.columns = [f'app_{i}' for i in range(apps_expanded.shape[1])]
        x.drop(columns=['application_name'], inplace=True)
        x = pd.concat([x, apps_expanded], axis=1)

    for ip_col in ['src_ip', 'dst_ip']:
        if ip_col in x.columns:
            ip_expanded = pd.DataFrame(x[ip_col].tolist(), index=x.index)
            ip_expanded.columns = [f'{ip_col}_part_{i}' for i in range(ip_expanded.shape[1])]
            x.drop(columns=[ip_col], inplace=True)
            x = pd.concat([x, ip_expanded], axis=1)

    scaler = StandardScaler()
    x[numeric_cols] = scaler.fit_transform(x[numeric_cols])

    return x.values, y


def name_to_int(name: str) -> int:
    return cityhash.CityHash64(name)

# def protocol_one_hot_vector(protocol:int, protocol_list = None)->list[int]:
#     if protocol_list is None:protocol_list = [1, 2, 6, 17, 58, -1]
#     if protocol not in protocol_list:
#         print ("Unknown protocol")
#         protocol = -1
#     return [1 if i==protocol else 0 for i in protocol_list]
#
# def apps_one_hot_vector(app:str,apps_list = None)->list[int]:
#     # TODO: Ajouter les applications manquantes
#     if apps_list is None : apps_list = ['TLS.Google','DNS','NetBIOS','NTP','TLS','LDAP','DNS.Microsoft','DNS.Google','HTTP','Unknown']
#     if app not in apps_list:
#         print ("Invalid application")
#         return [0 for i in apps_list]
#     return [1 if i==app else 0 for i in apps_list]

def generic_one_hottizator(value:str, possible_values_set :set)->list[int]:
    value = str(value)
    if value not in possible_values_set:
        raise ValueError(f"Valeur inconnue : {value}")
    return [1 if i==value else 0 for i in possible_values_set]

def ip_split(ip:str)->list[int]:
    try:
        return [int(i) for i in ip.split('.')]
    except:
        return [0,0,0,0]



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
    #vectorized_flow.append(protocol_one_hot_vector(flow['protocol']))
    vectorized_flow.append(generic_one_hottizator(flow['protocol'], []))
    #vectorized_flow.append(apps_one_hot_vector(flow['application_name']))
    vectorized_flow.append(generic_one_hottizator(flow['application_name'], []))
    vectorized_flow.append(ip_split(flow['src_ip']))
    vectorized_flow.append(ip_split(flow['dst_ip']))

    return vectorized_flow
