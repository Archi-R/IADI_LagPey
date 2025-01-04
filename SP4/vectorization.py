import pandas as pd
from sklearn.preprocessing import StandardScaler

def vectorize_flows(df, categorical_cols, numeric_cols, label_col, unique_values: dict):
    """
    Transforme les flux en vecteurs de caractéristiques numériques à partir d'un DataFrame directement.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les flux.
        categorical_cols (list): Colonnes catégoriques à vectoriser.
        numeric_cols (list): Colonnes numériques à normaliser.
        label_col (str): Nom de la colonne contenant les labels.
        unique_values (dict): Dictionnaire contenant les ensembles de valeurs uniques pour les colonnes catégoriques.

    Returns:
        pd.DataFrame: DataFrame vectorisé avec les colonnes numériques normalisées et les colonnes catégoriques one-hot.
    """
    # Vérification des colonnes manquantes
    missing_cols = [col for col in numeric_cols + categorical_cols + [label_col] if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns {missing_cols} not in index")

    # Séparer les labels et les features
    y = df[label_col].values
    x = df[categorical_cols + numeric_cols].copy()

    # Transformation des colonnes catégoriques
    for col in categorical_cols:
        if col in ['src_ip', 'dst_ip']:
            # Décomposition IP en parties
            x[col] = x[col].apply(ip_split)
            ip_expanded = pd.DataFrame(x[col].tolist(), index=x.index)
            ip_expanded.columns = [f'{col}_part_{i}' for i in range(ip_expanded.shape[1])]
            x.drop(columns=[col], inplace=True)
            x = pd.concat([x, ip_expanded], axis=1)
        else:
            # One-hot encoding pour les autres colonnes catégoriques
            one_hot_vectors = x[col].apply(generic_one_hottizator, possible_values_set=unique_values[col])
            one_hot_df = pd.DataFrame(one_hot_vectors.tolist(), index=x.index)
            one_hot_df.columns = [f"{col}_onehot_{i}" for i in range(one_hot_df.shape[1])]
            x.drop(columns=[col], inplace=True)
            x = pd.concat([x, one_hot_df], axis=1)

    # Normalisation des colonnes numériques
    scaler = StandardScaler()
    x[numeric_cols] = scaler.fit_transform(x[numeric_cols])

    # Ajouter la colonne label au DataFrame final
    x['label'] = y

    return x


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
    """
    Retourne un vecteur one-hot pour la valeur spécifiée, avec une colonne supplémentaire indiquant si la valeur est inconnue.

    Args:
        value (str): La valeur à encoder.
        possible_values_set (set): Ensemble des valeurs possibles pour l'encodage.

    Returns:
        list[int]: Vecteur one-hot avec une colonne supplémentaire pour les valeurs inconnues.
    """
    is_unknown = int(value not in possible_values_set)  # 1 si la valeur est inconnue
    one_hot = [1 if i == value else 0 for i in possible_values_set]
    one_hot.append(is_unknown)  # Ajouter la colonne pour les valeurs inconnues
    return one_hot

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
