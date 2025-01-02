from pcapLoader import csv_to_reader

########### tool temporaire
def quels_champs_sont_constants(csv_path: str)->dict:
    # on charge le fichier
    reader = csv_to_reader(csv_path)
    # on récupère les noms des champs
    fields = reader[0].keys()
    # on recupère le nombre de lignes
    nb_lignes = len(reader)
    # on crée un dictionnaire pour stocker les valeurs des champs
    values = {}
    # on initialise le dictionnaire avec des ensembles vides
    for field in fields:
        values[field] = set()
    # on parcourt les lignes
    for row in reader:
        # on parcourt les champs
        for field in fields:
            # on ajoute la valeur du champ à l'ensemble correspondant
            values[field].add(row[field])

    constants = {}
    faibles = {}
    moyens = {}

    # on parcourt les champs
    for field in fields:
        if len(values[field]) == 1:
            constants[field] = values[field]
        elif len(values[field]) < 10:
            faibles[field] = values[field]
        elif len(values[field]) < 50:
            moyens[field] = values[field]

    print("Champs constants : ")
    for field in constants:
        print(f"\n\t{field}, valeurs : ")
        for value in constants[field]:
            print("\t\t"+value)
    print("Champs faibles : ")
    for field in faibles:
        print(f"\n\t{field}, valeurs : ")
        for value in faibles[field]:
            print("\t\t"+value)
    print("Champs moyens : ")
    for field in moyens:
        print(f"\n\t{field}, valeurs : ")
        for value in moyens[field]:
            print("\t\t"+value)

# quels_champs_sont_constants("C:\\Projets_GIT_C\\ENSIBS\\ia_detection\\IADI_LagPey\\pcap_folder\\dataset\\csv\\trace_a_10.csv")

def valeurs_uniques(csv_path: str, field:str)->set:
    # on charge le fichier
    reader = csv_to_reader(csv_path)
    values = set()
    # on parcourt les lignes
    for row in reader:
        # on parcourt les champs
            values.add(row[field])

    return values


def fix_ligne10000(csv_path: str)->str:
    import csv
    # réécrire tout le csv sauf la ligne 10000
    # créer un nouveau fichier
    new_csv_path = csv_path.replace(".csv", "_fix10000.csv")
    # on charge le fichier
    reader = csv_to_reader(csv_path)
    # parcour les lignes
    with open(new_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=reader[0].keys())
        writer.writeheader()
        for i in range(len(reader)):
            if i != 10000:
                writer.writerow(reader[i])
            else:
                print("Ligne 10000")
                print(reader[i])
    # fermer
    f.close()
    # retourner le nouveau fichier




    # # on charge le fichier
    # reader = csv_to_reader(csv_path)
    # # on récupère les noms des champs
    # #aller à la ligne 10000
    # fields:list = reader[0].keys()
    #
    # new_raw = reader[10000].copy()
    # i=0
    # for field in fields:
    #     if field not in ["id", "expiration_id", "fan_in", "fan_out", "label"]:
    #         new_raw[field] = reader[10000][field[i-2]
    #

    return new_csv_path

# fix_ligne10000("C:\\Projets_GIT_C\\ENSIBS\\ia_detection\\IADI_LagPey\\pcap_folder\\dataset\\csv\\all_data_with_fan_labeled.csv")

def subset_divizor(df, list_of_values, field):
    """
    Divise un dataframe en sous-dataframes selon les valeurs d'un champ
    :param df:
    :param list_of_values:
    :param field:
    :return:
    """
    dict_apps = {}
    for app in list_of_values:
        dict_apps[app] = df[(df[field] == app) | (df[field].str.contains(app))]

    return dict_apps