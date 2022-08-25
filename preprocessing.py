"""# PREPROCESSING DATASET"""

from keras.utils import to_categorical
from collections import OrderedDict
import pandas as pd
import numpy as np
import math

def adapt_file_onehotencode(nome_file_ori, nome_file_mod, nome_file_onehotencoded):
    count = 0
    case_ids = []
    lista_eventi = []
    lista_risorse = []
    max_time = 0.0

    # ------------------------------
    # File Mod, contiamo i case id, le attività, le risorse e il max time
    # ------------------------------
    # Leggiamo il dataset d'entrata
    with open(nome_file_ori) as a:
        # Si crea un nuovo file, un dataset modificato
        with open(nome_file_mod, 'w') as s:
            for x in a:
                x = x.strip('\n')  # Togliamo i vuoti e il salto di linea per riga
                x = x.replace(",", "").replace("  ", " ")  # Togliamo le virgole e i doppi spazi
                if x:  # ignora righe vuote
                    if x.startswith('XP'):  # Cambio di case
                        count += 1
                    elif x.startswith('v '):  # è un vertice
                        case_ids.append(count)
                        splitted_x = x.split(" ")
                        lista_eventi.append(splitted_x[2])  # Aggiungiamo l'attivita in una lista
                        if len(splitted_x) == 5:  # Il file contiene attributi tempo e risorse
                            lista_risorse.append(splitted_x[4])  # Aggiungiamo le risorse in una lista
                            max_time = float(splitted_x[3]) if max_time < float(
                                splitted_x[3]) else max_time  # Salvo il tempo più grande
                    # Scriviamo la linea sul nuovo file
                    str1 = ''.join(x)
                    s.write(str1 + '\n')

    # Lista ordinata con le distinte attività presenti nel dataset
    set_eventi = OrderedDict.fromkeys(lista_eventi)

    # Lista ordinata con le distinte risorse presenti nel dataset
    set_risorse = OrderedDict.fromkeys(lista_risorse)
    esistono_attributi = len(set_risorse.keys()) > 0

    # ------------------------------
    # One Hot Encoding
    # ------------------------------
    # Trasforma i nomi attività in interi e lo salva in un nuovo file
    with open(nome_file_mod, 'r') as s:
        with open(nome_file_onehotencoded, 'w') as o:
            lines = s.readlines()
            str_lines = ''.join(lines)
            for i, s in enumerate(set_eventi.keys()):
                # Cerca l'attivita dentro il set e lo scambia per l'indice della posizione sul set
                str_lines = str_lines.replace(s, str(i))
            o.writelines(str_lines)

    return case_ids, set_eventi, set_risorse, max_time, esistono_attributi


def create_dataset(max_time, esistono_attributi, nome_file_onehotencoded):
    # ------------------------------
    # Costruiamo il nostro dataset
    # ------------------------------
    count = 0
    dataset = []
    with open(nome_file_onehotencoded) as a:
        for x in a:
            x = x.strip('\n')  # Togliamo i vuoti e il salto di linea per riga
            if x.startswith('XP'):  # Cambio di case
                count += 1
                dataset.append([])
                dataset[count - 1].append(
                    [])  # Aggiungiamo una lista per i vertici   ->  dataset[i][0] : i del case, 0 per i vertici
                dataset[count - 1].append(
                    [])  # Aggiungiamo una lista per i nodi      ->  dataset[i][0] : i del case, 1 per i nodi
            elif x.startswith('v '):
                x = x.strip('v ')  # Togliamo la v iniziale della riga
                splitted_x = x.split(" ")
                if esistono_attributi:
                    temp_list = list(map(int, splitted_x[:2]))
                    temp_list.append(float(splitted_x[2]) / max_time)
                    temp_list.append(int(splitted_x[3]))
                    dataset[count - 1][0].append(temp_list)
                else:
                    dataset[count - 1][0].append(list(map(int, splitted_x)))
            else:
                x = x.strip('e ')
                splitted_x = x.replace('__', " ").split(" ")
                dataset[count - 1][1].append(list(map(int, splitted_x)))

    np_dataset = np.array(dataset)
    return np_dataset


def creaMatrici(case_ids, set_eventi, set_risorse):
    matriceinput = np.zeros((len(case_ids), len(set_eventi.keys())), dtype='bool')
    matricenome = np.zeros((len(case_ids), len(set_eventi.keys())), dtype='int')
    matriceoutput = np.zeros((len(case_ids), len(set_eventi.keys())), dtype='bool')
    matricetempo = np.zeros((len(case_ids), 1), dtype='float')
    matricerisorse = np.zeros((len(case_ids), len(set_risorse.keys())), dtype='int')
    return matriceinput, matricenome, matriceoutput, matricerisorse, matricetempo


def preprocessing(case_ids, set_eventi, set_risorse, np_dataset, esistono_attributi, nome_file_preprocessed):
    # ------------------------------
    # Preprocess del dataset
    # Ignora nodi isolati (non hanno archi entrati o uscenti)
    # ------------------------------
    case_ids2 = []
    matriceinput, matricenome, matriceoutput, matricerisorse, matricetempo = creaMatrici(case_ids, set_eventi,
                                                                                         set_risorse)
    count = 0

    for i, case in enumerate(np_dataset):
        # per ogni nodo in una traccia (case_id)
        for j, node in enumerate(case[0]):  # case[0] -> 0 per la lista dei nodi
            delete = True
            nod = node[1]
            matricenome[count] = to_categorical(nod, num_classes=len(set_eventi.keys()), dtype='int')
            # Se esistono gli attributi, si usano altre matrici
            if esistono_attributi:
                tempo = node[2]
                risorsa = node[3]
                matricetempo[count] = tempo
                matricerisorse[count] = to_categorical(risorsa - 1, num_classes=len(set_risorse.keys()), dtype='int')
            case_ids2.append(i)

            temp = np.zeros(len(case_ids), dtype='bool')

            # itero su tutti gli archi del case-id i-esimo
            for k, edge in enumerate(case[1]):  # case[1] -> 1 per la lista degli archi

                # controllo se il nodo ha archi entranti
                # se ci sono si salva i nodi da cui è partito l'arco (nodi di input al nodo)
                if nod == edge[3]:
                    delete = False
                    temp = to_categorical(edge[2], num_classes=len(set_eventi.keys()), dtype='bool')
                    matriceinput[count] = np.sum([matriceinput[count], temp], axis=0, dtype='bool')

                # controllo se il nodo ha archi uscenti
                # se ci sono salva i nodi dove arriva l'arco (nodi di output al nodo)
                if nod == edge[2]:
                    delete = False
                    temp = to_categorical(edge[3], num_classes=len(set_eventi.keys()), dtype='bool')
                    matriceoutput[count] = np.sum([matriceoutput[count], temp], axis=0, dtype='bool')

            matriceinput = matriceinput.astype('int')
            matriceoutput = matriceoutput.astype('int')

            # se il nodo i-esimo non ha nessun arco entrante o uscente, quindi è isolato, lo eliminiamo
            if delete:
                matricenome = np.delete(matricenome, count, axis=0)
                matriceinput = np.delete(matriceinput, count, axis=0)
                matriceoutput = np.delete(matriceoutput, count, axis=0)
                # Se esistono gli attributi
                if esistono_attributi:
                    matricetempo = np.delete(matricetempo, count, axis=0)
                    matricerisorse = np.delete(matricerisorse, count, axis=0)
                del case_ids2[count]
            else:
                count += 1

    # Nodi scartati perché isolati, numero e percentuale
    print("Nodi scartati perché isolati: " + str(
        len(case_ids) - len(case_ids2)) + '\n' + "Percentuale rispetto al totale: {:.3f}".format(
        (len(case_ids) - len(case_ids2)) / len(case_ids) * 100) + " %")

    # Concatena le matrici
    preprocessed = np.hstack((matricenome, matriceinput, matriceoutput))

    # Salviamo il dataset in un file
    df_preprocessed = pd.DataFrame(preprocessed)
    df_preprocessed.to_csv(nome_file_preprocessed, index=False, header=False)

    if esistono_attributi:
        return case_ids2, np.hstack((matricetempo, matricerisorse))
    else:
        return case_ids2


"""# IMPORT DATASET"""


def split_dataset(dataset, case_ids, split_ratio):
    split = math.ceil(np.ma.size(dataset, axis=0) * split_ratio)

    i = 0
    while case_ids[split + i - 1] == case_ids[split + i]:
        i += 1

    datatrain = dataset[: split + i]
    datatest = dataset[split + i:]
    case_ids_train = case_ids[: split + i]
    case_ids_test = case_ids[split + i:]
    print('pos:', str(split) + ' + ' + str(i))
    return datatrain, datatest, case_ids_train, case_ids_test
