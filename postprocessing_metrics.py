#POST-PROCESSING AND COMPUTE AND PLOT OF METRICS

import numpy as np
import math
from sklearn.metrics import classification_report, multilabel_confusion_matrix, ConfusionMatrixDisplay, jaccard_score
import matplotlib.pyplot as plt
from lstm import log


def threshold_range(train_predictions_decoder, test_predictions_decoder):
    log("  Train")
    log("    0.01 -> " + str(np.count_nonzero(train_predictions_decoder > 0.01)))
    log("    0.2 -> " + str(np.count_nonzero(train_predictions_decoder > 0.2)))
    log("    0.5 -> " + str(np.count_nonzero(train_predictions_decoder > 0.5)))
    log("    0.8 -> " + str(np.count_nonzero(train_predictions_decoder > 0.8)))
    log("  Test")
    log("    0.01 -> " + str(np.count_nonzero(test_predictions_decoder > 0.01)))
    log("    0.2 -> " + str(np.count_nonzero(test_predictions_decoder > 0.2)))
    log("    0.5 -> " + str(np.count_nonzero(test_predictions_decoder > 0.5)))
    log("    0.8 -> " + str(np.count_nonzero(test_predictions_decoder > 0.8)))

"""# METRICS"""


def metrics_preprocess(t, train_predictions_decoder, test_predictions_decoder, train_label_predictions,
                       test_label_predictions):
    threshold = t
    log("  Applichiamo la threshold: " + str(threshold))

    train_predictions_decoder[train_predictions_decoder <= threshold] = 0
    train_predictions_decoder[train_predictions_decoder > threshold] = 1

    test_predictions_decoder[test_predictions_decoder <= threshold] = 0
    test_predictions_decoder[test_predictions_decoder > threshold] = 1

    size = math.floor(np.ma.size(train_predictions_decoder, axis=1) / 3)

    #se nel dataset sono presenti gli attributi vanno gestiti in modo opportuno nel fare lo split
    # Matrici Train
    matrice_nome_train = train_predictions_decoder[:, :size]
    matrice_input_train = train_predictions_decoder[:, size:2 * size]
    matrice_output_train = train_predictions_decoder[:, 2 * size:]

    label_matrice_nome_train = train_label_predictions[:, :size]
    label_matrice_input_train = train_label_predictions[:, size:2 * size]
    label_matrice_output_train = train_label_predictions[:, 2 * size:]

    # Matrici Test
    matrice_nome_test = test_predictions_decoder[:, :size]
    matrice_input_test = test_predictions_decoder[:, size:2 * size]
    matrice_output_test = test_predictions_decoder[:, 2 * size:]

    label_matrice_nome_test = test_label_predictions[:, :size]
    label_matrice_input_test = test_label_predictions[:, size:2 * size]
    label_matrice_output_test = test_label_predictions[:, 2 * size:]

    return matrice_nome_train, matrice_input_train, matrice_output_train, label_matrice_nome_train, label_matrice_input_train, label_matrice_output_train, matrice_nome_test, matrice_input_test, matrice_output_test, label_matrice_nome_test, label_matrice_input_test, label_matrice_output_test


def plot_confusion_matrix(matrice_confusione, eventi, name_matrix):
    f, axes = plt.subplots(3, 5, figsize=(25, 15))
    axs = axes.ravel()
    for i, e in enumerate(eventi.keys()):
        disp = ConfusionMatrixDisplay(matrice_confusione[i], display_labels=['N', 'Y'])
        disp.plot(ax=axs[i], values_format='.4g')
        disp.ax_.set_title(f'{i} - {e}')
        if i < 10:
            disp.ax_.set_xlabel('')
        if i % 5 != 0:
            disp.ax_.set_ylabel('')
        disp.im_.colorbar.remove()

    plt.subplots_adjust(wspace=0.10, hspace=0.1)
    f.colorbar(disp.im_, ax=axs)
    # Rimuoviamo i ricuadri in bianco
    axes[2][3].set_axis_off()
    axes[2][4].set_axis_off()
    plt.savefig(name_matrix + ".png")
    #plt.show()


def compute_metrics(m_nome_train, m_input_train, m_output_train, l_m_nome_train, l_m_input_train, l_m_output_train,
                    m_nome_test, m_input_test, m_output_test, l_m_nome_test, l_m_input_test, l_m_output_test, eventi,
                    w):
    log("   ------------")
    log("   ---TRAIN----")
    log("   ------------")
    log("   TITOLO: y_true = matrice_nome_train, y_pred = label_matrice_nome_train ")
    nome_train_report = classification_report(y_true=m_nome_train, y_pred=l_m_nome_train, output_dict=True)
    log(classification_report(y_true=m_nome_train, y_pred=l_m_nome_train, output_dict=False))
    log("   TITOLO: y_true = matrice_input_train, y_pred = label_matrice_input_train ")
    input_train_report = classification_report(y_true=m_input_train, y_pred=l_m_input_train, output_dict=True)
    log(classification_report(y_true=m_input_train, y_pred=l_m_input_train, output_dict=False))
    log("   TITOLO: y_true = matrice_output_train, y_pred = label_matrice_output_train ")
    output_train_report = classification_report(y_true=m_output_train, y_pred=l_m_output_train, output_dict=True)
    log(classification_report(y_true=m_output_train, y_pred=l_m_output_train, output_dict=False))

    log("   -----------")
    log("   ---TEST----")
    log("   -----------")
    log("   TITOLO: y_true = matrice_nome_test, y_pred = label_matrice_nome_test ")
    nome_test_report = classification_report(y_true=m_nome_test, y_pred=l_m_nome_test, output_dict=True)
    log(classification_report(y_true=m_nome_test, y_pred=l_m_nome_test, output_dict=False))
    log("   TITOLO: y_true = matrice_input_test, y_pred = label_matrice_input_test ")
    input_test_report = classification_report(y_true=m_input_test, y_pred=l_m_input_test, output_dict=True)
    log(classification_report(y_true=m_input_test, y_pred=l_m_input_test, output_dict=False))
    log("   TITOLO: y_true = matrice_output_test, y_pred = label_matrice_output_test ")
    output_test_report = classification_report(y_true=m_output_test, y_pred=l_m_output_test, output_dict=True)
    log(classification_report(y_true=m_output_test, y_pred=l_m_output_test, output_dict=False))

    # Calcoliamo le matrici di confusione
    #SE SI VUOLE FARE IL PLOT DEI VALORI CONTENUTI NELLA CONFUSION MATRIX AGGIUNGERE NEL RETURN LE VARIABILI IN BASSO E GESITTRLE NEL MAIN APPENDENDOLE ALLE RELATIVE LISTE
    log("   MATRICI DI CONFUSIONE MULTILABEL:")
    log("    TITOLO: matrice_confusione_nome_train: -> y_true = label_matrice_nome_train, y_pred = matrice_nome_train")
    matrice_confusione_nome_train = multilabel_confusion_matrix(y_true=l_m_nome_train, y_pred=m_nome_train)
    log(''.join(map(str, matrice_confusione_nome_train)))
    log(
        "    TITOLO: matrice_confusione_input_train: -> y_true = label_matrice_input_train, y_pred = matrice_input_train")
    matrice_confusione_input_train = multilabel_confusion_matrix(y_true=l_m_input_train, y_pred=m_input_train)
    log(''.join(map(str, matrice_confusione_input_train)))
    log(
        "    TITOLO: matrice_confusione_output_train: -> y_true = label_matrice_output_train, y_pred = matrice_output_train")
    matrice_confusione_output_train = multilabel_confusion_matrix(y_true=l_m_output_train, y_pred=m_output_train)
    log(''.join(map(str, matrice_confusione_output_train)))

    log("    TITOLO: matrice_confusione_nome_test: -> y_true = label_matrice_nome_test, y_pred = matrice_nome_test")
    matrice_confusione_nome_test = multilabel_confusion_matrix(y_true=l_m_nome_test, y_pred=m_nome_test)
    log(''.join(map(str, matrice_confusione_nome_test)))
    log("    TITOLO: matrice_confusione_input_test: -> y_true = label_matrice_input_test, y_pred = matrice_input_test")
    matrice_confusione_input_test = multilabel_confusion_matrix(y_true=l_m_input_test, y_pred=m_input_test)
    log(''.join(map(str, matrice_confusione_input_test)))
    log(
        "    TITOLO: matrice_confusione_output_test: -> y_true = label_matrice_output_test, y_pred = matrice_output_test")
    matrice_confusione_output_test = multilabel_confusion_matrix(y_true=l_m_output_test, y_pred=m_output_test)
    log(''.join(map(str, matrice_confusione_output_test)))

    # Salviamo file immagini delle matrici confusione
    plot_confusion_matrix(matrice_confusione_nome_train, eventi, "img/matrice_confusione_nome_train_window" + str(w))
    plot_confusion_matrix(matrice_confusione_input_train, eventi, "img/matrice_confusione_input_train_window" + str(w))
    plot_confusion_matrix(matrice_confusione_output_train, eventi, "img/matrice_confusione_output_train_window" + str(w))

    plot_confusion_matrix(matrice_confusione_nome_test, eventi, "img/matrice_confusione_nome_test_window" + str(w))
    plot_confusion_matrix(matrice_confusione_input_test, eventi, "img/matrice_confusione_input_test_window" + str(w))
    plot_confusion_matrix(matrice_confusione_output_test, eventi, "img/matrice_confusione_output_test_window" + str(w))

    #JACCARD INDEX
    #train
    log("   JI: y_true = matrice_nome_train, y_pred = label_matrice_nome_train ")
    ji_nome_train = np.append(jaccard_score(y_true = m_nome_train, y_pred = l_m_nome_train, average = 'weighted'), jaccard_score(y_true = m_nome_train, y_pred = l_m_nome_train, average = 'samples'))

    log("   JI: y_true = matrice_input_train, y_pred = label_matrice_input_train ")
    ji_input_train = np.append(jaccard_score(y_true = m_input_train, y_pred = l_m_input_train, average = 'weighted'), jaccard_score(y_true = m_input_train, y_pred = l_m_input_train, average = 'samples'))

    log("   JI: y_true = matrice_output_train, y_pred = label_matrice_output_train ")
    ji_output_train = np.append(jaccard_score(y_true = m_output_train, y_pred = l_m_output_train, average = 'weighted'), jaccard_score(y_true = m_output_train, y_pred = l_m_output_train, average = 'samples'))

    #test
    log("   JI: y_true = matrice_nome_test, y_pred = label_matrice_nome_test ")
    ji_nome_test = np.append(jaccard_score(y_true = m_nome_test, y_pred = l_m_nome_test, average = 'weighted'), jaccard_score(y_true = m_nome_test, y_pred = l_m_nome_test, average = 'samples'))

    log("   JI: y_true = matrice_input_test, y_pred = label_matrice_input_test ")
    ji_input_test = np.append(jaccard_score(y_true = m_input_test, y_pred = l_m_input_test, average = 'weighted'), jaccard_score(y_true = m_input_test, y_pred = l_m_input_test, average = 'samples'))

    log("   JI: y_true = matrice_output_test, y_pred = label_matrice_output_test ")
    ji_output_test = np.append(jaccard_score(y_true = m_output_test, y_pred = l_m_output_test, average = 'weighted'), jaccard_score(y_true = m_output_test, y_pred = l_m_output_test, average = 'samples'))

    return nome_train_report, input_train_report, output_train_report, nome_test_report, input_test_report, output_test_report, ji_nome_train, ji_input_train, ji_output_train, ji_nome_test, ji_input_test, ji_output_test

'''
# plot tp,fp,fn,tn al variare della window size per ogni classe
def plot_line_chart(attività_train, attività_test, input_train, input_test, output_train, output_test, set_eventi, list_windows_size):
    matrici = {"attività test": attività_test, "input test": input_test, "output test": output_test,
               "attività train": attività_train, "input train": input_train, "output train": output_train}

    for i, evento in enumerate(set_eventi.keys()):
        for chiave in matrici.keys():
            tn = []
            fp = []
            fn = []
            tp = []
            matrice = matrici.get(chiave)
            for j, window in enumerate(list_windows_size):
                divisore = np.sum(matrice[j][i])
                tn.append(matrice[j][i][0, 0] / divisore)
                fp.append(matrice[j][i][0, 1] / divisore)
                fn.append(matrice[j][i][1, 0] / divisore)
                tp.append(matrice[j][i][1, 1] / divisore)

            plt.figure(figsize=(12, 8))
            plt.title("nodi " + chiave + " classe " + str(i) + " " + evento + " al variare del ws")
            plt.plot(list_windows_size, tn, marker='+', label="True negative")
            plt.plot(list_windows_size, fp, marker='x', label="False positive")
            plt.plot(list_windows_size, fn, marker='d', label="False negative")
            plt.plot(list_windows_size, tp, marker='o', label="True positive")
            plt.xticks(np.arange(min(list_windows_size), max(list_windows_size) + 2, step=2))
            plt.yticks(np.arange(0, 1.1, step=0.1))

            plt.grid(True)
            plt.legend()
            plt.savefig('img/' + chiave.replace(' ', '_') + '_classe' + str(i) + '.png', dpi=200)
            plt.cla()
            plt.clf()
            plt.close()
'''

def plot_line_chart_ji(attività_train, attività_test, input_train, input_test, output_train, output_test, list_windows_size):
    matrici = {"attività test": attività_test, "input test": input_test, "output test": output_test, "attività train": attività_train, "input train": input_train, "output train": output_train}

    for chiave in matrici.keys():
        ji_weigthed = []
        ji_samples = []
        matrice = matrici.get(chiave)
        for i, window in enumerate(list_windows_size):
            ji_weigthed.append(matrice[i][0])
            ji_samples.append(matrice[i][1])

        plt.figure(figsize=(12,8))
        plt.title("Media nodi " + chiave + " al variare della ws")
        plt.plot(list_windows_size, ji_weigthed,  marker = 'o', label = "Jaccard Index Weigthed")
        plt.plot(list_windows_size, ji_samples,  marker = 'd', label = "Jaccard Index Samples")
        plt.xticks(np.arange(min(list_windows_size), max(list_windows_size) + 2, step = 2))
        plt.yticks(np.arange(0, 1.1, step = 0.1))

        plt.grid(True)
        plt.legend()
        plt.savefig('img/'+ chiave.replace(' ','_') + '_JI' + '.png', dpi = 200)
        plt.cla()
        plt.clf()
        plt.close()

# plot classification report (precision, recall, f1-score) al variare della window size per ogni classe
def plot_line_chart_classification_report(nome_train_report, nome_test_report, input_train_report, input_test_report,
                                          output_train_report, output_test_report, set_eventi, list_windows_size):
    matrici = {"attività train classification report": nome_train_report,
               "input train classification report": input_train_report,
               "output train classification report": output_train_report,
               "attività test classification report": nome_test_report,
               "input test classification report": input_test_report,
               "output test classification report": output_test_report}
    
    for i, evento in enumerate(set_eventi.keys()):
        for chiave in matrici.keys(): #per ogni matrice
            precision = []
            recall = []
            f1score = []
            matrice = matrici.get(chiave)
            for j, window in enumerate(list_windows_size):
                precision.append(matrice[j].get(str(i)).get('precision'))
                recall.append(matrice[j].get(str(i)).get('recall'))
                f1score.append(matrice[j].get(str(i)).get('f1-score'))

            plt.figure(figsize=(12,8))
            plt.title("nodi " + chiave + " classe " + str(i) + " " + evento + " al variare della ws")
            plt.plot(list_windows_size,  precision,  marker = 'o',  label = "Precision")
            plt.plot(list_windows_size, recall,   marker = 'x', label = "Recall")
            plt.plot(list_windows_size, f1score,  marker = 'd', label = "F1-score")
            plt.xticks(np.arange(min(list_windows_size), max(list_windows_size) + 2, step = 2))
            plt.yticks(np.arange(0, 1.1, step = 0.1))

            plt.grid(True)
            plt.legend()
            plt.savefig('img/'+ chiave.replace(' ','_') + '_classe'+ str(i) + '.png', dpi = 200)
            plt.cla()
            plt.clf()
            plt.close()

    #plot del classification report medi al variare della window size
    for chiave in matrici.keys(): #per ogni matrice
        precision_samples = []
        precision_weighted = []
        recall_samples = []
        recall_weighted = []
        f1score_samples = []
        f1score_weighted = []

        matrice = matrici.get(chiave)
        for j, window in enumerate(list_windows_size):
            precision_samples.append(matrice[j].get('samples avg').get('precision'))
            recall_samples.append(matrice[j].get('samples avg').get('recall'))
            f1score_samples.append(matrice[j].get('samples avg').get('f1-score'))
            precision_weighted.append(matrice[j].get('weighted avg').get('precision'))
            recall_weighted.append(matrice[j].get('weighted avg').get('recall'))
            f1score_weighted.append(matrice[j].get('weighted avg').get('f1-score'))

        plt.figure(figsize=(12,8))
        plt.title("media nodi " + chiave + " al variare della ws")
        plt.plot(list_windows_size,  precision_samples,  marker = 'o',  label = "Precision samples")
        plt.plot(list_windows_size, recall_samples,   marker = 'x', label = "Recall samples")
        plt.plot(list_windows_size, f1score_samples,  marker = 'd', label = "F1-score samples")
        plt.plot(list_windows_size,  precision_weighted,  marker = 's',  label = "Precision weighted")
        plt.plot(list_windows_size, recall_weighted,   marker = '*', label = "Recall weighted")
        plt.plot(list_windows_size, f1score_weighted,  marker = 'p', label = "F1-score weighted")
        plt.xticks(np.arange(min(list_windows_size), max(list_windows_size) + 2, step = 2))
        plt.yticks(np.arange(0, 1.1, step = 0.1))

        plt.grid(True)
        plt.legend()
        plt.savefig('img/'+ chiave.replace(' ','_') + '_avg' + '.png', dpi = 200)
        plt.cla()
        plt.clf()
        plt.close()
