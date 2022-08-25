"""# LSTM - Rete"""

from pandas import read_csv
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np


# Riceve un dataset, una windows size e il lag
def adeguare_data(dataset, window, lag):
  x_data = []
  y_data = []
  for data_case_id in dataset:
    num_line = data_case_id.shape[0]
    i_last = window
    i_next = lag
    for i, row in enumerate(data_case_id):
      # La windows_size supera le dimensioni dei dati
      if (i_last >= num_line):
        break
      else:
        # Rispetto le dimensioni dei dati
        # Controllo se la linea da predirre esiste
        if (i_next < num_line):
          # Se esiste aggiungo la linea corrispondente
          x_temp = []
          [x_temp.append(x[:-1]) for x in data_case_id[i:i_last]]
          x_data.append(x_temp)
          y_data.append(row[:-1])
        i_last += 1
        i_next += 1
  return x_data, y_data

def create_target(dataset, window, lag, case_id):
  ignora = window - 1 + lag
  risultato = []

  # Dividiamo il dataset per case
  case_id = len(dataset[0])-1
  # Crea una lista dei dati per ogni case
  # datase_diviso[0] -> dati del case 0
  dataset_diviso = [dataset[dataset[:, case_id]==k] for k in np.unique(dataset[:, case_id])]

  # Ignoro le righe corrispondenti e il resto le aggingo in una lista
  for data_case_id in dataset_diviso:
    [risultato.append(x[:-1]) for x in data_case_id[ignora:]]
  return np.array(risultato)

def lstm_preprocess(w, lag, esistono_attributi):
  # Prendo i dati embebed dei passaggi precedenti
  if esistono_attributi:
    train_series = read_csv('train_attribute_embeddings.csv', header=None, float_precision='None')
    test_series = read_csv('test_attribute_embeddings.csv', header=None, float_precision='None')
  else:
    train_series = read_csv('train_embeddings.csv', header=None, float_precision='None')
    test_series = read_csv('test_embeddings.csv', header=None, float_precision='None')

  np_train_series = train_series.to_numpy()
  np_test_series = test_series.to_numpy()

  # Cerco la colonna dove si trova il case id
  case_id = np_train_series.shape[1]-1

  # Strutturo l'array per gruppo di case
  trainset = [np_train_series[np_train_series[:, case_id]==k] for k in np.unique(np_train_series[:, case_id])]
  testset = [np_test_series[np_test_series[:, case_id]==k] for k in np.unique(np_test_series[:, case_id])]
  num_features = trainset[0].shape[1] - 1

  # Gruppi per la LSTM
  window_size = w          #variare la window size

  # TRAIN
  x_data, y_data = adeguare_data(trainset, window_size, lag)
  x_train = np.array(x_data)
  y_train = np.array(y_data)

  # TEST
  x_data, y_data = adeguare_data(testset, window_size, lag)
  x_test = np.array(x_data)
  y_test = np.array(y_data)
  print(x_test.shape, y_test.shape)

  return num_features, case_id, x_train, y_train, x_test, y_test

def lstm_model(w, units, features, x_train, y_train, x_test, y_test, lag, case_id, structural_decoder, attribute_decoder, label_dataset_train, label_dataset_test, esistono_attributi, set_risorse):
  window_size = w
  num_units_lstm = units
  num_features = features

  # Initialising the RNN
  model = Sequential()
  # Per usare più di una layer bisogna 'return_sequences' metterlo a true, con false è per un solo layer
  model.add(LSTM(units=num_units_lstm, input_shape=(window_size, num_features), return_sequences=True))

  # Second leyer
  model.add(LSTM(num_units_lstm))

  # Adding the output layer
  # For Full connection layer we use dense
  # Layer obligatoria
  model.add(Dense(units=num_features))

  #compile and fit the model on 30 epochs
  model.compile(loss='mean_squared_error', optimizer='nadam')
  log(str(model.summary()))

  best_save_lstm = ModelCheckpoint('LSTM_best.hdf5', save_best_only=True, save_weights_only= False, monitor='val_loss', mode='min')

  model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 50, batch_size = 32, callbacks = [best_save_lstm])
  # Addestramento terminato

  model.save('LSMT_last.hdf5')
  best_model_lstm = load_model('LSTM_best.hdf5')
  last_model_lstm = load_model('LSMT_last.hdf5')

  best_model_lstm.evaluate(x_train, y_train)
  last_model_lstm.evaluate(x_train, y_train)

  res_best_model = best_model_lstm.evaluate(x_test, y_test)
  res_last_model = last_model_lstm.evaluate(x_test, y_test)

  if res_best_model < res_last_model:
    train_predictions =  best_model_lstm.predict(x_train)
    test_predictions =  best_model_lstm.predict(x_test)
  else:
    train_predictions =  last_model_lstm.predict(x_train)
    test_predictions =  last_model_lstm.predict(x_test)

  log("  train embeddings MSE = " + str(np.mean(((train_predictions - y_train)**2))))
  log("  test embeddings MSE = " + str(np.mean(((test_predictions - y_test)**2))))

  #se ci sono gli attributi dobbiamo fare due volte il decoder CAGE, togliendo i risultati del primo decoder e concatenarli per colonna con i risultati del secondo decoder
  if esistono_attributi:
    intermedio_train = attribute_decoder.predict(x = train_predictions)
    intermedio_test = attribute_decoder.predict(x = test_predictions)

    attributi_train = intermedio_train[:, -len(set_risorse) - 1:]
    attributi_test = intermedio_test[:, -len(set_risorse) - 1:]

    struttura_train = structural_decoder.predict(x=intermedio_train[:, : -len(set_risorse) - 1])
    struttura_test = test_predictions_decoder = structural_decoder.predict(
      x=intermedio_test[:, : -len(set_risorse) - 1])

    train_predictions_decoder = np.hstack((struttura_train, attributi_train))
    test_predictions_decoder = np.hstack((struttura_test, attributi_test))

  else:
    train_predictions_decoder = structural_decoder.predict(x = train_predictions)
    test_predictions_decoder = structural_decoder.predict(x = test_predictions)

  df_train_predictions_decoder = pd.DataFrame(test_predictions_decoder)
  df_test_predictions_decoder = pd.DataFrame(train_predictions_decoder)

  df_train_predictions_decoder.to_csv("train_predictions_decoder.csv", index = False, header = False)
  df_test_predictions_decoder.to_csv("test_predictions_decoder.csv", index = False, header = False)

  # Comparo i predetti dal modello con il reale
  train_label_predictions = create_target(label_dataset_train, window_size, lag, case_id)
  test_label_predictions = create_target(label_dataset_test, window_size, lag, case_id)

  # Salviamo in un csv il label dataset
  pd.DataFrame(train_label_predictions).to_csv('target_train.csv', index = False, header = False)
  pd.DataFrame(test_label_predictions).to_csv('target_test.csv', index = False, header = False)

  str_temp = str(np.mean(((train_predictions_decoder - train_label_predictions)**2)))
  log("train MSE = " + str_temp)
  str_temp = str(np.mean(((test_predictions_decoder - test_label_predictions)**2)))
  log("test MSE = " + str_temp)

  return train_predictions_decoder, test_predictions_decoder, train_label_predictions, test_label_predictions

#log on file in folder risultati
def log(msg):
  with open('risultati/risultati.txt', 'a') as r:
    r.write(msg + '\n')