"""# CAGE"""

from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Nadam
from keras.losses import MeanSquaredError
from keras.callbacks import ModelCheckpoint
import pandas as pd


#[#set-eventi] + [#set-eventi] + [#set-eventi] + 1 (float tempo) + [#set-risorse] se ci sono gli attributi
#[#set-eventi] + [#set-eventi] + [#set-eventi] se non ci sono gli attributi

def get_encoder(inputshape, dimensionEmbedding, hiddenLayer, denseUnit, activationFunction):
  """
  Args:
          inputshape:
          dimensionEmbedding: dimension of the embedding
          hiddenLayer: number of hidden layers in encoder/decoder
          denseUnit: dimension of dense layers, not including the units in the embedding layer
  """
  # Input
  x = Input(shape=(inputshape,))
  # Encoder layers
  y = [None] * (hiddenLayer + 1)
  y[0] = x  # y[0] is assigned the input
  for i in range(hiddenLayer - 1):
      y[i + 1] = Dense(denseUnit, activation=activationFunction)                   (y[i])
  # The layer of number hiddenLayer have shape of dimensionEmbedding and it is connected to latest layer of encoding
  y[hiddenLayer] = Dense(dimensionEmbedding, activation=activationFunction)        (y[hiddenLayer - 1])
  # Encoder model
  encoder = Model(inputs=[x], outputs=[y[hiddenLayer]], name = "encoder")
  return encoder

def get_decoder(outputshape, dimensionEmbedding, hiddenLayer, denseUnit, activationFunction):
  """
  Args:
          outputshape:
          dimensionEmbedding: dimension of the embedding
          hiddenLayer: number of hidden layers in encoder/decoder
          denseUnit: dimension of dense layers, not including the units in the embedding layer
  """
  # Input
  y = Input(shape=(dimensionEmbedding,))
  # Decoder layers
  y_hat = [None] * (hiddenLayer + 1)
  y_hat[hiddenLayer] = y
  for i in range(hiddenLayer - 1, 0, -1):
      y_hat[i] = Dense(denseUnit, activation='relu')                   (y_hat[i + 1])
  y_hat[0] = Dense(units=outputshape, activation=activationFunction)                        (y_hat[1])
  # Output
  x_hat = y_hat[0]  # decoder's output is also the actual output
  # Decoder Model
  decoder = Model(inputs=[y], outputs=[x_hat], name = "decoder")
  return decoder

def get_autoencoder(shapeInputOutput, encoder, decoder):
  # Input
  x = Input(shape=(shapeInputOutput,))
  # Generate embedding
  y = encoder(x)
  # Generate reconstruction
  x_hat = decoder(y)
  # Autoencoder Model
  autoencoder = Model(inputs=[x], outputs=[x_hat])
  return autoencoder

"""# TRAINING E TEST STRUTTURALE"""

def CAGE_structural_train(codifica, dim_embedding, datatrain, datatest):
  structural_encoder = get_encoder(codifica, dim_embedding, 2, 256, 'sigmoid') #embedding 8
  structural_decoder = get_decoder(codifica, dim_embedding, 2, 256, 'sigmoid') #embedding 8 #3 128
  structural_autoencoder = get_autoencoder(codifica, structural_encoder, structural_decoder)

  #structural_encoder.summary()
  #structural_decoder.summary()
  #structural_autoencoder.summary()

  structural_autoencoder.compile(optimizer=Nadam(), loss=[MeanSquaredError()]) #https://keras.io/api/metrics/accuracy_metrics/#binaryaccuracy-class
  #https://keras.io/api/optimizers/Nadam/

  best_save = ModelCheckpoint('best_structural_autoencoder.hdf5', save_best_only=True, save_weights_only= False, monitor='val_loss', mode='min')

  structural_autoencoder.fit(x=datatrain, y = datatrain, validation_data = (datatest,datatest), epochs=50, shuffle=True, batch_size=32, callbacks=[best_save])
  structural_autoencoder.save('last_structural_autoencoder.hdf5')

def CAGE_structural_test(datatrain, datatest):
  best_model = load_model('best_structural_autoencoder.hdf5')
  last_model = load_model('last_structural_autoencoder.hdf5')

  best_structural_encoder = best_model.get_layer("encoder")
  last_structural_encoder = last_model.get_layer("encoder")

  best_structural_decoder = best_model.get_layer("decoder")
  last_structural_decoder = last_model.get_layer("decoder")

  best_model.evaluate(x = datatrain, y = datatrain)
  last_model.evaluate(x = datatrain, y = datatrain)

  best_model_results = best_model.evaluate(x = datatest, y = datatest)
  last_model_results = last_model.evaluate(x = datatest, y = datatest)

  if best_model_results < last_model_results:
    train_embeddings = best_structural_encoder.predict(x = datatrain, batch_size=128, verbose=1)
    test_embeddings = best_structural_encoder.predict(x = datatest, batch_size=128, verbose=1)
    structural_decoder = best_structural_decoder
  else:
    train_embeddings = last_structural_encoder.predict(x = datatrain, batch_size=128, verbose=1)
    test_embeddings = last_structural_encoder.predict(x = datatest, batch_size=128, verbose=1)
    structural_decoder = last_structural_decoder
  return train_embeddings, test_embeddings, structural_decoder

"""# TRAINING E TEST ATTRIBUTI

"""

def CAGE_attribute_train(codifica, dim_embedding, datatrain, datatest):
  attribute_encoder = get_encoder(codifica,dim_embedding,2,256, 'sigmoid')
  attribute_decoder = get_decoder(codifica,dim_embedding,2,256, 'sigmoid')
  attribute_autoencoder = get_autoencoder(codifica,attribute_encoder, attribute_decoder)

  attribute_autoencoder.compile(optimizer=Nadam(), loss=[MeanSquaredError()])

  best_save_finale = ModelCheckpoint('best_attribute_autoencoder.hdf5', save_best_only=True, save_weights_only= False, monitor='val_loss', mode='min')

  attribute_autoencoder.fit(x=datatrain, y = datatrain, validation_data = (datatest,datatest), epochs=50, shuffle=True, batch_size=32, callbacks=[best_save_finale])
  attribute_autoencoder.save('last_attribute_autoencoder.hdf5')

def CAGE_attribute_test(datatrain, datatest):
  best_attribute_autoencoder = load_model('best_attribute_autoencoder.hdf5')
  last_attribute_autoencoder = load_model('last_attribute_autoencoder.hdf5')

  best_attribute_encoder = best_attribute_autoencoder.get_layer("encoder")
  last_attribute_encoder = last_attribute_autoencoder.get_layer("encoder")

  best_attribute_decoder = best_attribute_autoencoder.get_layer("decoder")
  last_attribute_decoder = last_attribute_autoencoder.get_layer("decoder")

  best_attribute_autoencoder.evaluate(x = datatrain, y = datatrain)
  last_attribute_autoencoder.evaluate(x = datatrain, y = datatrain)

  best_model_results = best_attribute_autoencoder.evaluate(x = datatest, y = datatest)
  last_model_results = last_attribute_autoencoder.evaluate(x = datatest, y = datatest)

  if best_model_results < last_model_results:
    train_embeddings = best_attribute_encoder.predict(x = datatrain, batch_size=128, verbose=1)
    test_embeddings = best_attribute_encoder.predict(x = datatest, batch_size=128, verbose=1)
    attribute_decoder = best_attribute_encoder
  else:
    train_embeddings = last_attribute_encoder.predict(x = datatrain, batch_size=128, verbose=1)
    test_embeddings = last_attribute_encoder.predict(x = datatest, batch_size=128, verbose=1)
    attribute_decoder = last_attribute_decoder
  return train_embeddings, test_embeddings, attribute_decoder

"""# SAVE EMBEDDINGS"""

def save_embeddings(train_embeddings, test_embeddings, case_ids_train, case_ids_test, esistono_attributi, attribute):
  train_prediction = pd.DataFrame(train_embeddings)
  test_prediction = pd.DataFrame(test_embeddings)

  if not esistono_attributi:
    train_prediction["case_id"] = case_ids_train
    test_prediction["case_id"] = case_ids_test

  if attribute: #se devo salvare anche gli embedding degli attributi
    train_prediction["case_id"] = case_ids_train
    test_prediction["case_id"] = case_ids_test
    train_prediction.to_csv('train_attribute_embeddings.csv', index=False, header = False)
    test_prediction.to_csv('test_attribute_embeddings.csv', index=False, header = False)
  else:
    train_prediction.to_csv('train_embeddings.csv', index=False, header = False)
    test_prediction.to_csv('test_embeddings.csv', index=False, header = False)