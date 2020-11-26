import sys
import os
import pandas as pd
import numpy as np
import json
import shutil
from keras.preprocessing.image import ImageDataGenerator,  load_img, img_to_array
from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import time
import datetime
import csv
import subprocess


start_time = time.time()
datetimeObj = datetime.datetime.now()
timestampStr = datetimeObj.strftime("%Y%m%d-%H%M%S")
datestampStr = datetimeObj.strftime("%Y-%m-%d")

def exec_time(start):
      end = time.time()
      dur = end-start

      if dur<60:
          print("Execution Time:",dur,"seconds")
      elif dur>60 and dur<3600:
          dur=dur/60
          print("Execution Time:",dur,"minutes")
      else:
          dur=dur/(60*60)
          print("Execution Time:",dur,"hours")

datetimeObj = datetime.datetime.now()
timestampStr = datetimeObj.strftime("%Y%m%d-%H%M%S")

img_width, img_height = 150, 150

def get_network_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    network_name = config["network"]
    network_config = config[network_name]
    return network_name, network_config

def get_network(config_path):
    network_name, network_config = get_network_config(config_path)

    return network_name, network_config

class Classifier:

  def __init__(self, config_path):
        self.config = config_path
        class_file = self.config["classes"]
        with open(class_file) as file:
            csv_reader = csv.reader(file, delimiter=',')
            self.labels_to_names = {}
            self.names_to_labels = {}
            for line, row in enumerate(csv_reader):
                class_name, class_id = row
                self.labels_to_names[int(class_id)] = class_name
                self.names_to_labels[class_name] = int(class_id)


  def train(self, development):

    if development:
      epochs = 2
    else:
      epochs = self.config['epochs']

    train_data_path = self.config["annotations"]
    print('TRAIN DATA', train_data_path)
    test_data_path = self.config["valAnnotations"]

    train_data = pd.read_csv(train_data_path, header=None, names=['filename','class'])
    test_data = pd.read_csv(test_data_path, header=None, names=['filename','class'])

    num_train_samples = len(train_data)
    num_test_samples = len(test_data)

    """
    Read parameters from config
    """
    batch_size = self.config['batchSize']
    samples_per_epoch = self.config['samples_per_epoch']
    train_steps = num_train_samples // batch_size
    test_steps = num_test_samples // batch_size
    nb_filters1 = self.config['nb_filters1']
    nb_filters2 = self.config['nb_filters2']
    conv1_size = self.config['conv1_size']
    conv2_size = self.config['conv2_size']
    pool_size = self.config['pool_size']
    classes_num = len(self.labels_to_names)
    if classes_num > 1:
      lr = self.config['lr']
      print(classes_num, 'classes found in train data of size', train_data.iloc[:,1].unique())

      model = Sequential()
      model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, border_mode ="same", input_shape=(img_width, img_height, 3)))
      model.add(Activation("relu"))
      model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

      #model.add(Convolution2D(nb_filters2, conv2_size, conv2_size, border_mode ="same"))
      #model.add(Activation("relu"))
      #model.add(MaxPooling2D(pool_size=(pool_size, pool_size), dim_ordering='th'))

      model.add(Flatten())
      model.add(Dense(256))
      model.add(Activation("relu"))
      model.add(Dropout(0.5))
      model.add(Dense(classes_num, activation='softmax'))

      print('MODEL SUMMARY:')
      model.summary()

      model.compile(loss='categorical_crossentropy',
                    optimizer=optimizers.RMSprop(lr=lr),
                    metrics=['accuracy'])

      train_datagen = ImageDataGenerator(
          rescale=1. / 255,
          shear_range=0.2,
          zoom_range=0.2,
          horizontal_flip=True)

      test_datagen = ImageDataGenerator(rescale=1. / 255)

      train_generator = train_datagen.flow_from_dataframe(
          train_data,
          directory=None,
          x_col="filename",
          y_col="class",
          target_size=(img_height, img_width),
          batch_size=batch_size,
          class_mode='categorical')

      test_generator = test_datagen.flow_from_dataframe(
          test_data,
          x_col="filename",
          y_col="class",
          target_size=(img_height, img_width),
          batch_size=batch_size,
          class_mode='categorical')

      print('TRAINING CLASSES:',train_generator.class_indices.keys())
      print('TESTING CLASSES:',train_generator.class_indices.keys())

      """
      Tensorboard log
      """
      target_dir = self.config['targetDir']
      log_dir = self.config['logDir']
      tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
      early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=100, mode='min', verbose=1)
      checkpoint = ModelCheckpoint(target_dir + os.sep + 'model-{epoch:03d}.h5', verbose=1, \
        monitor='accuracy',save_best_only=True, mode='max')
      cbks = [tb_cb,early_stop,checkpoint]

      history = model.fit_generator(
          train_generator,
          samples_per_epoch=samples_per_epoch,
          steps_per_epoch=train_steps,
          epochs=epochs,
          validation_data=test_generator,
          #validation_data=bm_generator,
          #validation_data=[test_generator,bm_generator],
          callbacks=cbks,
          validation_steps=test_steps,
          workers=1,
          use_multiprocessing=False)

      if not os.path.exists(target_dir):
        os.mkdir(target_dir)
      model_json = model.to_json()
      with open(target_dir + os.sep + "model.json", "w") as json_file:
        json_file.write(model_json)
      model.save(target_dir + os.sep +'model_complete.h5')
      model.save_weights(target_dir + os.sep + 'weights_complete.h5')

      # summarize history for accuracy
      plt.plot(history.history['accuracy'])
      plt.plot(history.history['val_accuracy'])
      plt.title('model accuracy')
      plt.ylabel('accuracy')
      plt.xlabel('epoch')
      plt.legend(['train', 'test'], loc='upper left')
      plt.savefig(target_dir + os.sep + 'model_accuracy.png')
      plt.close()

      # summarize history for loss
      plt.plot(history.history['loss'])
      plt.plot(history.history['val_loss'])
      plt.title('model loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['train', 'test'], loc='upper left')
      plt.savefig(target_dir + os.sep + 'model_loss.png')
      plt.close()

      """
      Generate RaSpect_class_progr.json
      """
      progr_fields = {
          "git_hash":subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
          "date":datestampStr,
          "training_time":(time.time() - start_time)/3600
          }
      with open(os.path.dirname(self.config['annotations']) + os.sep + 'class_progr.json', "w") as outfile:
          json.dump(progr_fields, outfile, indent=2)
      print('class_progr printed at : {}'.format(os.path.dirname(self.config['annotations']) + os.sep + 'class_progr.json'))
      print("-------------------------------------")
      print('Training Time: ', (time.time() - start_time)/3600)

    else:
      print('There is only one class in your training data!')


  def evaluate(self, model_path, model_weights_path):
    model = load_model(model_path)
    model.load_weights(model_weights_path)

    val_path = self.config["valAnnotations"]
    acc_label = 'test-'
    acc_file = acc_label + 'accuracy-' + timestampStr + ".txt"

    val_data = pd.read_csv(val_path, header=None, names=['filename','class'])
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    val_generator = val_datagen.flow_from_dataframe(
      val_data,
      directory=None,
      x_col="filename",
      y_col="class",
      target_size=(img_height, img_width),
      batch_size=self.config['batchSize'],
      class_mode='categorical')

    results = model.evaluate_generator(val_generator)
    with open(os.path.join(os.path.dirname(os.path.dirname(model_path)),acc_file), "w") as outfile:
      writer = csv.writer(outfile)
      outfile.write('Training annotations: ')
      outfile.write(self.config["annotations"])
      outfile.write('\n')
      outfile.write('Ground truth annotations: ')
      outfile.write(val_path)
      outfile.write('\n')
      print('Accuracy on', len(val_data),'validation data from', acc_label[:-1], \
        'dataset is:', (results[1].round(2)))
      outfile.write('{:.0f} images with accuracy '.format(len(val_data)))
      outfile.write(str(results[1].round(2)))

      # update RaSpect_progr.json
    with open(os.path.dirname(self.config['annotations']) + os.sep + 'RaSpect_class_progr.json', "r+") as read_file:
      progr_fields = json.load(read_file)
      progr_fields.update({'model_path': model_path})
      progr_fields.update({acc_label[:-1] + '_accuracy_file': os.path.join(acc_label + 'accuracy-' + timestampStr + ".txt")})
      with open(os.path.dirname(self.config['annotations']) + os.sep + 'RaSpect_class_progr.json', "w") as write_file:
          json.dump(progr_fields, write_file, indent=2)
      print('------------------------------------------------------------------------------------')
      print('Updated RaSpect_class_progr.json at : {}'.format(os.path.dirname(self.config['annotations']) + os.sep + 'RaSpect_progr.json'))


  def predict_class(self, model_path, model_weights_path, test_dir):
    model = load_model(model_path)
    model.load_weights(model_weights_path)
    class_file = self.config["classes"]
    print(class_file)

    def predict_file(file):
      x = load_img(file, target_size=(img_width,img_height))
      x = img_to_array(x)
      x = np.expand_dims(x, axis=0)
      array = model.predict(x)
      result = array[0]
      print(result)
      answer = np.argmax(result)
      print(answer)

      return answer

    res_filenames = []
    res_classes = []
    for i, ret in enumerate(os.walk(test_dir)):
      for i, filename in enumerate(ret[2]):
        if filename.startswith("."):
          continue
        elif filename.endswith(".JPG") or filename.endswith(".jpg"):
          res_filename = ret[0] + '/' + filename
          answer = predict_file(ret[0] + '/' + filename)
          res_class = self.labels_to_names[int(answer)]
          print(res_filename, '-->', res_class)
          res_filenames.append(res_filename)
          res_classes.append(res_class)

    results = pd.DataFrame({"filename":res_filenames,\
                            "classes":res_classes})

    results.to_csv(test_dir + os.sep + 'preds_bt.csv', index=False, header=False)
    print('Results are printed at', test_dir + os.sep + 'preds.csv')

    exec_time(start_time)
