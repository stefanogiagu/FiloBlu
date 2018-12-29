from __future__ import absolute_import
# 
# SAna_CNN  1D CNN for Sentiment Analysis (with word embedding)
# 
# NOTE: assume TensorFLow backend for keras
#
# -*- coding: utf-8 -*-
__author__ = 'Stefano Giagu'
__email__ = 'stefano.giagu@roma1.infn.it'
__version__ = '1.0.0'
__status__ = 'prod'

# Imports:
import numpy as np
import tensorflow as tf
import keras
print ("TensorFlow version: " + tf.__version__)
print ("Keras version: " + keras.__version__)
import keras.backend as K

from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, InputSpec, Dense, Conv1D, Dropout, Activation, BatchNormalization, Flatten, MaxPooling1D, LeakyReLU, Concatenate, concatenate, Embedding, GlobalMaxPooling1D
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from keras.datasets import imdb
from keras.preprocessing import sequence

from sklearn.model_selection import KFold, RepeatedKFold

#from batch_renorm import BatchRenormalization

SEED = 1234
np.random.seed(SEED)  # for reproducibility

# set memory usage 'on demand'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))

# User Settable Parameters
VERBOSE = 1 #0 do not print during training

# learning rate decay parameters (if LR_ST=LR_END not used)
LR_ST = 1e-4
LR_END = 1e-4

#number of epochs
EPOCHS   = 10
# batch size (number of traingin events after each weight update)
BATCH_SIZE = 128 
#fraction of events used for validation
VALI_EVENTS_FRACTION = 0.2

# k-Fold cross validation splits
KFOLD = False
#KFOLD = True
KF_NSPL = 10
KF_NREP = 0
#KF_NREP = 10

# sample parameters
USE_IMDB = True #True to use IMDB based example, False to use user input data
#USE_IMDB = False
SAMP_DIM = 0 #sample size: updated when reading input
MAX_WORDS = 10000 #max number of significative words considered
MAX_LENGTH = 500  #max lenght of the msg (in number of words)
EMBEDDING_SIZE = 128 #Dimension of the dense embedding

IS_FIRST = True

#from Chollet
def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results

# Load text data (already vectorized in vectors of integers: each integer corresponding to a specific word and 
# 0 used for any unknown word), labels are sentiment wrt the given text and are encoded as 0 or 1.
# Encode vectors in one-hot
#
# NOTE: assumes the input vectors contain only the first most frequent MAX_WORDS (ID=3 means 3rd most frequent 
#       word (done at the level of the preparartyion of the Italian dictonary-ID file)
# 
# NOTE: this implementation uses word embedding layer (ToDo: use pretrained word embedding) to map words into 
#       the geometric space used by the CNN
def load_data(inp_file='db_data.npz'):
    global SAMP_DIM

    if USE_IMDB == True:
       # TEST
       from keras.datasets import imdb
       (msgs, label), (test_data, test_labels) = imdb.load_data(num_words=MAX_WORDS)
    else:
       #
       # DATASET
       #
       npzf = np.load(inp_file)
       npzf.files
       msgs_uncut = npzf['msgs']
       labels = npzf['labels']
       msgs = [[w for w in x if w<=MAX_WORDS] for x in msgs_uncut]

    #label = np.asarray(labels).astype('float32') 

    # transform data in input pads (samples x time) for word embedding
    data = sequence.pad_sequences(msgs, maxlen=MAX_LENGTH)

    SAMP_DIM = msgs.shape[0] 

    print(msgs.shape, 'msgs sample')
    print(label.shape, 'label sample')
    print(data.shape, 'data sample')

    folds = []

    if KFOLD == False:
        print('Using manual verification training/validation with VALI_EVENTS_FRACTION: ', VALI_EVENTS_FRACTION)
        vt = []
        vv = []
        n_vali = int(SAMP_DIM*VALI_EVENTS_FRACTION) 
        v_tmp = np.arange(SAMP_DIM) #randomize data
        np.random.shuffle(v_tmp)
        for idt in range(SAMP_DIM):
            if idt >= n_vali:
               	vt.append(v_tmp[idt])
            else:
               	vv.append(v_tmp[idt])
        folds.append((vt,vv))
    else:
        if KF_NREP == 0:
           print('Using k-Fold x-validation with K: ', KF_NSPL)
           kfold = KFold(n_splits=KF_NSPL, shuffle=True, random_state=SEED)
        else:
           print('Using k-Fold repeated x-validation with K/REP: ', KF_NSPL, ' / ', KF_NREP)
           kfold = RepeatedKFold(n_splits=KF_NSPL, n_repeats=KF_NREP , random_state=SEED)
        folds = list(kfold.split(data,label))

    return folds, data, label

def model():
    #
    # NNET Architecture
    # 

    #sequential (aka Feed-Forward Convolutional Neural Network) (functional approach)

    Input_txt = Input(shape=(MAX_LENGTH,), name='input_txt')

    # simple multilayer dense architecture

    channel_axis = -1

    # embedding layer
    x = Embedding(MAX_WORDS, EMBEDDING_SIZE, input_length=MAX_LENGTH, name='embedding')(Input_txt)

    # conv1D_1
    x = Conv1D(32, 7, name='conv_1')(x)
    x = Activation('relu', name='activation_1')(x)

    # maxpooling_1
    x = MaxPooling1D(5, name='maxpooling_1')(x)

    # conv1D_2
    x = Conv1D(32, 7, name='conv_2')(x)
    x = Activation('relu', name='activation_2')(x)

    # global_maxpooling_2
    x = GlobalMaxPooling1D(name='global_maxpooling_2')(x)

    # dense_3 (output layer)
    out = Dense(1, name='dense_3')(x)

    # model:
    nnet = Model(inputs=[Input_txt], outputs=out)

    #
    # Model Compile
    #

    # Optimizer: Adam with decaying learnign rate
    #OPTIMIZER = SGD(lr=LR_ST)
    OPTIMIZER = RMSprop(lr=LR_ST)
    #OPTIMIZER = Adam(lr=LR_ST)

    nnet.compile(optimizer=OPTIMIZER, loss='binary_crossentropy', metrics=['accuracy'])

    return nnet


def get_callbacks(w_name):
    
    # learning rate scheduler
    LR_SCHEDULER = LearningRateScheduler(lambda epoch: LR_ST * (LR_END/LR_ST)** (epoch/EPOCHS))

    WGHT_SAVE = ModelCheckpoint(w_name, save_best_only=True, verbose=VERBOSE, monitor='val_loss', mode='min')

    if LR_ST == LR_END:
       return [WGHT_SAVE]
    else:
       return [LR_SCHEDULER, WGHT_SAVE]


def train_model(train_data, vali_data, kindex=0):
    global IS_FIRST

    # create model
    nnet = model()
    if IS_FIRST == True:
       nnet.summary()
    IS_FIRST = False
    

    # Callbacks
    w_name = "SAna_CNN_trained_" + str(kindex) + "_weights.h5"
    callbacks = get_callbacks(w_name)

    # training
    print(train_data[0].shape)
    print(train_data[1].shape)
    print(vali_data[0].shape)
    print(vali_data[1].shape)
    history = nnet.fit(train_data[0], train_data[1], 
                       batch_size=BATCH_SIZE, epochs=EPOCHS,
                       validation_data=(vali_data[0], vali_data[1]),
                       verbose=VERBOSE,
                       callbacks=callbacks)
    score = nnet.evaluate(vali_data[0], vali_data[1], verbose=VERBOSE)

    return history, score

def run_training():

    # Run Training 
    folds, data, label = load_data()
    
    all_loss_score = []
    all_metric_score = []
    all_loss_err = []
    all_metric_err = []

    for j, (vt, vv) in enumerate(folds):

       print("k-Fold: ", j)

       train_data = []
       vali_data = []
       test_data = []

       t_data = data[vt]
       train_label = label[vt]
       train_data.append(t_data)
       train_data.append(train_label)

       v_data = data[vv]
       vali_label = label[vv]
       vali_data.append(v_data)
       vali_data.append(vali_label)

       test_data.append(data)
       test_data.append(label)

       history, score = train_model(train_data, vali_data, kindex=j)

       print('Test loss:', score[0])
       print('Test metric:', score[1])

       all_loss_score.append(score[0])
       all_metric_score.append(score[1])

       loss_history = history.history['val_loss']
       metric_history = history.history['val_acc']   

       all_loss_err.append(loss_history)
       all_metric_err.append(metric_history)


    # Average's
    average_loss = [np.mean([x[i] for x in all_loss_err]) for i in range(EPOCHS)]
    average_metric = [np.mean([x[i] for x in all_metric_err]) for i in range(EPOCHS)]

    print('Loss average score: ', np.mean(all_loss_score))
    print('Metric average score: ', np.mean(all_metric_score))

    import matplotlib.pyplot as plt
    plt.plot(range(1, len(average_loss) + 1), average_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.savefig("SAna_CNN_val_loss.pdf")
    plt.clf()
    plt.plot(range(1, len(average_metric) + 1), average_metric)
    plt.xlabel('Epochs')
    plt.ylabel('Validation Metric')
    plt.savefig("SAna_CNN_val_metric.pdf")
   

# test
run_training()
