"""
Yeni eklemeler: loss function ve metric tanımlandı r_squared hem loss function hem de metrik içerisinde kullanıldı.
"""

import os,sys,fnmatch,re
import math
import pandas as pd
import json
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.optimizers.legacy import Adam
from keras import backend as K
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Dense, Activation,Dropout,Conv2D,Conv1D,MaxPooling1D,MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from tensorflow.keras.metrics import MeanSquaredError,RootMeanSquaredError,MeanAbsoluteError,MeanAbsolutePercentageError
from keras.callbacks import CSVLogger
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def r_squared(y_true, y_pred):
    residual = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred))) # gerçek değerler ile tahmin edilen değerlerin farklarının karesi
    total = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true)))) #gerçek değerlerin ortalamadan uzaklığı
    r2_score = 1 - tf.divide(residual, total + K.epsilon()) # 1'e ne kadar yakın olursa o kadar iyi
    return r2_score

### yukarıdaki fonksiyon hem loss'ta hem de metrics'te kullanılabilir.
np.random.seed(42)
ProjName = "test_23"
file_path = 'data/THDMLS.csv'

datafile = pd.read_csv(file_path)

labels = ['Lambda1Input', 'Lambda2Input', 'Lambda3Input', 'Lambda4Input', 'Lambda5Input', 'M12input', 'TanBeta', 'betaH', 'alphaH', 'mhh_1', 'mhh_2', 'mAh_2', 'mHm_2', 'xs_gg13_h1', 'xs_gg13_h2', 'xs_gg13_Ah2', 'B_XsGamma', 'B0s_mumu']

datafile = datafile[labels]

dataframe = pd.DataFrame(datafile)

X_input = dataframe.drop({'betaH', 'alphaH', 'mhh_1', 'mhh_2', 'mAh_2', 'mHm_2', 'xs_gg13_h1', 'xs_gg13_h2', 'xs_gg13_Ah2', 'B_XsGamma', 'B0s_mumu'}, axis=1)

Y_output = dataframe['mhh_1']

X_train, X_test, y_train, y_test = train_test_split(X_input, Y_output, test_size=0.2, random_state=42)

def build_model(hp):
    model = tf.keras.Sequential()
    model.add(Dense(units=7))
    lays = hp.Int('num_layers', 1, 11)
    for l in range(1, lays+1):
        units = hp.Int('units_'+str(l), min_value=32, max_value=512, step=32)
        activation = hp.Choice("activation_" + str(l), ["relu", "tanh", "linear"])
        model.add(Dense(units=units, activation=activation, name='Dense_'+str(l)))
        npdr=np.random.randint(0,2)
        dr=hp.Int('droprate_'+str(l),min_value=npdr,max_value=npdr)
        npbn=np.random.randint(0,2)
        bn=hp.Int('bn_'+str(l),min_value=npbn,max_value=npbn)
        if dr == 1:
            fdrop=np.random.uniform(0.01,0.99)
            drate=hp.Float('dropout_'+str(l),min_value=fdrop,max_value=fdrop,step=0.1)
            model.add(Dropout(drate,name='Dropout_'+str(l)))
        if bn == 1:
            model.add(BatchNormalization(momentum=0.9))
    model.add(Dense(1, kernel_initializer='normal'))
    optimizer_choice = hp.Choice('optimizer', ['adam', 'adamax', 'sgd'])
    if optimizer_choice == 'adam':
        optimizer = tf.keras.optimizers.legacy.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5]))
    elif optimizer_choice == 'adamax':
        optimizer = tf.keras.optimizers.legacy.Adamax(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5]))
    elif optimizer_choice == 'sgd':
        optimizer = tf.keras.optimizers.legacy.SGD(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5]))
    else:
        optimizer = tf.keras.optimizers.legacy.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5]))
    model.compile(optimizer=optimizer, loss=r_squared, metrics=['mae', 'mse',r_squared])
    return model

maxtr=np.random.randint(1,2)
sd=np.random.randint(1,1000000)
epch=np.random.randint(50,250)
bs=np.random.randint(32,1000)
tuner = kt.RandomSearch(
    build_model,
    objective='val_mae',
    max_trials=maxtr,
    seed=sd,
    overwrite=True,
    directory="test_23",
    project_name="test_23"
)

tuner.search(X_train, y_train, epochs=epch, validation_data=(X_test,y_test))
best_hp = pd.DataFrame([tuner.get_best_hyperparameters(num_trials=1)[0].values])
best_hp.to_csv("ML_"+ProjName+'_config.csv',index=False)
best_hp_nec=best_hp[["num_layers","learning_rate"]]
bms=tuner.get_best_models()[0]
bms.build(X_train.shape)
csv_logger = CSVLogger("ML_"+ProjName+"_history_log.csv", append=False)
history = bms.fit(X_train, y_train, batch_size=bs, epochs=epch, verbose=2, validation_data=(X_test,y_test),callbacks=[csv_logger])

dftrain=pd.read_csv("ML_"+ProjName+"_history_log.csv")
TLoss=dftrain["loss"][len(dftrain)-1]
TValLoss=dftrain["val_loss"][len(dftrain)-1]

pred = bms.predict(X_test)
score=r2_score(y_test,pred.T[0])

score1=MeanSquaredError()
score1.update_state(y_test,pred.T[0])
score1=score1.result().numpy()

score2=RootMeanSquaredError()
score2.update_state(y_test,pred.T[0])
score2=score2.result().numpy()

score3=MeanAbsoluteError()
score3.update_state(y_test,pred.T[0])
score3=score3.result().numpy()

score4=MeanAbsolutePercentageError()
score4.update_state(y_test,pred.T[0])
score4=score4.result().numpy()

best_hp_nec.insert(loc=len(best_hp_nec.columns.values),column="Epochs",value=[epch])
best_hp_nec.insert(loc=len(best_hp_nec.columns.values),column="Batch_Size",value=[bs])
best_hp_nec.insert(loc=len(best_hp_nec.columns.values),column="Max_Trial",value=[maxtr])
best_hp_nec.insert(loc=len(best_hp_nec.columns.values),column="Seed",value=[sd])
best_hp_nec.insert(loc=len(best_hp_nec.columns.values),column="Loss",value=[TLoss])
best_hp_nec.insert(loc=len(best_hp_nec.columns.values),column="ValLoss",value=[TValLoss])
best_hp_nec.insert(loc=len(best_hp_nec.columns.values),column="R2_Score",value=[score])
best_hp_nec.insert(loc=len(best_hp_nec.columns.values),column="MSE",value=[score1])
best_hp_nec.insert(loc=len(best_hp_nec.columns.values),column="RMSE",value=[score2])
best_hp_nec.insert(loc=len(best_hp_nec.columns.values),column="MAE",value=[score3])
best_hp_nec.insert(loc=len(best_hp_nec.columns.values),column="MAPE",value=[score4])

best_hp_nec.to_csv("ML_"+ProjName+'.csv',index=False)

with open("ML_"+ProjName+'_summary.txt', 'w') as f:
    bms.summary(print_fn=lambda x: f.write(x + '\n'))

bms.save("ML_"+ProjName+"_weights.h5")

