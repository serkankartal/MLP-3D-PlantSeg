import os
import gc
import random
import numpy as np
import pandas as pd
import open3d as o3d
import keras
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# Konfigürasyon ayarları
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
columns = [0, 1, 2, 3, 4, 5]
min_layer_count = 2
max_layer_count = 3
neuron_count = 10
activation = "relu"
learning_rate = 0.001
patience = 2
max_iter = 10
plant_type = "all_species"

model_name = "mlp" + activation
MODEL_DIR = "./models/" + model_name
DATA_DIR = '../data/training_data/' + plant_type + "_txt/"


def getData(data_dir):
    source_directory_list = os.listdir(DATA_DIR + data_dir)
    data = None
    for file in source_directory_list:
        temp = pd.read_csv(DATA_DIR + data_dir+"/" + file,sep=" ").iloc[:,:6].to_numpy()
        if data is None:
            data = temp
        else:
            data = np.concatenate((data, temp), axis=0)
    return data

def getTrainTestValDataPerc():
    temp_plant=getData("/plant/")
    temp_plant[:,2]= temp_plant[:,2]+20
    temp_back=getData("/background/")
    temp_back[:,2]= temp_back[:,2] -20


    if temp_back.shape[0] > (temp_plant.shape[0] *5):
        print("more than 5 times")
        bacg_count = np.min([temp_back.shape[0], temp_plant.shape[0] ])*5
        temp_back = temp_back[np.random.choice(temp_back.shape[0], bacg_count), :]

    backgr_class=np.zeros((temp_back.shape[0],1))
    plant_class=np.ones((temp_plant.shape[0],1))
    train_X = np.concatenate(( temp_back, temp_plant), axis=0)
    train_y=np.concatenate((backgr_class,plant_class),axis=0)

    #Shuffle
    combined_data = np.column_stack((train_X, train_y.reshape(-1, 1)))
    np.random.shuffle(combined_data)
    shuffled_train_X = combined_data[:, :-1]
    shuffled_train_y = combined_data[:, -1]

    shuffled_train_X, shuffled_val_X,  shuffled_train_y,shuffled_val_y = train_test_split(shuffled_train_X, shuffled_train_y, test_size=0.4, random_state=1)
    shuffled_val_X,shuffled_test_X, shuffled_val_y,shuffled_test_y = train_test_split( shuffled_val_X, shuffled_val_y, test_size=0.5, random_state=1)

    return shuffled_train_X,shuffled_train_y,shuffled_val_X,shuffled_val_y,shuffled_test_X,shuffled_test_y
    # return shuffled_train_X,shuffled_train_y,shuffled_val_X,shuffled_val_y,shuffled_test_X,shuffled_test_y

def trainModel():
    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)
    train_X, train_y,val_X, val_y,test_X, test_y=getTrainTestValDataPerc()
    # train_X, train_y,val_X, val_y,test_X, test_y=getTrainTestValData()

    for layer_c in range(min_layer_count, max_layer_count):
        model = Sequential()
        parameter_path = model_name + "_day_len_"  + str(layer_c)  + '_' + str(neuron_count) + '_' + activation
        print("***********************")
        print(parameter_path)
        print("***********************")

        for k in range(layer_c):
            model.add(Dense(neuron_count, activation=activation, input_dim=train_X[:,columns].shape[1]))


        model.add(Dense(1,activation="sigmoid"))

        if not os.path.exists(MODEL_DIR + "/" + parameter_path):
            os.makedirs(MODEL_DIR + "/" + parameter_path)
        # model.add(Activation("softmax"))


        model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])

        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience),
            keras.callbacks.ModelCheckpoint(
                # filepath='../models/' + plant_type + '_NN_model_{epoch:02d}.h5',  # Include epoch in the filename
                filepath='../models/'+plant_type+'_NN_model.h5',
                save_weights_only=False,
            # keras.callbacks.ModelCheckpoint(filepath='./models/' + parameter_path + '.h5', save_weights_only=False,
                                                                            save_best_only=True, mode='min', monitor='val_loss', period=1),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.2, patience=1, verbose=1, min_lr=0.0000001
            ),
            keras.callbacks.CSVLogger('../logs/' + parameter_path + '.csv', append=True, separator=';'),
        ]

        model.fit(
            train_X[:,columns],
            train_y,
            validation_data=(val_X[:,columns], val_y),
            batch_size=32768,
            epochs=max_iter,
            verbose=1,
            callbacks=callbacks,
            shuffle=True  ##soradan ekledim testlerde yoktu
        )

def ModelTestonRawData(folder,epoch=""):

    if epoch=="":
        model = keras.models.load_model('../models/'+plant_type+'_NN_model.h5')
        folder_out = DATA_DIR + "/" + folder
    else:
        if not os.path.exists('../models/'+plant_type+'_NN_model_'+epoch+'.h5'):
            return
        model = keras.models.load_model('../models/'+plant_type+'_NN_model_'+epoch+'.h5')
        folder_out=DATA_DIR+"/"+folder+"_"+epoch

    raw_data_file=DATA_DIR+"/"+folder
    if not os.path.exists(folder_out+"_out/"):
        os.makedirs(folder_out+"_out/")
    source_directory_list = os.listdir(raw_data_file)
    for data_dir in source_directory_list:
        # data = pd.read_csv(DATA_DIR +raw_data_file+"/" + data_dir,sep=" ").iloc[:,:6].to_numpy()
        pcd = o3d.io.read_point_cloud(raw_data_file+"/" + data_dir)
        data= np.concatenate((np.asarray(pcd.points), np.asarray(pcd.colors) * 255), axis=1)
        Y = model.predict(data[:,columns])

        # pred_y=np.where(Y[:, 0] <= 0.5, 0, 1)
        condition = Y[:, 0] <= 0.5
        pred_y = np.zeros((len(Y),4))
        pred_y[condition] = [72, 60, 50,0]
        pred_y[~condition] = [34, 139, 24,1]

        # predicted_Data = np.concatenate((data,  pred_y.reshape(-1,1)), axis=1)
        predicted_Data = np.concatenate((data[:,:3],  pred_y), axis=1)

        if epoch == "":
            output_folder = raw_data_file+"_out/"+data_dir[:-4]
        else:
            output_folder=raw_data_file+ "_"+epoch+ "_out"+"/" + data_dir[:-4]
        np.savetxt(output_folder+"_predicted.txt",predicted_Data,fmt="%.2f")
        np.savetxt(output_folder+"_background.txt",data[condition],fmt="%.2f")
        np.savetxt(output_folder+"_plant.txt",data[~condition],fmt="%.2f")

        del predicted_Data
        del data
        gc.collect()
        # saveSegmentedDataNumpy(predicted_Data[~condition], predicted_Data[condition], DATA_DIR +raw_data_file+"/" + data_dir,data[~condition],"MLP")


def main():
    trainModel()
    # ModelTestonRawData("9001")
    # ModelTestonRawData("CSV22")
    ModelTestonRawData("Chakti")

if __name__ == "__main__":
    main()


