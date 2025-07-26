import os
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from RequiredFunctions import evaluateResults
import Preprocess_paris

# General Settings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Configurations
min_layer_count = 3
max_layer_count = 4
neuron_count = 30
learning_rate = 0.001
patience = 10
max_iter = 30
activation = "relu"

dataset = "paris"
model_name = f"{dataset}_mlp{activation}"
MODEL_DIR = f"../models/{model_name}"
DATA_DIR = f"../data/{dataset}_out/"
train_data_file = ["Lille1_1.txt"]
f_index = 3
perc = 0.20  # test oranı

# ------------------------------------------------------------------

def getTrainTestData(test_data_file="Lille1_2"):
    # read backgorund
    source_directory_list = os.listdir(DATA_DIR + "5_smooted")
    train_data = None
    X_train, X_test, y_train, y_test=None,None,None,None
    for file in source_directory_list:
        data = pd.read_csv(DATA_DIR + "5_smooted/" + file).to_numpy()

        # seperate plant and background data
        backgr_data = data[data[:, 4] == 1, :4]
        plant_data = data[data[:, 4] != 1, :4]

        backgr_class = np.ones((backgr_data.shape[0], 1))
        plant_class = np.zeros((plant_data.shape[0], 1))

        X = np.concatenate((backgr_data, plant_data), axis=0)
        Y = np.concatenate((backgr_class, plant_class), axis=0)
        X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X, Y, test_size=perc, random_state=42)

        if X_train is None:
            X_train, X_test, y_train, y_test =X_train_t, X_test_t, y_train_t, y_test_t
        else:
            X_train = np.concatenate((X_train, X_train_t), axis=0)
            X_test = np.concatenate((X_test, X_test_t), axis=0)
            y_train = np.concatenate((y_train, y_train_t), axis=0)
            y_test = np.concatenate((y_test, y_test_t), axis=0)
    return X_train, X_test, y_train, y_test

def trainModelWithOneFile():
    source_directory_list = os.listdir(DATA_DIR + "5_smooted")

    scaler=MinMaxScaler()
    X_train=None
    for file in source_directory_list:
        if  file  in train_data_file:
            X_temp = pd.read_csv(DATA_DIR + "5_smooted/" + file).to_numpy()

            y_temp= np.zeros((X_temp.shape[0], 1))
            y_temp[X_temp[:, 4] == 1] = 1
            X_temp = scaler.fit_transform(X_temp)
            if X_train is None:
                X_train=X_temp[:,f_index:-1]
                y_train=y_temp
            else:
                X_train = np.concatenate((X_train, X_temp[:,f_index:-1]), axis=0)
                y_train = np.concatenate((y_train, y_temp), axis=0)

    for layer_c in range(min_layer_count, max_layer_count):
        model = Sequential()
        parameter_path = model_name + "_day_len_"  + str(layer_c)  + '_' + str(neuron_count) + '_' + activation
        print("***********************")
        print(parameter_path)
        print("***********************")

        for k in range(layer_c):
            model.add(Dense(neuron_count, activation=activation, input_dim=X_train.shape[1]))


        model.add(Dense(1,activation="sigmoid"))

        if not os.path.exists(MODEL_DIR + "/" + parameter_path):
            os.makedirs(MODEL_DIR + "/" + parameter_path)
        # model.add(Activation("softmax"))


        model.compile(loss="binary_crossentropy", optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience),
            keras.callbacks.ModelCheckpoint(filepath='../models/'+dataset+'_NN_model_'+str(train_data_file)+'.h5', save_weights_only=False,
            # keras.callbacks.ModelCheckpoint(filepath='./models/' + parameter_path + '.h5', save_weights_only=False,
                                                                            save_best_only=True, mode='min', monitor='val_loss', period=3),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.2, patience=5, verbose=0, min_lr=0.0000001
            ),
            keras.callbacks.CSVLogger('../logs/' + parameter_path + '.csv', append=True, separator=';'),
        ]

        model.fit(
            X_train,
            y_train,
            batch_size=512,
            epochs=max_iter,
            verbose=2,
            validation_split=0.20,
            callbacks=callbacks,
            shuffle=True)  ##soradan ekledim testlerde yoktu

def trainModelWithPercData():

    X_train, X_test, y_train, y_test = getTrainTestData()

    for layer_c in range(min_layer_count, max_layer_count):
        model = Sequential()
        parameter_path = model_name + "_day_len_"  + str(layer_c)  + '_' + str(neuron_count) + '_' + activation
        print("***********************")
        print(parameter_path)
        print("***********************")

        for k in range(layer_c):
            model.add(Dense(neuron_count, activation=activation, input_dim=X_train.shape[1]))


        model.add(Dense(1,activation="sigmoid"))

        if not os.path.exists(MODEL_DIR + "/" + parameter_path):
            os.makedirs(MODEL_DIR + "/" + parameter_path)
        # model.add(Activation("softmax"))


        model.compile(loss="binary_crossentropy", optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience),
            keras.callbacks.ModelCheckpoint(filepath='../models/'+dataset+'_NN_model_'+str(perc*100)+'.h5', save_weights_only=False,
            # keras.callbacks.ModelCheckpoint(filepath='./models/' + parameter_path + '.h5', save_weights_only=False,
                                                                            save_best_only=True, mode='min', monitor='val_loss', period=3),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.2, patience=5, verbose=0, min_lr=0.0000001
            ),
            keras.callbacks.CSVLogger('../logs/' + parameter_path + '.csv', append=True, separator=';'),
        ]

        model.fit(
            X_train,
            y_train,
            batch_size=512,
            epochs=max_iter,
            verbose=2,
            validation_split=0.20,
            callbacks=callbacks,
            shuffle=True  ##soradan ekledim testlerde yoktu
        )

        # y_pred = model.predict(X_test)
        # plant_data = data[y_pred <=0.5, :4]
        # backgr_data_t = data[y_pred>=0.5, :4]
        #
        # accuracy, precision, recall, f1 = evaluateResults(y_test, y_pred)
        # if not os.path.exists(DATA_DIR+"train_test_result/"):
        #     os.makedirs(DATA_DIR+"train_test_result/")
        #
        # np.savetxt(DATA_DIR+"train_test_result/"+str(layer_c)+"_train_background.txt",backgr_data_t,fmt="%.1f")
        # np.savetxt(DATA_DIR+"train_test_result/"+str(layer_c)+"_train_plants.txt",plant_data,fmt="%.1f")
        #
        # with open(DATA_DIR+"train_test_result/"+dataset+"_evaluation_metrics.txt", "w") as file:
        #     file.write("Accuracy: {:.4f}\n".format(accuracy))
        #     file.write("Precision: {:.4f}\n".format(precision))
        #     file.write("Recall: {:.4f}\n".format(recall))
        #     file.write("F1 Score: {:.4f}\n".format(f1))

def modelTest():
    X_train, X_test, y_train, y_test = getTrainTestData()
    # data,X_train, X_test, y_train, y_test = getTrainTestDataFile()
    model = load_model('../models/' + dataset + '_NN_model_' + str(perc * 100) + '.h5')
    y_pred = model.predict(X_test)

    y_pred_binary = np.where(y_pred <= 0.5, 0, 1)
    plant_data = X_test[y_pred_binary[:, 0] == 0, :4]
    backgr_data_t = X_test[y_pred_binary[:, 0] != 0, :4]

    accuracy, precision, recall, f1 = evaluateResults(y_test, y_pred_binary)
    print(
        "accuracy:" + str(accuracy) + " precision:" + str(precision) + " recall:" + str(recall) + " f1:" + str(f1))
    if not os.path.exists(DATA_DIR + "ML_result_"+ str(perc * 100) +"/"):
        os.makedirs(DATA_DIR + "ML_result_"+ str(perc * 100) +"/")

    np.savetxt(DATA_DIR + "ML_result_"+ str(perc * 100) +"/train_background.txt", backgr_data_t, fmt="%.1f")
    np.savetxt(DATA_DIR + "ML_result_"+ str(perc * 100) +"/train_plants.txt", plant_data, fmt="%.1f")

    # print("-----" + file + "-----")
    Rotate_and_merge_paris.write2Excel(DATA_DIR + "ML_result_"+ str(perc * 100) +"/evaluation_metrics.xlsx", "all files", accuracy,
                                       precision, recall, f1)

def modelTestWithOneFile():
    X_test=None
    y_test=None
    excel_data=None
    source_directory_list = os.listdir(DATA_DIR + "5_smooted")

    model = load_model('../models/' + dataset + '_NN_model_'+str(train_data_file)+'.h5')

    scaler = MinMaxScaler()
    # for file in source_directory_list:
    #     if  file in train_data_file:
    #         X_temp = pd.read_csv(DATA_DIR + "5_smooted/" + file).to_numpy()
    #         X_temp = scaler.fit_transform(X_temp)

    for file in source_directory_list:
        X_temp= pd.read_csv(DATA_DIR + "5_smooted/" + file).to_numpy()

        y_test = np.zeros((X_temp.shape[0], 1))
        y_test[X_temp[:, 4] == 1] = 1
        X_temp = scaler.fit_transform(X_temp)
        X_test=X_temp[:,f_index:-1]

        y_pred = model.predict(X_test)
        y_pred_binary = np.where(y_pred <= 0.5, 0, 1)
        plant_data = X_test[y_pred_binary[:, 0] == 0, :4]
        backgr_data_t = X_test[y_pred_binary[:, 0] != 0, :4]

        accuracy, precision, recall, f1 = evaluateResults(y_test, y_pred_binary)
        print("-----" + file + "-----")
        print("accuracy:" + str(accuracy) + " precision:" + str(precision) + " recall:" + str(recall) + " f1:" + str(f1))
        if not os.path.exists(DATA_DIR + "ML_result_"+ str(train_data_file) +"/"):
            os.makedirs(DATA_DIR + "ML_result_"+ str(train_data_file) +"/")

        np.savetxt(DATA_DIR + "ML_result_"+str(train_data_file)+"/train_background_"+file+".txt", backgr_data_t, fmt="%.1f")
        np.savetxt(DATA_DIR + "ML_result_"+str(train_data_file)+"/train_plants_"+file+".txt", plant_data, fmt="%.1f")
        # Create a new DataFrame with the new data
        new_data = pd.DataFrame({
            "Accuracy": [accuracy],
            "Precision": [precision],
            "Recall": [recall],
            "F1": [f1]
        })

        # Append the new data to the existing DataFrame
        if excel_data is None:
            excel_data=new_data
        else:
            excel_data = pd.concat([excel_data, new_data], ignore_index=True)

    # Rotate_and_merge_paris.write2Excel(DATA_DIR + "ML_result_"+ str(train_data_file) +"/evaluation_metrics.xlsx", "all files", accuracy,  precision, recall, f1)
    excel_data.to_excel(DATA_DIR + "ML_result_" + str(train_data_file) + "/evaluation_metrics.xlsx", sheet_name="all files", index=False)

if __name__ == "__main__":
    # trainModelWithPercData()
    # modelTest()
    trainModelWithOneFile()
    modelTestWithOneFile()