import keras.models
import keras
from tensorflow.keras.layers import Dense
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import random
from sklearn.model_selection import train_test_split
import keras_tuner as kt
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adadelta
from RequiredFunctionPhenospex import *

import matplotlib.pyplot as plt
import seaborn as sns
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from sklearn.preprocessing import MinMaxScaler, StandardScaler

strategy = tf.distribute.MirroredStrategy()
import gc

columns=[0,1,2,3,4,5]
min_layer_count=2
max_layer_count=3
neuron_count=10
model_name="MLP"
learning_rate=0.001
patience=2
max_iter=10

activation="relu"
model_name="mlp"+activation #svm ,tree,knn
MODEL_DIR="./models/"+model_name
config="config"
#vijay Chickpea, 9001 Mungbean, Chackti Pearl millet, CSV22 Sorghum  sorghum_pearmillet
plant_type="all_species_ply"

version=4
DATA_DIR = '../data/training_data/'+plant_type+"/"

def getDataTxt(data_dir):
    source_directory_list = os.listdir(DATA_DIR + data_dir)
    data = None
    for file in source_directory_list:
        temp = pd.read_csv(DATA_DIR + data_dir+"/" + file,sep=" ").iloc[:,:6].to_numpy()
        if data is None:
            data = temp
        else:
            data = np.concatenate((data, temp), axis=0)
    return data


def getDataPly(data_dir):
    source_directory_list = os.listdir(DATA_DIR + data_dir)
    data = None
    for file in source_directory_list:
        # PLY file loading using PlyFile
        ply = PlyFile()
        ply.load(DATA_DIR + data_dir + "/" + file)

        if version == 1:
            # Only RGB channels
            temp = np.concatenate((
                ply.vertices["red"].reshape(-1, 1),
                ply.vertices["green"].reshape(-1, 1),
                ply.vertices["blue"].reshape(-1, 1)
            ), axis=1)

        elif version == 2:
            # XYZ and RGB channels, filtering z < 200
            temp = np.concatenate((
                ply.vertices["x"].reshape(-1, 1),
                ply.vertices["y"].reshape(-1, 1),
                ply.vertices["z"].reshape(-1, 1),
                ply.vertices["red"].reshape(-1, 1),
                ply.vertices["green"].reshape(-1, 1),
                ply.vertices["blue"].reshape(-1, 1)
            ), axis=1)

        elif version==4:
            # XYZ, RGB and NIR channels, filtering z < 200
            temp = np.concatenate((
                ply.vertices["red"].reshape(-1, 1),
                ply.vertices["green"].reshape(-1, 1),
                ply.vertices["blue"].reshape(-1, 1),
                ply.vertices["nir16"].reshape(-1, 1),
                ply.vertices["nir940"].reshape(-1, 1)
            ), axis=1)

        else:
            # XYZ, RGB and NIR channels, filtering z < 200
            temp = np.concatenate((
                ply.vertices["x"].reshape(-1, 1),
                ply.vertices["y"].reshape(-1, 1),
                ply.vertices["z"].reshape(-1, 1),
                ply.vertices["red"].reshape(-1, 1),
                ply.vertices["green"].reshape(-1, 1),
                ply.vertices["blue"].reshape(-1, 1),
                ply.vertices["nir16"].reshape(-1, 1),
                ply.vertices["nir940"].reshape(-1, 1)
            ), axis=1)

        # Concatenate the current file's data with the overall data
        if data is None:
            data = temp
        else:
            data = np.concatenate((data, temp), axis=0)

    return data

def getTrainTestValDataPerc():
    temp_plant=getDataPly("/plant/")
    temp_plant[:,2]= temp_plant[:,2]+20
    temp_back=getDataPly("/background/")
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
            model.add(Dense(neuron_count, activation=activation, input_dim=train_X.shape[1]))


        model.add(Dense(1,activation="sigmoid"))

        if not os.path.exists(MODEL_DIR + "/" + parameter_path):
            os.makedirs(MODEL_DIR + "/" + parameter_path)
        # model.add(Activation("softmax"))


        model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])

        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience),
            keras.callbacks.ModelCheckpoint(
                filepath='../models/' + plant_type + '_NN_model_'+str(version)+'_{epoch:02d}.h5',  # Include epoch in the filename
                # filepath='../models/'+plant_type+'_NN_model.h5',
                save_weights_only=False,
            # keras.callbacks.ModelCheckpoint(filepath='./models/' + parameter_path + '.h5', save_weights_only=False,
                                                                            save_best_only=True, mode='min', monitor='val_loss', period=1),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.2, patience=1, verbose=1, min_lr=0.0000001
            ),
            keras.callbacks.CSVLogger('../logs/' + parameter_path + '.csv', append=True, separator=';'),
        ]

        model.fit(
            train_X,
            train_y,
            validation_data=(val_X, val_y),
            batch_size=32768,
            epochs=max_iter,
            verbose=1,
            callbacks=callbacks,
            shuffle=True  ##soradan ekledim testlerde yoktu
        )

def build_model(hp):
    global train_X
    model = Sequential()

    # Gizli katman sayısı (1-5 arası)
    num_hidden_layers = hp.Int("num_hidden_layers", min_value=1, max_value=3, step=1)

    # Her katmandaki nöron sayısı (5, 10, 20, 50)
    for i in range(num_hidden_layers):
        units = hp.Choice(f"units_{i}", [10, 20, 50])
        activation = hp.Choice(f"activation_{i}", ["relu", "sigmoid", "tanh"])
        if i == 0:
            model.add(Dense(units, activation=activation, input_dim=train_X.shape[1]))
        else:
            model.add(Dense(units, activation=activation))

    # Çıkış katmanı
    model.add(Dense(1, activation="sigmoid"))

    # Öğrenme oranı
   # learning_rate = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])

    # Optimizatör seçimi
    # optimizer_choice = hp.Choice("optimizer", ["SGD", "RMSprop", "Adam", "Adadelta"])
    optimizer_choice = hp.Choice("optimizer", ["SGD", "RMSprop", "Adam"])

    if optimizer_choice == "SGD":
        optimizer = SGD(learning_rate=learning_rate)
    elif optimizer_choice == "RMSprop":
        optimizer = RMSprop(learning_rate=learning_rate)
    elif optimizer_choice == "Adam":
        optimizer = Adam(learning_rate=learning_rate)
    # elif optimizer_choice == "Adadelta":
    #     optimizer = Adadelta(learning_rate=learning_rate)

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model

def trainModelWithTuner():
    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)

    # Load the training, validation, and test datasets
    global train_X
    train_X, train_y, val_X, val_y, test_X, test_y = getTrainTestValDataPerc()

    # with strategy.scope():
        # Set up Keras Tuner for hyperparameter search
    # tuner = kt.BayesianOptimization(
    #     build_model_voxel,
    #     max_trials=500,
    #     objective="val_loss",  # Optimize based on validation loss
    #     directory="../Results/tuner_results_voxel",  # Save results in this directory
    #     project_name="keras_tuner_optimization_"+str(version),  # Project name for organization
    #     overwrite = True
    # )
    tuner = kt.BayesianOptimization(
        build_model,
        max_trials=500,
        objective="val_loss",  # Optimize based on validation loss
        directory="../Results/tuner_results",  # Save results in this directory
        project_name="keras_tuner_optimization_" + str(version),  # Project name for organization
        overwrite=True
    )

    # Define callbacks for the tuner search
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience),
    ]

    # Start hyperparameter search
    tuner.search(
        train_X, train_y,
        validation_data=(val_X, val_y),
        epochs=50,  # Maximum number of epochs
        batch_size=16384,  # Use large batch size
        callbacks=callbacks,
        shuffle=True  # Enable shuffling
    )


        # Get the best model from the tuner
    best_model = tuner.get_best_models(num_models=1)[0]

    # Summarize the best model
    best_model.summary()

    # Evaluate the best model on the test set
    test_loss, test_acc = best_model.evaluate(test_X, test_y)
    print(f"Best-Test accuracy: {test_acc}")
    print(f"Best-Test loss: {test_loss}")

    # Save the best model
    model_dir = "../models/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    best_model.save(os.path.join(model_dir, 'best_NN_model_'+str(version)+'.h5'))
    print("Model saved to ../models/best_NN_model_"+str(version)+".h5")

def print_model_hyperparameters(model):
    # Print layer-wise hyperparameters (activation functions, units, etc.)
    for layer in model.layers:
        print(f"Layer Name: {layer.name}")
        print(f"Layer Type: {layer.__class__.__name__}")

        # Check and print the activation function if it exists
        if hasattr(layer, 'activation'):
            print(f"Activation Function: {layer.activation.__name__}")

        # Print number of units if it's a dense layer or similar
        if hasattr(layer, 'units'):
            print(f"Units: {layer.units}")

        # Print kernel and bias initializers if available
        if hasattr(layer, 'kernel_initializer'):
            print(f"Kernel Initializer: {layer.kernel_initializer}")

        if hasattr(layer, 'bias_initializer'):
            print(f"Bias Initializer: {layer.bias_initializer}")

        print("-" * 40)

    # Print optimizer, loss, and metrics information if the model is compiled
    if model.optimizer:
        optimizer_config = model.optimizer.get_config()
        print(f"Optimizer: {optimizer_config['name']}")

    if model.loss:
        print(f"Loss Function: {model.loss}")

    if model.metrics_names:
        print(f"Metrics: {model.metrics_names}")

def testAllBestModelsWithTuner():
    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)

    # Load the training, validation, and test datasets
    global train_X
    global version
    model_dir = "../models/"
    results_dir = "../Results/"

    # Ensure the results directory exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for version in range(1, 5):
        train_X, train_y, val_X, val_y, test_X, test_y = getTrainTestValDataPerc()
        best_model = keras.models.load_model(os.path.join(model_dir, 'best_NN_model_' + str(version) + '.h5'))

        # Evaluate model and print results
        test_loss, test_acc = best_model.evaluate(test_X, test_y)
        print("*************__" + str(version) + "__*************")
        print(best_model.summary())
        print_model_hyperparameters(best_model)

        # Predict the values
        y_pred = best_model.predict(test_X)

        # Apply 0.5 threshold to predictions to convert to binary values (0 or 1)
        y_pred_binary = (y_pred > 0.5).astype(int)

        # Create the confusion matrix
        cm = confusion_matrix(test_y, y_pred_binary)
        print(f"Confusion Matrix for version {version}:\n", cm)

        # Save the confusion matrix as a heatmap

        plt.ylabel('Actual', fontsize=14)
        plt.figure(figsize=(5, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greys",   xticklabels=[0, 1], yticklabels=[0, 1],annot_kws={"size": 14}, cbar=False)
        plt.xlabel('Predicted Values', fontsize=14)
        plt.ylabel('True Values', fontsize=14)

        # Save the confusion matrix plot to ../results/ directory
        plt.savefig(os.path.join(results_dir, f'confusion_matrix_version_{version}.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

def testAllBestModelsWithTuner_old():
    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)

    # Load the training, validation, and test datasets
    global train_X
    global  version
    model_dir = "../models/"
    for version in range(1,4,1):
        train_X, train_y, val_X, val_y, test_X, test_y = getTrainTestValDataPerc()
        best_model = keras.models.load_model(os.path.join(model_dir, 'best_NN_model_' + str(version) + '.h5'))
        test_loss, test_acc = best_model.evaluate(test_X, test_y)
        print("*************__"+str(version)+"__*************")
        print(best_model.summary())
        print_model_hyperparameters(best_model)
        # print(f"Best-Test accuracy: {test_acc}")
        # print(f"Best-Test loss: {test_loss}")

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

def drawViolinPlots():
    # Get the train, validation, and test data
    train_X, train_y, val_X, val_y, test_X, test_y = getTrainTestValDataPerc()

    scaler = MinMaxScaler()  # or StandardScaler() for Z-score normalization

    # train_X = scaler.fit_transform(train_X[:, :7])
    # val_X= scaler.transform(val_X[:, :7])
    # test_X= scaler.transform(test_X[:, :7])
    # Create a directory to save the plots if it doesn't exist
    results_dir = "../Results/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Helper function to create and save violin plots
    def create_and_save_violin_plot(X, y, dataset_type):
        # Create DataFrame
        df = pd.DataFrame(X[:, :7], columns=['x', 'y', 'z', 'red', 'green', 'blue', 'nir16'])

        # Add the class labels to the DataFrame
        df['label'] = y

        # Convert label values to class names
        df['label'] = df['label'].map({0: 'background', 1: 'plant'})

        # Reshape the DataFrame for Seaborn violinplot
        df_melted = pd.melt(df, id_vars="label", value_vars=[ 'red', 'green', 'blue'],
                            var_name="Features", value_name="Values")

        # Create the violin plot
        plt.figure(figsize=(10, 6))
        sns.violinplot(x="Features", y="Values", hue="label", data=df_melted, split=True, inner="quart", palette="Set2")

        # Add title and labels
        plt.title(f"Violin Plot of Features for {dataset_type.capitalize()} Dataset")
        plt.xlabel("Features")
        plt.ylabel("Values")

        # Save the plot to the results folder
        plot_filename = os.path.join(results_dir, f"violin_plot_{dataset_type}.png")
        plt.savefig(plot_filename)
        plt.close()

    def create_and_save_box_plot(X, y, dataset_type):
        df = pd.DataFrame(X[:, :7], columns=['x', 'y', 'z', 'red', 'green', 'blue', 'nir16'])
        df['label'] = y
        df['label'] = df['label'].map({0: 'background', 1: 'plant'})

        df_melted = pd.melt(df, id_vars="label", value_vars=['x', 'y', 'z', 'red', 'green', 'blue', 'nir16'],
                            var_name="Features", value_name="Values")

        plt.figure(figsize=(10, 6))
        sns.boxplot(x="Features", y="Values", hue="label", data=df_melted, palette="Set2")
        plt.title(f"Box Plot of Features for {dataset_type.capitalize()} Dataset")
        plt.xlabel("Features")
        plt.ylabel("Values")
        plot_filename = os.path.join(results_dir, f"box_plot_{dataset_type}.png")
        plt.savefig(plot_filename)
        plt.close()

    def create_and_save_pair_plot(X, y, dataset_type):
        df = pd.DataFrame(X[:, :7], columns=['x', 'y', 'z', 'red', 'green', 'blue', 'nir16'])
        df['label'] = y
        df['label'] = df['label'].map({0: 'background', 1: 'plant'})

        sns.pairplot(df, hue='label', palette="Set2")
        plt.suptitle(f"Pair Plot of Features for {dataset_type.capitalize()} Dataset")
        plot_filename = os.path.join(results_dir, f"pair_plot_{dataset_type}.png")
        plt.savefig(plot_filename)
        plt.close()


    # Create and save violin plots for train, validation, and test datasets
    # create_and_save_violin_plot(train_X, train_y, "train")
    # create_and_save_violin_plot(val_X, val_y, "validation")
    create_and_save_violin_plot(test_X, test_y, "test")

    # create_and_save_box_plot(train_X, train_y, "train")
    # create_and_save_box_plot(val_X, val_y, "validation")
    # create_and_save_box_plot(test_X, test_y, "test")
    #
    # create_and_save_pair_plot(train_X, train_y, "train")
    # create_and_save_pair_plot(val_X, val_y, "validation")
    create_and_save_pair_plot(test_X, test_y, "test")

    print("Violin plots saved to the 'results' folder.")

def drawCombinedViolinPlots():
    results_dir = "../Results/"
    # Train, validation ve test verilerini al
    train_X, train_y, val_X, val_y, test_X, test_y = getTrainTestValDataPerc()

    # Tüm veri setlerini birleştir
    combined_X = np.vstack((train_X, val_X, test_X))
    combined_y = np.hstack((train_y, val_y, test_y))
    # combined_X =test_X
    # combined_y =test_y

    # DataFrame oluştur
    df = pd.DataFrame(combined_X[:, :7], columns=['x', 'y', 'z', 'red', 'green', 'blue', 'nir16'])
    df['label'] = combined_y

    # Class label'lerini anlamlı hale getir
    df['label'] = df['label'].map({0: 'background', 1: 'plant'})

    # Özellik gruplarına göre violin plot çizdirme
    feature_groups = {
        "Coordinates (x, y, z)": ['x', 'y', 'z'],
        "Colors (r, g, b)": ['red', 'green', 'blue'],
        "NIR": ['nir16']
    }

    for group_name, features in feature_groups.items():
        # Melt işlemi
        df_melted = pd.melt(df, id_vars="label", value_vars=features,
                            var_name="Features", value_name="Values")

        # Violin plot oluştur
        plt.figure(figsize=(10, 6))
        sns.violinplot(x="Features", y="Values", hue="label", data=df_melted, split=True, inner="quart", palette="Set2")

        # Başlık ve etiketler
        plt.title(f"Violin Plot of {group_name} for Plant and Background Classes")
        plt.xlabel("Features")
        plt.ylabel("Values")

        # Grafiği kaydet

        plot_filename = os.path.join(results_dir, f'violin_plot_{group_name.replace(" ", "_").lower()}.png')
        plt.savefig(plot_filename)
        plt.close()

# trainModelWithTuner()
testAllBestModelsWithTuner()
# drawCombinedViolinPlots()

# ModelTestonRawData("9001")
# ModelTestonRawData("CSV22")
# ModelTestonRawData("Chakti")

# # Model_test("train")
# # Model_test("val")
# # Model_test("test")

# # RGBSegmentation()
# # segmentRansac()
