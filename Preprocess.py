import os
import glob
import gc
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess

from collections import Counter
from scipy.spatial import KDTree
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.backend import clear_session
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model  # Used for loading Keras models
import tensorflow as tf

import plyfile  # For reading PLY files
import pycuda.driver as cuda  # Optional: required only if CUDA operations are performed

from RequiredFunctionPhenospex import *  # Your custom utility functions

# ----------------------------- Configuration -----------------------------

start_column = 0  # Starting column index for features (e.g., to skip ID columns)
expNumber = 50    # Experiment number identifier
plant_type = "all_species"  # Set both plant type and Raw_Data_Dir accordingly

# Example raw data directories:
# "./data/new/Exp56_1_Chakti", "./data/new/Exp56_1_CSV22", etc.
# For data count experiments:
Raw_Data_Dir = "./data/Data_Count/ChickpeaExp57"  # Chickpea experiment (Exp57)

Ground_Data_Dir = "./data/new/"  # Path to ground truth/reference data

# Derived directories
DATA_DIR = Raw_Data_Dir + "_out/"  # Output directory generated from raw data
org_Data_Dir = DATA_DIR            # Original data directory used for backup or reference
source_dir = DATA_DIR              # Directory for main processing operations



def visualize_and_save_model(model_path, save_path='model_plot.png'):
    # Modeli yükle
    model = load_model(model_path)

    # Modeli görselleştir ve görseli dosyaya kaydet
    plot_model(model, to_file=save_path, show_shapes=True, show_layer_names=True,rankdir='LR')
    print(f"Model görselleştirmesi '{save_path}' olarak kaydedildi.")

def writeAllPhenospexResults2OneFile():
    global version
    output_file='GroundTruth_HCv1_v2_predicted_'+str(version)+'.xlsx'
    inputFolder = DATA_DIR + "9_Biomass_prediction_Phenospex_"+str(version)
    source_directory_list = os.listdir(inputFolder)

    if os.path.exists(DATA_DIR + output_file):
        excel_file_path = DATA_DIR+output_file
        df_excel = pd.read_excel(excel_file_path,engine='openpyxl', dtype={"unit": str})
    else:
        excel_file_path = Ground_Data_Dir + 'GroundTruth_HCv1_v2.xlsx'
        df_excel = pd.read_excel(excel_file_path,sheet_name='SerkanGT', engine='openpyxl', dtype={"unit": str})
    # Extract values from column 2 in the "154:10:01" format
    date_values =df_excel.iloc[:, 0][:228]
    for i,date_value in enumerate(date_values):
        # Define the prefix and suffix
        date_value=date_value.split("_")
        prefix = date_value[0]
        suffix = "_parameters"

        # Use glob to find CSV files that match the specified pattern
        pattern = f'{inputFolder}/{prefix}*{suffix}.csv'
        csv_files = glob.glob(pattern)
        if len(csv_files) == 0:
            continue
        try:
            df_csv = pd.read_csv(csv_files[0])
            df_csv = df_csv[df_csv['x'] == int(date_value[2])]
            df_csv = df_csv[df_csv['y'] == int(date_value[1])]
            df_excel.at[i, "Phenospex prediction"] = df_csv['leaf_area'].iloc[0]
            df_excel.at[i, "Phenospex Voxel"] = df_csv['voxel_volume'].iloc[0]
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print(f"Error: CSV file {csv_files[0]} not found or empty.")

    # Save the updated DataFrame to the Excel file
    df_excel.to_excel(DATA_DIR + output_file, index=False)

    #*********************************results**************************
    # Calculate MAE (Mean Absolute Error)+

def writeAllResults2OneFile(version):
    output_file='GroundTruth_HCv1_v2_predicted_'+str(version)+'.xlsx'
    source_directory_list = os.listdir(DATA_DIR + "8_Biomass_prediction_7_"+str(version))
    inputFolder = DATA_DIR + "8_Biomass_prediction_7_"+str(version)

    if os.path.exists(DATA_DIR + output_file):
        excel_file_path = DATA_DIR+output_file
        df_excel = pd.read_excel(excel_file_path,engine='openpyxl', dtype={"unit": str})
    else:
        excel_file_path = Ground_Data_Dir + 'GroundTruth_HCv1_v2.xlsx'
        df_excel = pd.read_excel(excel_file_path,sheet_name='SerkanGT', engine='openpyxl', dtype={"unit": str})
    # Extract values from column 2 in the "154:10:01" format
    date_values =df_excel.iloc[:, 0][:228]
    for i,date_value in enumerate(date_values):
        # Define the prefix and suffix
        date_value=date_value.split("_")
        prefix = date_value[0]
        suffix = str(int(date_value[2])-1)+'_'+str(int(date_value[1])-1)+"_parameters"

        # Use glob to find CSV files that match the specified pattern
        pattern = f'{inputFolder}/{prefix}*{suffix}.csv'
        csv_files = glob.glob(pattern)
        if len(csv_files) == 0:
            continue
        try:
            df_csv = pd.read_csv(csv_files[0])
            df_excel.at[i, "Serkans prediction"] = df_csv['leaf_area'].iloc[0]
            df_excel.at[i, "Serkans Voxel"] = df_csv['voxel_volume'].iloc[0]
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print(f"Error: CSV file {csv_files[0]} not found or empty.")

    # Save the updated DataFrame to the Excel file
    df_excel.to_excel(DATA_DIR + output_file, index=False)

    #*********************************results**************************
    # Calculate MAE (Mean Absolute Error)

    df_excel = df_excel.dropna()
    mae_serkan = np.abs(df_excel['GroundTruthfor3D_LA (mm2)'] - df_excel['Serkans prediction']).mean()
    mape_serkan = np.mean(np.abs((df_excel['GroundTruthfor3D_LA (mm2)'] - df_excel['Serkans prediction']) / df_excel['GroundTruthfor3D_LA (mm2)'])) * 100
    mse_serkan = ((df_excel['GroundTruthfor3D_LA (mm2)'] - df_excel['Serkans prediction']) ** 2).mean()

    mae_HCphenaV2 = np.abs(df_excel['GroundTruthfor3D_LA (mm2)'] - df_excel['HCphenaV2']).mean()
    mape_HCphenaV2 = np.mean(np.abs((df_excel['GroundTruthfor3D_LA (mm2)'] - df_excel['HCphenaV2']) / df_excel['GroundTruthfor3D_LA (mm2)'])) * 100
    mse_HCphenaV2 = ((df_excel['GroundTruthfor3D_LA (mm2)'] - df_excel['HCphenaV2']) ** 2).mean()
    # Print the results
    print(f'MAE HCphenaV2: {mae_HCphenaV2:.2f} MAE Serkan: {mae_serkan:.2f}')
    print(f'MAPE HCphenaV2: {mape_HCphenaV2:.2f}% MAPE Serkan: {mape_serkan:.2f}%')
    print(f'MSE HCphenaV2: {mse_HCphenaV2:.2f} MSE Serkan: {mse_serkan:.2f}')

def calculateBiomass4PhenoPred():
    #apply preprocessing
    source_directory_list = os.listdir(org_Data_Dir + "3_merged")
    outputFolder = DATA_DIR + "3_merged_Phenospex"
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    for file in source_directory_list:
        # Specify the path to the executable and the command-line arguments
        executable_path = './PhenoSpexExe/3dproc-v2.1'
        input_file = org_Data_Dir + "3_merged/" + file
        output_file = outputFolder + "/" + file
        config_file = './PhenoSpexExe/preprocessing.json'

        # Use subprocess to call the executable with the provided arguments
        try:
            subprocess.run([executable_path, '-i', input_file, '-o', output_file, '-c', config_file], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")


    #apply parameter extraction
    source_directory_list = os.listdir(DATA_DIR + "3_merged_Phenospex")
    outputFolder = DATA_DIR + "9_Biomass_prediction_Phenospex_"+str(version)
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    for file in source_directory_list:
        if "full" not in file:
            continue
        # Specify the path to the executable and the command-line arguments
        executable_path = './PhenoSpexExe/3dproc-v2.1'
        input_file = DATA_DIR + "3_merged_Phenospex/"+file
        output_file = outputFolder + "/" +file
        config_file = './PhenoSpexExe/parameter_extraction_merged.json'

        # Use subprocess to call the executable with the provided arguments
        try:
            subprocess.run([executable_path, '-i', input_file, '-o', output_file, '-c', config_file], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")

def calculateBiomass():
    source_directory_list = os.listdir(DATA_DIR + "6_segmentation_result_plants_MLP_P")
    outputFolder = DATA_DIR + "8_Biomass_prediction_6"
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    for file in source_directory_list:
        # Specify the path to the executable and the command-line arguments
        executable_path = './PhenoSpexExe/3dproc-v2.1'
        input_file = DATA_DIR + "6_segmentation_result_plants_MLP_P/"+file
        output_file = outputFolder + "/" +file
        config_file = './PhenoSpexExe/parameter_extraction_merged.json'

        # Use subprocess to call the executable with the provided arguments
        try:
            subprocess.run([executable_path, '-i', input_file, '-o', output_file, '-c', config_file], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")

def calculateBiomassSeperately(version):
    source_directory_list = os.listdir(DATA_DIR + "7_partitioned_trays4plants_MLP_P_"+str(version))
    outputFolder = DATA_DIR + "8_Biomass_prediction_7_"+str(version)
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    for file in source_directory_list:
        # Specify the path to the executable and the command-line arguments
        executable_path = './PhenoSpexExe/3dproc-v2.1'
        input_file = DATA_DIR + "7_partitioned_trays4plants_MLP_P_"+str(version)+"/"+file
        output_file = outputFolder + "/" +file
        config_file = './PhenoSpexExe/parameter_extraction.json'

        # Use subprocess to call the executable with the provided arguments
        try:
            subprocess.run([executable_path, '-i', input_file, '-o', output_file, '-c', config_file], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")

def convertOurOutput2PhenospexFormat6():
    source_directory_list = os.listdir(DATA_DIR + "6_segmentation_result_plants_MLP")
    outputFolder = DATA_DIR + "6_segmentation_result_plants_MLP_P"
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    for file in source_directory_list:
        if os.path.isfile(DATA_DIR + "6_segmentation_result_plants_MLP/"+file):
            convertOurOutput2PhenospexFormat_OneFile(DATA_DIR + "6_segmentation_result_plants_MLP/"+file, outputFolder + "/" +file)
        elif os.path.isdir(DATA_DIR + "6_segmentation_result_plants_MLP/"+file):
            sub_directory_list = os.listdir(DATA_DIR + "6_segmentation_result_plants_MLP/"+file)
            for sub_file in sub_directory_list:
                convertOurOutput2PhenospexFormat_OneFile(DATA_DIR + "6_segmentation_result_plants_MLP/" + file+"/"+sub_file,
                                                         outputFolder + "/" + sub_file)

def convertOurOutput2PhenospexFormat7(version=3):
    source_directory_list = os.listdir(DATA_DIR + "7_partitioned_trays4plants_MLP_"+str(version))
    outputFolder = DATA_DIR + "7_partitioned_trays4plants_MLP_P_"+str(version)
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    for file in source_directory_list:
        if os.path.isfile(file):
            convertOurOutput2PhenospexFormat_OneFile(DATA_DIR + "7_partitioned_trays4plants_MLP_"+str(version)+"/"+file, outputFolder + "/" +file)
        elif os.path.isdir(DATA_DIR + "7_partitioned_trays4plants_MLP_"+str(version)+"/"+file):
            sub_directory_list = os.listdir(DATA_DIR + "7_partitioned_trays4plants_MLP_"+str(version)+"/"+file)
            for sub_file in sub_directory_list:
                convertOurOutput2PhenospexFormat_OneFile(DATA_DIR + "7_partitioned_trays4plants_MLP_"+str(version)+"/"+file+"/"+sub_file,
                                                         outputFolder + "/" + sub_file)

def preprocess():
    #Modify raw .ply data
    outputFolder = org_Data_Dir + "1_modified"
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    subfile_directory = os.listdir(Raw_Data_Dir)
    for i, subfile in enumerate(subfile_directory):
        plyfile_directory = os.listdir(Raw_Data_Dir+"/"+subfile)
        for i, plyfile in enumerate(plyfile_directory):
            if not os.path.exists(outputFolder +"/"+subfile):
                os.makedirs(outputFolder +"/"+subfile)
            modify_ply_from_Phenospex(Raw_Data_Dir+"/"+subfile + "/" + plyfile, outputFolder +"/"+subfile+ "/" + plyfile)

    #Rotate modified data
    outputFolder = org_Data_Dir + "2_rotated"
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    subfile_directory = os.listdir(org_Data_Dir + "1_modified")
    for i, subfile in enumerate(subfile_directory):
        plyfile_directory = os.listdir(DATA_DIR+ "1_modified"+"/"+subfile)
        for i, plyfile in enumerate(plyfile_directory):
            if not os.path.exists(outputFolder +"/"+subfile):
                os.makedirs(outputFolder +"/"+subfile)
            rotate2originPhenospex(DATA_DIR + "1_modified/"+subfile + "/"+ plyfile, outputFolder + "/" +subfile + "/"+  plyfile,line_seperator=",")

    # Merge Rotated data
    outputFolder = org_Data_Dir + "3_merged"
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    subfile_directory = os.listdir(org_Data_Dir + "2_rotated")
    for i, subfile in enumerate(subfile_directory):
        mergePlyWithinFolderPhenospex(org_Data_Dir + "2_rotated" + "/" + subfile, outputFolder + "/" + subfile)

def preprocessWithPhenospex():

    # apply preprocessing
    source_directory_list = os.listdir(org_Data_Dir + "3_merged")
    outputFolder = DATA_DIR + "4_voxelized_Phenospex"
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    for file in source_directory_list:
        # Specify the path to the executable and the command-line arguments
        executable_path = './PhenoSpexExe/3dproc-v2.1'
        input_file = org_Data_Dir + "3_merged/" + file
        output_file = outputFolder + "/" + file
        config_file = './PhenoSpexExe/preprocessing_voxelization.json'

        # Use subprocess to call the executable with the provided arguments
        try:
            subprocess.run([executable_path, '-i', input_file, '-o', output_file, '-c', config_file], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")


    # Smoothing Voxelized data
    outputFolder = DATA_DIR + "5_smooted_Phenospex"
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    plyfile_directory = os.listdir(DATA_DIR + "4_voxelized_Phenospex")
    for i, plyfile in enumerate(plyfile_directory):
        if "voxel" in plyfile:
            continue
        smoothingPhenospex(DATA_DIR + "4_voxelized_Phenospex/" + plyfile, outputFolder + "/" + plyfile)

def filter_point_cloud(pcd_plants,nb_neighbors=100,std_ratio=2.0):
    # PointCloud verisini numpy array'e dönüştürme
    points = np.asarray(pcd_plants.points)
    colors = np.asarray(pcd_plants.colors)
    plants_data = np.column_stack((points, colors))

    # Mevcut z değerlerini al
    z_values = plants_data[:, 2]  # z değerlerinin 3. kolon olduğunu varsayıyoruz

    # max_z değerini bul
    max_z = np.max(z_values)

    # max_z-40'tan küçük z değerleri için bir maske oluştur
    mask_max_z = z_values < (max_z - 80)

    # Orijinal plants_data dizisinden max_z-40'tan küçük z değerlerini al
    plants_data_max_z = plants_data[mask_max_z]

    # Statistical Outlier işlemi ile elde edilen pcd_plants dizisinden max_z-40'tan büyük z değerlerini al
    pcd_plants_outliers = pcd_plants.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)[0]
    points_outliers = np.asarray(pcd_plants_outliers.points)
    colors_outliers = np.asarray(pcd_plants_outliers.colors)
    plants_outliers = np.column_stack((points_outliers, colors_outliers))

    z_values = plants_outliers[:, 2]  #
    mask_outliers = z_values >= (max_z - 80)
    plants_outliers_filtered = plants_outliers[mask_outliers]

    # Yeni plants_data dizisini oluştur
    new_plants_data = np.vstack((plants_data_max_z, plants_outliers_filtered))

    # Yeni numpy array'den PointCloud oluşturma
    new_pcd_plants = o3d.geometry.PointCloud()
    new_pcd_plants.points = o3d.utility.Vector3dVector(new_plants_data[:, :3])
    new_pcd_plants.colors = o3d.utility.Vector3dVector(new_plants_data[:, 3:])

    return new_pcd_plants

def segmentWithMLPPhenospex(version=2,is_smooted=1):
    #Start Segmentation with Pre-Trained model
    global  DATA_DIR
    global org_Data_Dir
    source_dir=DATA_DIR
    if is_smooted:
        source_directory_list = os.listdir(DATA_DIR + "5_smooted_Phenospex")
        source_dir=DATA_DIR + "5_smooted_Phenospex"
        model = load_model('./models/best_NN_model_' + str(version)+ '.h5')


    else:
        source_directory_list = os.listdir(DATA_DIR + "4_voxelized_Phenospex")
        source_dir=DATA_DIR + "4_voxelized_Phenospex"
        DATA_DIR=DATA_DIR[:-4]+"_NonSmoothed"+DATA_DIR[-4:]
        model = load_model('./models/best_NN_model_' + str(version)+ '.h5')
    # model = load_model('./models/all_species_NN_model.h5')
    for file in source_directory_list:
        if "voxel" in file:
            continue
        print("Segmenting : ",file)
        ply = PlyFile()
        ply.load(source_dir + "/" + file)

        X_org = np.concatenate((ply.vertices["x"].reshape(-1, 1), ply.vertices["y"].reshape(-1, 1), ply.vertices["z"].reshape(-1, 1)), axis=1)
        X_org = np.concatenate((X_org,ply.vertices["red"].reshape(-1, 1), ply.vertices["green"].reshape(-1, 1), ply.vertices["blue"].reshape(-1, 1)), axis=1)
        X_org =  np.concatenate((X_org,ply.vertices["nir16"].reshape(-1, 1),ply.vertices["nir940"].reshape(-1, 1)), axis=1)
        X_org = X_org[X_org[:, 2] < 200]

        if version==1:
            X = X_org.copy()[:,3:6]
        elif version==2:
            X = X_org.copy()[:,:6]
        else:
            X = X_org.copy()

        X = X[:, start_column:]
        Y = model.predict(X)

        bacg = X_org[Y[:, 0] <= 0.5]
        plants = X_org[Y[:, 0] > 0.5]
        "Inversion of Z coordinate"
        # bacg[:, 2] = bacg[:, 2]
        # plants[:, 2] = plants[:, 2]

        pcd_plants = o3d.geometry.PointCloud()
        pcd_plants.points = o3d.utility.Vector3dVector(plants[:, :3])
        pcd_plants.colors = o3d.utility.Vector3dVector(plants[:, 3:6] / 255)
        # pcd_plants = pcd_plants.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)[0]

        pcd_plants=filter_point_cloud(pcd_plants,nb_neighbors=200,std_ratio=2.0)

        pcd_back = o3d.geometry.PointCloud()
        pcd_back.points = o3d.utility.Vector3dVector(bacg[:, :3])
        pcd_back.colors = o3d.utility.Vector3dVector(bacg[:, 3:6] / 255)

        saveSegmentedDataTXT(pcd_plants, pcd_back, file,plants,"MLP_"+str(version))
        # Clean up
        clear_session()
        del X, X_org, Y, bacg, plants, pcd_plants, pcd_back
        gc.collect()

    del model
    gc.collect()

def saveSegmentedDataTXT(pcd_plants,pcd_back,file,plants,model_name="MLP"):
    if not os.path.exists(DATA_DIR + "6_segmentation_result_background_" + model_name + "/"):
        os.makedirs(DATA_DIR + "6_segmentation_result_background_" + model_name + "/")
    if not os.path.exists(DATA_DIR + "6_segmentation_result_plants_" + model_name + "/"):
        os.makedirs(DATA_DIR + "6_segmentation_result_plants_" + model_name + "/")

    if not os.path.exists(DATA_DIR + "7_partitioned_trays4plants_" + model_name + "/" + file[:-4]):
        os.makedirs(DATA_DIR + "7_partitioned_trays4plants_" + model_name + "/" + file[:-4])
    if not os.path.exists(DATA_DIR + "7_partitioned_trays4background_" + model_name + "/" + file[:-4]):
        os.makedirs(DATA_DIR + "7_partitioned_trays4background_" + model_name + "/" + file[:-4])

    points = np.asarray(pcd_plants.points)
    colors = np.asarray(pcd_plants.colors)*255
    plants_data = np.column_stack((points, colors))
    np.savetxt(DATA_DIR + "6_segmentation_result_plants_" + model_name + "/" + file[:-4] + "_plants.txt",
               plants_data, delimiter=',', fmt='%.2f')

    points = np.asarray(pcd_back.points)
    colors = np.asarray(pcd_back.colors)*255
    back_data = np.column_stack((points, colors))
    np.savetxt(DATA_DIR + "6_segmentation_result_background_" + model_name + "/" + file[:-4] + "_background.txt",
               back_data, delimiter=',', fmt='%.2f')

    "Partition of cleaned data"
    # Use glob to find CSV files that match the specified pattern
    pattern = f'{DATA_DIR + "1_modified/" + file[:7]}/*{file[-20:]}'
    csv_files = glob.glob(pattern)
    # header_obj_info = read_ply_header_obj_info(csv_files[0])
    # fileGlobals = convert_obj_info_2_dict(header_obj_info)
    #     field_y_origin = fileGlobals["field_y_origin"]
    #     field_y_period = fileGlobals["field_y_period"]
    #     field_x_period = fileGlobals["field_x_period"]
    #     y_sectors = int(fileGlobals["y_sectors"])
    fileGlobals = read_parameter_json()
    field_y_origin = fileGlobals["field"]["origin_y"]
    field_y_period = fileGlobals["field"]["period_y"]
    field_x_period = fileGlobals["field"]["period_x"]
    y_sectors = int(fileGlobals["field"]["y_units"])
    # if os.path.exists(DATA_DIR + "1_modified/" + file[:-4] + "_M.ply"):
    #     header_obj_info = read_ply_header_obj_info((DATA_DIR + "1_modified/" + file[:-4] + "_M.ply"))
    # else:
    #     header_obj_info = read_ply_header_obj_info((DATA_DIR + "1_modified/" + file[:-4] + "_S.ply"))


    raw1 = plants[plants[:, 0] <= 0]
    raw2 = plants[plants[:, 0] > 0]
    raw1[:, 0] = raw1[:, 0] + field_x_period / 2
    raw2[:, 0] = raw2[:, 0] - field_x_period / 2

    coor_y = field_y_origin
    file_sp = file.split("_")
    for i in range(y_sectors):
        tray = np.copy(raw1[raw1[:, 1] > coor_y])
        tray = tray[tray[:, 1] <= (coor_y + field_y_period)]
        tray[:, 1] = tray[:, 1] - (coor_y + field_y_period / 2)

        part_plant = o3d.geometry.PointCloud()
        part_plant.points = o3d.utility.Vector3dVector(tray[:, :3])
        part_plant.colors = o3d.utility.Vector3dVector(tray[:, 3:6] / 255)

        points = np.asarray(part_plant.points)
        colors = np.asarray(part_plant.colors)*255
        temp_plant = np.column_stack((points, colors))
        np.savetxt(DATA_DIR + "7_partitioned_trays4plants_" + model_name + "/" + file[:-4] + "/" + #str(expNumber) + "_" +
            file_sp[0] + "_0_" + str(i)  + ".txt",temp_plant, delimiter=',', fmt='%.2f')

        coor_y = coor_y + field_y_period

    coor_y = field_y_origin
    for i in range(y_sectors):
        tray = np.copy(raw2[raw2[:, 1] > coor_y])
        tray = tray[tray[:, 1] <= (coor_y + field_y_period)]
        tray[:, 1] = tray[:, 1] - (coor_y + field_y_period / 2)

        part_back = o3d.geometry.PointCloud()
        part_back.points = o3d.utility.Vector3dVector(tray[:, :3])
        part_back.colors = o3d.utility.Vector3dVector(tray[:, 3:6] / 255)
        points = np.asarray(part_back.points)
        colors = np.asarray(part_back.colors)*255
        temp_plant = np.column_stack((points, colors))
        np.savetxt( DATA_DIR + "7_partitioned_trays4plants_" + model_name + "/" + file[:-4] + "/" + #str(expNumber) + "_" +
            file_sp[0] + "_1_" + str(i)  +".txt", temp_plant, delimiter=',', fmt='%.2f')

        coor_y = coor_y + field_y_period

    print(model_name, "---Segmentation process completed --- check Data/Data2Clean/6_segmentation_result_plants/")

def saveSegmentedDataPCD(pcd_plants,pcd_back,file,plants,model_name="MLP"):
    if not os.path.exists(DATA_DIR + "6_segmentation_result_background_"+model_name+"/"):
        os.makedirs(DATA_DIR + "6_segmentation_result_background_"+model_name+"/")
    if not os.path.exists(DATA_DIR + "6_segmentation_result_plants_"+model_name+"/"):
        os.makedirs(DATA_DIR + "6_segmentation_result_plants_"+model_name+"/")

    if not os.path.exists(DATA_DIR + "7_partitioned_trays4plants_"+model_name+"/" + file[:-4]):
        os.makedirs(DATA_DIR + "7_partitioned_trays4plants_"+model_name+"/" + file[:-4])
    if not os.path.exists(DATA_DIR + "7_partitioned_trays4background_"+model_name+"/" + file[:-4]):
        os.makedirs(DATA_DIR + "7_partitioned_trays4background_"+model_name+"/" + file[:-4])

    o3d.io.write_point_cloud(DATA_DIR + "6_segmentation_result_plants_"+model_name+"/" + file[:-4] + "_plants.pcd", pcd_plants)
    o3d.io.write_point_cloud(DATA_DIR + "6_segmentation_result_background_"+model_name+"/" + file[:-4] + "_background.pcd", pcd_back)

    "Partition of cleaned data"
    # Use glob to find CSV files that match the specified pattern
    pattern = f'{DATA_DIR + "1_modified/" + file[:7] }/*{file[-20:]}'
    csv_files = glob.glob(pattern)
    header_obj_info = read_ply_header_obj_info(csv_files[0])

    # if os.path.exists(DATA_DIR + "1_modified/" + file[:-4] + "_M.ply"):
    #     header_obj_info = read_ply_header_obj_info((DATA_DIR + "1_modified/" + file[:-4] + "_M.ply"))
    # else:
    #     header_obj_info = read_ply_header_obj_info((DATA_DIR + "1_modified/" + file[:-4] + "_S.ply"))

    fileGlobals = convert_obj_info_2_dict(header_obj_info)
    field_y_origin = fileGlobals["field_y_origin"]
    field_y_period = fileGlobals["field_y_period"]
    field_x_period = fileGlobals["field_x_period"]
    y_sectors = int(fileGlobals["y_sectors"])

    raw1 = plants[plants[:, 0] <= 0]
    raw2 = plants[plants[:, 0] > 0]
    raw1[:, 0] = raw1[:, 0] + field_x_period / 2
    raw2[:, 0] = raw2[:, 0] - field_x_period / 2

    coor_y = field_y_origin
    file_sp = file.split("_")
    for i in range(y_sectors):
        tray = np.copy(raw1[raw1[:, 1] > coor_y])
        tray = tray[tray[:, 1] <= (coor_y + field_y_period)]
        tray[:, 1] = tray[:, 1] - (coor_y + field_y_period / 2)

        part_plant = o3d.geometry.PointCloud()
        part_plant.points = o3d.utility.Vector3dVector(tray[:, :3])
        part_plant.colors = o3d.utility.Vector3dVector(tray[:, 3:] / 255)
        o3d.io.write_point_cloud(
            DATA_DIR + "7_partitioned_trays4plants_"+model_name+"/" + file[:-4] + "/" + #str(expNumber) + "_" +
             file_sp[0] + "_0_" + str(i) + "_" + file_sp[1] + ".pcd", part_plant)

        coor_y = coor_y + field_y_period

    coor_y = field_y_origin
    for i in range(y_sectors):
        tray = np.copy(raw2[raw2[:, 1] > coor_y])
        tray = tray[tray[:, 1] <= (coor_y + field_y_period)]
        tray[:, 1] = tray[:, 1] - (coor_y + field_y_period / 2)

        part_back = o3d.geometry.PointCloud()
        part_back.points = o3d.utility.Vector3dVector(tray[:, :3])
        part_back.colors = o3d.utility.Vector3dVector(tray[:, 3:] / 255)
        o3d.io.write_point_cloud(
            DATA_DIR + "7_partitioned_trays4plants_"+model_name+"/" + file[:-4] + "/" + #str(expNumber) + "_" +
             file_sp[0] + "_1_" + str(i) + "_" + file_sp[1] + ".pcd", part_back)
        coor_y = coor_y + field_y_period


    print(model_name,"---Segmentation process completed --- check Data/Data2Clean/6_segmentation_result_plants/")

def segmentRansac(distance_threshold = 2):
    # Start Segmentation with Pre-Trained model
    input_dir=DATA_DIR + "5_smooted"
    source_directory_list = os.listdir(input_dir)

    # Set the parameters for region growing segmentation.
     # Distance threshold for plane segmentation
    ransac_n = 10  # RANSAC algorithm parameter
    num_iterations = 100 # Number of RANSAC iterations
    cluster_epsilon = 0.2  # DBSCAN clustering parameter

    for file in source_directory_list:
        pcd = o3d.io.read_point_cloud(input_dir + "/" + file)

        X = np.concatenate((np.asarray(pcd.points), np.asarray(pcd.colors) * 255), axis=1)
        pcd = pcd.select_by_index(np.where(X[:, 2] < 200)[0])
        X = X[X[:, 2] < 200]
        X_org = X.copy()

        # Perform plane segmentation to remove ground or large planar surfaces.
        inliers, outliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)
        pcd_plants = pcd.select_by_index(outliers, invert=True)
        pcd_back = pcd.select_by_index(outliers, invert=False)

        bacg = X_org[outliers]
        plants= np.concatenate((np.asarray(pcd_plants.points), np.asarray(pcd_plants.colors) * 255), axis=1)
        "Inversion of Z coordinate"
        bacg[:, 2] = bacg[:, 2] * -1
        plants[:, 2] = plants[:, 2] * -1

        saveSegmentedData(pcd_plants, pcd_back, file,plants,"Ransac")

def RGBSegmentation(smoothness = 20,cls_size=30):
    # Start Segmentation with Pre-Trained model
    input_dir=DATA_DIR + "3_merged"
    source_directory_list = os.listdir(input_dir)

    for file in source_directory_list:
        pcd = o3d.io.read_point_cloud(input_dir + "/" + file)

        # X = np.concatenate((np.asarray(pcd.points), np.asarray(pcd.colors) * 255), axis=1)
        X=np.asarray(pcd.points).astype(np.float32)
        pcd = pcd.select_by_index(np.where(X[:, 2] < 200)[0])
        X = X[X[:, 2] < 200]
        X_org = X.copy()

        cloud = pcl.PointCloud()
        cloud.from_array(X)

        curvature = 100
        tree = cloud.make_kdtree()
        segment = cloud.make_RegionGrowing(ksearch=50)
        segment.set_MinClusterSize(30)
        segment.set_MaxClusterSize(100000)
        segment.set_NumberOfNeighbours(30)
        segment.set_SmoothnessThreshold((smoothness / 180.0) * (3.14))
        segment.set_CurvatureThreshold(curvature)
        segment.set_SearchMethod(tree)
        clusters = segment.Extract()

        solid_index=[]
        for i, a in enumerate(clusters):
            if (cls_size < len(clusters[i])):
                solid_index.extend(clusters[i])


        pcd_plants = pcd.select_by_index(solid_index, invert=True)
        pcd_back = pcd.select_by_index(solid_index)

        plants= np.concatenate((np.asarray(pcd_plants.points), np.asarray(pcd_plants.colors) * 255), axis=1)
        "Inversion of Z coordinate"
        plants[:, 2] = plants[:, 2] * -1
        saveSegmentedData(pcd_plants, pcd_back, file,plants,"RGBSegmentation")

def tempFunc():
    source_directory_list = os.listdir(DATA_DIR + "6_segmentation_result_plants_MLP")
    outputFolder = DATA_DIR + "6_segmentation_result_plants_MLP_InverseZ"
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    for file in source_directory_list:
        matrix = np.loadtxt(DATA_DIR + "6_segmentation_result_plants_MLP" + "/" + file, delimiter=',')

        # Step 2: Multiply the third column by -1
        matrix[:, 2] *= -1

        # Step 3: Save the modified matrix back to a file
        np.savetxt(outputFolder + "/" + file, matrix, delimiter=',', fmt='%.2f')

def updateJson_max_z(json_file_path,max_z):
    # JSON dosyasını oku
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # "resolution" değerini güncelle
    data['crop']['max_z'] = max_z # Yeni değeri burada belirleyin

    # Güncellenmiş JSON verisini dosyaya geri yaz
    with open(json_file_path, 'w') as file:
        json.dump(data, file, indent=4)

## calculate r2 MAe MAPE vs. for all predictions
# Fonksiyonlar
def calculate_metrics(true_values, predicted_values):
    r2 = r2_score(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    mape = np.mean(np.abs((true_values - predicted_values) / true_values)) * 100
    return r2, mae, rmse, mape

def process_file3(file_path):
    df = pd.read_excel(file_path)

    # Sadece gerekli sütunlar üzerinde işlem yapacağız
    relevant_columns = ['GroundTruthfor3D_LA (mm2)', 'Phenospex prediction', 'Serkans prediction', 'HCphenaV2']
    df = df[relevant_columns].dropna()  # NaN değerleri düşüreceğiz

    # Değerler
    true_values = df['GroundTruthfor3D_LA (mm2)']
    phenospex_prediction = df['Phenospex prediction']
    serkans_prediction = df['Serkans prediction']
    hcphena_v2 = df['HCphenaV2']

    # Phenospex prediction için metrikler
    phenospex_prediction_metrics = calculate_metrics(true_values, phenospex_prediction)

    # Serkans prediction için metrikler
    serkans_prediction_metrics = calculate_metrics(true_values, serkans_prediction)

    # HCphenaV2 prediction için metrikler
    hcphena_v2_metrics = calculate_metrics(true_values, hcphena_v2)

    return {
        'Phenospex prediction_R²': phenospex_prediction_metrics[0],
        'Phenospex prediction_MAE': phenospex_prediction_metrics[1],
        'Phenospex prediction_RMSE': phenospex_prediction_metrics[2],
        'Phenospex prediction_MAPE': phenospex_prediction_metrics[3],
        'Serkans_prediction_R²': serkans_prediction_metrics[0],
        'Serkans_prediction_MAE': serkans_prediction_metrics[1],
        'Serkans_prediction_RMSE': serkans_prediction_metrics[2],
        'Serkans_prediction_MAPE': serkans_prediction_metrics[3],
        'HCphenaV2_R²': hcphena_v2_metrics[0],
        'HCphenaV2_MAE': hcphena_v2_metrics[1],
        'HCphenaV2_RMSE': hcphena_v2_metrics[2],
        'HCphenaV2_MAPE': hcphena_v2_metrics[3]
    }

def process_file(file_path):
    df = pd.read_excel(file_path)

    # Sadece gerekli sütunlar üzerinde işlem yapacağız
    relevant_columns = ['GroundTruthfor3D_LA (mm2)', 'Phenospex prediction', 'Serkans prediction']
    df = df[relevant_columns].dropna()  # NaN değerleri düşüreceğiz

    # Değerler
    true_values = df['GroundTruthfor3D_LA (mm2)']
    hcphena_v2 = df['Phenospex prediction']
    serkans_prediction = df['Serkans prediction']

    # Phenospex prediction için metrikler
    hcphena_v2_metrics = calculate_metrics(true_values, hcphena_v2)

    # Serkans prediction için metrikler
    serkans_prediction_metrics = calculate_metrics(true_values, serkans_prediction)

    return {
        'Phenospex prediction_R²': hcphena_v2_metrics[0],
        'Phenospex prediction_MAE': hcphena_v2_metrics[1],
        'Phenospex prediction_RMSE': hcphena_v2_metrics[2],
        'Phenospex prediction_MAPE': hcphena_v2_metrics[3],
        'Serkans_prediction_R²': serkans_prediction_metrics[0],
        'Serkans_prediction_MAE': serkans_prediction_metrics[1],
        'Serkans_prediction_RMSE': serkans_prediction_metrics[2],
        'Serkans_prediction_MAPE': serkans_prediction_metrics[3]
    }

def process_all_files(base_folder):
    # subfolders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
    subfolders = [str(i) for i in range(50, 70)]

    all_results = []

    for subfolder in subfolders:
        folder_path = os.path.join(base_folder, subfolder)
        file_name = 'GroundTruth_HCv1_v2_predicted_'+str(version)+'.xlsx'  # Excel dosya adı sabit
        file_path = os.path.join(folder_path, file_name)

        if os.path.exists(file_path):
            print(file_path)
            # metrics = process_file(file_path)
            metrics = process_file3(file_path)
            result_entry = {
                'Resolution': f'{subfolder}',
                **metrics
            }
            all_results.append(result_entry)

    return all_results

def save_results(results,version):
    results_file = DATA_DIR[:-4] + '/all_results_'+str(version)+'.xlsx'
    results_df = pd.DataFrame(results)
    results_df.to_excel(results_file, index=False)
    print(f"Processed all files. Results saved to {results_file}")

    generate_plots4OneResolution((DATA_DIR[:-4]), "Serkans_prediction", "Phenospex prediction",resolution=60)

# visualize results

def generate_plots(output_dir, serkans_prediction_col, hcphena_v2_col):
    # Verileri yükle
    global  version
    file_path = os.path.join(output_dir, 'all_results_'+str(version)+'.xlsx')
    df = pd.read_excel(file_path)
    fs=14
    def plot_metric(metric, title, ylabel, file_name,y_max):
        plt.figure(figsize=(12, 6))
        x = range(len(df['Resolution']))
        width = 0.3

        bars1 =plt.bar([p - width for p in x], df[f'{hcphena_v2_col}_{metric}'], width=width, label="Traditional Method",   color='#1f77b4')
        bars2 =plt.bar(x, df[f'{serkans_prediction_col}_{metric}'], width=width, label= "AI Based Method", color='#ff7f0e')

        plt.xlabel('Resolution', fontsize=fs+2)
        plt.ylabel(ylabel, fontsize=fs+2)
        if "Vijay" in Raw_Data_Dir:
            title="Chickpea"
        elif "CSV22" in Raw_Data_Dir:
            title="Sorghum"
        elif "9001" in Raw_Data_Dir:
            title="Mungbean"
        elif "Chakti" in Raw_Data_Dir:
            title="Pearl Millet"

        plt.title(title, fontsize=16)
        plt.xticks(x, df['Resolution'], rotation=90, fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.legend(loc='upper right', fontsize=fs)
        # plt.legend(loc='upper right', fontsize=fs)
        plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)

        plt.ylim(top=y_max)

        # Değerleri çubukların üzerine yaz
        def add_labels(bars):
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom', ha='center', fontsize=fs-4)

        add_labels(bars1)
        add_labels(bars2)

        # Grafiği kaydet
        output_path = os.path.join(output_dir, file_name)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

    # Mean Absolute Error (MAE) grafiği
    plot_metric('MAE', 'Mean Absolute Error', 'Mean Absolute Error', 'mean_absolute_error_'+str(version)+'.png',16000)

    # R² grafiği
    plot_metric('R²', 'R²', 'R²', 'r2_'+str(version)+'.png',1)

    # RMSE grafiği
    plot_metric('RMSE', 'RMSE', 'RMSE', 'rmse_'+str(version)+'.png',18000)

    # MAPE grafiği
    plot_metric('MAPE', 'MAPE', 'MAPE', 'mape_'+str(version)+'.png',100)

    print("Grafikler oluşturuldu ve kaydedildi.")

def generate_plots_for_all_folders(output_dir, folder_list, resolution_value, serkans_prediction_col, traditional_col,version):
    # Hata metrikleri listesi
    metrics = ['MAE', 'R²', 'RMSE', 'MAPE']

    # Her hata metriği için bir figure oluştur
    for metric in metrics:
        plt.figure(figsize=(6, 4))
        plt.title(f'{metric} for Resolution {resolution_value}', fontsize=12)
        plt.xlabel('Folders', fontsize=12)
        plt.ylabel(metric, fontsize=12)

        traditional_values = []
        ai_values = []
        folder_names = []

        # Her klasörden veriyi oku
        for folder in folder_list:
            file_path = os.path.join(output_dir,folder)
            file_path = os.path.join(file_path+"_out", 'all_results_'+str(version)+'.xlsx')
            df = pd.read_excel(file_path)

            # Belirtilen resolution değerini filtrele
            df_filtered = df[df['Resolution'] == resolution_value]

            if "Vijay" in folder:
                title = "Chickpea"
            elif "CSV22" in folder:
                title = "Sorghum"
            elif "9001" in folder:
                title = "Mungbean"
            elif "Chakti" in folder:
                title = "Pearl Millet"
            if not df_filtered.empty:
                traditional_values.append(df_filtered[f'{traditional_col}_{metric}'].values[0])
                ai_values.append(df_filtered[f'{serkans_prediction_col}_{metric}'].values[0])
                folder_names.append(os.path.basename(title))  # Klasör ismini listeye ekle

        # Bar plot çizimi
        x = range(len(folder_names))
        width = 0.35

        plt.bar([p - width / 2 for p in x], traditional_values, width=width, label="Traditional Method",
                color='#1f77b4')
        plt.bar([p + width / 2 for p in x], ai_values, width=width, label="AI Based Method", color='#ff7f0e')

        plt.xticks(x, folder_names, rotation=45, fontsize=12)
        plt.legend(loc='upper right', fontsize=12)
        plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)

        # Grafiği kaydet
        plt.tight_layout()
        plt.savefig(f'./Results/{metric}_comparison_for_resolution_{resolution_value}.png', bbox_inches='tight', dpi=300)
        plt.close()

    print("Grafikler oluşturuldu ve kaydedildi.")

def generate_plots3(output_dir, serkans_prediction_col, hcphena_v2_col, ground_truth_col):
    global version
    # Verileri yükle
    file_path = os.path.join(output_dir, 'all_results.xlsx')
    df = pd.read_excel(file_path)

    def plot_metric(metric, title, ylabel, file_name):
        plt.figure(figsize=(18, 9))
        x = range(len(df['Resolution']))
        width = 0.2

        bars1 = plt.bar([p - width for p in x], df[f'{hcphena_v2_col}_{metric}'], width=width, label=hcphena_v2_col, color='#1f77b4')
        bars2 = plt.bar(x, df[f'{serkans_prediction_col}_{metric}'], width=width, label=serkans_prediction_col, color='#ff7f0e')
        bars3 = plt.bar([p + width for p in x], df[f'{ground_truth_col}_{metric}'], width=width, label=ground_truth_col, color='#2ca02c')

        plt.xlabel('Resolution', fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.title(title, fontsize=16)
        plt.xticks(x, df['Resolution'], rotation=90, fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)

        # Değerleri çubukların üzerine yaz
        def add_labels(bars):
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom', ha='center', fontsize=10)

        add_labels(bars1)
        add_labels(bars2)
        add_labels(bars3)

        # Grafiği kaydet
        output_path = os.path.join(output_dir, file_name)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

    # Mean Absolute Error (MAE) grafiği
    plot_metric('MAE', 'Mean Absolute Error', 'Mean Absolute Error', 'mean_absolute_error_'+str(version)+'.png')

    # R² grafiği
    plot_metric('R²', 'R²', 'R²', 'r2_'+str(version)+'.png')

    # RMSE grafiği
    plot_metric('RMSE', 'RMSE', 'RMSE', 'rmse_'+str(version)+'.png')

    # MAPE grafiği
    plot_metric('MAPE', 'MAPE', 'MAPE', 'mape_'+str(version)+'.png')

    print("Grafikler oluşturuldu ve kaydedildi.")

def create_filtered_ply(ply, indices, output_filename):
    # Yeni bir PlyFile nesnesi oluştur
    filtered_ply = PlyFile()

    # Header'ı kopyala
    filtered_ply.header = ply.header
    filtered_ply.elements = ply.elements.copy()

    # Filtrelenmiş vertex sayısını güncelle
    filtered_ply.elements["vertex"]["count"] = len(indices)

    # Filtrelenmiş vertexleri oluştur
    filtered_vertices = ply.vertices[indices]
    filtered_ply.vertices = filtered_vertices

    # Yeni header'ı oluştur
    filtered_ply.build_header()

    # Yeni ply dosyasını yaz
    filtered_ply.write(output_filename)
    print(f"Yeni ply dosyası '{output_filename}' oluşturuldu.")

def createPhenospexTrainSetFaster(plant_t="mungbean4train_out",data_t="plant"):

    DATA_DIR="./data/new/"+plant_t+"/" + str(65) + "/"
    outputFolder ="./data/new/"+plant_t+"/" + str(65) + "/"  + "4_voxelized_Phenospex_4Train/"+data_t
    # outputFolder ="./data/new/"+plant_t+"/" + str(65) + "/"  + "5_smooted_Phenospex_4Train/"+data_t
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    plyfile_directory = os.listdir( DATA_DIR+  "4_voxelized_Phenospex")
    for i, plyfile in enumerate(plyfile_directory):
        if "voxel" in plyfile:
            continue
        input_filename=DATA_DIR + "4_voxelized_Phenospex/" +plyfile
        output_filename=outputFolder + "/" + plyfile

        file_path = "./data/training_data/all_species_txt/" + data_t + "/" + plyfile[:-3] + "txt"
        # Check if the file exists
        if os.path.exists(file_path):
            numpy_data  =  pd.read_csv(file_path,sep=' ').iloc[:,:6].to_numpy()
        else:
            print("Could not found "+ file_path)
            continue

        numpy_points = numpy_data[:, 0:3].astype(np.float32)
        # read file with phenospex
        ply = PlyFile()
        ply.load(input_filename)
        ply_points= np.concatenate((ply.vertices["x"].reshape(-1, 1), ply.vertices["y"].reshape(-1, 1), ply.vertices["z"].reshape(-1, 1)), axis=1)

        tree = KDTree(ply_points)
        indices = []
        for point in numpy_points:
            dist, idx = tree.query(point)
            if dist < 1e-1:  # Mesafe belirlenen eşik değerinden küçükse
                indices.append(idx)

        create_filtered_ply(ply, indices,output_filename)

def writeNumberOfPlantPoints4AI(version):
    """Write AI segmented plant point counts to an Excel file."""

    # Define paths and files
    source_directory = os.path.join(DATA_DIR, f"7_partitioned_trays4plants_MLP_{version}")
    output_file = f"GroundTruth_HCv1_v2_predicted_{version}.xlsx"
    ground_truth_file = os.path.join(DATA_DIR, output_file)
    updated_output_path=os.path.join(DATA_DIR, f"GroundTruth_HCv1_v2_predicted_{version}_points.xlsx")

    # Load Excel file
    if os.path.exists(ground_truth_file):
        df_excel = pd.read_excel(ground_truth_file, engine='openpyxl', dtype={"unit": str})
    else:
        raise Exception("ground truth dosyası yok")

    # Extract file names from the Excel file
    date_values = df_excel.iloc[:, 0][:228]  # Adjust range if needed

    for i, date_value in enumerate(date_values):
        # Parse file name parts
        date_value_parts = date_value.split("_")
        prefix = date_value_parts[0]
        suffix = f"{int(date_value_parts[2]) - 1}_{int(date_value_parts[1]) - 1}"

        # Search for corresponding .txt file recursively
        pattern = os.path.join(source_directory, '**', f"{prefix}_{suffix}.txt")
        txt_files = glob.glob(pattern, recursive=True)

        if len(txt_files) == 0:
            continue

        try:
            # Read the .txt file as a NumPy array and count rows
            data = np.loadtxt(txt_files[0], delimiter=',')
            row_count = data.shape[0]

            # Update the Excel file
            df_excel.at[i, "Serkans prediction Point Count"] = row_count

        except (FileNotFoundError, IOError, ValueError) as e:
            print(f"Error: Could not process file {txt_files[0]} - {e}")

    # Save the updated Excel file
    df_excel.to_excel(updated_output_path, index=False, engine='openpyxl')
    print(f"Updated Excel file saved to {updated_output_path}")

    """ write Phenospex segmented plant points """

def combine_excels_into_sheets(version):
    """
    Combine Excel files from subdirectories into a single Excel file with sheets for each subdirectory.

    Args:
        version (str): The version string to construct the Excel file names.
    """
    # Define the base directory and output file path
    output_file_name = f"GroundTruth_HCv1_v2_predicted_{version}_ALL_Combined.xlsx"
    output_path = os.path.join(org_Data_Dir, output_file_name)
    writer = pd.ExcelWriter(output_path, engine='openpyxl')

    # Iterate through subdirectories 55 to 66
    for folder_num in range(55, 66):  # Iterates from 55 to 66 inclusive
        folder_path = os.path.join(org_Data_Dir, str(folder_num))
        excel_file_name = f"GroundTruth_HCv1_v2_predicted_{version}_points.xlsx"
        excel_file_path = os.path.join(folder_path, excel_file_name)

        if os.path.exists(excel_file_path):
            try:
                # Read the Excel file
                df = pd.read_excel(excel_file_path, engine='openpyxl')
                # Add it as a new sheet in the output Excel file
                df.to_excel(writer, sheet_name=str(folder_num), index=False)
                print(f"Added data from {excel_file_path} to sheet: {folder_num}")
            except Exception as e:
                print(f"Error processing {excel_file_path}: {e}")
        else:
            print(f"File not found: {excel_file_path}")

    # Save the combined Excel file

    # writer.to_excel(output_path, index=False, engine='openpyxl')
    writer.close()
    print(f"Combined Excel file saved to {output_path}")


def write_whole_file_plant_counts(version):
    output_file = os.path.join(org_Data_Dir, "whole_file_plant_counts_"+str(version)+".xlsx")
    results = []

    for folder_num in range(60, 61):  # Iterates through 55 to 65 inclusive
        base_folder = os.path.join(org_Data_Dir, str(folder_num))
        segmentation_folder = os.path.join(base_folder, "6_segmentation_result_plants_MLP_"+str(version))
        phenospex_folder = os.path.join(base_folder, "9_Biomass_prediction_Phenospex_"+str(version))

        if os.path.exists(segmentation_folder):
            for file in os.listdir(segmentation_folder):
                if file.endswith(".txt"):
                    numpy_file_path = os.path.join(segmentation_folder, file)

                    try:
                        # Read numpy data and count rows
                        numpy_data = np.loadtxt(numpy_file_path, delimiter=',')
                        row_count = numpy_data.shape[0]

                        # Find corresponding PLY file
                        ply_pattern = os.path.join(phenospex_folder, f"{file.split('.')[0][:-6]}mesh.ply")
                        if os.path.exists(ply_pattern):
                            ply_file_path = ply_pattern
                            ply = PlyFile()
                            ply.load(ply_file_path)
                            vertex_count = ply.elements["vertex"]["count"]
                        else:
                            vertex_count = None  # PLY file not found

                        # Add to results
                        results.append([file, row_count, vertex_count])

                    except Exception as e:
                        print(f"Error processing {file}: {e}")

    # Write results to Excel
    df = pd.DataFrame(results, columns=["file name", "AI Segmented Plant Count", "Coordinate Based Segmented Plant Count"])
    df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"Results written to {output_file}")


def updateJson(json_file_path,resolution):
    # JSON dosyasını oku
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # "resolution" değerini güncelle
    data['image3d']['resolution'] = resolution # Yeni değeri burada belirleyin

    # Güncellenmiş JSON verisini dosyaya geri yaz
    with open(json_file_path, 'w') as file:
        json.dump(data, file, indent=4)


def updatePhenospexJson(resolution):
    global DATA_DIR

    rounded_value = round(resolution * 100)
    DATA_DIR = org_Data_Dir + str(int(rounded_value)) + "/"
    # JSON dosyasının yolu
    json_file_path = './PhenoSpexExe/parameter_extraction.json'
    updateJson(json_file_path,resolution)
    json_file_path = './PhenoSpexExe/parameter_extraction_merged.json'
    updateJson(json_file_path,resolution)
    json_file_path = './PhenoSpexExe/preprocessing.json'
    updateJson(json_file_path,resolution-0.01)
    json_file_path = './PhenoSpexExe/preprocessing_voxelization.json'
    updateJson(json_file_path,resolution-0.01)

    # set max z
# Exp56_1_Vijay= Chickpea,  Exp56_1_MB_9001 : mungbean,Exp56_1_Chakti : Pearl millet,Exp56_1_CSV22 : Sorghum
    if Raw_Data_Dir.__contains__("Chakti")  :
        cutting_z=80
    elif Raw_Data_Dir.__contains__("Vijay")  :
        cutting_z=110
    elif Raw_Data_Dir.__contains__("CSV22") or Raw_Data_Dir.__contains__("9001") :
        cutting_z=80
    else:
        #cutting_z = 80
        cutting_z = 118 # for data count exp 57 chickpea
    updateJson_max_z('./PhenoSpexExe/preprocessing.json',cutting_z)
    updateJson_max_z('./PhenoSpexExe/preprocessing_voxelization.json',cutting_z)

if __name__ == "__main__":
    global version
    # When changing the height threshold, it is sufficient to update "max_z": 120 only inside the preprocess section.
    # When changing the resolution, all 4 JSON files must be updated accordingly.
    # After each test, all files after the "merged" step should be saved, including the result Excel file.

    # preprocess()
    version =3# 3 means x,y,z,red,green,blue + nir
    #
    for resolution in [round(0.60 + i * 0.01, 2) for i in range(1)]:
        updatePhenospexJson(resolution)
        # preprocessWithPhenospex()
        # segmentWithMLPPhenospex(version=version,is_smooted=1)
        # convertOurOutput2PhenospexFormat7(version=version)
        calculateBiomassSeperately(version=version)
        writeAllResults2OneFile(version=version)
        calculateBiomass4PhenoPred()
        writeAllPhenospexResults2OneFile()
        gc.collect()
        writeNumberOfPlantPoints4AI(version=version)

        visualize_and_save_model(model_path='./models/best_NN_model_' + str(version)+ '.h5')

    write_whole_file_plant_counts(version)
    combine_excels_into_sheets(version)

    #********** Önceden etiketlenmiş verileri yeni oluşturulan phenspoex verielri içerisinden seç***********
    # createPhenospexTrainSetFaster("mungbean4train_out")
    # createPhenospexTrainSetFaster("mungbean4train_out","background")


    # #Other Funcitons
    # all_results = process_all_files(DATA_DIR[:-4]) # for voxel  based operatioon
    # all_results = process_all_files(org_Data_Dir)
    # save_results(all_results,version)
    # generate_plots((DATA_DIR[:-4]), "Serkans_prediction", "Phenospex prediction")
    # folder_list = ['Exp56_1_Vijay', 'Exp56_1_MB_9001', 'Exp56_1_Chakti', 'Exp56_1_CSV22']  # Klasör isimleri
    # generate_plots_for_all_folders('./data/new/',folder_list, resolution_value=60, serkans_prediction_col="Serkans_prediction", traditional_col="Phenospex prediction",version=version)
    # #içinde 3 olanlar phenoıspex in kendi ilk yaptıgı degerleri de grafige ekleyip görmek için. Bunu degiştirirsen process_all_files içindeki process_file3 olarak degiştir
    # generate_plots3(org_Data_Dir, "Serkans_prediction", "Phenospex prediction", "HCphenaV2")
    # Fonksiyonu çağır
    # tempFunc()
    # resolution için hem voxel size ı hemde 3 json dosyasını güncelle
    # segmentRansac()
    # RGBSegmentation()