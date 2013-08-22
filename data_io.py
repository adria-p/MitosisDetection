"""
Created on Jun 21, 2013
    
    Dummy functions to get the paths of the files and folders 
    used by this project.
    
@author: Bibiana and Adria 
"""

import csv
import json
import os
import pickle
from skimage.io import ImageCollection

def get_paths():
    paths = json.loads(open("SETTINGS.json").read())
    for key in paths:
        paths[key] = os.path.expandvars(paths[key])
    return paths

def get_train_df():
    train_path = get_paths()["train_data_path"]
    train_format = get_paths()["train_data_format"]
    real_path = os.path.join(os.getcwd(), train_path)
    imageCollections = [ImageCollection(os.path.join(train_path, subdirname, train_format))   for _, dirnames, _ in os.walk(real_path) for subdirname in dirnames]
    return imageCollections

def get_savez_name():
    file_name = get_paths()["savez_file_name"]
    return file_name

def get_positive_folder():
    file_name = get_paths()["positive_folder"]
    return file_name

def get_negative_folder():
    file_name = get_paths()["negative_folder"]
    return file_name

def get_valid_df():
    valid_path = get_paths()["valid_data_path"]
    train_format = get_paths()["train_data_format"]
    real_path = os.path.join(os.getcwd(), valid_path)
    imageCollections = [ImageCollection(os.path.join(valid_path, subdirname, train_format))   for _, dirnames, _ in os.walk(real_path) for subdirname in dirnames]
    print imageCollections
    return imageCollections
    # return pd.read_csv(valid_path, converters=converters)

def save_model(model):
    out_path = get_paths()["model_path"]
    pickle.dump(model, open(out_path, "w"))

def load_model():
    in_path = get_paths()["model_path"]
    return pickle.load(open(in_path))

def write_submission(names, coord, predictions):
    prediction_path = get_paths()["prediction_path"]
    writer = csv.writer(open(prediction_path, "w"), lineterminator="\n")
    count = 0
    for name in names:
        #writer.writerow([name] + coord[count] + [predictions[count]])
        count += 1
        
def write_submission_nice(names, coord, predictions):
    currentImage = names[0]
    csvToWrite = names[0].split(".")[0] + get_paths()["prediction_path"]
    writer = csv.writer(open(csvToWrite, "w"), lineterminator="\n")
    count = 0
    for name in names:
        if name != currentImage:
            currentImage = name
            csvToWrite = name.split(".")[0] + get_paths()["prediction_path"]
            writer = csv.writer(open(csvToWrite, "w"), lineterminator="\n")
        if(round(predictions[count]) == 1):
            pass#writer.writerow(coord[count])
        count += 1

def get_test_folder():
    file_name = get_paths()["test_folder"]
    return file_name

def get_testing_folder():
    file_name = get_paths()["testing_folder"]
    return file_name

def get_training_folder():
    file_name = get_paths()["training_folder"]
    return file_name

def get_wndchrm_dataset():
    file_name = get_paths()["wndchrm_dataset"]
    return file_name

def get_wndchrm_datafit():
    file_name = get_paths()["wndchrm_datafit"]
    return file_name

def get_wndchrm_datafit_test():
    file_name = get_paths()["wndchrm_datafit_test"]
    return file_name

def get_training_old_folder():
    file_name = get_paths()["training_old_folder"]
    return file_name

def get_testing_old_folder():
    file_name = get_paths()["testing_old_folder"]
    return file_name

def get_savez_name_test():
    file_name = get_paths()["savez_file_name_test"]
    return file_name

def get_wndchrm_dataset_test():
    file_name = get_paths()["wndchrm_dataset_test"]
    return file_name