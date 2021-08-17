import os
import csv
from os import walk
import pandas as pd


def create_csv_for_ast():
    """
    Creates csv file required for the AST pipeline in the form of:
        folder,filename,label
        hc,006B_feher.wav,3
    """
    audio_folder = '/media/jvel/data/audio/demencia94B-wav16k'
    audio_list = os.listdir(audio_folder)
    for_csv = []

    df_labels = pd.read_csv("data/labels.csv", dtype=str)
    labels = df_labels.label.values

    label_map = {'1': 'alzheimer', '2': 'mci', '3': 'hc'}

    for class_id, filename in zip(labels, audio_list):
        for_csv.append([label_map[class_id], filename, class_id])
    with open("dementia_meta.csv", "w+") as my_csv:
        csv_writer = csv.writer(my_csv, delimiter=',')
        csv_writer.writerow(['folder', 'filename', 'label'])
        csv_writer.writerows(for_csv)


def create_class_label_idx_csv():
    """
    Creates csv file required for the AST pipeline in the form of:
        index,mid,display_name
        3, /m/21rwj03, hc
    """
    audio_folder = 'data/dlrdata/audio'
    for_csv = []
    for index, folder in enumerate(class_folders):
        foldername = folder
        for (dirpath, dirnames, filenames) in walk(os.path.join(data_folder, folder)):
            wavfiles = [f for f in filenames if f.endswith('wav')]
        for wavfile in wavfiles:
            class_label = index
            for_csv.append([class_label, '/m/21rwj' + str(class_label).zfill(2), folder])
    with open("data/dlr_class_label.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerow(['index', 'mid', 'display_name'])
        csvWriter.writerows(for_csv)
