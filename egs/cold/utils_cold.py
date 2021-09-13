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
    audio_folder = '/media/jvel/data/audio/cold2'
    audio_list = os.listdir(audio_folder)
    for_csv = []

    df_labels = pd.read_csv("data/labels.csv", dtype=str)
    labels = df_labels.label.values

    label_map = {'1': 'cold', '0': 'healthy'}

    for class_id, filename in zip(labels, audio_list):
        for_csv.append([label_map[class_id], filename, class_id])
    for_csv.sort()
    with open("data/cold_meta.csv", "w+") as my_csv:
        csv_writer = csv.writer(my_csv, delimiter=',')
        csv_writer.writerow(['folder', 'filename', 'label'])
        csv_writer.writerows(for_csv)


def create_class_label_idx_csv():
    """
    Creates csv file required for the AST pipeline in the form of:
        index,mid,display_name
        3, /m/21rwj03, hc
    """
    audio_folder = '/media/jvel/data/audio/cold2'
    audio_list = os.listdir(audio_folder)
    for_csv = []

    df_labels = pd.read_csv("data/labels.csv", dtype=str)
    labels = df_labels.label.values

    label_map = {'1': 'cold', '0': 'healthy'}

    for class_id, filename in zip(labels, audio_list):
        for_csv.append([class_id, '/m/21rwj' + str(class_id).zfill(2), label_map[class_id]])
        # for_csv.append([class_label, '/m/21rwj' + str(class_label).zfill(2), folder])
    with open("data/cold_class_label_indices.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerow(['index', 'mid', 'display_name'])
        csvWriter.writerows(for_csv)


create_csv_for_ast()
create_class_label_idx_csv()