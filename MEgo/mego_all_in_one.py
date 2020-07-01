# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 21:13:31 2020

@author: User
"""


#%% Set working directory
import os
import sys
import pandas as pd
import numpy as np
from .mego_functions import *

#%% Read in data\
#data = pd.read_excel("data.xlsx", sheet_name="JS")
#CSV_PATH = 'report/jiseong.csv'
#data = pd.read_csv(CSV_PATH,encoding='latin-1')
#username = os.path.basename(CSV_PATH)[:-4]

def get_report(csv_path, username):
    data = pd.read_csv(csv_path, encoding='latin-1')

    event_whole, thought_whole, people_whole, experience_tokenized, people_tokenized_spacing = preprocess_all(data)

    wordcloud_all(event_whole, thought_whole)

    get_people_count(people_whole)

    things_joy, things_sadness, things_fear = get_similar_words(experience_tokenized, 25, 5, 2, 4, 0)

    count_matrix, term_frequency, feature_names = get_dtm(experience_tokenized, 0.1)

    co_matrix, term_id = create_co_occurences_matrix(feature_names, experience_tokenized)

    word_cluster_plot(co_matrix, feature_names, term_frequency, 13)

    list_of_people = np.unique(people_tokenized_spacing)
    ego_matrix_exp = get_ego_matrix_exp(data, list_of_people)

    pos_people_close, pos_people_far, neg_people_close, neg_people_far = plot_ego_network_exp(ego_matrix_exp, list_of_people, 30)

    report_data = {
        "things_joy": things_joy,
        "thing_sadness": things_sadness, 
        "things_fear": things_fear,
        "pos_people_close":pos_people_close,
        "pos_people_far":pos_people_far,
        "neg_people_close":neg_people_close,
        "neg_people_far":neg_people_far,
        }
    
    return report_data

#report_data = get_report(CSV_PATH,username)

#print(report_data)