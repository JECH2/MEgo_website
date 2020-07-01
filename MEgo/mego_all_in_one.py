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
from pathlib import Path

#%% Read in data\
#data = pd.read_excel("data.xlsx", sheet_name="JS")
#CSV_PATH = 'report/jiseong.csv'
#data = pd.read_csv(CSV_PATH,encoding='latin-1')
#username = os.path.basename(CSV_PATH)[:-4]

def get_report(username):

    dirname = os.path.realpath(__file__)
    wd = dirname[:-len("mego_all_in_one.py")] + "report"

    os.chdir(wd)
    data = pd.read_csv(wd+'/'+username+'.csv', encoding='latin-1')

    event_whole, thought_whole, people_whole, experience_tokenized, people_tokenized_spacing = preprocess_all(data)

    wordcloud_all(event_whole, thought_whole, wd, username)

    get_people_count(people_whole, username)

    things_joy, things_sadness, things_fear = get_similar_words(experience_tokenized, 25, 5, 2, 4, 0)

    count_matrix, term_frequency, feature_names = get_dtm(experience_tokenized, 0.1)

    co_matrix, term_id = create_co_occurences_matrix(feature_names, experience_tokenized)

    word_cluster_plot(co_matrix, feature_names, term_frequency, 13, username)

    list_of_people = np.unique(people_tokenized_spacing)
    ego_matrix_exp = get_ego_matrix_exp(data, list_of_people)

    pos_people_close, pos_people_far, neg_people_close, neg_people_far = plot_ego_network_exp(ego_matrix_exp, list_of_people, 30,username)

    report_data = {
        "things_joy": things_joy,
        "things_sadness": things_sadness,
        "things_fear": things_fear,
        "pos_people_close":pos_people_close,
        "pos_people_far":pos_people_far,
        "neg_people_close":neg_people_close,
        "neg_people_far":neg_people_far,
        }
    
    return report_data

#print(report_data)
#report_data = get_report(data, username, wd)