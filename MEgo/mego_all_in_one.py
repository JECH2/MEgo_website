# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 21:13:31 2020

@author: User
"""


#%% Set working directory
import os
import sys

homedir = os.path.expanduser("~")
project = r"\Project II" 
username = os.environ.get("USERNAME")
platform = sys.platform

if username != "양지성":
    if platform != "win32":
        homedir = homedir.replace("\\", "/")
        project = project.replace("\\", "/")
        wd = homedir + r"/Dropbox" + project
    else:
        if username == "최은진":
            wd = r"{}\Dropbox{}".format(homedir, project)
        else:
            wd = r"{}\Dropbox\Shared Folder\Courses\GCT501{}".format(homedir, project)
else:
    wd = r"|{}\Dropbox\Shared Folder\Courses\GCT501{}".format(homedir, project)

os.chdir(wd); os.getcwd()

#%% Read in data\
import pandas as pd
data = pd.read_excel("data.xlsx", sheet_name="JS")

def get_report(data):
    import pandas as pd
    import numpy as np
    import mego_functions as f

    event_whole, thought_whole, people_whole, experience_tokenized, people_tokenized_spacing = f.preprocess_all(data)

    f.wordcloud_all(event_whole, thought_whole)

    f.get_people_count(people_whole)

    things_joy, things_sadness, things_fear = f.get_similar_words(experience_tokenized, 25, 5, 2, 4, 0)

    count_matrix, term_frequency, feature_names = f.get_dtm(experience_tokenized, 0.1)

    co_matrix, term_id = f.create_co_occurences_matrix(feature_names, experience_tokenized)

    f.word_cluster_plot(co_matrix, feature_names, term_frequency, 13)

    list_of_people = np.unique(people_tokenized_spacing)
    ego_matrix_exp = f.get_ego_matrix_exp(data, list_of_people)

    pos_people_close, pos_poeple_far, neg_people_close, neg_people_far = f.plot_ego_network_exp(ego_matrix_exp, list_of_people, 30)

    report_data = {
        "things_joy": things_joy,
        "thing_sadness": things_sadness, 
        "things_fear": things_fear
        }
    
    return report_data, pos_people_close, pos_poeple_far, neg_people_close, neg_people_far

report_data, pos_people_close, pos_poeple_far, neg_people_close, neg_people_far = get_report(data)