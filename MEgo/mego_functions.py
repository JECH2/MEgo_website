# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 22:04:46 2020

@author: User
"""


# Reduce list of lists
def reduce(arr):
    reduced_list = []
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            reduced_list.append(arr[i][j])
        
    return reduced_list


# Preprocess and tokenize
def preprocess(series):
    # Lowercase all letters
    series = series.str.lower()
    series.fillna("")
    
    # Tokenize all words
    series_tokenized = [str(series[i]).split(" ") for i in range(series.shape[0])]
    
    # Remove stopwords
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
    from spacy.lang.en.stop_words import STOP_WORDS
    from stop_words import get_stop_words
    
    # Set stopwords
    custom_stopwords = ["when", "where", "what", "how", "why"]
    stopwords = list(set(custom_stopwords + get_stop_words('en') + stopwords.words('english') + list(STOP_WORDS) + list(ENGLISH_STOP_WORDS)))
    
    # Remove stopwords
    for i in range(len(series_tokenized)):
        for j in range(len(series_tokenized[i])):
            for k in range(len(stopwords)):
                if stopwords[k] == series_tokenized[i][j]:
                    series_tokenized[i][j] = ""
                    
    # Remove empty strings
    import numpy as np
    
    series_tokenized = [list(filter(None, series_tokenized[i])) for i in range(len(series_tokenized))]
    
    # Remove non-alphabetic characters.
    import re
    
    nonalpha_re = re.compile("[^a-zA-Z0-9]")
    for i in range(len(series_tokenized)):
        for j in range(len(series_tokenized[i])):
            nonalpha = re.findall(nonalpha_re, series_tokenized[i][j])
            for k in range(len(nonalpha)):
                series_tokenized[i][j] = series_tokenized[i][j].replace(nonalpha[k], "")
 
    # Remove stopwords
    for i in range(len(series_tokenized)):
        for j in range(len(series_tokenized[i])):
            for k in range(len(stopwords)):
                if stopwords[k] == series_tokenized[i][j]:
                    series_tokenized[i][j] = ""
                    
    # Remove empty strings
    import numpy as np
    
    series_tokenized = [list(filter(None, series_tokenized[i])) for i in range(len(series_tokenized))]
    
    # Lemmatize all tokens
    from nltk import WordNetLemmatizer, pos_tag
    nltk.download('averaged_perceptron_tagger')
    
    WNlemma = nltk.WordNetLemmatizer()
    for i in range(len(series_tokenized)):
        for j in range(len(series_tokenized[i])):
            series_tokenized[i][j] = WNlemma.lemmatize(series_tokenized[i][j], 'v')
            series_tokenized[i][j] = WNlemma.lemmatize(series_tokenized[i][j], 'n')
            
    return series_tokenized

# Preprocess all
def preprocess_all(data):
    import pandas as pd
    
    event = data.event
    thought = data.thoughts
    #emotion = data.emotion
    emotion_label = data.emotion_label
    people = data.related_people
    people_tokenized = [[people.iloc[i].replace(" ", "")] if type(people.iloc[i]) == str else [","] for i in range(len(people))]
    people_tokenized_spacing = reduce([people.iloc[i].split(",") if type(people.iloc[i]) == str else [","] for i in range(len(people))])
    people_tokenized_spacing = [people_tokenized_spacing[i].strip()  for i in range(len(people)) if people_tokenized_spacing[i].strip() != ","]
    experience = pd.Series(["{} {} {} {}".format(event[i], thought[i], emotion_label[i], people_tokenized[i][0].replace(","," ")) for i in range(data.shape[0])])
    
    # Get tokens from each series
    event_tokenized = preprocess(event)
    thought_tokenized = preprocess(thought)
    #emotion_tokenized = preprocess(emotion)
    emotion_label_tokenized = preprocess(emotion_label)
    experience_tokenized = preprocess(experience)
    
    # Apply bigrams
    # event_tokenized_bi = f.ngram_tokenize(event_tokenized, 2)
    # thought_tokenized_bi = f.ngram_tokenize(thought_tokenized, 2)
    # experience_tokenized_bi = f.ngram_tokenize(experience_tokenized, 2)
    
    # Merge tokens
    event_whole = reduce(event_tokenized)
    thought_whole = reduce(thought_tokenized)
    # emotion_whole = f.reduce(emotion_tokenized)
    # emotion_label_whole = f.reduce(emotion_label_tokenized)
    people_whole = reduce([reduce(people_tokenized)[i].split(",") for i in range(len(reduce(people_tokenized))) if reduce(people_tokenized)[i] != ","])
    experience_whole = reduce(experience_tokenized)
    
    return event_whole, thought_whole, people_whole, experience_tokenized, people_tokenized_spacing

# Wordcloud_all
def wordcloud_all(event_whole, thought_whole, wd):
    import os
    
    event_wordcloud = wordcloud(event_whole, "event", wd)
    thought_wordcloud = wordcloud(thought_whole, "thought", wd)

# Ngram tokenization
def ngram_tokenize(list_of_tokens, n):
    from nltk import ngrams
    for i in range(len(list_of_tokens)):
         list_of_ngrams = list(ngrams(list_of_tokens[i], n))
         for t1, t2 in list_of_ngrams:
             list_of_tokens[i].append(" ".join([t1, t2]))
    return list_of_tokens

# Plot wordcloud
def wordcloud(tokens_whole, name, wd):
    from wordcloud import WordCloud
    import os
    
    font_path = "C:\Windows\Fonts"
    os.chdir(font_path)
    wordcloud = WordCloud(font_path="ariblk.ttf", background_color='#A67D97', colormap="BuPu")
    array_wordcloud = wordcloud.generate_from_text(" ".join(tokens_whole)).to_array()
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    #plt.imshow(array_wordcloud, interpolation="bilinear")
    plt.axis('off')
    #plt.show()
    
    os.chdir(wd)
    plt.savefig("wordcloud_{}.png".format(name), transparent=True)

# Get similar words by emotions
def get_similar_words(experience_tokenized, size, window, min_count, workers, sg):
    from gensim.models import Word2Vec

    # size, window, min_count, workers, sg = 25, 5, 2, 4, 0
    # event_model = Word2Vec(sentences=event_tokenized, size=size, window=window, min_count = min_count , workers=workers, sg=sg)
    # thought_model = Word2Vec(sentences=thought_tokenized, size=size, window=window, min_count = min_count , workers=workers, sg=sg)
    # emotion_model = Word2Vec(sentences=emotion_tokenized, size=size, window=window, min_count = min_count , workers=workers, sg=sg)
    # emotion_label_model = Word2Vec(sentences=emotion_label_tokenized, size=size, window=window, min_count = min_count , workers=workers, sg=sg)
    experience_model = Word2Vec(sentences=experience_tokenized, size=size, window=window, min_count = min_count , workers=workers, sg=sg)
    joy = experience_model.most_similar("joy")
    sadness = experience_model.most_similar("sadness")
    fear = experience_model.most_similar("fear")
    
    return " ".join([joy[:5][i][0] for i in range(5)]), " ".join([sadness[:5][i][0] for i in range(5)]), " ".join([fear[:5][i][0] for i in range(5)])

# Get people count
def get_people_count(people_whole):
    import nltk
    import seaborn as sns
    from matplotlib import pyplot as plt
    import pandas as pd
    
    # Get the unigram distribution and convert it to the dataframe
    fd = nltk.FreqDist(people_whole).most_common(20)
    terms_uni = []
    counts_uni = []
    for term, count in fd:
        terms_uni.append(term)
        counts_uni.append(count)
    fd_dict = {
        'terms': terms_uni,
        'counts': counts_uni
        }
    fd_df_uni = pd.DataFrame(data=fd_dict, columns=fd_dict.keys()).sort_values(by=["counts", "terms"], ascending=False).reset_index(drop=True)
    
    # Plot
    with sns.axes_style({"font.family": ["Arial"]}):
        sns.catplot(y="terms", x="counts", kind="bar", palette="BuPu_d", edgecolor=".6", data=fd_df_uni).savefig("people_count.png", transparent=True)
        # g.fig.suptitle("Your Experience Keywords")

# Get DTM
def get_dtm(doc_tokenized, df):
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd

    vect = CountVectorizer(min_df=df, ngram_range=(1, 1))
    doc_tokenized_by_doc = [" ".join(doc_tokenized[i]) for i in range(len(doc_tokenized))]
    doc_vec = vect.fit_transform(doc_tokenized_by_doc)
    feature_names = vect.get_feature_names()
    dense = doc_vec.todense()
    denselist = dense.tolist()
    count_matrix = pd.DataFrame(denselist, columns=feature_names)
    term_frequency = {count_matrix.columns[i]: count_matrix.sum(axis=0)[i] for i in range(count_matrix.shape[1])}
    
    return count_matrix, term_frequency, feature_names


# Cooccurrence matrix from DTM
def create_co_occurences_matrix(allowed_words, documents):
    import numpy as np
    import itertools
    from scipy.sparse import csr_matrix


    #print(f"allowed_words:\n{allowed_words}")
    #print(f"documents:\n{documents}")
    word_to_id = dict(zip(allowed_words, range(len(allowed_words))))
    documents_as_ids = [np.sort([word_to_id[w] for w in doc if w in word_to_id]).astype('uint32') for doc in documents]
    row_ind, col_ind = zip(*itertools.chain(*[[(i, w) for w in doc] for i, doc in enumerate(documents_as_ids)]))
    data = np.ones(len(row_ind), dtype='uint32')  # use unsigned int for better memory utilization
    max_word_id = max(itertools.chain(*documents_as_ids)) + 1
    docs_words_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(len(documents_as_ids), max_word_id))  # efficient arithmetic operations with CSR * CSR
    words_cooc_matrix = docs_words_matrix.T * docs_words_matrix  # multiplying docs_words_matrix with its transpose matrix would generate the co-occurences matrix
    words_cooc_matrix.setdiag(0)
    
    #print(f"words_cooc_matrix:\n{words_cooc_matrix.todense()}")
    return words_cooc_matrix, word_to_id

## Community plotting
def community_layout(g, partition):
    import numpy as np
    import matplotlib.pyplot as plt
    import networkx as nx

    """
    Compute the layout for a modular graph.


    Arguments:
    ----------
    g -- networkx.Graph or networkx.DiGraph instance
        graph to plot

    partition -- dict mapping int node -> int community
        graph partitions


    Returns:
    --------
    pos -- dict mapping int node -> (float x, float y)
        node positions

    """

    pos_communities = _position_communities(g, partition, scale=3.)

    pos_nodes = _position_nodes(g, partition, scale=1.)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos

def _position_communities(g, partition, **kwargs):
    import networkx as nx
    
    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos

def _find_between_community_edges(g, partition):
    import networkx as nx

    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges

def _position_nodes(g, partition, **kwargs):
    import networkx as nx
    import math
    """
    Positions nodes within communities.
    """

    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs, k=10)
        pos.update(pos_subgraph)

    return pos

# Plot the network
def word_cluster_plot(co_matrix, feature_names, term_frequency, k):
    # to install networkx 2.0 compatible version of python-louvain use:
    # pip install -U git+https://github.com/taynaud/python-louvain.git@networkx2
    import community
    import networkx as nx
    import numpy as np
    import matplotlib.pyplot as plt
    
    A = co_matrix.todense()
    G = nx.from_numpy_matrix(A)
    node_mapping = {key: value for key, value in enumerate(feature_names)}
    partition = community.community_louvain.best_partition(G)
    pos = community_layout(G, partition)
    pos_terms = {node_mapping[terms]: value for terms, value in pos.items()}
    k = len(feature_names)*k
    nx.relabel.relabel_nodes(G, mapping=node_mapping, copy=False)
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    nx.draw_networkx(G, pos=pos_terms, font_size=14, font_family = "fantasy", font_weight="bold",
                     font_color="#FFFFFF", node_size=np.multiply(list(term_frequency.values()),k), 
                     node_color=list(partition.values()), cmap=plt.cm.Set1,
                     edge_color="#FFFFFF"); plt.savefig("word clustering.png", transparent=True, dpi=600)
    return

# Convert
def convert_time(ms):
    from unit_converter.converter import convert, converts
    h = converts(str(ms)+" ms", 'h')
    h = round(float(h), 2)
    return h

# Get the experience sign (pos/neg)
def get_exp_sign(emotion):
    exp_sign_mapping = {
    "positive": ["joy", "trust", "surprise", "anticipation", "ecstasy", "serenity", "admiration", "vigilance", "amazement", "interest", "acceptance", "distraction"],
    "negative": ["fear", "sadness", "disgust", "anger", "rage", "terror", "grief", "loathing", "annoyance", "apprehension", "pensiveness", "boredom"]
    }
    if emotion in exp_sign_mapping["positive"]:
        sign = "+"
    else:
        sign = "-"
    return sign

def get_kakao_data(list_of_filenames, wd):
    import os
    
    os.chdir(wd)
    frequency_kakao_incoming_list = []
    length_kakao_incoming_list = []
    frequency_kakao_outgoing_list = []
    length_kakao_outgoing_list = []
    recent_list = []
    
    for filename in list_of_filenames:
        with open(filename, mode="r", encoding="utf-8") as f:
            content = f.readlines()
        
        # Make the text into tables
        from datetime import datetime as dt
        from datetime import timedelta
        import re
        
        line_re = re.compile("^(January|Feburary|March|April|May|June|July|August|September|October|November|December)")
        content = [content[4:][i].replace(",",":::").replace(" :", ":::").replace("\n", "") for i in range(len(content[4:])) if content[4:][i] != "\n" and len(content[4:][i]) > 24 and re.match(line_re, content[4:][i])]
        content = [content[i].split(":::") for i in range(len(content)) if len(content[i].split(":::")) == 5]
        date = ["".join(content[i][:2]) for i in range(len(content))]
        date_merged = [dt.strptime(date[i], "%B %d %Y") for i in range(len(content))]
        
        # Create an empty dataframe and fill in
        import pandas as pd 
        import numpy as np
        
        kakao = pd.DataFrame(np.zeros(shape=(len(content),5)), columns=["month_day", "year", "time", "name", "text"])
        for i in range(kakao.shape[0]):
            for j in range(kakao.shape[1]):
                kakao.iloc[i,j] = content[i][j]
        
        kakao.drop(["month_day", "year", "time"], axis=1, inplace=True)
        kakao["date"] = date_merged
        kakao = kakao[["date", "name", "text"]]
        
        # Incoming messages
        kakao_incoming = kakao[kakao.name != " you"]
        frequency_kakao_incoming = len(kakao_incoming.text)
        length_kakao_incoming = sum(kakao_incoming.text.apply(len))
        
        # Outcoming messages
        kakao_outgoing = kakao[kakao.name == " you"]
        frequency_kakao_outgoing = len(kakao_outgoing.text)
        length_kakao_outgoing = sum(kakao_outgoing.text.apply(len))
        
        # Recent
        kakao.reset_index(drop=True, inplace=True)
        add_day = timedelta(days=2)
        today = dt.strptime("20200626", "%Y%m%d") 
        try:
            recent = min([today-kakao.date[i] for i in range(len(kakao.date))]) + add_day
        except:
            recent = "No Kakao history"
        
        # Fill in the list
        frequency_kakao_incoming_list.append(frequency_kakao_incoming)
        length_kakao_incoming_list.append(length_kakao_incoming)
        frequency_kakao_outgoing_list.append(frequency_kakao_outgoing)
        length_kakao_outgoing_list.append(length_kakao_outgoing)
        recent_list.append(recent)
        
        # Craete matrix and transpose
        kakao_array = [frequency_kakao_incoming_list, length_kakao_incoming_list, frequency_kakao_outgoing_list, length_kakao_outgoing_list, recent_list]
        kakao_array_T = np.transpose(kakao_array)
                
    return kakao_array_T

def get_call_data(social_data, call_people, list_of_people):
    import re
    import pandas as pd
    from datetime import datetime as dt
    from datetime import timedelta
    import numpy as np
    
    call_re = re.compile("^(call)")
    call_people_mapping = {key: value for key, value in zip(list_of_people, call_people)}
    call_log = [social_data[i] for i in range(len(social_data)) if re.match(call_re, social_data[i])]
    call_log = pd.read_csv(call_log[0])
    call_date = [dt.strptime(call_log.Date[i], "%m/%d/%y %H:%M %p") for i in range(len(call_log.Date))]
    call_log.drop(["Number"], inplace=True, axis=1)
    call_log.Date = call_date
    call_log = call_log[["Date", "Name", "Type", "Duration(secs)"]]
    call_log.columns = ["Date", "Name", "Type", "Duration_secs"]
    call_log_list = []
    recent_list = []
    for person in list_of_people:
        call_log_by_person = call_log[(call_log.Name == call_people_mapping[person]) & (call_log.Type != "Missed") & (call_log.Duration_secs > 0)]
        call_log_list.append(call_log_by_person)
        
    frequency_call_incoming_list = []
    length_call_incoming_list = []
    frequency_call_outgoing_list = []
    length_call_outgoing_list = []
        
    for person in call_log_list:
        # Incoming calls
        frequency_call_incoming = person[person.Type == "Incoming"].shape[0]
        length_call_incoming = sum(person.Duration_secs[person.Type == "Incoming"])
        
        # Outgoing calls
        frequency_call_outgoing = person[person.Type == "Outgoing"].shape[0]
        length_call_outgoing = sum(person.Duration_secs[person.Type == "Outgoing"])
        
        # Recent
        person.reset_index(drop=True, inplace=True)
        add_day = timedelta(days=2)
        today = dt.strptime("20200626", "%Y%m%d")
        try:
            recent = min([today-person.Date[i] for i in range(len(person.Date))]) + add_day
        except:
            recent = "No call history"
        
        # Fill in lists
        frequency_call_incoming_list.append(frequency_call_incoming)
        length_call_incoming_list.append(length_call_incoming)
        frequency_call_outgoing_list.append(frequency_call_outgoing)
        length_call_outgoing_list.append(length_call_outgoing)
        recent_list.append(recent)
        
        # Craete matrix and transpose
        call_array = [frequency_call_incoming_list, length_call_incoming_list, frequency_call_outgoing_list, length_call_outgoing_list, recent_list]
        call_array_T = np.transpose(call_array)
   
    return call_array_T

def people_regex(list_of_people):
    import re
    
    regex = "("
    for person in list_of_people:
        if person != list_of_people[-1]:
            regex += person+"|" 
        else:
            regex += person+")"
    return re.compile(regex)

def common_experience(data, list_of_people):
    import pandas as pd
    import numpy as np

    emotions = ["rage", "vigilance", "ecstasy", "admiration", "terror", "amazement", 
                "grief", "loathing", "anger", "anticipation", "joy", "trust", "fear",
                "surprise", "sadness","disgust", "annoyance", "interest", "serenity",
                 "acceptance", "apprehension", "distraction", "pensiveness","boredom"]
    emotion_label = data.emotion_label
    emotion_label_1st = pd.Series([emotion_label[i].split(',')[0] for i in range(len(emotion_label))])
    data.emotion_label = emotion_label_1st
    data["exp_sign"] = emotion_label_1st.apply(get_exp_sign)
    data[["related_people", "importance", "exp_sign"]]
    people_exp = data[["related_people", "importance", "exp_sign"]]
    people_exp.dropna(inplace=True, axis=0)
    people_exp.reset_index(drop=True, inplace=True)
    people_re = people_regex(list_of_people)
    people_exp = people_exp[people_exp.related_people.str.contains(people_re, regex=True)].reset_index(drop=True)
    
    # Positive experience
    people_exp_pos = people_exp[people_exp.exp_sign == "+"].reset_index(drop=True)
    
    frequency_common_positive_event_list = []
    avg_positive_importance_list = []
    for person in list_of_people:
        count = 0
        importance = []
        for i in range(people_exp_pos.related_people.shape[0]):
            if person in people_exp_pos.related_people[i]:
                count += 1
                importance.append(people_exp_pos.importance[i])
        frequency_common_positive_event_list.append(count)
        if importance:
            avg_positive_importance_list.append(round(sum(importance)/count,2))
        else:
            avg_positive_importance_list.append(0)
    
    # Negative experience
    people_exp_neg = people_exp[people_exp.exp_sign == "-"].reset_index(drop=True)
    frequency_common_negative_event_list = []
    avg_negative_importance_list = []
    for person in list_of_people:
        count = 0
        importance = []
        for i in range(people_exp_neg.related_people.shape[0]):
            if person in people_exp_neg.related_people[i]:
                count += 1
                importance.append(people_exp_neg.importance[i])
        frequency_common_negative_event_list.append(count)
        if importance:
            avg_negative_importance_list.append(round(sum(importance)/count,2))
        else:
            avg_negative_importance_list.append(0)
    
    # Craete matrix and transpose
    exp_array = [frequency_common_positive_event_list, avg_positive_importance_list, frequency_common_negative_event_list, avg_negative_importance_list]
    exp_array_T = np.transpose(exp_array)

    return exp_array_T

# Save modeling process as matrix
def get_modeling_matrix(ego_matrix, k):
    import numpy as np
    
    weight_medium = {
        "call": 0.7,
        "kakao": 0.3
        }
    weight_direction = {
        "incoming": 0.7,
        "outgoing": 0.3
        }

    frequency_contact = [weight_medium["kakao"]*((weight_direction["incoming"]*ego_matrix.n_kakao_in[i])+(weight_direction["outgoing"]*ego_matrix.n_kakao_out[i])) 
                        + weight_medium["call"]*((weight_direction["incoming"]*ego_matrix.n_call_in[i])+(weight_direction["outgoing"]*ego_matrix.n_call_out[i])) for i in range(ego_matrix.shape[0])]
    length_contact = [weight_medium["kakao"]*((weight_direction["incoming"]*ego_matrix.l_kakao_in[i])+(weight_direction["outgoing"]*ego_matrix.l_kakao_out[i])) 
                        + weight_medium["call"]*((weight_direction["incoming"]*ego_matrix.l_call_in[i])+(weight_direction["outgoing"]*ego_matrix.l_call_out[i])) for i in range(ego_matrix.shape[0])]
    intensity_contact = [np.log10((frequency_contact[i]*length_contact[i])/(int(ego_matrix.recenticy[i])+1)) for i in range(ego_matrix.shape[0])]
    positive_experience = [ego_matrix.n_pos_exp[i]*ego_matrix.a_pos_imp[i] for i in range(ego_matrix.shape[0])]
    negative_experience = [ego_matrix.n_neg_exp[i]*ego_matrix.a_neg_imp[i] for i in range(ego_matrix.shape[0])]
    intensity_experience = [0.01 if ((positive_experience[i] == float(0) and negative_experience[i] == float(0)) or (positive_experience[i] == negative_experience[i])) else np.log10((positive_experience[i]+1)/(negative_experience[i] + 1)) for i in range(ego_matrix.shape[0])]
    intensity_relationship = np.multiply(intensity_contact, intensity_experience)
    intensity_relationship_abs = [abs(intensity_relationship[i]) for i in range(ego_matrix.shape[0])]
    distance_relationship = [(max(intensity_relationship_abs) + 1 - intensity_relationship_abs[i])*k for i in range(ego_matrix.shape[0])]
    intensity_sign = ["+" if np.sign(intensity_relationship[i]) == 1.0 else "-" for i in range(ego_matrix.shape[0])]
    
    model_array = [frequency_contact, length_contact, intensity_contact, 
                   positive_experience, negative_experience, intensity_experience, 
                   intensity_relationship, intensity_relationship_abs, distance_relationship, intensity_sign]
    model_array_T = np.transpose(model_array)
    
    return model_array_T

def get_modeling_matrix_exp(exp_matrix, k):
    import numpy as np
    
    weight_medium = {
        "call": 0.7,
        "kakao": 0.3
        }
    weight_direction = {
        "incoming": 0.7,
        "outgoing": 0.3
        }

    frequency_contact = [0 for i in range(exp_matrix.shape[0])]
    length_contact = [0 for i in range(exp_matrix.shape[0])]
    intensity_contact = [0 for i in range(exp_matrix.shape[0])]
    positive_experience = [exp_matrix.n_pos_exp[i]*exp_matrix.a_pos_imp[i] for i in range(exp_matrix.shape[0])]
    negative_experience = [exp_matrix.n_neg_exp[i]*exp_matrix.a_neg_imp[i] for i in range(exp_matrix.shape[0])]
    intensity_experience = [0.01 if ((positive_experience[i] == float(0) and negative_experience[i] == float(0)) or (positive_experience[i] == negative_experience[i])) else np.log10((positive_experience[i]+1)/(negative_experience[i] + 1)) for i in range(exp_matrix.shape[0])]
    intensity_relationship = [0 for i in range(exp_matrix.shape[0])]
    intensity_relationship_abs = [0 for i in range(exp_matrix.shape[0])]
    distance_relationship = [0 for i in range(exp_matrix.shape[0])]
    intensity_sign = ["+" if np.sign(intensity_experience[i]) == 1.0 else "-" for i in range(exp_matrix.shape[0])]
    
    model_array = [frequency_contact, length_contact, intensity_contact, 
                   positive_experience, negative_experience, intensity_experience, 
                   intensity_relationship, intensity_relationship_abs, distance_relationship, intensity_sign]
    model_array_T = np.transpose(model_array)
    
    return model_array_T

def get_ego_matrix_exp(data, list_of_people):
    #%% Preprocess Kakao
    import re
    import os
    import pandas as pd
    import numpy as np
    
    # Load the data
    # list_of_people = ["Eunjin Choi", "Dayoung Im", "Jiwon Yoon", "Seunghwan Kim", "Soyoung Park", "Jaedong Yang"]
    # list_of_people_abb = ["EJC", "DYI", "JWY", "SHK", "SYP", "JDY"]
    # social_data = os.listdir()
    # kakao_re = re.compile("^(kakao)")
    # kakao_files = [social_data[i] for i in range(len(social_data)) if re.match(kakao_re, social_data[i])]
    
    #%%
    # kakao_matrix = get_kakao_data(kakao_files, wd)
    
    #%% Preprocess Call history
    # call_people = ["최은진", "임다영", "지원", "김승환", "엄마", "아버지"]
    
    # call_matrix = f.get_call_data(social_data, call_people, list_of_people)
    
    #%% Common experience
    exp_matrix = pd.DataFrame(common_experience(data, list_of_people))
    empty = pd.DataFrame(np.zeros(shape=(len(list_of_people), 8)))
    exp_matrix_columns = ["n_kakao_in", "l_kakao_in", "n_kakao_out", "l_kakao_out",
                                   "n_call_in", "l_call_in", "n_call_out", "l_call_out",
                                   "n_pos_exp", "a_pos_imp", "n_neg_exp", "a_neg_imp", "recenticy"]
    recenticy = pd.DataFrame(np.zeros(shape=(len(list_of_people), 1)))
    exp_matrix = pd.concat([empty, exp_matrix, recenticy], axis=1)
    
    exp_matrix.columns = exp_matrix_columns
    exp_matrix.index = list_of_people
    modeling_array = get_modeling_matrix_exp(exp_matrix, 10)
    modeling_array_columns = ["frequency_contact", "length_contact", "intensity_contact",	"positive_experience",	"negative_experience",	
                          "intensity_experience", "intensity_relationship",	"intensity_relationship_abs",	"distance_relationship", "sign"]

    modeling_matrix = pd.DataFrame(modeling_array, columns=modeling_array_columns, index=list_of_people)
    
    #%% Create Ego Matrix
    # ego_matrix_columns = ["n_kakao_in", "l_kakao_in", "n_kakao_out", "l_kakao_out", "r_kakao",
                                       # "n_call_in", "l_call_in", "n_call_out", "l_call_out", "r_call",
                                       # "n_pos_exp", "a_pos_imp", "n_neg_exp", "a_neg_imp"]
    # ego_matrix = pd.DataFrame(np.concatenate([kakao_matrix, call_matrix, exp_matrix], axis=1),
                              # columns=ego_matrix_columns,
                              # index=list_of_people) 
    # recenticity = [str(min(ego_matrix.r_kakao[i], ego_matrix.r_call[i]))[:-14] for i in range(ego_matrix.shape[0])]
    # ego_matrix["recenticy"] = recenticity
    # ego_matrix.drop(["r_kakao", "r_call"], axis=1, inplace=True)
    # ego_matrix.to_csv("ego_matrix.csv", index=True)
    
    #%% Compute relationship score
    # modeling_array = f.get_modeling_matrix(ego_matrix, 10)
    
    
    ego_matrix_exp = pd.concat([exp_matrix, modeling_matrix], axis=1)
    # ego_matrix_final = pd.concat([ego_matrix, modeling_matrix], axis=1)
    
    return ego_matrix_exp

def plot_ego_network(ego_matrix_final, list_of_people_abb, marker_size):
    import matplotlib as mpl
    import matplotlib.pylab as plt
    from numpy.random import default_rng
    import numpy as np
    from scipy.spatial import distance
    from random import randint, uniform
    
    # Create matrix for positive people
    rng = default_rng()
    pos_people = ego_matrix_final[ego_matrix_final.sign == "+"]
    # pos_people_coord_x = rng.choice([x for x in range(-10, 11) if x != 0], size=pos_people.shape[0], replace=False)
    # pos_people_coord_x  = rng.choice([y for y in range(1, 11) if y != 0], size=pos_people.shape[0], replace=False)
    pos_people_coord_x = []
    pos_people_coord_y = []
    for i in range(pos_people.shape[0]):
        x = 0
        y = 0
        while round(distance.euclidean((0,0), (x, y)), 0) != round(float(pos_people.distance_relationship[i]), 0):
            x = uniform(-50, 55)
            y = uniform(10, 55)
            #print("Current distance: ", round(distance.euclidean((0,0), (x, y)), 0))
            #print("Target distance: ", round(float(pos_people.distance_relationship[i]), 0))
        print("\nPoint {} found".format(i+1))
        
        if (round(x, 2) not in pos_people_coord_x) and (round(y, 2) not in pos_people_coord_y):
            pos_people_coord_x.append(round(x,2))
            pos_people_coord_y.append(round(y,2))
    
    # colors = plt.cm.plasma(np.linspace(0,1,6))
    # for i in range(pos_people.shape[0]):
    #     plt.plot(pos_people_coord_x[i], pos_people_coord_y[i], color=colors[i], ls="None", marker="o", markersize=25)
    # plt.plot(0, 0, color="yellow", marker="o", markersize=25, zorder=2)
    # plt.text(0, 0, "Me", fontsize=10, horizontalalignment="center", verticalalignment="center", zorder=2)
    # plt.margins(0.1, 0.1)
    # plt.xlim([-25, 25])
    # plt.ylim([-25, 25])
    # plt.axis("off")
    # plt.axhline(0, ls="dashed", zorder=1)
    # for i, x, y in zip(range(len(pos_people_coord_x)), pos_people_coord_x, pos_people_coord_y):
    #     plt.text(x, y, list_of_people_abb[i], fontsize=10, horizontalalignment="center", verticalalignment="center")
    
    # Negative people
    neg_people = ego_matrix_final[ego_matrix_final.sign == "-"]
    neg_people_coord_x = []
    neg_people_coord_y = []
    for i in range(neg_people.shape[0]):
        x = 0
        y = 0
        while round(distance.euclidean((0,0), (x, y)), 0) != round(float(neg_people.distance_relationship[i]), 0):
            x = uniform(-50, 55)
            y = uniform(-10, -55)
            #print("Current distance: ", round(distance.euclidean((0,0), (x, y)), 0))
            #print("Target distance: ", round(float(neg_people.distance_relationship[i]), 0))
        print("\nPoint {} found".format(i+1))
        
        if (round(x, 2) not in neg_people_coord_x) and (round(y, 2) not in neg_people_coord_y):
            neg_people_coord_x.append(round(x,2))
            neg_people_coord_y.append(round(y,2))
    
    coord_x = pos_people_coord_x + neg_people_coord_x
    coord_y = pos_people_coord_y + neg_people_coord_y
    
    colors = plt.cm.plasma(np.linspace(0.5,1,6))
    plt.figure(figsize=(10,10))
    for i in range(len(list_of_people_abb)):
        plt.plot(coord_x[i], coord_y[i], color=colors[i], ls="None", marker="o", markersize=marker_size)
    plt.plot(0, 0, color="#781E4F", marker="o", markersize=marker_size, zorder=2)
    plt.text(0, 0, "Me", fontsize=15, horizontalalignment="center", verticalalignment="center", zorder=2)
    plt.margins(0.1, 0.1)
    plt.xlim([-50, 50])
    plt.ylim([-50, 50])
    plt.axis("off")
    plt.axhline(0, ls="dashed", zorder=1, color="#A67D97")
    for i, x, y in zip(range(len(coord_x)), coord_x, coord_y):
        plt.text(x, y, list_of_people_abb[i], fontsize=15, horizontalalignment="center", verticalalignment="center")
    # plt.show()
    plt.savefig("ego_network.png", transparent=True)

def plot_ego_network_exp(ego_matrix_exp, list_of_people, marker_size):
    import matplotlib as mpl
    import matplotlib.pylab as plt
    from numpy.random import default_rng
    import numpy as np
    from scipy.spatial import distance
    from random import randint, uniform
    
    # Intenstiy relationship
    intensity_experience_abs = [abs(float(ego_matrix_exp.intensity_experience[i])) for i in range(ego_matrix_exp.shape[0])]
    distance_experience = [(max(intensity_experience_abs) + 1 - intensity_experience_abs[i])*20 for i in range(ego_matrix_exp.shape[0])]
    ego_matrix_exp["intensity_experience_abs"] = intensity_experience_abs
    ego_matrix_exp["distance_experience"] = distance_experience
    
    # Create matrix for positive people
    rng = default_rng()
    pos_people = ego_matrix_exp[ego_matrix_exp.sign == "+"]
    # pos_people_coord_x = rng.choice([x for x in range(-10, 11) if x != 0], size=pos_people.shape[0], replace=False)
    # pos_people_coord_x  = rng.choice([y for y in range(1, 11) if y != 0], size=pos_people.shape[0], replace=False)
    pos_people_coord_x = []
    pos_people_coord_y = []
    for i in range(pos_people.shape[0]):
        x = 0
        y = 0
        while round(distance.euclidean((0,0), (x, y)), 0) != round(float(pos_people.distance_experience[i]), 0):
            x = uniform(-50, 55)
            y = uniform(10, 55)
            #print("Current distance: ", round(distance.euclidean((0,0), (x, y)), 0))
            #print("Target distance: ", round(float(pos_people.distance_experience[i]), 0))
        print("\nPoint {} found".format(i+1))
        
        if (round(x, 2) not in pos_people_coord_x) and (round(y, 2) not in pos_people_coord_y):
            pos_people_coord_x.append(round(x,2))
            pos_people_coord_y.append(round(y,2))
    
    # colors = plt.cm.plasma(np.linspace(0,1,6))
    # for i in range(pos_people.shape[0]):
    #     plt.plot(pos_people_coord_x[i], pos_people_coord_y[i], color=colors[i], ls="None", marker="o", markersize=25)
    # plt.plot(0, 0, color="yellow", marker="o", markersize=25, zorder=2)
    # plt.text(0, 0, "Me", fontsize=10, horizontalalignment="center", verticalalignment="center", zorder=2)
    # plt.margins(0.1, 0.1)
    # plt.xlim([-25, 25])
    # plt.ylim([-25, 25])
    # plt.axis("off")
    # plt.axhline(0, ls="dashed", zorder=1)
    # for i, x, y in zip(range(len(pos_people_coord_x)), pos_people_coord_x, pos_people_coord_y):
    #     plt.text(x, y, list_of_people_abb[i], fontsize=10, horizontalalignment="center", verticalalignment="center")
    
    # Negative people
    neg_people = ego_matrix_exp[ego_matrix_exp.sign == "-"]
    neg_people_coord_x = []
    neg_people_coord_y = []
    for i in range(neg_people.shape[0]):
        x = 0
        y = 0
        while round(distance.euclidean((0,0), (x, y)), 0) != round(float(neg_people.distance_experience[i]), 0):
            x = uniform(-50, 55)
            y = uniform(-10, -55)
            #print("Current distance: ", round(distance.euclidean((0,0), (x, y)), 0))
            #print("Target distance: ", round(float(neg_people.distance_experience[i]), 0))
        print("\nPoint {} found".format(i+1))
        
        if (round(x, 2) not in neg_people_coord_x) and (round(y, 2) not in neg_people_coord_y):
            neg_people_coord_x.append(round(x,2))
            neg_people_coord_y.append(round(y,2))
    
    coord_x = pos_people_coord_x + neg_people_coord_x
    coord_y = pos_people_coord_y + neg_people_coord_y
    
    colors = plt.cm.plasma(np.linspace(0.5,1,len(list_of_people)))
    plt.figure(figsize=(10,10))
    for i in range(len(list_of_people)):
        plt.plot(coord_x[i], coord_y[i], color=colors[i], ls="None", marker="o", markersize=marker_size)
    plt.plot(0, 0, color="#781E4F", marker="o", markersize=marker_size, zorder=2)
    plt.text(0, 0, "Me", fontsize=15, horizontalalignment="center", verticalalignment="center", zorder=2)
    plt.margins(0.1, 0.1)
    plt.xlim([-50, 50])
    plt.ylim([-50, 50])
    plt.axis("off")
    plt.axhline(0, ls="dashed", zorder=1, color="#A67D97")
    for i, x, y in zip(range(len(coord_x)), coord_x, coord_y):
        plt.text(x, y, (list(pos_people.index) + list(neg_people.index))[i], fontsize=15, horizontalalignment="center", verticalalignment="center")
    # plt.show()
    plt.savefig("ego_network.png", transparent=True)
    
    pos_people_close = pos_people.index[pos_people.distance_experience <= 30]
    pos_poeple_far = pos_people.index[pos_people.distance_experience > 30]
    neg_people_close = neg_people.index[neg_people.distance_experience <= 30]
    neg_people_far = neg_people.index[neg_people.distance_experience > 30]
    
    return pos_people_close, pos_poeple_far, neg_people_close, neg_people_far


def get_url_list(report_data):
    import numpy as np

    queries = [list(report_data.values())[i].split(" ") for i in range(len(report_data.keys()))]
    urls = np.empty(shape=(3, 5), dtype="object")
    for i in range(len(queries)):
        for j in range(len(queries[i])):
            youtube_url = "https://www.youtube.com/results?search_query={}".format(str(queries[i][j]))
            urls[i][j] = youtube_url
            print(youtube_url)

    return urls