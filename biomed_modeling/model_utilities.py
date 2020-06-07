# Author: Thomas Porturas <thomas.porturas.eras@gmail.com>

from time import time
from datetime import datetime
import os, sys

import numpy as np
from scipy.stats.mstats import gmean
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as hc
import pandas as pd
import pickle
import gensim
import spacy
import scispacy
from sklearn import linear_model
from sklearn.manifold import TSNE
import glob
import re

#plotting tools
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib import transforms
from mpl_toolkits.mplot3d import Axes3D
from wordcloud import WordCloud
from cycler import cycler
import seaborn as sns

"""
The :mod:'model_utilities' module contains many functions that are useful for processing and graphing the topic modeling results.
It also includes the SilentPrinting and Timing classes that you use with the 'with' statement. SilentPrinting stops printing to
the terminal, but is unable to silence MALLET. 
"""

def plot_model_comparison(paths, x_column, y_columns, x_label, y_label, graph_title, show=True, fig_save_path=None, csv_save_path=None):
    # Use this to combine multiple CompareModels csv outputs into 1 with a mean and standard devation
    # Main variables
    data_dict = {}
    mean_sd_dict = {}
    x_data = None
    # Setup data_dict y_column keys
    for column in y_columns:
        data_dict[column] = {}
    # Read each file in paths    
    for path in paths:
        df = pd.read_csv(path)
        # Setup data_dict x keys and values if not yet done
        if x_data is None:
            x_data = df[x_column].tolist()
            for column in y_columns:
                for x in x_data:
                    data_dict[column][x] = []
        # Add file's data to list in data_dict
        for column in y_columns:
            data = df[column].tolist()    
            for x in x_data:
                data_dict[column][x].append(data.pop(0))
    # Calculate mean and Standard deviation for each y value
    for y_column in data_dict:
        mean_sd_dict[y_column] = {'X':[], 'MEAN':[], 'STDV':[]}
        for x in data_dict[y_column]:
            mean_sd_dict[y_column]['X'].append(x)
            mean_sd_dict[y_column]['MEAN'].append(np.mean(data_dict[y_column][x]))
            mean_sd_dict[y_column]['STDV'].append(np.std(data_dict[y_column][x]))
    # Plot graph of  x VS y with standard deviation for error bars
    plt.figure(figsize=(12, 8))
    for y_column in mean_sd_dict:
        plt.errorbar(mean_sd_dict[y_column]['X'], mean_sd_dict[y_column]['MEAN'], 
                    yerr=mean_sd_dict[y_column]['STDV'], label=y_column, 
                    marker='o', markersize=5, capsize=5, markeredgewidth=1)
    plt.title(graph_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(title='Models', loc='best')
    # Saving figure if fig_save_path is entered
    if fig_save_path is not None:
        plt.savefig(fig_save_path, bbox_inches='tight')
    # Saving a CSV file of the means and standard deviations if csv_save_path is entered
    if csv_save_path is not None:
        dataframe_dict= {}
        
        for y_column in y_columns:
            dataframe_dict[x_column] = mean_sd_dict[y_column]['X']
            dataframe_dict[" ".join([y_column, "MEAN"])] = mean_sd_dict[y_column]['MEAN']
            dataframe_dict[" ".join([y_column, "STDV"])] = mean_sd_dict[y_column]['STDV']
        data = pd.DataFrame.from_dict(dataframe_dict)
        data.to_csv(csv_save_path, index=False)
    if show:
        plt.show()
    plt.close() # Closes and deletes graph to free up memory

def dominant_doc_topic_df(model, nlp_data, num_keywords=10):
    topics_df = pd.DataFrame()

    for i, row_list in enumerate(model[nlp_data.gensim_lda_input()]):
        row = row_list[0] if model.per_word_topics else row_list

        row = sorted(row, key=lambda x:(x[1]), reverse=True)

        for j, (topic_num, prop_topic) in enumerate(row):
            if j==0:
                wp = model.show_topic(topic_num, topn=num_keywords)
                topic_keywords = ", ".join([word for word, prop in wp])
                topics_df = topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    topics_df.columns = ["Dominant Topic", "Contribution", "Topic Keywords"]
    contents = pd.Series(nlp_data.get_token_text())
    topics_df = pd.concat([topics_df, contents], axis=1)
    topics_df = topics_df.reset_index()
    topics_df.columns = ["Document", "Dominant Topic", "Contribution", "Topic Keywords", "Document Tokens"]
    topics_df["Document"] += 1
    topics_df["Dominant Topic"] = 1 + topics_df["Dominant Topic"].astype(int)
    return topics_df

def best_doc_for_topic(dom_top_df):
    sorted_df = pd.DataFrame()
    dom_top_df_grouped = dom_top_df.groupby('Dominant Topic')
    for i, grp in dom_top_df_grouped:
        sorted_df = pd.concat([sorted_df, grp.sort_values(['Contribution'], ascending=False).head(1)], axis=0)
    sorted_df.reset_index(drop=True, inplace=True)
    sorted_df.columns = ["Best Document", "Topic Number", "Contribution", "Topic Keywords", "Document Tokens"]
    sorted_df = sorted_df[["Topic Number", "Contribution", "Topic Keywords", "Best Document", "Document Tokens"]]
    return sorted_df

def plot_doc_token_counts_old(dom_top_df=None, nlp_data=None, show=True, fig_save_path=None, bins=None):
    if dom_top_df is not None:
        doc_lens = [len(doc) for doc in dom_top_df["Document Tokens"]]
    
    if nlp_data is not None:
        doc_lens = np.array(nlp_data.sklearn_lda_input().sum(axis=1)).flatten()

    fig = plt.figure(figsize=(12,7), dpi=160)
    plt.hist(doc_lens, bins = 500, color='navy')
    # Prints texts on the graph at x=400
    x = 400
    plt.text(x, 120, "Documents")
    text = plt.text(x, 110, "Total Tokens")
    plt.text(x, 100, "Mean")
    plt.text(x,  90, "Median")
    plt.text(x,  80, "Stdev")
    plt.text(x,  70, "1%ile")
    plt.text(x,  60, "99%ile")
    #This is for offsetting the data so it will appear even
    canvas = fig.canvas
    text.draw(canvas.get_renderer())
    ex = text.get_window_extent()
    t = transforms.offset_copy(text.get_transform(), x=ex.width, units='dots')
    # This prints the statistics
    plt.text(x, 120, " : " + str(len(doc_lens)), transform=t)
    plt.text(x, 110, " : " + str(np.sum(doc_lens)), transform=t)
    plt.text(x, 100, " : " + str(round(np.mean(doc_lens), 1)), transform=t)
    plt.text(x,  90, " : " + str(round(np.median(doc_lens), 1)), transform=t)
    plt.text(x,  80, " : " + str(round(np.std(doc_lens),1)), transform=t)
    plt.text(x,  70, " : " + str(np.quantile(doc_lens, q=0.01)), transform=t)
    plt.text(x,  60, " : " + str(np.quantile(doc_lens, q=0.99)), transform=t)


    plt.gca().set(xlim=(0, 500), ylabel='Number of Documents', xlabel='Document Token Count')
    plt.tick_params(size=16)
    plt.xticks(np.linspace(0,500,11))
    plt.title('Distribution of Document Token Counts', fontdict=dict(size=22))
    plt.tight_layout()

    if fig_save_path is not None:
        plt.savefig(fig_save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_doc_token_counts(dom_top_df=None, nlp_data=None, show=True, fig_save_path=None, bins=None):
    if dom_top_df is not None:
        doc_lens = [len(doc) for doc in dom_top_df["Document Tokens"]]
    
    if nlp_data is not None:
        doc_lens = np.array(nlp_data.sklearn_lda_input().sum(axis=1)).flatten()

    if bins is None:
        bins = 50 * math.ceil(max(doc_lens)/50)
        if max(doc_lens) - np.quantile(doc_lens, q=0.99) < bins * 0.2:
            bins += 50 * math.ceil((bins*0.25)/50)
    bin_list = [i+1 for i in range(bins)]
    fig = plt.figure(figsize=(12,7), dpi=160)
    plt.hist(doc_lens, bins = bin_list, color='navy', rwidth=None)
    # Prints texts on the graph at position x
    x = 0.79
    t = fig.transFigure
    plt.text(x, 0.88, "Documents", transform=t)
    text = plt.text(x, 0.85, "Total Tokens", transform=t)
    plt.text(x, 0.82, "Mean", transform=t)
    plt.text(x, 0.79, "Median", transform=t)
    plt.text(x, 0.76, "Stdev", transform=t)
    plt.text(x, 0.73, "1%ile", transform=t)
    plt.text(x, 0.70, "99%ile", transform=t)
    #This is for offsetting the data so it will appear even
    canvas = fig.canvas
    text.draw(canvas.get_renderer())
    ex = text.get_window_extent()
    t = transforms.offset_copy(text.get_transform(), x=ex.width, units='dots')
    # This prints the statistics
    plt.text(x, 0.88, " : " + str(len(doc_lens)), transform=t)
    plt.text(x, 0.85, " : " + str(np.sum(doc_lens)), transform=t)
    plt.text(x, 0.82, " : " + str(round(np.mean(doc_lens), 1)), transform=t)
    plt.text(x, 0.79, " : " + str(round(np.median(doc_lens), 1)), transform=t)
    plt.text(x, 0.76, " : " + str(round(np.std(doc_lens),1)), transform=t)
    plt.text(x, 0.73, " : " + str(np.quantile(doc_lens, q=0.01)), transform=t)
    plt.text(x, 0.70, " : " + str(np.quantile(doc_lens, q=0.99)), transform=t)


    plt.gca().set(xlim=(0, bins), ylabel='Number of Documents', xlabel='Document Token Count')
    plt.tick_params(size=16)
    #plt.xticks(np.linspace(0,500,11))
    plt.title('Distribution of Document Token Counts', fontdict=dict(size=22))
    plt.tight_layout()

    if fig_save_path is not None:
        plt.savefig(fig_save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def create_wordcloud(topic, model, nlp_data, seed=100, num_w=20, fig_dpi=400, topic_names=None,
                    show=True, fig_save_path=None, colormap='tab10', horizontal_pref=0.8):
    cloud = WordCloud(background_color='white', width=1000, height=1000, max_words=num_w, colormap=colormap,
                        prefer_horizontal=horizontal_pref, random_state=seed)
    topics = model.show_topics(num_topics=-1, num_words=num_w, formatted=False)
    cloud.generate_from_frequencies(dict(topics[topic-1][1]), max_font_size=300)
    plt.figure(figsize=(2,2), dpi=fig_dpi)
    plt.imshow(cloud)
    if topic_names is None:
        plt.title('Topic {}'.format(topic+1), fontdict=dict(size=16), pad=10)
    else:
        plt.title(topic_names[topic+1], fontdict=dict(size=16), pad=10)
    plt.axis('off')
    plt.tight_layout()

    if fig_save_path is not None:
        plt.savefig(fig_save_path, bbox_inches='tight')
    if show: # It shows ugly but the actual save file looks good.
        plt.show()
    plt.close()

def create_multi_wordclouds(n_topics, n_horiz, model, nlp_data, seed=100, num_w=20, fig_dpi=400, topic_names=None, title_font=15,
                            show=True, fig_save_path=None, colormap='tab10', horizontal_pref=0.8):
    if isinstance(n_topics, int):
        topics_list = list(range(n_topics))
    else:
        topics_list = [i-1 for i in n_topics]
        n_topics = len(topics_list)

    cloud = WordCloud(background_color='white', width=1000, height=1000, max_words=num_w, colormap=colormap,
                        prefer_horizontal=horizontal_pref, random_state=seed)
    topics = model.show_topics(num_topics=-1, num_words=num_w, formatted=False)

    x_len = n_horiz
    y_len = math.ceil(n_topics/n_horiz)

    fig, axes = plt.subplots(y_len, x_len, figsize=(2*x_len,2*y_len), dpi=fig_dpi, 
                                sharex=True, sharey=True, squeeze=False, constrained_layout=True)
    for i, ax in enumerate(axes.flatten()):
        if i < n_topics:
            fig.add_subplot(ax)
            topic_words = dict(topics[topics_list[i]][1])
            cloud.generate_from_frequencies(topic_words, max_font_size=300)
            plt.gca().imshow(cloud)
            if topic_names is None:
                plt.gca().set_title('Topic {}'.format(topics_list[i]+1), fontdict=dict(size=title_font), pad=10)
            else:
                plt.gca().set_title(topic_names[topics_list[i]+1], fontdict=dict(size=title_font), pad=10)
            plt.gca().axis('off')
        else:
            fig.add_subplot(ax)
            plt.gca().axis('off')

    #plt.suptitle('Topic Wordclouds', fontdict=dict(size=16))
    plt.axis('off')
    plt.margins(x=0, y=0)

    if fig_save_path is not None:
        plt.savefig(fig_save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def color_doc_topics(model, doc, nlp_data, max_chars=120, dpi=150, show=True, fig_save_path=None, topics=5, min_phi=None,
            topic_names=None, incl_perc=False, highlight=False, highlight_topic_names=False): 
    # The output file looks better than show
    colors = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    if topics > 10: # There are only 10 colors so the max is 10. Change above to add more colors for more topics
        topics = 10
    # This is for the lemmetazation step
    doc_prep = gensim.utils.simple_preprocess(str(doc), deacc=True, min_len=2, max_len=30)

    #This is for processing the string while retaining the original characters since simple_preprocess removes punctuation and accents
    #It splits the string by ' ' and then individually processes the chunks into tokens and finds there location in the string
    #Finally a list is made with strings that directly translate to tokens and preserves non-token strings
    doc_raw_split = str(doc).split()
    doc_raw_word_list = []
    raw_token_dict = {}
    for string_piece in doc_raw_split:
        tokens = gensim.utils.simple_preprocess(str(string_piece), deacc=True, min_len=1, max_len=30)
        working_string = gensim.utils.deaccent(string_piece.lower())
        output_string = string_piece
        for token in tokens:
            if token in working_string:
                start_index = working_string.find(token)
                end_index = start_index + len(token)
                front_part = output_string[:start_index]
                token_part = output_string[start_index:end_index]
                output_string = output_string[end_index:]
                working_string = working_string[end_index:]
                if len(front_part) > 0:
                    doc_raw_word_list.append(front_part)
                    raw_token_dict[front_part] = False
                doc_raw_word_list.append(token_part)
                raw_token_dict[token_part] = token
        if len(output_string) > 0: # This saves strings that do not become tokens, False prevents them from being in the wordset
            doc_raw_word_list.append(output_string)
            raw_token_dict[output_string] = False

    # This is for finding all index locations of the tokens within the original raw string list
    wordset = set([raw_token_dict[word] for word in raw_token_dict.keys() if raw_token_dict[word]])
    doc_index_dict = {}
    for word in wordset:
        word_indexes = [i for i, w in enumerate(doc_raw_word_list) if raw_token_dict[w] == word]
        doc_index_dict[word] = word_indexes

    token_index_dict = {}
    token_list = []
    # This is for lemmitazation of the text and linking the lemma to its original token index locations
    nlp = spacy.load(nlp_data.spacy_lib, disable=['parser','ner'])
    allowed_postags = ['NOUN', 'ADJ', 'VERB','ADV']

    for word in doc_prep:
        if word not in nlp_data.stopwords:
            token = nlp(word)[0]
            if token.pos_ in allowed_postags and token.lemma_ not in ['-PRON-']:
                token_list.append(token.lemma_)
                if token.lemma_ in token_index_dict:
                    token_index_dict[token.lemma_] = list(set(token_index_dict[token.lemma_] + doc_index_dict[word]))
                else:
                    token_index_dict[token.lemma_] = doc_index_dict[word]
    for token in token_index_dict:
        token_index_dict[token] = sorted(set(token_index_dict[token]))
    # This processes the n-grams based on the model's n-gram settings and combines index locations for the n-gram
    processed_tokens = nlp_data.process_ngrams_([token_list])[0]
    final_token_dict = {}
    for token in processed_tokens:
        if token not in final_token_dict:
            final_token_dict[token] = []
        split_tokens = token.split('_')
        for split_token in split_tokens:
            final_token_dict[token].append(token_index_dict[split_token].pop(0))
    # This is where the text is processed by the model and the top n models are saved
    topic_perc, wordid_topics, wordid_phivalues = model.get_document_topics(
        nlp_data.gensim_lda_input([" ".join(processed_tokens)])[0], per_word_topics=True, 
        minimum_probability=0.001, minimum_phi_value=min_phi)
    topic_perc_sorted = sorted(topic_perc, key=lambda x:(x[1]), reverse=True)
    top_topics = [topic[0] for i, topic in enumerate(topic_perc_sorted) if i < topics]
    top_topics_color = {top_topics[i]:i for i in range(len(top_topics))}
    word_dom_topic = {}
    # This links the individual word lemmas to its best topic within available topics
    for wd, wd_topics in wordid_topics:
        for topic in wd_topics:
            if topic in top_topics:
                word_dom_topic[model.id2word[wd]] = topic
                break    
    
    # Links the index location to a color
    index_color_dict = {}
    for token in final_token_dict:
        if token in word_dom_topic:
            for i in final_token_dict[token]:
                index_color_dict[i] = top_topics_color[word_dom_topic[token]]
    # this is for assembling the individual lines of the graph based on character length and position of punctuation
    add_lines = math.ceil(len(top_topics_color)/5)
    last_index = len(doc_raw_word_list) - 1
    line_len = 0
    line_num = 0
    doc_raw_lines = [[]]
    no_space_list = [".", ",", ")", ":", "'"]
    for i, word in enumerate(doc_raw_word_list):
        word_len = len(word)
        if line_len + word_len < max_chars or (word in no_space_list and line_len <= max_chars):
            if word == '(':
                if i != last_index:
                    if (line_len + word_len + len(doc_raw_word_list[i+1]) + 1 >= max_chars 
                        and doc_raw_word_list[i+1] not in no_space_list):
                        line_num += 1
                        line_len = 0
                        doc_raw_lines.append([])
        else:
            line_num += 1
            line_len = 0
            doc_raw_lines.append([])
        line_len += word_len + 1
        doc_raw_lines[line_num].append(i)
    line_num += 1

    # This creates the figure and subplots
    lines = line_num + add_lines
    fig, axes = plt.subplots(lines + 1, 1, figsize=(math.ceil(max_chars/8), math.ceil(lines/2)), dpi=dpi,
                                squeeze=True, constrained_layout=True)
    axes[0].axis('off')
    plt.axis('off')

    
    indent = 0
    # This is the loop for drawing the text
    for i, ax in enumerate(axes):
        t = ax.transData
        canvas = ax.figure.canvas
        if i > add_lines:
            x = 0.06
            line = i - add_lines - 1
            for index in doc_raw_lines[line]:
                word = doc_raw_word_list[index]
                if word[-1] == "(":
                    pass
                elif index != last_index:
                    if doc_raw_word_list[index+1][0] not in no_space_list:
                        word = word + " "
                if index in index_color_dict:
                    color = colors[index_color_dict[index]]
                else:
                    color = 'black'
                
                if highlight:
                    bbox=dict(facecolor=color, edgecolor=[0,0,0,0], pad=0, boxstyle='round')
                    text = ax.text(x, 0.5, word, horizontalalignment='left',
                            verticalalignment='center', fontsize=16, color='black',
                            transform=t, fontweight=700)
                    if color != 'black':
                        text.set_bbox(bbox)
                else:
                    text = ax.text(x, 0.5, word, horizontalalignment='left',
                            verticalalignment='center', fontsize=16, color=color,
                            transform=t, fontweight=700)
                text.draw(canvas.get_renderer())
                ex = text.get_window_extent()
                t = transforms.offset_copy(text.get_transform(), x=ex.width, units='dots')
                ax.axis('off')

        elif i < add_lines:
            x = 0.06
            if i == 0:
                word = "Topics: "
                color = 'black'
                text = ax.text(x, 0.5, word, horizontalalignment='left',
                        verticalalignment='center', fontsize=16, color=color,
                        transform=t, fontweight=700)
                text.draw(canvas.get_renderer())
                ex = text.get_window_extent()
                t = transforms.offset_copy(text.get_transform(), x=ex.width, units='dots')
                indent = ex.width
            else:
                color = 'black'
                text = ax.text(x, 0.5, "", horizontalalignment='left',
                        verticalalignment='center', fontsize=16, color=color,
                        transform=t, fontweight=700)
                text.draw(canvas.get_renderer())
                ex = text.get_window_extent()
                t = transforms.offset_copy(text.get_transform(), x=indent, units='dots')

            for num, index in enumerate(range(i*5, len(top_topics))):
                if num < 5:
                    if topic_names is None:
                        word = "Topic {}, ".format(top_topics[index]+1)
                    else:
                        word = topic_names[top_topics[index]+1] + ", "
                    if incl_perc:
                        topic_perc_dict = dict(topic_perc_sorted)
                        word = "{:.1f}% ".format(topic_perc_dict[top_topics[index]]*100) + word
                    color = colors[top_topics_color[top_topics[index]]]
                    
                    if highlight_topic_names:
                        bbox=dict(facecolor=color, edgecolor=[0,0,0,0], pad=0, boxstyle='round')
                        text = ax.text(x, 0.5, word, horizontalalignment='left',
                                verticalalignment='center', fontsize=16, color='black',
                                transform=t, fontweight=700)
                        if color != 'black':
                            text.set_bbox(bbox)
                    else:
                        text = ax.text(x, 0.5, word, horizontalalignment='left',
                                verticalalignment='center', fontsize=16, color=color,
                                transform=t, fontweight=700)

                    text.draw(canvas.get_renderer())
                    ex = text.get_window_extent()
                    t = transforms.offset_copy(text.get_transform(), x=ex.width, units='dots')
            ax.axis('off')
        else:
            ax.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle('Document Colored by Top {} Topics'.format(topics), 
                    fontsize=22, y=0.95, fontweight=700)
    # This saves and/or shows the plot. Note: Saved file looke better than the drawn plot
    if fig_save_path is not None:
        plt.savefig(fig_save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()    

def docs_per_topic(model, nlp_data=None, doc_list=None, corpus=None):
    if corpus is None:
        if doc_list is None:
            corpus = nlp_data.gensim_lda_input()
        else:
            corpus = nlp_data.process_new_corpus(doc_list)['gensim']
    
    num_topics = model.num_topics
    dominant_topics = []
    topic_percantages = []
    for i, corp in enumerate(corpus):
        topic_perc, wordid_topics, wordidphvalues = model.get_document_topics(
                                corp, per_word_topics=True)
        dominant_topic = sorted(topic_perc, key = lambda x: x[1], reverse=True)[0][0]
        dominant_topics.append((i, dominant_topic))
        topic_percantages.append(topic_perc)
    
    df = pd.DataFrame(dominant_topics, columns=['Document', 'Dominant Topic'])
    docs_by_dom_topic = df.groupby('Dominant Topic').size()
    df_docs_by_dom_topic = docs_by_dom_topic.to_frame().reset_index()
    df_docs_by_dom_topic.columns = ['Dominant Topic', 'Document Count']
    present_topics = df_docs_by_dom_topic['Dominant Topic'].tolist()
    absent_topics = [i for i in range(num_topics) if i not in present_topics]
    add_rows = {'Dominant Topic':absent_topics, 'Document Count':[]}
    for t in absent_topics:
        add_rows['Document Count'].append(0)
    if len(absent_topics) > 0:
        df_add_rows = pd.DataFrame(add_rows)
        df_docs_by_dom_topic = df_docs_by_dom_topic.append(df_add_rows, ignore_index=True)
    df_docs_by_dom_topic.sort_values('Dominant Topic', inplace=True) 
    df_docs_by_dom_topic['Dominant Topic'] += 1



    topic_weight_doc = pd.DataFrame([dict(t) for t in topic_percantages])
    df_topic_weight_doc = topic_weight_doc.sum().to_frame().reset_index()
    df_topic_weight_doc.columns = ['Topic', 'Document Weight']
    present_topics = df_topic_weight_doc['Topic'].tolist()
    absent_topics = [i for i in range(num_topics) if i not in present_topics]
    add_rows = {'Topic':absent_topics, 'Document Weight':[]}
    for t in absent_topics:
        add_rows['Document Weight'].append(0.0)
    if len(absent_topics) > 0:
        df_add_rows = pd.DataFrame(add_rows)
        df_topic_weight_doc = df_topic_weight_doc.append(df_add_rows, ignore_index=True)
    df_topic_weight_doc['Topic'] += 1
    df_topic_weight_doc.sort_values('Topic', inplace=True)
    df_topic_weight_doc.reset_index(drop=True, inplace=True)

    return df_docs_by_dom_topic, df_topic_weight_doc

def doc_topics_per_time(model, nlp_data, year_res=5, df=None, data_column=None, year_column=None, year_list=None, 
                        year_start=None, year_end=None):
    if df is not None:
        data = nlp_data.process_new_corpus(df[data_column].tolist())['gensim']
        year_list = df[year_column]
    elif year_list is not None:
        data = nlp_data.gensim_lda_input()
    else:
        print("No year/data given")
        return None
    
    grouped_df = pd.DataFrame(list(zip(data, year_list)), columns=['data', 'year']).groupby('year')
    year_doc_dict = {}
    for year, group in grouped_df:
        if year_start is None:
            year_doc_dict[int(year)] = group['data'].tolist()
        elif year >= year_start:
            year_doc_dict[int(year)] = group['data'].tolist()
    years = sorted(year_doc_dict.keys())
    final_year_doc_dict = {}
    if year_start is None:
        year_start = years[0]
    if year_end is None:
        year_end = years[-1]
    all_years = list(range(year_start, year_end+1))
    for year in all_years:
        if year not in years:
            final_year_doc_dict[year] = []
        else:
            final_year_doc_dict[year] = year_doc_dict[year]
    years = sorted(final_year_doc_dict.keys())
    intervals = {}
    year_range = []
    years_label = None
    num_years = len(years)
    num_intervals = math.ceil(num_years / year_res)
    print("Number of years: {} \nNumber of intervals: {}".format(num_years, num_intervals))
    n = year_res
    for i in range(num_intervals):
        index = i*n
        year_range = [years[index] + num for num in range(year_res)]
        if index + year_res <= num_years:
            years_label = str(years[index]) + " to " + str(years[index + n - 1])
        else:
            years_label = str(years[index]) + " to " + str(years[-1])
        intervals[years_label] = []
        for year in year_range:
            if year in years:
                intervals[years_label].extend(final_year_doc_dict[year]) 
    master_dict_tn = {}
    master_dict_tw = {}
    for key in intervals:
        print("Processing {} docs from {}...".format(len(intervals[key]), key))
        df_topic_num, df_topic_weights = docs_per_topic(model, corpus=intervals[key])
        
        master_dict_tn['Topic'] = df_topic_num['Dominant Topic'].tolist()
        master_dict_tn[key] = df_topic_num['Document Count'].tolist()

        master_dict_tw['Topic'] = df_topic_weights['Topic'].tolist()
        master_dict_tw[key] = df_topic_weights['Document Weight'].tolist()

    df_doc_counts_by_year = pd.DataFrame(master_dict_tn)
    df_doc_weights_by_year = pd.DataFrame(master_dict_tw)
    return df_doc_counts_by_year, df_doc_weights_by_year

def plot_doc_topics_per_time(df_data, n_topics, n_horiz=5, fig_dpi=150, ylabel=None, xlabel=None, topic_names=None, show=True, 
                fig_save_path=None, relative_val=True, x_val=None, xtick_space=None, xmintick_space=None, hide_x_val=True, 
                df_data2=None, relative_val2=True, ylabel2=None, colors=['tab:blue', 'tab:orange'], linear_reg=False):
    # df_data needs to be one of the outputs from doc_topics_per_time dataframes or data frame with topics in first column and labeled 'Topic'
    columns = list(df_data.columns)[1:]
    column_totals = df_data.loc[:,columns[0]:].sum(axis=0)
    column_totals_list = list(column_totals)
    graphs = {}
    graphs2 = {}
    if isinstance(n_topics, int):
        topics_list = list(range(1, n_topics + 1))
    else:
        topics_list = [i for i in n_topics].sort()

    for topic in topics_list:
        data = df_data.loc[df_data['Topic'] == topic, columns[0]:]
        data2 = None
        plot2 = False
        if relative_val:
            data = data / column_totals_list
            data.fillna(0, inplace=True)
            graphs[topic] = data.values.flatten().tolist()
        else:
            graphs[topic] = data.values.flatten().tolist()
        if  df_data2 is not None:
            data2 = df_data2.loc[df_data2['Topic'] == topic, columns[0]:]
            plot2 = True
            if relative_val2:
                data2 = data2 / column_totals_list
                graphs2[topic] = data2.values.flatten().tolist()
            else:
                graphs2[topic] = data2.values.flatten().tolist()

    # Plotting
    x_len = n_horiz
    y_len = math.ceil(len(topics_list)/n_horiz)
    if x_val is None:
        x_val = list(range(1, len(columns)+1))

    diff_axis = False
    if not relative_val == relative_val2:
        diff_axis = True
    ax2_list = []

    fig, axes = plt.subplots(y_len, x_len, figsize=(2*x_len, 1.5*y_len), dpi=fig_dpi, 
                                sharex=True, sharey=True, squeeze=False, constrained_layout=True)
    for i, ax in enumerate(axes.flatten()):
        if i < n_topics:
            ax.plot(x_val, graphs[topics_list[i]], color=colors[0])
            if plot2 and diff_axis: 
                ax2 = ax.twinx()
                ax2_list.append(ax2)
                ax2_list[0].get_shared_y_axes().join(*ax2_list)
                ax2.plot(x_val, graphs2[topics_list[i]], color=colors[1])
                if (i + 1) % x_len > 0 and (i + 1) != len(topics_list):
                    ax2.set_yticklabels([])
            elif plot2:
                ax.plot(x_val, graphs2[topics_list[i]], color=colors[1])

            if topic_names is not None:
                ax.title.set_text(topic_names[i+1])
            else:
                ax.title.set_text('Topic {}'.format(topics_list[i]))
            if xtick_space is not None: ax.xaxis.set_major_locator(ticker.MultipleLocator(xtick_space))
            if xmintick_space is not None: ax.xaxis.set_minor_locator(ticker.MultipleLocator(xmintick_space))
            if hide_x_val:ax.set_xticklabels([])
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha('right')
        else:
            ax.axis('off')
    if plot2 and diff_axis and False:
        print(len(ax2_list))
        ax2_list[0].get_shared_y_axes().join(*ax2_list)
    #plt.tight_layout()
    if xlabel is not None:
        fig.text(0.5, 0, xlabel, ha='center', va='top', fontsize=14)
    if ylabel is not None:
        fig.text(0, 0.5, ylabel, ha='right', va='center', fontsize=14, rotation=90)
    if ylabel2 is not None and plot2 and diff_axis:
        fig.text(1, 0.5, ylabel2, ha='left', va='center', fontsize=14, rotation=90)

    if fig_save_path is not None:
        plt.savefig(fig_save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

    if linear_reg:
        x = np.array(range(len(columns))).reshape(-1, 1)
        lr_dict = {
            'Topic':[],
            'Coefficient':[],
            'R^2':[]
        }
        for topic in graphs:
            lin_reg_mod = linear_model.LinearRegression()
            lin_reg_mod.fit(x, graphs[topic])
            if topic_names is not None:
               lr_dict['Topic'].append(topic_names[topic])
            else: 
                lr_dict['Topic'].append(topic)
            lr_dict['Coefficient'].append(lin_reg_mod.coef_[0])
            lr_dict['R^2'].append(lin_reg_mod.score(x, graphs[topic]))
        df_lr = pd.DataFrame(lr_dict)
        return df_lr

def graph(x, y, title=None, x_label=None, y_label=None, show=False, fig_save_path=None):
    plt.figure(figsize=(4,3), dpi=300)
    plt.plot(x, y, marker='.')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=30, ha='right')
    if fig_save_path is not None:
        plt.savefig(fig_save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def graph_multi(x_list, y_list, label_list, legend=None, legend_params={'loc':'best'}, title=None, x_label=None, y_label=None, show=False, fig_save_path=None):
    plt.figure(figsize=(4,3), dpi=300)
    for i, label in enumerate(label_list):
        plt.plot(x_list[i], y_list[i], label=label, marker='.')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(title=legend, **legend_params)
    plt.xticks(rotation=30, ha='right')
    if fig_save_path is not None:
        plt.savefig(fig_save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_tsne_doc_cluster3d(model, nlp_data, doc_list=None, corpus=None, min_tw=None, marker_size=1, seed=2020,
                            show_topics=False, custom_titles=None, show=True, fig_save_path=None):
    if corpus is None:
        if doc_list is None:
            corpus = nlp_data.gensim_lda_input()
        else:
            corpus = nlp_data.process_new_corpus(doc_list)['gensim']
    n_topics = model.num_topics
    topic_weights= {}
    for i in range(n_topics):
        topic_weights[i] = []
    for i, row_list in enumerate(model.get_document_topics(corpus)):
        temp_dict = {t:w for t, w in row_list}
        for topic in range(n_topics):
            if topic in temp_dict:
                topic_weights[topic].append(temp_dict[topic])
            else:
                topic_weights[topic].append(0)
    
    arr = pd.DataFrame(topic_weights).fillna(0).values
    
    if min_tw is not None:
        arr = arr[np.amax(arr, axis=1) >= min_tw]

    topic_num = np.argmax(arr, axis=1)



    tsne_model = TSNE(n_components=3, verbose=1, random_state=seed, angle=0.99, init='pca', n_jobs=-1)
    tsne_lda = tsne_model.fit_transform(arr)

    # Calculate geometric mean of doc coordinates to place topic titles
    topic_positions = {}
    if show_topics:
        for topic in range(n_topics):
            topic_arr = topic_num == topic
            coord_arr = tsne_lda[topic_arr]
            topic_loc = np.median(coord_arr, axis=0)
            topic_positions[topic] = topic_loc

    colors = np.array([color for name, color in mcolors.XKCD_COLORS.items()])

    title = "t-SNE Clustering of {} Topics".format(n_topics)
    x = tsne_lda[:,0]
    y = tsne_lda[:,1]
    z = tsne_lda[:,2]
    color = colors[topic_num]
    fig = plt.figure(figsize=(6,6), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, color=color, marker='.', s=marker_size)
    
    center_dots_x = []
    center_dots_y = []
    for topic in topic_positions:
        x, y, z = topic_positions[topic]
        center_dots_x.append(x)
        center_dots_y.append(y)
        if custom_titles is not None:
            text = custom_titles[topic+1]
        else:
            text = "Topic {}".format(topic+1)
        bbox=dict(facecolor=[1,1,1,0.5], edgecolor=colors[topic], boxstyle='round')
        txt_box = ax.text(x, y, z, text, horizontalalignment='center', verticalalignment='center', fontsize=5,
                            )
        txt_box.set_bbox(bbox)

    fig.suptitle(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    if fig_save_path is not None:
        plt.savefig(fig_save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_tsne_doc_cluster(model, nlp_data, doc_list=None, corpus=None, min_tw=None, marker_size=1, seed=2020,
                            dpi=450, show_topics=False, topic_names=None, show=True, show_legend=False, 
                            fig_save_path=None, tp_args=None, **kwargs):
    if corpus is None:
        if doc_list is None:
            corpus = nlp_data.gensim_lda_input()
        else:
            corpus = nlp_data.process_new_corpus(doc_list)['gensim']
    n_topics = model.num_topics
    topic_weights= {}
    for i in range(n_topics):
        topic_weights[i] = []
    for i, row_list in enumerate(model.get_document_topics(corpus)):
        temp_dict = {t:w for t, w in row_list}
        for topic in range(n_topics):
            if topic in temp_dict:
                topic_weights[topic].append(temp_dict[topic])
            else:
                topic_weights[topic].append(0)
    
    arr = pd.DataFrame(topic_weights).fillna(0).values
    
    if min_tw is not None:
        arr = arr[np.amax(arr, axis=1) >= min_tw]

    topic_num = np.argmax(arr, axis=1)



    tsne_model = TSNE(n_components=2, verbose=1, random_state=seed, angle=0.99, init='pca', n_jobs=-1)
    tsne_lda = tsne_model.fit_transform(arr)

    topic_positions = {}

    if show_topics:
        for topic in range(n_topics):
            topic_arr = topic_num == topic
            coord_arr = tsne_lda[topic_arr]
            topic_loc = np.median(coord_arr, axis=0)
            topic_positions[topic] = topic_loc

    colors = np.array([color for name, color in mcolors.XKCD_COLORS.items()])

    title = "t-SNE Clustering of {} Topics".format(n_topics)
    x = tsne_lda[:,0]
    y = tsne_lda[:,1]
    color = colors[topic_num]
    fig = plt.figure(figsize=(6,6), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.scatter(x, y, color=color, marker='.', s=marker_size)
    
    center_dots_x = []
    center_dots_y = []
    if tp_args is None:
        tp_args = {
            'fontsize':8,
            'weight':'bold'
        }
    for topic in topic_positions:
        x, y = topic_positions[topic]
        center_dots_x.append(x)
        center_dots_y.append(y)
        text = "T{}".format(topic+1)
        bbox=dict(facecolor=[1,1,1,0.6], edgecolor=[0,0,0,0], pad=0, boxstyle='round')
        txt_box = ax.text(x, y, text, horizontalalignment='center', verticalalignment='center', **tp_args
                            )
        txt_box.set_bbox(bbox)

    legend_list = []
    for topic in range(n_topics):
        if topic_names is None:
            text = "Topic {}".format(topic+1)
            legend_list.append(text)
        else:
            text = topic_names[topic+1]
            legend_list.append(text)
    if show_legend:
        kwargs2 = kwargs.copy()
        if 'size' in kwargs2:
            kwargs2['size'] += 8
        else:
            kwargs2['size'] = 'xx-large'
        t = ax.transAxes
        canvas = ax.figure.canvas
        y = 0.985
        add_offset = 0.002
        for i, topic in enumerate(legend_list):        
            ax.text(1.025, y, '\u2219', color=colors[i], transform=t, ha='center', va='center', **kwargs2)
            txt_box = ax.text(1.05, y, topic, color='black', transform=t, ha='left', va='center', **kwargs)
            txt_box.draw(canvas.get_renderer())
            ex = txt_box.get_window_extent()
            t = transforms.offset_copy(txt_box.get_transform(), y=-ex.height, units='dots')
            y -= add_offset

    fig.suptitle(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    if fig_save_path is not None:
        plt.savefig(fig_save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def rows_per_df_grp(df, grouping_column): # Takes a dataframe and returns a dict of dataframes by the grouping column, and a list of counts
    grp_df = df.groupby(grouping_column)
    grouped_row_df_dict = {}
    row_counts = []
    for grp, data in grp_df:
        grouped_row_df_dict[grp] = data
        row_counts.append((grp, len(data)))
    return grouped_row_df_dict, row_counts

def graph_coherence(data_path_list, title=None, x_label=None, y_label=None, show=False, fig_save_path=None, box_plot=True, **kwargs):
    graphs = {}
    for data_path in data_path_list:
        df = pd.read_csv(data_path)
        columns = list(df.columns)[1:]
        for column in columns:
            if column in graphs:
                graphs[column].extend(df[column].tolist())
            else:
                graphs[column] = df[column].tolist()
    labels = list(graphs.keys())
    x_values = []
    for label in labels:
        x_values.append(graphs[label])

    # Show graph
    plt.figure(figsize=(12, 8)) 
    if box_plot:   
        plt.boxplot(x_values, labels=labels, **kwargs)
    else:
        mean_sd_dict = {'X':[], 'MEAN':[], 'STDV':[]}
        for label in graphs:
            mean_sd_dict['X'].append(label)
            mean_sd_dict['MEAN'].append(np.mean(graphs[label]))
            mean_sd_dict['STDV'].append(np.std(graphs[label]))
        # Plot graph of  x VS y with standard deviation for error bars
        params = dict(fmt='_', markersize=10, capsize=5, markeredgewidth=1, c='Black')
        for key in kwargs:
            params[key] = kwargs[key]
        plt.errorbar(mean_sd_dict['X'], mean_sd_dict['MEAN'], 
                    yerr=mean_sd_dict['STDV'], **params)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if fig_save_path is not None:
        plt.savefig(fig_save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()# Closes and deletes graph to free up memory

def plot_topic_groups(df_data, topic_groups, n_horiz=5, fig_dpi=150, ylabel=None, xlabel=None, show=True, merge_graphs=False,
                fig_save_path=None, relative_val=True, x_val=None, xtick_space=None, xmintick_space=None, hide_x_val=True, 
                colors=['tab:blue'], linear_reg=False):
    # df_data needs to be one of the outputs from doc_topics_per_time dataframes or data frame with topics in first column and labeled 'Topic'
    topics_list = [topic-1 for group in topic_groups for topic in topic_groups[group]]
    group_list = [group for group in topic_groups]
    n_groups = len(group_list)
    

    columns = list(df_data.columns)[1:]
    column_totals = df_data.loc[topics_list,columns[0]:].sum(axis=0)
    column_totals_list = list(column_totals)
    graphs = {}
    grouped_df = pd.DataFrame()

    for group in topic_groups:
        data = df_data.loc[df_data['Topic'].isin(topic_groups[group]), columns[0]:].sum(axis=0)
        if relative_val:
            data = data / column_totals_list
            data.fillna(0, inplace=True)
            graphs[group] = data.values.flatten().tolist()
        else:
            graphs[group] = data.values.flatten().tolist()
        data = data.to_frame().T
        data['Topic Group'] = group
        data['Topics'] = str(topic_groups[group])[1:-1]
        grouped_df = pd.concat([grouped_df, data])
    
    grouped_df.reset_index(drop=True, inplace=True)
    
    new_columns = ['Topic Group'] + ['Topics'] + columns
    grouped_df = grouped_df[new_columns]

    # Plotting
    x_len = n_horiz
    y_len = math.ceil(n_groups/n_horiz)
    if x_val is None:
        x_val = list(range(1, len(columns)+1))

    
    if merge_graphs:
        fig = plt.figure(figsize=(12, 8)) 
        if n_groups > 10:
            plt.gca().set_prop_cycle(cycler(color=plt.get_cmap('tab20').colors))
        for graph in graphs:
            plt.plot(columns, graphs[graph], label=graph)
        plt.legend(loc='best')

    else:
        fig, axes = plt.subplots(y_len, x_len, figsize=(2*x_len, 1.5*y_len), dpi=fig_dpi, 
                                    sharex=True, sharey=True, squeeze=False, constrained_layout=True)
        for i, ax in enumerate(axes.flatten()):
            if i < n_groups:
                ax.plot(x_val, graphs[group_list[i]], color=colors[0])
                ax.title.set_text(group_list[i])

                if xtick_space is not None: ax.xaxis.set_major_locator(ticker.MultipleLocator(xtick_space))
                if xmintick_space is not None: ax.xaxis.set_minor_locator(ticker.MultipleLocator(xmintick_space))
                if hide_x_val:ax.set_xticklabels([])
                for label in ax.get_xticklabels():
                    label.set_rotation(45)
                    label.set_ha('right')
            else:
                ax.axis('off')

    #plt.tight_layout()
    if xlabel is not None:
        fig.text(0.5, 0, xlabel, ha='center', va='top', fontsize=14)
    if ylabel is not None:
        fig.text(0, 0.5, ylabel, ha='right', va='center', fontsize=14, rotation=90)

    if fig_save_path is not None:
        plt.savefig(fig_save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

    if linear_reg:
        x = np.array(range(len(columns))).reshape(-1, 1)
        lr_dict = {
            'Topic Group':[],
            'Topics':[],
            'Coefficient':[],
            'R^2':[]
        }
        for group in topic_groups:
            lin_reg_mod = linear_model.LinearRegression()
            lin_reg_mod.fit(x, graphs[group])
            lr_dict['Topic Group'].append(group)
            lr_dict['Topics'].append(str(topic_groups[group])[1:-1])
            lr_dict['Coefficient'].append(lin_reg_mod.coef_[0])
            lr_dict['R^2'].append(lin_reg_mod.score(x, graphs[group]))
        df_lr = pd.DataFrame(lr_dict)
        return grouped_df, df_lr
    else:
        return grouped_df

def build_summary_df(df_bestdoc, df_nt, df_ty, topic_names=None, rel_val=True):
    df_lr = plot_doc_topics_per_time(df_ty, n_topics=len(df_bestdoc["Topic Number"]), show=False, relative_val=rel_val, linear_reg=True)
    
    columns_names = ["Topic", "Keywords", "Document Count", "Coefficient", "R^2"]
    topic_kw_counts_dict = {
        columns_names[0]:df_bestdoc["Topic Number"].to_list(),
        columns_names[1]:df_bestdoc["Topic Keywords"].to_list(),
        columns_names[2]:df_nt["Document Count"].tolist(),
        columns_names[3]:df_lr["Coefficient"].tolist(),
        columns_names[4]:df_lr["R^2"].tolist(),
    }
    if topic_names is not None:
        columns_names.insert(1, "Name")
        p = re.compile(r'T\d\d*:') # This removes the topic number that is in my topic labels e.g. 'T1:'
        topic_names_list = [
            topic_names[i+1][re.match(p, topic_names[i+1]).end():] 
            if re.match(p, topic_names[i+1]) is not None else topic_names[i+1] 
            for i in range(len(topic_names))
            ]
        topic_kw_counts_dict[columns_names[1]] = topic_names_list
        
    df_nt_kw_lr = pd.DataFrame(topic_kw_counts_dict)
    df_nt_kw_lr = df_nt_kw_lr[columns_names]
    return df_nt_kw_lr

def build_cooc_matrix_df(model, nlp_data, doc_list=None, corpus=None, min_tw=None):
    # This creates 2 co-occurence matrix dataframes
    # The first returned df is by topic weights
    # The second df ignores topic weights and a document has a topic if its weight is greater than min_tw or 0.1
    if corpus is None:
        if doc_list is None:
            corpus = nlp_data.gensim_lda_input()
        else:
            corpus = nlp_data.process_new_corpus(doc_list)['gensim']
    n_topics = model.num_topics
    topic_weights= {}
    for i in range(1, n_topics+ 1):
        topic_weights[i] = []
    for i, row_list in enumerate(model.get_document_topics(corpus, minimum_probability=0.001)):
        temp_dict = {t+1:w for t, w in row_list}
        for topic in range(1, n_topics+1):
            if topic in temp_dict:
                topic_weights[topic].append(temp_dict[topic])
            else:
                topic_weights[topic].append(0)
    
    arr = pd.DataFrame(topic_weights).fillna(0)
    
    if min_tw is not None:
        arr_n = arr[arr >= min_tw].fillna(0)
    else:
        arr_n = arr[arr >= 0.1].fillna(0) 
    arr_n[arr_n > 0] = 1

    df_cooc_w = arr.T.dot(arr)
    df_cooc_n = arr_n.T.dot(arr_n)
    np.fill_diagonal(df_cooc_w.values, 0)
    np.fill_diagonal(df_cooc_n.values, 0)

    return df_cooc_w, df_cooc_n

def plot_heatmap(df_matrix, topic_names=None, show=True, fig_save_path=None):
    #plots a heatmap of the passed co-occurence matrix
    #fig, ax = plt.subplots(figsize=(8,8))
    #plt.figure(figsize=(8,8), dpi=300, facecolor='w')
    sns.set(font_scale=0.8)
    sns.heatmap(df_matrix, linewidths = 0.5, square=True, cmap='YlOrRd', xticklabels=True, yticklabels=True)
    plt.tight_layout()

    # This saves and/or shows the plot. Note: Saved file looke better than the drawn plot
    length = len(df_matrix.index)
    plt.ylim(length, 0)
    plt.xlim(0, length) 

    #ax.set_ylim(length, 0)
    #ax.set_xlim(0, length) 
    if fig_save_path is not None:
        plt.savefig(fig_save_path, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    plt.close()    

def plot_clusterheatmap(df_matrix, topic_names=None, show=True, fig_save_path=None, **kwargs):
    #plots a cluster heatmap of the passed co-occurence matrix

    df1 = df_matrix.copy()
    # This relabels the columns and rows if topic names is passed
    if topic_names is not None:
        df1.index = [topic_names[topic] for topic in df1.index]
        df1.columns = [topic_names[int(topic)] for topic in df1.columns]
    # This if for calculating the linkage manually outside of the cluster method because
    # the co-occurence matrix is an uncondensed distance matrix. To properly calculate linkage
    # the matrix must be reprocessed with values closer to 0 indicating stronger association.
    # To accomplish this the max value in the matrix is used as the new 0 and all other values are
    # max value - matrix value. The diagnol is reassigned to 0, and then the matrix is transformed
    # into a condensed distance matrix as input to the linkage method and the result
    # is used for the clustering method. 
    df2 = df_matrix.values.max() - df_matrix
    np.fill_diagonal(df2.values, 0)
    df3 = hc.linkage(ssd.squareform(df2), method='average')
    sns.set(font_scale=0.9)
    # This makes the cluster graph and assigns a reference to it as sns_grid
    sns_grid = sns.clustermap(df1, 
        row_linkage = df3,
        col_linkage = df3,
        **kwargs)
    plt.tight_layout()
    #sns_grid.savefig("reports/main_a5/testing.png")
    
    # This adjusts the rotation and positions of the x and y tick labels
    sns_grid.ax_heatmap.set_yticklabels(sns_grid.ax_heatmap.get_yticklabels(), rotation=0)
    if topic_names is not None:
        sns_grid.ax_heatmap.set_xticklabels(sns_grid.ax_heatmap.get_xticklabels(), rotation=-60, ha='left')
        xd = -10/72
        offset = mpl.transforms.ScaledTranslation(xd, 0, sns_grid.fig.dpi_scale_trans)
        for label in sns_grid.ax_heatmap.get_xticklabels():
            label.set_transform(label.get_transform() + offset)
    # This is to ensure that the heatmap is of the appropriate size.
    # There were issues where part of the heatmap was cutoff
    length = len(df1.index)
    sns_grid.ax_heatmap.set_ylim(length, 0)
    sns_grid.ax_heatmap.set_xlim(0, length) 
    # This saves and/or shows the plot. 
    if fig_save_path is not None:
        plt.savefig(fig_save_path, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    plt.close()    

def import_topic_names(file_path_name):
    # imports topic names from a spreadsheet with the first columns containing topic numbers and 
    # the second containing topic names
    if file_path_name[-3:] == 'csv':
        df = pd.read_csv(file_path_name)
    elif file_path_name[-3:] in ['xls','lsx', 'lsm', 'lsb', 'odf']:
        df = pd.read_excel(file_path_name)
    else:
        raise NameError('Unsupported file format, please provide csv or excel file')
    
    topic_names_dict = dict(zip(df.iloc[:,0].values, df.iloc[:,1].values))
    return topic_names_dict

class SilentPrinting:
    def __init__(self, verbose=False):
        self.verbose=verbose
        
    def __enter__(self):
        if not self.verbose:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.verbose:
            sys.stdout.close()
            sys.stdout = self._original_stdout

class Timing:
    def __init__(self, text=None):
        if text is None:
            text = "Initiating..."
        self.text = text
        self.t = None
        
    def __enter__(self):
        self.t = time()
        print(self.text)

    def __exit__(self, exc_type, exc_val, exc_tb):
        comp_time =  time() - self.t                                       
        print("Done in {:.3f}s.".format(comp_time))

if __name__ == "__main__": # Code only runs if this file is run directly. This is for testing purposes
    pass

    # Example of how to combine multiple CompareModels outputs into 1 graph and file
    """
    coh_file_list = glob.glob('.\\reports\\t(5_50_5)test*coh.csv')
    time_file_list = glob.glob('.\\reports\\t(5_50_5)test*time.csv')

    plot_model_comparison(coh_file_list, 'Number of Topics', ['Gensim', 'Mallet', 'Sklearn'], 
        "Number of Topics", "C_V Coherence", "Model Coherence Comparison", show = True, 
        fig_save_path = 'reports/figures/test_coh_means_stdv.png', csv_save_path = 'reports/test_coh_means_stdv.csv')

    plot_model_comparison(time_file_list, 'Number of Topics', ['Gensim', 'Mallet', 'Sklearn'], 
        "Number of Topics", "Time (sec)", "Model Time Comparison", show = True, 
        fig_save_path = 'reports/figures/test_time_means_stdv.png', csv_save_path = 'reports/test_time_means_stdv.csv')
    """
    