# Author: Thomas Porturas <thomas.porturas.eras@gmail.com>
from time import time
import os, sys

import pandas as pd
import gensim
import pickle
import random
import pyLDAvis
import pyLDAvis.gensim

# These are the import statements for the custom modules. If you want to run this file seperately, switch imports.
from . import data_nl_processing
from . import model_utilities as mu
#import data_nl_processing
#import model_utilities as mu

"""
The :mod:'mallet_model' module contains the MalletModel class and the generate_mallet_models() function.
The MalletModel class is a container for the MALLET model used for topic modeling.
The generate_mallet_models() function produces and saves several predefined MalletModel objects along with
information and initial results for each generated model
"""

class MalletModel:
    """
    This class is for building a MALLET or gensim LDA model.

    Parameters
    ----------
    nlp_data : This is the processed corpus as a fitted NlpForLdaInput object

    topics : an integer of number of topics for the model to generate. Default 20

    seed : an integer of the random seed. Default 0

    topn : integer of number of top keywords to use for coherence calculations. Default 20

    coherence : string name of coherence model to use for the gensim.models.CoherenceModel class. Default 'c_v' 
    
    model_type : string name of model to use. Options are 'mallet' or 'gensim'. Default 'mallet'

    mallet_path : path to where you extracted the MALLET binaries. Default 'C:\\mallet\\bin\\mallet'

    **parameters : keyword arguments for the model parameters if you want to change the default parameters. 

    After initializing the class, run the start() method. The model is converted to and stored as a gensim 
    LDA object in self.model. The raw model is save as self.model_raw
    
    Use the save() method to save the model to the path variable. You can load the model by using pickle:
        with open('path', 'rb') as model:
            malletmodel = pickle.load(model)

    """
    def __init__(self, nlp_data, topics=20, seed=0, topn=20, coherence='c_v', model_type='mallet',
                    mallet_path='C:\\mallet\\bin\\mallet', **parameters):
        self.nlp_data = nlp_data
        self.topics = topics
        self.seed = seed
        self.mallet_path = mallet_path
        self.model = None
        self.model_raw = None
        self.coherence = coherence
        self.topn = topn
        self.model_type = model_type
        if model_type == 'mallet':
            self.parameters = {
                    'mallet_path':self.mallet_path,
                    'workers':5,
                    'random_seed':self.seed,
                    'alpha':50,
                    'iterations':1000,
                    'optimize_interval':0
                }
        elif model_type == 'gensim':
            if self.seed == 0:
                self.seed = None
            self.parameters = {
                    'decay':0.5,
                    'alpha':'asymmetric',
                    'eta':None,
                    'workers':None,
                    'random_state':self.seed,
                    'chunksize':100,
                    'passes':10,
                    'per_word_topics':True
                }
        else:
            raise ValueError("model_type must be either 'mallet' or 'gensim'")

        for key in parameters:
            self.parameters[key] = parameters[key]
    
    def start(self, verbose=True):

        print("Initiating model building...")
        t0 = time()
        if self.model_type == 'mallet':
            self.model_raw = self.mallet_model_(self.topics)
            self.model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(self.model_raw['model'])
        elif self.model_type == 'gensim':
            self.model_raw = self.gensim_model_(self.topics)
            self.model = self.model_raw['model']
        print("Model done in {:0.3f}s".format(time()-t0))

    def gensim_model_(self, topics):
        tm = time()

        print("Building Gensim model...")
        model = gensim.models.LdaMulticore(corpus=self.nlp_data.gensim_lda_input(),
                                                id2word=self.nlp_data.get_id2word(),
                                                num_topics=topics,
                                                **self.parameters)
        model_time =  (time() - tm)                                       
        print("Done in %0.3fs." % model_time)
        tc = time()
        print("Running Coherence model for Gensim...")
        model_topics_list = self.gensim_topic_words_(model.show_topics(formatted=False, num_words=self.topn, num_topics=-1))
        coh_model = gensim.models.CoherenceModel(topics=model_topics_list, texts=self.nlp_data.get_token_text(),
                                                        dictionary=self.nlp_data.get_id2word(), window_size=None,
                                                        coherence=self.coherence)
        model_coherence = coh_model.get_coherence()

        print("Done in %0.3fs." % (time() - tc))
        print("Gensim coherence for {} topics is: {:0.3f}".format(topics, model_coherence))
        return {"coherence":model_coherence, "time":model_time, "topics":model_topics_list, "model":model}

    def mallet_model_(self, topics):
        tm = time()
        print("Building Mallet model...")     
        model = gensim.models.wrappers.LdaMallet(corpus=self.nlp_data.gensim_lda_input(),
                                                    id2word=self.nlp_data.get_id2word(),
                                                    num_topics=topics,
                                                    **self.parameters)

        model_time =  (time() - tm)                                       
        print("Done in %0.3fs." % model_time)
        tc = time()
        print("Running Coherence model for Mallet...")
        model_topics_list = self.gensim_topic_words_(model.show_topics(formatted=False, num_words=self.topn, num_topics=-1))
        coh_model = gensim.models.CoherenceModel(topics=model_topics_list, texts=self.nlp_data.get_token_text(),
                                                        dictionary=self.nlp_data.get_id2word(), window_size=None,
                                                        coherence=self.coherence)
        model_coherence = coh_model.get_coherence()

        print("Done in %0.3fs." % (time() - tc))
        print("Mallet coherence for {} topics is: {:0.3f}".format(topics, model_coherence))
        return {"coherence":model_coherence, "time":model_time, "topics":model_topics_list, "model":model}

    def gensim_topic_words_(self, show_topics):
        show_topics.sort()
        topic_word_list = []
        for topic in show_topics:
            message = "Topic #%d: " % topic[0]
            new_list = list(word[0] for word in topic[1])
            message += ", ".join(new_list)
            topic_word_list.append(new_list)
            print(message)
        print()
        return topic_word_list

    def output_parameters(self, save=False, path=None):
        if save:
            with open(path, 'w') as file:
                file.write(str(self.parameters))
        return self.parameters

    def save(self, file_path_name):
        with open(file_path_name, 'wb') as file:            
            pickle.dump(self, file)

def generate_mallet_models(data_path, data_column, model_save_folder, figure_save_folder, topic_num, model_params, 
                            file_name_append=None, seed=None, **nlpargs):
    """
    This function is for building multiple mallet models and producing an initial set of results using pyLDAvis and
    wordcloud. All files are saved to the folders specified.

    Parameters
    ----------
    data_path : string to the data file.

    data_column : string of the column name containing the text data

    model_save_folder : string of the path to where to save the model

    figure_save_folder : string of the path to where to save the output figures.

    topic_num : an integer of number of topics for the models to generate

    model_params : a list of dictionaries of the model parameters to generate. You can also include the mallet_path
        entry in this dict if it differs from the default path 'C:\\mallet\\bin\\mallet'

    file_name_append: string to append to file names generated. Default None    

    seed : an integer of the random seed. Default None

    **nlpargs : keyword arguments for the nlp parameters if you want to change the default parameters. 

    Nothing is returned by this function.
    
    """
    with mu.Timing("Loading Data..."):
        df = pd.read_csv(data_path)
        data = df[data_column].tolist()

    with mu.Timing('Processing Data...'):
        nlp_params = dict(spacy_lib='en_core_sci_lg', max_df=.25, bigrams=True, trigrams=True, max_tok_len=30)
        for key in nlpargs:
            if key in nlp_params:
                nlp_params[key] = nlpargs[key]
        nlp_data = data_nl_processing.NlpForLdaInput(data, **nlp_params)
        nlp_data.start()

    with mu.Timing("Building Models..."):
        os.makedirs(model_save_folder, exist_ok=True)
        if seed is None:
            seed = int(time()*100)-158000000000
        if file_name_append is None:
            append = ''
        else:
            append = file_name_append
        models = model_params
        model_list = []
        for model in models:
            mallet_model = MalletModel(nlp_data, topics=topic_num, seed=seed, model_type='mallet', **model)
            mallet_model.start()
            save_path = model_save_folder + 'mallet_t{}a{}o{}{}'.format(topic_num, model['alpha'], model['optimize_interval'], append)
            mallet_model.save(save_path)
            model_list.append((mallet_model, save_path))

        with open(model_save_folder + 'mallet_parameters_{}T{}.txt'.format(topic_num, append), 'w') as para_file:
            file_string_list = []
            file_string_list.append("Model Parameters for {} Topics \n".format(topic_num))
            file_string_list.append("\n")
            for i in range(len(model_list)):
                file_string_list.append("Mallet model t{}a{}o{} Parameters: \n".format(topic_num, models[i]['alpha'], models[i]['optimize_interval']))
                file_string_list.append("Model {}/{} generated \n".format(i+1, len(model_list)))
                file_string_list.append("File Path: {} \n".format(model_list[i][1]))
                file_string_list.append("{} Coherence: {} \n".format(model_list[i][0].coherence, model_list[i][0].model_raw['coherence']))
                for key in model_list[i][0].parameters:
                    file_string_list.append("{}: {} \n".format(key, model_list[i][0].parameters[key]))
                file_string_list.append("\n")
            para_file.writelines(file_string_list)
    with mu.Timing("Creating Figures..."):
        os.makedirs(figure_save_folder, exist_ok=True)
        for i in range(len(model_list)):
            save_path = figure_save_folder + 'mallet_t{}a{}o{}s{}{}.html'.format(
                            topic_num, models[i]['alpha'], models[i]['optimize_interval'], seed, append)
            panel = pyLDAvis.gensim.prepare(model_list[i][0].model, model_list[i][0].nlp_data.gensim_lda_input(), model_list[i][0].nlp_data.get_id2word(), 
                                        mds='tsne', sort_topics=False)
            pyLDAvis.save_html(panel, save_path)

        for i in range(len(model_list)):
            save_path = figure_save_folder + 'mallet_wordcloud_t{}a{}o{}s{}{}.png'.format(
                            topic_num, models[i]['alpha'], models[i]['optimize_interval'], seed, append)
            mu.create_multi_wordclouds(topic_num, 8, model_list[i][0].model, model_list[i][0].nlp_data, num_w=20, fig_dpi=400,
                            show=False, fig_save_path=save_path)

if __name__ == "__main__":
    pass

    # Example code
    """
    data_path = 'data/processed/data.csv'
    data_column = 'title_abstract'
    addendum = "test"
    model_save_folder = 'models/'
    figure_save_folder = 'reports/figures/'
    topic_num = 40
    model_params = [
                {'alpha':5,'optimize_interval':0},
                {'alpha':50,'optimize_interval':0},
                {'alpha':5,'optimize_interval':200},
                {'alpha':50,'optimize_interval':200},
                ]
    generate_mallet_models(data_path, data_column, model_save_folder, figure_save_folder, topic_num, model_params, 
                        file_name_append=addendum)
    """