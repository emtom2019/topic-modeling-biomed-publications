# Author: Thomas Porturas <thomas.porturas.eras@gmail.com>

from time import time
from datetime import datetime

import pandas as pd
import gensim
import pickle
from sklearn.decomposition import LatentDirichletAllocation

import matplotlib.pyplot as plt

# These are the import statements for the custom modules. If you want to run this file seperately, switch imports.
from . import data_nl_processing
from .model_utilities import SilentPrinting
#import data_nl_processing
#from model_utilities import SilentPrinting


"""
The :mod:'compare_models' module contains the CompareModels class that builds multiple
LDA models based on input parameters using the scikit-learn, gensim, and MALLET packages
and compares them using gensim's CoherenceModel class. It also contains the run_model_comparison 
function which runs multiple iterations of the CompareModels class.
"""

class CompareModels:
    """
    This class is for building and comparing the coherence of multiple scikit-learn, gensim, and MALLET models 
    to help fine decide the best model to use and the optimal number of topics to generate.

    Parameters
    ----------
    nlp_data : This is the processed corpus as a fitted NlpForLdaInput object

    topics : a tuple of integers (start, end, step) indicating numbers of topics. Default (5,50,5)

    seed : integer for the random seed for all models generated. Default None

    gensim_lda: boolean value of whether to run gensim LDA model. Default True 
    
    mallet_lda: boolean value of whether to run MALLET LDA model. Default True 
    
    sklearn_lda: boolean value of whether to run scikit-learn LDA model. Default True 

    topn : integer of number of top keywords to use for coherence calculations. Default 20

    coherence : string name of coherence model to use for the gensim.models.CoherenceModel class. Default 'c_v' 
    
    mallet_path : path to where you extracted the MALLET binaries. Default 'C:\\mallet\\bin\\mallet'

    After initializing the class, run the start() method. You can save the models with the save() method.
    Graph results using the graph_results() method and save a csv file using the output_dataframe() method.
    
    After you are done creating models, the trained models are stored in the self.models dictionary. The models
    are seperated into dictionaries named 'Gensim', 'Mallet', or 'Sklearn', and are further seperated into 
    dictionaries named after the topic number as: 
        {"coherence":model_coherence, "time":model_time, "topics":model_topics_list, "model":model}
    
    Use the save() method to save the model to the path variable. You can load the model by using pickle:
        with open('path', 'rb') as model:
            comparemodels = pickle.load(model)

    WARNING: When running with gensim_lda=True, make sure to have 'if __name__ == "__main__":' before any
    executable code. Otherwise the gensim.models.LdaMulticore class will reload the module and cause
    an infinite loop of running models.
    """
    def __init__(self, nlp_data, topics=(5,50,5), seed=None, gensim_lda=True, mallet_lda=True, sklearn_lda=True, 
                    topn=20, coherence='c_v', mallet_path='C:\\mallet\\bin\\mallet'):
        self.nlp_data = nlp_data
        self.gensim_lda = gensim_lda
        self.mallet_lda = mallet_lda
        self.sklearn_lda = sklearn_lda
        self.topics_range = range(topics[0], topics[1]+topics[2], topics[2])
        self.seed = seed
        self.mallet_path = mallet_path
        self.models = {}
        self.coherence = coherence
        self.topn = topn
        if self.seed is None:
            mallet_seed = 0
        else:
            mallet_seed = self.seed
        self.parameters = {
            'gensim':{
                'decay':0.5,
                'alpha':'asymmetric',
                'eta':None,
                'workers':None,     # This uses all available cpu cores -1
                'random_state':self.seed,
                'chunksize':100,
                'passes':10,
                'per_word_topics':True
            },
            'mallet':{
                'mallet_path':self.mallet_path,
                'workers':5,  # Number of threads used during training. May need to change to CPU count -1
                'random_seed':mallet_seed,
                'alpha':50,
                'iterations':1000,
                'optimize_interval':0
            },
            'sklearn':{
                'max_iter':10,                  
                'learning_method':'online',     
                'random_state':self.seed,       
                'batch_size':128,               
                'evaluate_every':-1,            
                'n_jobs':-1,              # Number of CPUs used (-1 = all)
                'learning_decay':0.7,
                'learning_offset':10,
                'doc_topic_prior':None,         
                'topic_word_prior':None                 
            }
        }

    def start(self, verbose=True):
        print("Initiating model building...")
        t0 = time()
        if self.gensim_lda:
            self.models['Gensim'] = {}
        if self.mallet_lda:
            self.models['Mallet'] = {}
        if self.sklearn_lda:
            self.models['Sklearn'] = {}
        with SilentPrinting(verbose):
            self.run_models_(self.topics_range)
        print("All models done in {:0.3f}s".format(time()-t0))

    def run_models_(self, topics):
        for topic_num in topics:
            t1 = time()
            print("Building models with {} topics...".format(topic_num))
            
            if self.gensim_lda:
                self.models['Gensim'][topic_num] = self.gensim_model_(topic_num)

            if self.mallet_lda:
                self.models['Mallet'][topic_num] = self.mallet_model_(topic_num)
            
            if self.sklearn_lda:
                self.models['Sklearn'][topic_num] = self.sklearn_model_(topic_num)

            print("{}-topic models done in {:0.3f}s.".format(topic_num, time() - t1))

    def gensim_model_(self, topics):
        tm = time()

        print("Building Gensim model...")
        model = gensim.models.LdaMulticore(corpus=self.nlp_data.gensim_lda_input(),
                                                id2word=self.nlp_data.get_id2word(),
                                                num_topics=topics,
                                                **self.parameters['gensim'])
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
                                                    **self.parameters['mallet'])

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

    def sklearn_model_(self, topics):
        tm = time()

        print("Building Sklearn model...")

        model = LatentDirichletAllocation(n_components = topics,      # Number of topics
                                        **self.parameters['sklearn']
                                        )
        model.fit(self.nlp_data.sklearn_lda_input())

        model_time =  (time() - tm)                                       
        print("Done in %0.3fs." % model_time)
        tc = time()
        print("Running Coherence model for Sklearn...")
        model_topics_list = self.sklearn_topic_words_(model)
        coh_model = gensim.models.CoherenceModel(topics=model_topics_list, texts=self.nlp_data.get_token_text(),
                                                        dictionary=self.nlp_data.get_id2word(), window_size=None,
                                                        coherence=self.coherence)
        model_coherence = coh_model.get_coherence()

        print("Done in %0.3fs." % (time() - tc))
        print("Sklearn coherence for {} topics is: {:0.3f}".format(topics, model_coherence))
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

    def sklearn_topic_words_(self, model):
        topic_word_list = []
        feature_names = self.nlp_data.get_feature_names()
        for topic_idx, topic in enumerate(model.components_):
            message = "Topic #%d: " % topic_idx
            new_list = list(feature_names[i]
                                for i in topic.argsort()[:-self.topn - 1:-1])
            message += ", ".join(new_list)
            topic_word_list.append(new_list)
            print(message)
        print()
        return topic_word_list
    def graph_results(self, show=True, save=False, path=None):
        graphs = {}
        # Bulding graph x,y lists
        for key in self.models:
            graphs[key] = ([],[])
            for topic_num in self.models[key]:
                graphs[key][0].append(topic_num)
                graphs[key][1].append(self.models[key][topic_num]['coherence'])
        
        # Show graph
        plt.figure(figsize=(12, 8))
        for key in graphs:
            plt.plot(graphs[key][0], graphs[key][1], label=key)
        plt.title("Choosing Optimal LDA Model")
        plt.xlabel("Number of topics")
        plt.ylabel(self.coherence + " coherence")
        plt.legend(title='Models', loc='best')
        if save:
            plt.savefig(path, bbox_inches='tight')
        if show:
            plt.show()
        plt.close() # Closes and deletes graph to free up memory

    def output_dataframe(self, data_column="coherence", save=False, path=None):
        dataframe_dict = {}
        topics = None
        for key in self.models:
            data_list = []
            if topics is None:
                topics = list(self.models[key].keys())
                topics.sort()
                dataframe_dict["Number of Topics"] = topics
            for topic_num in topics:
                data_list.append(self.models[key][topic_num][data_column])
            dataframe_dict[key] = data_list
        data = pd.DataFrame.from_dict(dataframe_dict)
        if save:
            data.to_csv(path, index=False)
        return data

    def output_parameters(self, save=False, path=None):
        if save:
            with open(path, 'w') as file:
                file.write(str(self.parameters))
        return self.parameters

    def save(self, file_path_name):
        with open(file_path_name, 'wb') as file:            
            pickle.dump(self, file)

def run_model_comparison(runs, data_path, data_column, topic_range, add_info, models, 
    ngram_max=3, mallet_path='C:\\mallet\\bin\\mallet'):
    """
    This function is for running and saving multiple instances of the CompareModels class for the purpose of
    producing replications with random seeds generated and saved each time.

    Parameters
    ----------
    runs : Number of replications to run

    data_path : path to the data file

    data_column : name of the column containing the data in the data file
    
    topic_range : a tuple of integers (start, end, step) indicating number of topics to compare

    add_info : a string denoting additional information to append to file names

    models : a dictionary {'gensim_lda':True, 'mallet_lda':True, 'sklearn_lda':True} of which models to run

    ngram_max : integer of the maximum size of n-grams to produce. Default and maximum is 3.

    mallet_path : path to where you extracted the MALLET binaries. Default 'C:\\mallet\\bin\\mallet'

    This function saves files to the relative path  '\\models', '\\reports', and '\\reports\\figures'
    There will be an error if these folders do not exist.

    """
    for run in range(runs):
        
        if ngram_max > 2:
            trigrams = True
            bigrams = True
        elif ngram_max > 1:
            trigrams = False
            bigrams = True
        else:
            trigrams = False
            bigrams = False

        print("Loading dataset for CompareModels testing...")
        t0 = time()
        df = pd.read_csv(data_path)
        data = df[data_column].tolist()
        print("done in %0.3fs." % (time() - t0))

        spacy_library = 'en_core_sci_lg'
        nlp_data = data_nl_processing.NlpForLdaInput(data, spacy_lib=spacy_library, max_df=.25, bigrams=bigrams, trigrams=trigrams)
        nlp_data.start()

        model_seed = int(time()*100)-158000000000

        compare_models = CompareModels(nlp_data=nlp_data, topics=topic_range, seed=model_seed, coherence='c_v', 
            mallet_path=mallet_path, **models)
        compare_models.start()

        now = datetime.now().strftime("%m%d%Y%H%M")
        print("All models done at: " + now)
        
        compare_models.save('models/t({}_{}_{}){}{}mod'.format(*topic_range, add_info, model_seed))
        compare_models.output_dataframe(save=True, path='reports/t({}_{}_{}){}{}coh.csv'.format(*topic_range, add_info, model_seed))
        compare_models.output_dataframe(save=True, path='reports/t({}_{}_{}){}{}time.csv'.format(*topic_range, add_info, model_seed), data_column="time")
        compare_models.output_parameters(save=True, path='reports/t({}_{}_{}){}{}para.txt'.format(*topic_range, add_info, model_seed))
        compare_models.graph_results(show=False, save=True, path='reports/figures/t({}_{}_{}){}{}.png'.format(*topic_range, add_info, model_seed))

if __name__ == "__main__": # Prevents the following code from running when importing module
    pass

    # Example code for CompareModels:
    """
    print("Loading dataset for CompareModels testing...")
    t0 = time()
    df = pd.read_csv('data/processed/data.csv')
    data = df["title_abstract"].tolist()
    print("done in %0.3fs." % (time() - t0))

    nlp_data = data_nl_processing.NlpForLdaInput(data)
    nlp_data.start()

    compare_models = CompareModels(nlp_data=nlp_data, topics=(5,50,5), seed=2020, coherence='c_v')
    compare_models.start()
    compare_models.save('models/compare_models_test')
    compare_models.output_dataframe(save=True, path='reports/compare_models_test_coh.csv')
    compare_models.output_dataframe(save=True, path='reports/compare_models_test_time.csv', data_column="time")
    print(compare_models.output_parameters(save=True, path='reports/compare_models_test_params.txt'))
    compare_models.graph_results(save=True, path='reports/figures/compare_models_test_fig.png')
    """

    #Example code for run_model_comparison() function:
    """
    data_path = 'data/processed/data.csv'
    data_column = 'title_abstract'
    topic_range = (5, 50, 5)
    models = {'gensim_lda':True, 'mallet_lda':True, 'sklearn_lda':True}
    run_model_comparison(10, data_path, data_column, topic_range, "test", models)
    """
