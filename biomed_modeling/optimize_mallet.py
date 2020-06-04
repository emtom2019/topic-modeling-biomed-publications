from time import time
import pandas as pd
import re, gensim
import pickle
import matplotlib.pyplot as plt

import data_nl_processing 
from model_utilities import SilentPrinting

"""
The :mod:'optimize_mallet' module contains the CompareMalletModels class that builds multiple MALLET 
LDA models based on input parameters.
"""

class CompareMalletModels:
    """
    This class if for building and comparing the coherence of multiple MALLET models to help fine tune the 
    starting parameters.

    Parameters
    ----------
    nlp_data : This is the processed corpus as an NlpForLdaInput object

    topics : a list of numbers of topics for the models to generate. Default [40]

    alpha : a list of numbers for the alpha parameter to try. Default [50]
    
    opt_int : a list of numbers for the opt_int parameter to try. Default [0]

    iterations : a list of numbers for the iterations parameter to try. Default [1000]

    repeats : integer of number repetitions to produce per set of model parameters. Default 1

    topn : integer of number of top keywords to use for coherence calculations. Default 20

    coherence : string name of coherence model to use for the gensim.models.CoherenceModel class. Default 'c_v' 
    
    mallet_path : path to where you extracted the MALLET binaries. Default 'C:\\mallet\\bin\\mallet'

    After initializing the class, run the start() method. You can save the models with the save() method.
    Graph results using the graph_results() method and save a csv file using the output_dataframe() method.
    
    After you are done creating models, the trained gensim.models.wrappers.LdaMallet models are stored in
    a list in the self.models dictionary object. For example, the default settings model is stored as
    CompareMalletModels.models[t40a50o0i1000][0]
    

    """
    def __init__(self, nlp_data, topics=[40], alpha=[50], opt_int=[0], iterations=[1000],
                repeats=1, topn=20, coherence='c_v', mallet_path='C:\\mallet\\bin\\mallet'):
        self.nlp_data = nlp_data
        self.repeats = repeats
        self.mallet_path = mallet_path
        self.models = {}
        for t in topics:
            for a in alpha:
                for o in opt_int:
                    for i in iterations:
                        self.models['t{}a{}o{}i{}'.format(t, a, o, i)] = []
        self.coherence = coherence
        self.topn = topn
        self.testing_para = {
            'topics':topics,
            'alpha':alpha,
            'opt_int':opt_int,
            'iterations':iterations
        }
        self.parameters = {
            'mallet_path':self.mallet_path,
            'workers':5,
        }

    def start(self, verbose=True):
        # This starts running the models.
        print("Initiating model building...")
        t0 = time()
        with SilentPrinting(verbose):
            self.run_models_()
        print("All models done in {:0.3f}s".format(time()-t0))

    def run_models_(self):
        for run in range(self.repeats):
            t1 = time()
            seed = int(t1*100)-158000000000
            print("Initiating run {}/{}".format(run+1, self.repeats))
            parameters = {
                'mallet_path':self.parameters['mallet_path'],
                'workers':self.parameters['workers']
            }
            for t in self.testing_para['topics']:
                for a in self.testing_para['alpha']:
                    for o in self.testing_para['opt_int']:
                        for i in self.testing_para['iterations']:
                            parameters['num_topics'] = t
                            parameters['alpha'] = a
                            parameters['optimize_interval'] = o
                            parameters['iterations'] = i
                            parameters['random_seed'] = seed
                            model_name = 't{}a{}o{}i{}'.format(t, a, o, i)
                            print("Running model {} run {}...".format(model_name, run+1))
                            self.models[model_name].append(self.mallet_model_(parameters))

            print("Run {}/{} done in {:0.3f}s.".format(run+1, self.repeats, time() - t1))


    def mallet_model_(self, parameters):
        tm = time()
        print("Building Mallet model...")     
        model = gensim.models.wrappers.LdaMallet(corpus=self.nlp_data.gensim_lda_input(),
                                                    id2word=self.nlp_data.get_id2word(),
                                                    **parameters)

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
        print("Mallet coherence for {} topics is: {:0.3f}".format(parameters['num_topics'], model_coherence))
        return {"coherence":model_coherence, "time":model_time, "topics":model_topics_list, "model":model, 'seed':parameters['random_seed']}

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

    def graph_results(self, show=True, save=False, path=None):
        # This graphs results as a box and whiskers plot
        graphs = self.output_dataframe(seed=False).to_dict('list')
        labels = list(graphs.keys())
        x_values = []
        for label in labels:
            x_values.append(graphs[label])

        # Show graph
        plt.figure(figsize=(12, 8))    
        plt.boxplot(x_values, labels=labels)

        plt.title("Mallet Model Comparison")
        plt.xlabel("Models: t:topics, a:alpha, o:optimize interval, i:iterations")
        plt.ylabel(self.coherence + " coherence")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        if save:
            plt.savefig(path, bbox_inches='tight')
        if show:
            plt.show()
        plt.close() # Closes and deletes graph to free up memory

    def output_dataframe(self, data_column="coherence", seed=True, save=False, path=None):
        # This outputs a dataframe of the model results
        dataframe_dict = {}
        if seed:
            seed_filled = False
            dataframe_dict['seed'] = []
        else:
            seed_filled = True
        for key in self.models:
            dataframe_dict[key] = []
            for model in self.models[key]:
                dataframe_dict[key].append(model[data_column])
                if not seed_filled:
                    dataframe_dict['seed'].append(model['seed'])
            seed_filled = True
        data = pd.DataFrame.from_dict(dataframe_dict)
        if save:
            data.to_csv(path, index=False)
        return data

    def save(self, file_path_name):
        with open(file_path_name, 'wb') as file:            
            pickle.dump(self, file)

if __name__ == "__main__":
    pass
