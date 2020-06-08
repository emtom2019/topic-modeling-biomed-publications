# Author: Thomas Porturas <thomas.porturas.eras@gmail.com>

# For measuring time
from time import time

# Modules for NLP
import pandas as pd
import spacy
import scispacy # Not sure if import necessary, but you need to install it for the 'en_core_sci_md' library


# Gensim
import gensim
from gensim.corpora.dictionary import Dictionary

# sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# custom stop_words module containing stopwords from the nltk package (not installed)
from .stopword_lists import NLTK_STOPWORDS

"""
The :mod:'data_nl_processing' module contains the NlpForLdaInput class that provides natural language processing for LDA model input
that is compatible with both gensim and scikit-learn.
"""

class NlpForLdaInput:
    """This class is for processing a corpus into a bag of words input that is compatible with Gensim and Scikit-learn 
    Latent Dirichlet Allocation models to keep the input constant when comparing the different implementations of LDA.
    All numbers and symbols will be removed and only words with at least 2 characters will be kept.

    Parameters
    ----------
    data : list of documents in the form of a list of strings

    stop_words : list of stop word strings, default is the standard nltk.corpus + several words that were used in my project

    spacy_lib : name of spacy library to be used as a string, default is 'en_core_sci_lg'. This along with scispacy must be 
        downloaded from https://allenai.github.io/scispacy/ if it is not installed. Other libraries may be used, please 
        see the spaCy documentation at https://spacy.io/ 
    
    max_tok_len : integer value of maximum length of tokens. Default is 30.

    ngram_range : tuple of minimum to maximum number of word grams produced by the CountVectorizer class. Default is (1,1).
        This can produce bigrams and trigrams, and it can be used for both gensim and scikit-learn LDA models. However, the 
        tokenized text will not include these bigrams which is an issue when trying to calculate coherence with the gensim
        gensim.models.CoherenceModel class using 'c_v', and will produce inaccurate results. Instead use the bigrams and trigrams
        parameters.

    bigrams : Boolean, default is True. Bigrams will be generated when true and will be present with method get_token_text().

    trigrams : Boolean, default is True. Trigrams will be generated when true and will be present with method get_token_text().
        Note, bigrams must be set to True to enable Trigrams

    min_df : integer, default is 10. Ignores tokens that appear in less than min_df number of documents when building vocabulary
    
    max_df : float, default is 0.25. Ignores tokens that appear in more than max_df proportion of documents when building vocabulary

    You will need to run the start() method to process the text and fit the model. 
    
    For gensim LDA models and the gensim LDA Mallet wrapper you will need to use the get_id2word() and gensim_lda_input() methods 
    to get the id2word and corpus parameters respectively. For the sklearn LDA model, you will only need the sklearn_lda_input() 
    method for the X parameter when running the fit() method of the LatentDirichletAllocation class. All the models in this 
    package will only ask for the reference to the fitted NlpForLdaInput object.

    """

    def __init__(self, data, stop_words='default', spacy_lib='en_core_sci_lg', max_tok_len=30,
                    ngram_range=(1,1), bigrams=True, trigrams=True, min_df=10, max_df=0.25):
        self.data = data
        if stop_words == 'default':
            self.stopwords = set(NLTK_STOPWORDS)
            self.stopwords.update(['elsevier', 'copyright'])
        else:
            self.stopwords = stop_words
        self.spacy_lib = spacy_lib
        self.ngram_range = ngram_range # this is for the sklearn countvectorizer
        self.bigrams = bigrams
        self.bigram_mod = None
        self.trigrams = trigrams # You must set brigrams to True to use trigrams
        self.trigram_mod = None
        self.min_df = min_df
        self.max_df = max_df
        self.max_tok_len = max_tok_len
        self.vectorizer = None
        self.lda_input_data = None
        self.id2word = None
        self.lemmatized_text = None
        self.token_text = []
        self.gensim_corpus_vect = None
        self.token_pattern = '[a-zA-Z0-9_]{2,}' # This is for the sklearn CountVectorizer. Note: '_' must be present if bigrams=True
    
    def preprocess_(self, texts):
        for doc in texts:
            yield(gensim.utils.simple_preprocess(str(doc),deacc=True, max_len=self.max_tok_len)) #deacc=True removes punctuations

    def remove_stopwords_(self, text):
        processed_text = [[word for word in doc if word not in self.stopwords] for doc in text]
        return processed_text
    
    def lemmatization_(self, texts, allowed_postags=['NOUN', 'ADJ', 'VERB','ADV']):
    #Lemmatization using spaCy package for lemmatization (simpler than NLTK)
    #https://spacy.io/api/annotation
        output_text = []
        nlp = spacy.load(self.spacy_lib, disable=['parser','ner'])
        for text in texts:
                doc = nlp(" ".join(text))
                output_text.append(" ".join([
                    token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags
                    ]))
        return output_text

    def make_ngrams_(self, texts):
        bigram = gensim.models.Phrases(texts, min_count=self.min_df, threshold=100) # Higher threshold results in fewer phrases
        self.bigram_mod = gensim.models.phrases.Phraser(bigram)
        if self.trigrams:
            trigram = gensim.models.Phrases(bigram[texts], min_count=self.min_df, threshold=100)
            self.trigram_mod = gensim.models.phrases.Phraser(trigram)
        return self.process_ngrams_(texts)

    def process_ngrams_(self, texts):
        texts = [self.bigram_mod[doc] for doc in texts]
        if self.trigrams:
            texts = [self.trigram_mod[self.bigram_mod[doc]] for doc in texts]
        return texts

    def start(self, verbose=True):
        # This method starts the processing of the corpus
        if verbose: print("Preprocessing dataset...")
        if verbose: print('Creating Tokens For Lemmetization...')
        t0 = time()
        # Tokenize
        data_tokens = self.remove_stopwords_(list(self.preprocess_(self.data[:])))
        if verbose: print("done in %0.3fs." % (time() - t0))
        if verbose: print('Lemmetization in progress using spaCy...')
        t0 = time()
        # Lemmatization function using spaCy                
        self.lemmatized_text = self.lemmatization_(data_tokens)
        token_text = []
        for doc in self.lemmatized_text:
            token_text.append(doc.split())
        if self.bigrams:
            self.token_text = self.make_ngrams_(token_text)
            input_text = []
            for doc in self.token_text:
                input_text.append(" ".join(doc))
        else:
            self.token_text = token_text
            input_text = self.lemmatized_text

        if verbose: print("done in %0.3fs." % (time() - t0))
        # Vectorizing dataset for use with LDA algorithm
        if verbose: print('Vectorizing dataset...')
        t0 = time()
        self.vectorizer = CountVectorizer(analyzer='word',
                                min_df=self.min_df,                          
                                stop_words=self.stopwords,               
                                lowercase=True,  
                                token_pattern=self.token_pattern,
                                ngram_range=self.ngram_range, 
                                #max_features=50000,   
                                max_df=self.max_df            
                                )
        self.lda_input_data = self.vectorizer.fit_transform(input_text)
        self.gensim_corpus_vect = gensim.matutils.Sparse2Corpus(self.lda_input_data, documents_columns=False)
        self.id2word = Dictionary.from_corpus(self.gensim_corpus_vect, 
                id2word=dict((idn, word) for word, idn in self.vectorizer.vocabulary_.items()))
        if verbose: print("done in %0.3fs." % (time() - t0))

    def get_id2word(self):
        # Returns the id2word for use with gensim packages
        return self.id2word

    def get_lem_text(self, corpus=None):
        # Returns the lemmatized text
        # Optional Corpus as a list of strings to be lemmatized
        if corpus is None:
            return self.lemmatized_text
        else:
            return self.lemmatization_(corpus)

    def get_token_text(self, corpus=None):
        # Returns the token text
        if corpus is None:
            return self.token_text
        else:
            token_text = []
            for doc in corpus:
                token_text.append(doc.split())
            if self.bigrams:
                return self.process_ngrams_(token_text)
            else:
                return token_text

    def sklearn_lda_input(self, corpus=None):
        # Returns input for the the scikit-learn LDA model
        if corpus is None:
            return self.lda_input_data
        else:
            return self.vectorizer.transform(corpus)

    def gensim_lda_input(self, corpus=None):
        # Returns input for the the gensim LDA model
        if corpus is None:
            return self.gensim_corpus_vect
        else:
            return gensim.matutils.Sparse2Corpus(self.vectorizer.transform(corpus), documents_columns=False)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names()

    def process_new_corpus(self, corpus, verbose=True): 
        # Run this to process a new corpus with the trained vectorizer model after running the start method
        if verbose: print('Processing new text...')
        t0 = time()
        data_tokens = self.remove_stopwords_(list(self.preprocess_(corpus)))
        lem_text = self.get_lem_text(data_tokens)
        token_text = self.get_token_text(lem_text)
        vectorizer_input = []
        for doc in token_text:
            vectorizer_input.append(" ".join(doc))
        sklearn_lda_input = self.sklearn_lda_input(vectorizer_input)
        gensim_lda_input = self.gensim_lda_input(vectorizer_input)
        if verbose: print("done in %0.3fs." % (time() - t0))
        return {'lem text':lem_text, 'tokens': token_text, 'sklearn':sklearn_lda_input, 'gensim':gensim_lda_input}

if __name__ == "__main__": # Prevents the following code from running when importing module
    pass