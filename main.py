# Author: Thomas Porturas <thomas.porturas.eras@gmail.com>
import biomed_modeling as bm
import pandas as pd

"""
This file contains the sample code to run this package and analyze a set of data with LDA topic modeling.
A processed dataset and 2 raw datasets are provided for demonstration
Each step runs a set of functions to:
    1: process raw data from OVID (must be a csv file), 
    2: run different types of models and compare them over a span of topic numbers to find the 
        ideal topic number and model to use.
    3: optimize the model starting parameters
    4. generate your final models
    5. create figures and results from your model
You can run most steps by themselves and/or turn off certain steps by passing runstepX=False to the main() class.
"""

class main:
    def __init__(self, run_step1=True, run_step2=True, run_step3=True, run_step4=True, run_step5=True):
        if run_step1:
            self.step1()
        if run_step2:
            self.step2()
            #self.step2_alt()
        if run_step3:
            self.step3()
        if run_step4:
            self.step4()
        if run_step5:
            self.step5()

    def step1(self): # Process raw csv files from OVID into a single file compatible with this topic modeling project
        paths = [
            'data/raw/rawdata1.csv',
            'data/raw/rawdata2.csv',
            ]
        df = bm.process_files(paths, entry_types='Article')
        df.to_csv('data/processed/processed_data.csv',index=False)

    def step2(self): # Compare different LDA models and different numbers of topics from 5-50 using CompareModels
        df = pd.read_csv('data/processed/data.csv')
        data = df["title_abstract"].tolist()
        with bm.mu.Timing('Processing Data...'):
            nlp_data = bm.NlpForLdaInput(data)
            nlp_data.start()
        with bm.mu.Timing('Building Models...'):
            compare_models = bm.CompareModels(nlp_data=nlp_data, topics=(5,50,5), seed=2020, coherence='c_v')
            compare_models.start()
            compare_models.save('models/compare_models_test')
            compare_models.output_dataframe(save=True, path='reports/compare_models_test_coh.csv')
            compare_models.output_dataframe(save=True, path='reports/compare_models_test_time.csv', data_column="time")
            print(compare_models.output_parameters(save=True, path='reports/compare_models_test_params.txt'))
            compare_models.graph_results(show=False, save=True, path='reports/figures/compare_models_test_fig.png')

    def step2_alt(self): # Alternative step2 where you can use the run_model_comparison function instead
        # This will allow you to easily run multiple repetitions of CompareModels with different seeds
        # It is set to running 2 repetitions but you can change that number to whatever you want
        data_path = 'data/processed/data.csv'
        data_column = 'title_abstract'
        topic_range = (5, 50, 5)
        models = {'gensim_lda':True, 'mallet_lda':True, 'sklearn_lda':True}
        with bm.mu.Timing('Initiating model comparisons...'):
            bm.run_model_comparison(2, data_path, data_column, topic_range, "test", models)

    def step3(self): # Optimize mallet parameters and find median coh of 3 runs
        data_path = 'data/processed/data.csv'
        df = pd.read_csv(data_path)
        data = df['title_abstract'].tolist()
        nlp_data = bm.NlpForLdaInput(data)
        nlp_data.start()
        compare_mallet_models = bm.CompareMalletModels(nlp_data=nlp_data, topics=[40], alpha=[5, 50], 
            opt_int=[0,200], iterations=[1000], repeats=3)
        compare_mallet_models.start()
        compare_mallet_models.save('models/comparemalletmodels_test')
        compare_mallet_models.output_dataframe(save=True, path='reports/comparemalletmodels_test_coh.csv')
        compare_mallet_models.graph_results(show=False, save=True, path='reports/figures/comparemalletmodels_test_fig.png')

    def step4(self): # Run a final set of models with specified parameters
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
        bm.generate_mallet_models(data_path, data_column, model_save_folder, figure_save_folder, topic_num, model_params, 
                            file_name_append=addendum)

    def step5(self): # Run the pipeline to produce final figures and spreadsheets. Step 4 must be run before this step.
        model_path = 'models/mallet_t40a5o200test' # You need to run the sample code in mallet_model first
        data_path = 'data/processed/data.csv'
        data_column = 'title_abstract'
        year_column = 'year'
        journal_column = 'journal'
        main_path = 'reports/main/'
        with bm.mu.Timing('Running pipline...'):
            bm.run_pipeline(model_path, data_path, data_column, year_column, journal_column, 
                main_path=main_path, year_start=1980, year_res=5)

if __name__ == "__main__":
    main(run_step1=True, run_step2=True, run_step3=True, run_step4=True, run_step5=True)