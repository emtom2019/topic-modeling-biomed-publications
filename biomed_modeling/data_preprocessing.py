# Author: Thomas Porturas <thomas.porturas.eras@gmail.com>

from time import time

import pandas as pd
import numpy as np
import re

"""
The :mod:'data_preprocessing' module contains the process_files() function and CleanData class.
The process_files() function can combine raw CSV file output from the ovid database and process them into 
files usable by the topic models in this project.
The CleanData class can be used to edit the processed data and filter out entries based on text content 
and cut and paste text segments.
"""

class CleanData:
    """
    This class is for editing the processed data allowing you to filter out entries based on input strings
    and copying, cutting and pasting text surrounding keywords.

    Parameters
    ----------
    data : Path to the processed data as a csv file.


    Note: This class was meant to help split the abstacts to perform topic modeling on just
    the methods or introduction and conclusion segments of the abstracts. It introduced 
    an issue for topic modeling since we lost a significant portion of the corpus and the 
    resulting topics appeared to be of lower quality than the analysis on the
    complete abstract.

    """
    def __init__(self, data):
        self.df_data = pd.read_csv(data)
        self.processed_data = None

    def extract_column(self, col): # Returns desired column as a list
        return self.df_data[col].tolist()

    def todict(self): # Returns a dict object of the dateframe
        return self.df_data.to_dict()

    def search_(self, col, keywords): # Returns boolean dataframe of rows with keywords
        search_pattern = '|'.join(map(re.escape, keywords))
        return self.df_data[col].str.contains(search_pattern, na=False, case=False)

    def remove_rows_with(self, col, keywords=['case report']): 
        # Removes all rows where keywords are found
        search_results = self.search_(col, keywords)
        removed_rows = self.df_data.drop(self.df_data[~search_results].index)
        self.df_data.drop(self.df_data[search_results].index, inplace=True)
        return removed_rows

    def remove_rows_without(self, col, keywords=['methods:']): 
        # Removes all rows where keywords are not found
        search_results = self.search_(col, keywords)
        removed_rows = self.df_data.drop(self.df_data[search_results].index)
        self.df_data.drop(self.df_data[~search_results].index, inplace=True)
        return removed_rows

    def remove_columns(self,col): # Removes specified column
        self.df_data.drop(labels=col, axis=1, inplace=True)

    def copy_text(self, search_col, start_keywords=["methods:"], end_keywords=["results:"], dest_col='methods'):
        # Copies text between the first index postion of the first 'start_keywords' found 
        # and the first index position of the first 'end_keywords' found
        # If no end_end keyword found, then text copied until end of string
        new_col = []
        start_search_pattern = '|'.join(map(re.escape, start_keywords))
        end_search_pattern = '|'.join(map(re.escape, end_keywords))
        for row in self.df_data[search_col]:
            start = re.search(start_search_pattern, row, re.IGNORECASE)
            end = re.search(end_search_pattern, row, re.IGNORECASE)
            if start is None:
                start_index = start
            else:
                start_index = start.start()
            if end is None:
                end_index = end
            else:
                end_index = end.start()
            new_col.append(row[start_index:end_index])
        self.df_data[dest_col] = new_col

    def delete_text(self, search_col, dest_col, start_keywords=["methods:"], end_keywords=["conclusion:"]):
        # Deletes text between the first index postion of the first 'start_keywords' found 
        # and the first index position of the first 'end_keywords' found
        # If no end_end keyword found, then all text after 'start_keywords' is deleted
        new_col = []
        start_search_pattern = '|'.join(map(re.escape, start_keywords))
        end_search_pattern = '|'.join(map(re.escape, end_keywords))
        for row in self.df_data[search_col]:
            start = re.search(start_search_pattern, row, re.IGNORECASE)
            end = re.search(end_search_pattern, row, re.IGNORECASE)
            if start is None:
                start_index = start
            else:
                start_index = start.start()
            if end is None:
                end_index = end
            else:
                end_index = end.start()
            new_col.append(row[:start_index]+row[end_index:])
        self.df_data[dest_col] = new_col

    def save_data(self, file_path_name): # Saves data as CSV
        self.df_data.to_csv(file_path_name, index=False)
        print("Data saved to: " + file_path_name)

def process_files(paths, header_row=1, entry_types='Article|Conference Abstract|Conference Paper', 
                    fill_unknown_years=False, remove_unknown_years=True):
    """
    This function returns a processed dataframe that is compatible with this topic modeling project.
    It processes a single or multiple raw csv files to produce a single dataframe that contains all of
    the data required for topic modeling in this project.

    Note: The raw files are assumed to be complete citation entries downloaded from the OVID database and at the 
    minimum the following columns must be present: 'PT', 'AB', 'SO', 'TI','YR'. When downloading, the file will be
    an excel file, use save as to convert to a csv file.

    Parameters
    ----------
    paths : List of paths to the raw csv files

    header_row : the integer row number containing the column titles. 0 is the first row. In the default OVID files 
        the 2nd row or row '1' contains the header. Default 1

    entry_types : string as a regular expression of the keywords of types of articles to include seperated by '|'. 
        Default 'Article|Conference Abstract|Conference Paper'

    fill_unknown_years : Boolean. Requires column 'DC' to be present in the raw file. Attempts to replace missing 'YR' entries
        with data from the 'DC' column (date created). Otherwise missing years are replaced with 0. Default False

    remove_unknown_years : Boolean. Removes all entries with unknown years of publications or 'YR' = NAN. Default True

    This function will returns a pandas dataframe

    """
    df_main = pd.DataFrame(columns=["abstract", "journal", "title", "year"])
    entry_set = []
    for i, path in enumerate(paths):
        df_raw = pd.read_csv(path, header=header_row, index_col=0)
        print('File {} raw file header:'.format(i+1))
        print(df_raw.head(n=2))
        # Fill in NA values, Empty years will be replaced with a year from another column (DC)
        df_raw['PT'].fillna(u'NA', inplace=True)
        df_raw['YR'].fillna(int(0), inplace=True)
        entry_set.extend(df_raw['PT'].tolist())
        if fill_unknown_years: # Requires column 'DC' to be present in raw file
            year_replace = [int(x[0:4]) for x in df_raw['DC'].astype(str).tolist()]
            df_raw['YR'] = np.where(df_raw['YR'] == 0, year_replace, df_raw['YR'])
        if remove_unknown_years:
            df_raw = df_raw[df_raw['YR']!=0]
        # Filter in entries by entry type
        df = df_raw[df_raw['PT'].str.contains(entry_types)]
        df = df.filter(items = ['AB', 'SO', 'TI','YR'])
        df['SO'] = df['SO'].str.split(r'\.|=').str[0]
        df = df.rename(index=str, columns={"AB": "abstract", "SO": "journal", "TI": "title", "YR":"year"})
        
        
        print('File {} processed file Header:'.format(i+1))
        print(df.head(n=2))
        df_main = df_main.append(df, ignore_index=True)

    df_main = df_main.reset_index()
    print("All entry types present in raw files:")
    print(set(entry_set)) # This prints all variations of entry types present in column PT

    df_main['title_abstract'] = df_main[['title', 'abstract']].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    df_main = df_main.filter(items = ['title', 'abstract', 'title_abstract', 'journal', 'year'])
    df_main = df_main.reset_index(drop=True)
    
    print('')
    print('Number of entries: {}'.format(len(df_main.index)))
    print('Final file Header:')
    print(df_main.head(n=2))
    return df_main

if __name__ == "__main__": # Prevents the following code from running when importing module
    pass

    # Example usage of process_files() function
    """
    paths = [
        'data/raw/rawdata1.csv',
        'data/raw/rawdata2.csv',
        ]
    df = process_files(paths, entry_types='Article')
    df.to_csv('data/processed/processed_data.csv',index=False)
    """

    # Example usage of CleanData class
    """
    print("Loading dataset...")
    t0 = time()
    data = CleanData(data='data/processed/data.csv')
    print("done in %0.3fs." % (time() - t0))

    num_rows = len(data.df_data.index)
    print("Filtering %d rows..." % num_rows)
    t0 = time()
    df_filtered_out = data.remove_rows_with(col="title_abstract", keywords=["case report"])
    print("done in %0.3fs." % (time() - t0))
    new_num_rows = len(data.df_data.index)
    print("%d rows removed \n %d rows remain" % (num_rows - new_num_rows, new_num_rows ))
    data.save_data('data/external/data_filtered.csv')
    
    filtered_out_path = 'data/processed/data_casereport.csv'
    df_filtered_out.to_csv(filtered_out_path, index=False)
    print("Deleted data saved to: " + filtered_out_path)

    print("Filtering %d rows..." % len(data.df_data.index))
    t0 = time()
    df_filtered_out = data.remove_rows_without(col="title_abstract", keywords=["methods:","procedures:"])
    print("done in %0.3fs." % (time() - t0))
    new_num_rows = len(data.df_data.index)
    print("%d rows removed \n %d rows remain" % (num_rows - new_num_rows, new_num_rows ))

    filtered_out_path = 'data/processed/removed_data_no_methods.csv'
    df_filtered_out.to_csv(filtered_out_path, index=False)
    print("Deleted data saved to: " + filtered_out_path)

    print("Copying methods to new column...")
    t0 = time()
    data.copy_text(search_col = "title_abstract", 
                    dest_col='methods',
                    start_keywords=["methods:", "procedures:"], 
                    end_keywords=["results:", "findings:", "conclusion:", "conclusions:"], 
                    )
    print("done in %0.3fs." % (time() - t0))

    print("Deleting methods and results from origin column...")
    t0 = time()
    data.delete_text(search_col = "title_abstract", 
                    dest_col='title_abstract',
                    start_keywords=["methods:", "procedures:"], 
                    end_keywords=["conclusion:", "conclusions:"], 
                    )
    print("done in %0.3fs." % (time() - t0))
    data.save_data('data/external/data_methods_split.csv')
    """