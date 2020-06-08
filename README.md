# topic-modeling-biomed-publications
 This is a set of tools to analyze biomedical publications using Latent Dirichlet Allocation topic modeling.
    Version 1.1.0

Installation instructions:
    -This project was built on a windows machine and the instructions may not be correct for mac or linux

1. Download and install a 64-bit anaconda distribution from https://www.anaconda.com/products/individual
    - This project was done using the windows 64-bit python 3 distribution

2. Open the anaconda prompt and change directory to the project folder (or wherever lda_env.yml is)

3. Type: "conda env create -f lda_env.yml" and hit enter
    - This will create the lda_env environment that installs all required packages
    - On my windows PC, pip fails to install certain packages. Type in "conda activate lda_env" and hit enter.
    - Then type "conda list" and hit enter. This will print out all installed packages. 
    - Make sure scispacy and en-core-sci-lg are installed.
    - If not type and hit enter: 
        - "pip install scispacy==0.2.4"
        - "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz"

4. Download mallet from http://mallet.cs.umass.edu/download.php and follow the installation instructions
    - This project assumes that mallet is in C:\mallet and that the mallet path is C:\mallet\bin\mallet.
    - You will have to update the code if this is not the case.
    - Make sure you update the registry files because Mallet will not run properly otherwise
    - Note: I was not able to get mallet to work with interactive python/jupyter notbooks on my windows machine

5. Clone this repository from github

Usage instructions:
- This project was used to analyze data from the OVID database and tested using results from an embase search 
- A processed dataset and 2 small raw datasets are provided for demonstration purposes. 
- Please see main.py for the demonstration code. If everything is installed correctly, it should run and 
    produce MALLET lda models and figures of the analysis results
- To use your own dataset, search the OVID database for your criteria. Export the results as an excel file and either select complete reference or custom. If custom make sure you select 'PT' (publication type), 'AB' (abstract), 'SO' (source), 'TI' (title),'YR' (year of publication), and 'DC' (date created). If there is a limit to the number of entries you can download at once, you can group articles by year and download them seperately. Open each file and save as a .csv file. The process_files() function will combine them into a single file. Change the code as you see fit to use your dataset instead of the example datasets.

The code was written with the aid of the documentation for python and the individual packages, google, stackoverflow, and https://www.machinelearningplus.com/