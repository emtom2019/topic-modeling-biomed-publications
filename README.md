# topic-modeling-biomed-publications
 This is a set of tools to analyze biomedical publications using Latent Dirichlet Allocation topic modeling.


Installation instructions:
    -This project was built on a windows machine and the instructions may not be correct for mac or linux

1. Download and install a 64-bit anaconda distribution from https://www.anaconda.com/products/individual
    - This project was done using the windows 64-bit python 3 distribution

2. Open the anaconda prompt and change directory to the project folder (or wherever lda_env.yml is)

3. Type: "conda env create -f lda_env.yml" and hit enter
    - This will create the lda_env environment that installs all required packages
    - On my windows PC, pip fails to install certain packages. Type in "conda activate lda_env" and hit enter.
    - Then type "conda list" and hit enter. This will print out all installed packages. 
    - Make sure scispacy, en-core-sci-sm, en-core-sci-md, and en-core-sci-lg are installed.
    - If not type and hit enter: 
        - "pip install scispacy==0.2.4"
        - "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz"
        - "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_md-0.2.4.tar.gz"
        - "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz"

4. Download mallet from http://mallet.cs.umass.edu/download.php and follow the installation instructions
    - This project assumes that mallet is in C:\mallet and that the mallet path is C:\mallet\bin\mallet.
    - You will have to update the code if this is not the case.
    - Make sure you update the registry files because Mallet will not run properly otherwise
    - Note: I was not able to get mallet to work with interactive python/jupyter notbooks on my windows machine

5. Clone this repository from github

6. 