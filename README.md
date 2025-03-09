# SUMMER 2025 DENSION RESEARCH PROJECT: Using NLP Methods to Replace Complex Words with Simpler Alternatives in Health Texts
Existing NLP algorithms / methods were used to implement and test methods for
replacing complex language with simpler alternatives in health texts.

The primary models used were: 
- GloVe
- ModernBERT
- WordNet

## SETUP:
Due to the sheer size, the data used to incorporate the NLP algorithms / models
hasn't been included but can be accessed at the following links:
- GloVe preprocessed vectors : https://github.com/stanfordnlp/GloVe
    - The largest set of pre-processed word embeddings was utilised (CommonCrawl, 840B tokens) for this project

Some other useful resources:
- ModernBERT : https://huggingface.co/docs/transformers/main/quicktour 
- Spacy : https://spacy.io/models

Filepaths for all data has been assigned to global variables at the top of relevant
files (glove.py, filter.py, find_suggestions.py) for ease of updating.

Additionally, please make sure all the relevant Python packages have been installed.
From memory, the most important are:
- faiss [link : https://github.com/facebookresearch/faiss]
- nltk (for WordNet corpus)
- transformers
- numpy
- scipy
- (I will add more if I remember the specific modules I installed...)


## RUNNING THE PROGRAM:
The main program has been written in [find_suggestions.py], and by default runs
a one-layered search using GloVe as the search method and ModernBERT for scoring.

To select further options for searches please enter the command in the following format: <br>
python3 find_suggestions.py <search_1> <search_2> <sort_method>

The main methods tested over the summer were:
- Glove + Glove
- WordNet + None
- ModernBert + None

In order to add samples, please run add_samples.py to easily append to the existing collection sample sentences.

## OTHER NOTES:
The algorithm (at present) is immensely inefficient, especially with a two-layered GloVe search. Improving this
is a priority for future iterations, but please be aware of the (probably) exponentially long runtime when
running the program. It is highly advised to Not run the program using all the samples (./datafiles/all_samples.csv), but instead a smaller set of texts (./datafiles/extract.csv) to do a test run.
