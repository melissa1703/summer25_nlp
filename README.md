## SUMMER 2025 DENSION RESEARCH PROJECT: Using NLP Methods to Replace Complex Words with Simpler Alternatives in Health Texts
As part of a summer research project, I used existing NLP algorithms / methods
to test methods for replacing complex language with simpler alternatives in health
texts. 

The primary models used were: 
- GloVe [link]
- ModernBERT [link]
- WordNet

# SETUP:
In order to run the program smoothly, please make sure the following modules have been installed:
- cmudict
- numpy
- faiss
- nltk 
- scipy
- syllables
- tokenizers
- torch
- transformers

# RUNNING THE PROGRAM:
The main program has been written in [find_suggestions.py], and by default runs
a one-layered search using GloVe as the search method and ModernBERT for scoring.

To select further options for searches please enter the command in the following format:
python3 find_suggestions.py <search_1> <search_2> <sort_method>

The main methods tested over the summer were:
- Glove + Glove
- WordNet + None
- ModernBert + None
