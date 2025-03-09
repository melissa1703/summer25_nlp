from word import Word
import glove
import bert
import filters
import wordnet

import csv
import sys
import datetime

# FILEPATH VARIABLES FOR EASE:
GLOVE_VECTORS : str = "./data_files/commoncrawl.840B.300d.txt"
SAMPLES : str = "./datafiles/all_samples.csv"

def wordnet_only(ngsl : list[str], sentences : list[str]):
    print(f"> Suggestions will be found using WordNet only")

def find_suggestions(glove_data : tuple, ngsl : list[str], sentences : list[str],
                     search_1 : str, search_2 : str):
    
    print(f"> Suggestions will be found using {search_1}-{search_2}")

    # SETUP TO STORE ALTERNATIVE WORDS:
    # sentence : tuple(valid_alts[Word : list[Word]], invalid_alts[...])
    suggestions : dict[str : tuple[dict]] = {}
    count = 1

    for sentence in sentences:
        # SANITY CHECK:
        print(f"({count}) FINDING SUGGESTIONS FOR: '{sentence}")
        count += 1

        valid_alts : dict[Word : list[Word]] = {}
        invalid_alts : dict[Word : list[Word]] = {} # List of invalid alternatives stored for reference (currently not used)

        words = filters.get_words(sentence)

        for original in words:
            if (filters.skip(original, ngsl)):
                valid_alts[original] = ["Word skipped ; considered common."]
                invalid_alts[original] = ["Word skipped ; considered common."]
                continue

            # ELSE; look for suggestions and store as required
            # Initial search for alternative words:

            if (search_1 == "GLOVE"):
                first = glove.word_search(glove_data, ngsl, original)
                valid_alts[original] = first[0]
                invalid_alts[original] = first[1]
            elif (search_1 == "WORDNET"):
                valid_alts[original] = wordnet.word_search(ngsl, original)
                invalid_alts[original] = []
            elif (search_1 == "MODERNBERT"):
                valid_alts[original] = bert.word_search(ngsl, sentence, original, "MODERN")
                invalid_alts[original] = []
            elif (search_1 == "BIOBERT"):
                valid_alts[original] = bert.word_search(ngsl, sentence, original, "BIO")
                invalid_alts[original] = []

            # Conduct second search for more alternatives:
            if (search_2 == "NONE"):
                pass

            elif (search_2 == "GLOVE" or search_2 == "MODERNBERT" or search_2 == "BIOBERT"):
                second = glove.list_search(glove_data, ngsl, valid_alts[original], original)
                valid_alts[original] += second

                if (search_2 == "MODERNBERT" or search_2 == "BIOBERT"):
                    print("Second search cannot be conducted using BERT; defaulting to GloVe.")
            
            elif (search_2 == "WORDNET"):
                second = wordnet.list_search(ngsl, valid_alts[original], original)
                valid_alts[original] += second

        # Once search is complete for each word of a sentence, store both valid 
        # & invalid alternatives found
        suggestions[sentence] = (valid_alts, invalid_alts)

    return suggestions


def add_scores(suggestions : dict[str : dict], embeddings : dict, sort_by : str):
    print(f"> Adding scores to suggested alternatives and sorting by {sort_by}")
    count : int = 1

    for sentence in suggestions.keys():
        # [NOTE] Sanity check to ensure algorithm is running for each sentence...
        print(f"({count}) SCORING : '{sentence}'")
        count += 1 # increment
        
        # Get valid alts found
        valid_alts : dict= suggestions[sentence][0]

        for original in valid_alts.keys():
            alts : list[Word] = valid_alts[original]

            # SAFETY MEASURE:
            if (len(alts) == 0):
                continue

            # Words w/o alternatives have a str indicating why - skip
            elif (type(alts[0]) != Word):
                continue
            
            # ELSE:
            for alt in alts:
                # (1) Add GloVe score iff not alr scored:
                if (alt.get_g_score() == 99):
                    try:
                        g_score : float = glove.get_score(embeddings[original.word], embeddings[alt.word])
                    except KeyError:
                        # In the case a suggested word isn't in GloVe 
                        g_score : float = -1
                    alt.set_g_score(g_score)

                # (2) Add BERT score iff not alr scored:
                if (alt.get_b_score() == -1):
                    new_sentence : str = bert.substitute(sentence, original.word, alt.word)
                    b_score : float = bert.get_score(sentence, new_sentence)

                    alt.set_b_score(b_score)
                
            # SORT ONCE ALL SCORES HAVE BEEN SET
            # for GloVe, the smaller the distance, the more accurate
            if (sort_by.upper() == "GLOVE"):
                valid_alts[original].sort(key = lambda x : x.glove_score)
            # for BERT, the higher the % the more accurate
            if (sort_by.upper() == "BERT"):
                valid_alts[original].sort(key = lambda x : x.bert_score, reverse = True)
    
    print(f"> All alternatives scored and sorted by {sort_by}")


def record_results(suggestions : dict[str : dict], tumestamp, k : int,
                     search_1 : str, search_2 : str, sort_by : str):
    
    root : str = f"./output/{search_1.upper()}-{search_2.upper()}-{timestamp}"
    txtfile = open(f"{root}.txt", "a")
    
    txtfile.write(f"TIMESTAMP : {timestamp}\n")
    txtfile.write(f"SEARCH METHODS : {search_1.upper()}-{search_2.upper()}\n")
    txtfile.write(f"SORTED BY : {sort_by}\n")
    txtfile.write("--------------------------------------------------\n")

    csvfile = open(f"{root}.csv", "a")
    fields = ["WORD", "SUGGESTIONS"]

    identified : int = 0
    suggestions_made : int = 0

    csv_data : list[dict] = []
    alr_logged : list[str] = []

    for sentence in suggestions.keys():
        txtfile.write(sentence)
        txtfile.write("\n")

        valid_alts : dict[str : list[Word]] = suggestions[sentence][0]
        # SHORTLIST : only record the first k words
        for word in valid_alts.keys():
            shortlisted : list[str] = []
            counted : bool = False 

            txtfile.write(f"[{word.word} | {word.type}]\n")

            # (1) Check if suggestion / comment has been made ; if len == 0,
            #     assume that word was flagged as complex but no simpler alts
            #     were found
            if (len(valid_alts[word]) == 0):
                identified += 1 # increment count
                counted = True # safety measure
                txtfile.write("No simpler alternatives were found for this word\n")
                continue

            for val in valid_alts[word][:k]:
                # (2) Doesn't contain Words -> write relevant tag
                if (type(val) != Word):
                    txtfile.write(val)
                else:
                    txtfile.write(val.get_str())
                    shortlisted.append(val.get_word())

                    # If there is at least one suggestion, +1 count to words
                    if (not counted):
                        identified += 1
                        suggestions_made += 1

                        counted = True

                txtfile.write("\n")
            
            if (len(shortlisted) > 0):
                # THIS LINE SKIPS DUPLICATE WORDS
                # if (word.word not in alr_logged):
                entry : dict = {"WORD" : word.word.lower(), "SUGGESTIONS" : (",").join(shortlisted)}
                csv_data.append(entry)
                alr_logged.append(word.word.lower())

        txtfile.write("--------------------------------------------------\n")

    # note how many words were identified as complex + had alternatives found
    txtfile.write(f"{identified} words were identified as complex.\n"
                  f"Alternatives were found for {suggestions_made} complex words.\n")
        
    writer = csv.DictWriter(csvfile, fieldnames = fields)
    writer.writeheader()
    writer.writerows(csv_data)
    
    print(f"> Results have been saved to {root}")


def arg_parse(arg : str, arg_idx : int) -> str:
    if (arg.upper() == "GLOVE"):
        return "GLOVE"
    if (arg.upper() == "WORDNET"):
        return "WORDNET"
    
    if (arg.upper() == "BERT"):
        return "MODERNBERT"
    if (arg.upper() == "MODERNBERT"):
        return "MODERNBERT"
    if (arg.upper() == "BIOBERT"):
        return "BIOBERT"
    if (arg.upper() == "NONE"):
        return "NONE"

    # ELSE:
    if (arg_idx == 1):
        return "GLOVE"
    # ELSE: default to no search
    if (arg_idx == 2):
        return "NONE"
    if (arg_idx == 3):
        return "BERT"
    

def get_samples(sample_sentences : str) -> list :
    sentences : list = []

    with open(sample_sentences, "r") as data:
        sentences = [line.rstrip() for line in data]
    
    return sentences


# python3 find_suggestions.py [glove/wordnet/bert] [glove/wordnet/bert] [glove/bert]
# specify none for search_2 if you want to skip it !!!
if __name__ == "__main__":
    # Load in relevant items:
    glove_data : tuple = glove.get_faiss_vectors(GLOVE_VECTORS)
    ngsl : list[str] = filters.get_freq().keys()
    sentences = get_samples(SAMPLES)
    timestamp = datetime.datetime.now()    # Timestamp for recording results


    # Determine parameters for search (DEFAULT : GloVe 1-layer search)
    search_1 : str = "GLOVE"
    search_2 : str = "GLOVE"
    sort_by : str = "BERT"

    # At least 2 args;
    if (len(sys.argv) > 1):
        search_1 : str = arg_parse(sys.argv[1], 1)
    # At least 3 args;
    if (len(sys.argv) > 2):
        search_2 : str = arg_parse(sys.argv[2], 2)
    if (len(sys.argv) > 3):
        sort_by : str = arg_parse(sys.argv[3], 3)

    suggestions = find_suggestions(glove_data, ngsl, sentences, 
                                   search_1, search_2)

    add_scores(suggestions, glove_data[1], sort_by)
    
    # RECORD THE TOP 15 WORDS
    record_results(suggestions, timestamp, 15, search_1, search_2, sort_by)
