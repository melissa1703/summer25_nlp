import numpy as np
import datetime
import faiss
from scipy import spatial

import filters
from word import Word

def get_depth(filepath : str) -> int:
    """
    Returns the expected depth of vectors from given file based on filename
    """
    items : list = filepath.split(".")
    digit : str = items[-2][:-1]

    return int(digit)

def validate_data(filepath : str = "./data_files/commoncrawl.840B.300d.txt") -> int:
    timestamp = datetime.datetime.now()
    depth : int = get_depth(filepath)
    count = 1

    with open(filepath, "r") as data:
        for line in data:
            values = line.split()
            word = values[0]
            vector = values[1:]
            with open(f"results/words_only-{timestamp}", "a") as f:
                f.write(word)
                f.write("\n")

            if (len(vector) != depth):
                with open(f"results/data_validation-{timestamp}", "a") as f:
                    f.write(f"TOTAL LENGTH : {len(values)}\n")
                    f.write(f"VECTOR LENGHT : {len(vector)}\n")
                    f.write(f"LINE : {line}")
                    f.write("\n")
            else : 
                if (count < 100):
                    with open(f"results/data_validation-{timestamp}", "a") as f:
                        f.write(f"TOTAL LENGTH : {len(values)}\n")
                        f.write(f"VECTOR LENGHT : {len(vector)}\n")
                        f.write(f"LINE : {line}")
                        f.write("\n")
                
                count += 1

    print(f"TOTAL WORDS : {count}")

def get_faiss_vectors(filepath : str = "./data_files/commoncrawl.840B.300d.txt") -> tuple[dict]:
    """
    Given filepath of GloVe vector file, parses and stores vectors.

    RETURNS : FAISS index for searching and word embedding & id dictionaries 
    to access the words + vectors.
    """
    print(f"> Creating FAISS index from {filepath}")

    depth : int = get_depth(filepath)   # find expected depth of vectors
    idx : int = 1                        # initialise id count for index

    # process GloVe data
    word_embeddings : dict = {}  # word -> vector for search
    word_id : dict = {}          # id -> word to retrieve correct word

    with open(filepath, "r") as data:
        for line in data:
            values = line.split()
            word = values[0]
            vector = values[1:]

            if (len(vector) == depth):
                try:
                    arr = np.array(values[1:], "float32")
                # IN CASE OF ERROR VALUES -> skip (temp measure)
                except:
                    continue

                # add to dictionary IFF word hasn't been added already
                if (word not in word_embeddings.keys()):
                    word_embeddings[word] = arr
                    word_id[idx] = word
                    idx += 1
    
    # SETUP FOR FAISS INDEX:
    faiss_index = faiss.IndexFlatL2(depth)
    faiss_index = faiss.IndexIDMap(faiss_index)

    embedding_arr = np.array(list(word_embeddings.values()))
    id_arr = np.array(list(word_id.keys()))

    faiss_index.add_with_ids(embedding_arr, id_arr)

    print("> FAISS index created from GloVe data")

    return (faiss_index, word_embeddings, word_id)


def find_k_closest(index, embeddings : dict, ids : dict, word : str, k : int) -> list:
    """
    Given a word and value k, returns the k closest neighbours to the word
    based on GloVe embeddings.
    """
    
    closest : list = []
    
    try:
        # +1 to account for the fact that the word itself will always be closest
        result = index.search(np.atleast_2d(embeddings[word]), k + 1)[1]
        
        for i in result[0]:
            closest.append(ids[i])

    except KeyError:
        return []
    
    return closest
        

def get_score(a, b) -> float:
    """
    Calculates the distance between provided vectors 'a' and 'b' to
    determine how close provided words are - the lower the score, the more
    accurate for GloVe

    RETURNS :  rounded to three d.p for convenience.
    """
    dist = spatial.distance.euclidean(a, b)

    return round(dist, 3)


def word_search(glove_data : tuple, ngsl : list[str], original : Word) -> tuple[list[Word]]:
    """
    [FIRST SEARCH] Using GloVe, do an initial search for alternatives for a
    given word; returns a list of suggested alternatives simpler than the
    original.

    - glove_data : tuple containing FAISS index, word embeddings and IDs
    - ngsl : list of words in NGSL
    - original : tuple containing original word (token[0]) and type (token[1])

    RETURNS : list of simpler alternatives for original.
    """
    index = glove_data[0]
    embeddings = glove_data[1]
    ids = glove_data[2]
    k : int = 50

    neighbours : list[str] = find_k_closest(index, embeddings, ids, original.word, k)

    # Filter the neighbours found:
    filtered : tuple[list[Word]] = filters.sort_suggestions(neighbours, ngsl, original)

    return filtered


def list_search(glove_data : tuple, ngsl : list[str], current : list[Word], 
                original : Word):
    """
    [SECOND SEARCH] Given a list of words from a first search, finds further
    alternatives by branching off from initial suggestions given.

    - glove_data : tuple containing FAISS index, word embeddings and IDs
    - ngsl : list of words in NGSL
    - original : tuple containing original word (token[0]) and type (token[1])
    - current : list of current alternatives suggested

    RETURNS : list of additional simpler alternatives
    """
    index = glove_data[0]
    embeddings : dict = glove_data[1]
    ids : dict = glove_data[2]

    new : list[Word] = []

    for word in current:
        k : int = 25

        neighbours : list[str] = find_k_closest(index, embeddings, ids, word.word, k)
        filtered : list[Word] = filters.sort_suggestions(neighbours, ngsl, original)

        old_suggestions = [word.word for word in current]
        new_suggestions = [word.word for word in new]

        for valid in filtered[0]:
            # Only add to suggestions IFF it
            if (valid.word not in old_suggestions and valid.word not in new_suggestions):
                new.append(valid)
    
    return new

def search(glove_data : tuple) :
    index = glove_data[0]
    embeddings = glove_data[1]
    ids = glove_data[2]

    freq : dict = filters.get_freq()
    print("TO QUIT ENTER Q")
    while (True):
        to_check : str = input("WORD: ").strip().lower()
        if (to_check == "q"):
            break
        else:
            to_get : int = int(input("NO. OF TERMS TO GET: ").strip().lower())

            if (to_check in freq.keys()):
                print("This word is COMMON as per the NGSL.")
            else:
                print("This word is UNCOMMON as per the NGSL")
            
            print(f"Finding {to_get} closest terms to {to_check}...")
            closest = find_k_closest(index, embeddings, ids, to_check, to_get)

            for val in closest:
                dist = get_score(embeddings[to_check], embeddings[val])
                print(f"> {val.upper()} || {dist}")


if __name__ == "__main__":
    # data : str = "./data_files/wikipedia.6B.300d.txt"
    # data : str = "./data_files/twitter.27B.200d.txt"
    # data : str = "./data_files/commoncrawl.42B.300d.txt"
    data : str = "./data_files/commoncrawl.840B.300d.txt"

    glove_data : tuple = get_faiss_vectors(data)
    search(glove_data)
