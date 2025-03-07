from word import Word
import filters
from nltk.corpus import wordnet
from nltk.corpus import verbnet
import nltk

nltk.download('wordnet')

TAGS : dict[str : str] = {
    "n" : "NOUN",
    "v" : "VERB",
    "a" : "ADJ",
    "r" : "ADV",
    "s" : "ADJ",
    "u" : "X"
}

def convert_tags(tags : list[str]) -> list[str]:
    """
    Given a list of POS tags from WordNet, converts them into Spacy POS format 
    for consistency / clarity.
    """
    updated : list[str] = []

    for tag in tags:
        try:
            updated.append(TAGS[tag].upper())
        except KeyError:
            updated.append("X")
    
    return updated

def convert_tag(tag : str) -> str:
    """
    Given a POS tag from WordNet, converts it into Spacy POS format for
    consistency / clarity.
    """
    try:
        return TAGS[tag].upper()
    except KeyError:
        return "X"

def get_word_tags(word : str) -> list[str]:
    synsets = wordnet.synsets(word)
    possible = []

    for set in synsets:
        possible.append(set.pos())

    return convert_tags(possible)


def get_word_types(word : str) -> list[str]:
    """
    Given a word, returns the type of word it is.
    """
    word_types : list[str] = []
    synsets = wordnet.synsets(word)

    for set in synsets:
        type : str = set.lexname().split(".")[0]
        
        if (type not in word_types):
            word_types.append(type.lower())
    
    return word_types

def get_lemmas(word : str, type : str) -> list[str]:
    lemmas : list[str] = []
    synsets = wordnet.synsets(word)

    for set in synsets:
        lemma_type : str = set.lexname().split(".")[0]

        if (lemma_type.lower() == type.lower()):
            to_add = [lemma for lemma in set.lemma_names() if 
                      lemma not in lemmas and lemma.lower() != word.lower()]
            lemmas += to_add
    
    return lemmas

def word_search(ngsl : list[str], original : Word) -> list[Word]:
    """
    [FIRST SEARCH] Given a word, returns a list of synonymous words from WordNet.
    """
    synonyms : list[str] = []
    synsets = wordnet.synsets(original.word)

    for set in synsets:
        pos_tag : str = convert_tag(set.pos())
        if (pos_tag.upper() == original.type.upper()):
            # Only add word if it hasn't already been added
            parsed : list[str] = [lemma for lemma in set.lemma_names() if
                                  lemma.lower() not in synonyms and 
                                  lemma.lower() != original.word.lower()]
            
            synonyms += parsed
    
    filtered : tuple[list[Word]] = filters.sort_suggestions(synonyms, ngsl, original)

    return filtered[0]

def word_search_no_filter(original : Word) -> list[str]:
    synonyms : list[str] = []
    synsets = wordnet.synsets(original.word)

    for set in synsets:
        pos_tag : str = convert_tag(set.pos())
        if (pos_tag.upper() == original.type.upper()):
            synonymous_words = [word.word.lower() for word in synonyms]

            # Only add word if it hasn't already been added
            parsed : list[str] = [Word(lemma, pos_tag) for lemma in set.lemma_names() if
                                  lemma.lower() not in synonymous_words and 
                                  lemma.lower() != original.word.lower()]
            
            synonyms += parsed

    return synonyms


def list_search(ngsl : list[str], current : list[Word], original : Word) -> list[Word]:
    """
    [SECOND SEARCH] Given a list of existing suggestions, finds more synonyms 
    from WordNet
    """
    new : list[str] = []

    for word in current:
        synsets = wordnet.synsets(word.word)
        for set in synsets:
            pos_tag : str = convert_tag(set.pos())

            if (pos_tag.upper() == original.type.upper()):
                old_suggestions : list[str] =  [word.word.lower() for word in current]

                parsed : list[str] = [lemma for lemma in set.lemma_names() if
                                      lemma.lower() not in old_suggestions and 
                                      lemma.lower() not in new]
                
                new += parsed
    
    filtered : tuple[list[Word]] = filters.sort_suggestions(new, ngsl, original)

    return filtered[0]


if __name__ == "__main__" :
    
    while True:
        word : str = input("> ")

        if (word.lower() == "quit"):
            print("> Exiting program.")
            break

        synsets = wordnet.synsets(word)
        for arr in synsets:
            print(f"{arr} || {arr.definition()}")
            print(arr.lemma_names())
            print(arr.lexname())
