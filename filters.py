import cmudict
import spacy
import csv
from syllables import estimate

from wordnet import get_word_tags
from word import Word

# SET FILEPATH FOR FREQUENCY PATH HERE:
FREQ_FILE : str = "./datafiles/NGSL_1.2_stats.csv"

# Load in Spacy NLP For global access:
nlp = spacy.load("en_core_web_trf")

# https://stackoverflow.com/questions/49581705/using-cmudict-to-count-syllables
def count_syllables(word : str) -> int:
    count = 0
    phonetic = cmudict.dict().get(word)
    if (phonetic):
        # arbitrarily select the first one to check
        to_check = phonetic[0]
        for sound in to_check:
            if (sound[-1:].isdigit()):
                count += 1

    return count

def bert_format(sentence : str) -> str:
    """
    Formats a given sentence to be parsed by BERT.
    """
    parts : list[str] = []
    doc = nlp(sentence)
    for token in doc:
        parts.append(token.text.lower())
    
    formatted = " ".join(parts)
    return formatted

def get_freq(ngsl : str = FREQ_FILE) -> dict:
    print("> Processing NGSL")
    frequency : dict = {}

    with open(ngsl, "r") as data:
        reader = csv.DictReader(data)

        for row in reader:
            key : str = row["Lemma"]
            freq : int = int(row["SFI Rank"])
            frequency[key] = freq
        
    print("> Finished processing NGSL")
    
    return frequency


def is_simple(original : str, alt : str, ngsl : list[str]) -> bool:
    """
    Determines whether a given alternative word is 'simpler' than the original
    word based on frequency, length and syllable count.

    Returns True if alternative is more simple than original, False otherwise.
    """
    common : bool = alt.lower() in ngsl
    shorter : bool = (len(alt) <= len(original))
    # Set to 2 as per Julie's suggestion, but alter as required:
    few_syllables : bool = (estimate(alt) <= 2)

    if (shorter or common or few_syllables):
        return True
    
    return False

def skip(original : Word, ngsl : list[str]) -> bool:
    word : str = original.word
    type : str = original.type

    # (1) specified word types don't need to be changed
    valid_type : bool = (type == "PRON" or type == "AUX" or type == "PART"
                         or type == "ADP" or type == "PUNCT" or type == "DET"
                         or type == "NUM")
    
    # (2) if in NGSL, no need to simplify
    common : bool = word.lower() in ngsl

    # (3) if a word is one syllable
    one_syllable : bool = (estimate(word) == 1)

    return (valid_type or common or one_syllable)


def same_pos(pos : str, alt : str) -> bool:
    """
    Given a words POS, checks whether provided alternate word matches using
    Spacy - backup measure in the case a word cannot be found using WordNet.

    RETURNS : True if POS matches, False otherwise.
    """
    doc = nlp(alt)
    alt_pos : str = doc[0].pos_

    if (alt_pos.lower() == pos.lower()):
        return True
    
    return False


def same_type(type : str, alt : str) -> bool:
    """
    Given a word type, checks whether provided alternate word matches.

    RETURNS: True if alternate word is of the correct type, False otherwise.
    """
    wordtypes = get_word_tags(alt)
    
    # If alt word can't be found with WordNet, default to using Spacy
    if ((len(wordtypes) == 0)):
        return same_pos(type, alt)
    
    if (type.upper() in wordtypes):
        return True
    
    return False


def valid_format(string : str) -> bool:
    """
    Determines if suggested string is valid (alphabetical, hyphenated words 
    permitted).

    Strings such as urls / links, random alphanumerical sequences, etc. are
    considered invalid as per above.
    """
    for char in string:
        if (not char.isalpha() and char != "-"):
            return False
        
    return True


def sort_suggestions(suggested : list[str], ngsl : list[str], original : Word) -> list[Word]:
    valid : list[Word] = []
    invalid : list[Word] = []

    for word in suggested:
        formatted = valid_format(word)
        simple = is_simple(original.word, word, ngsl)
        type_match = same_type(original.type, word)
        # MAKE SURE ORIGINAL WORD DOESN'T GET ADDED AGAIN...
        not_same = (word.lower() != original.word.lower())

        if (formatted and simple and type_match and not_same):
            new_word = Word(word, original.type)
            valid.append(new_word)
        else:
            # [TEMP] mark type as - to indicate diff type for now...
            new_word = Word(word, "-")
            invalid.append(new_word)
    
    return (valid, invalid)

def rm_punctuation(word : str) -> str:
    if (not word[-1].isalpha()):
        return word[:-1]
    
    return word

def get_tokens(sentence : str) -> list[tuple]:
    tokens : list[tuple] = []
    doc = nlp(sentence)

    for token in doc:
        type : str = token.pos_ # get word POS
        word : str = token.text # get the word itself

        tokens.append((word, type))
    
    return tokens

def get_words(sentence : str) -> list[Word]:
    words : list[Word] = []
    doc = nlp(sentence)
    
    for token in doc:
        word : str = token.text # get the word itself
        type : str = token.pos_ # get POS as type in context of sentence

        # Create + append new Word obj
        words.append(Word(word, type))

    return words
