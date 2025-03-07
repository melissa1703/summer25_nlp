from transformers import pipeline
from evaluate import load

from word import Word
import filters

modern_bert = pipeline("fill-mask", model = "answerdotai/ModernBERT-large")
bio_bert = pipeline("fill-mask", model = "dmis-lab/biobert-base-cased-v1.2")
bertscore = load("bertscore")

def substitute(sentence : str, old : str, new : str) -> str:
    """
    Given a sentence, finds the old word and replaces it with the new.
    """
    parts = sentence.split(old)

    return new.join(parts)


def get_ids(words : list, to_mask : str) -> list[int]:
    """
    Given a list of words in a sentence, finds the index(es) of the word which
    needs to be masked for BERT.
    """
    ids = []
    i : int = 0

    while (i < len(words)):
        current : str = words[i]
        if (current.lower() == to_mask.lower()):
            ids.append(i)
        
        i += 1
    
    return ids


def mask_particle(sentence : str, word : str) -> list[str]:
    """
    Given a sentence and a word, masks the word preceding it as well.
    """
    variant : list[str] = []
    
    to_check : list[str] = sentence.split()
    idx = to_check.index(word)

    split_pt = to_check[idx - 1]

    masked_1 : str = add_mask(sentence, word) # Mask original word
    masked_2 : str = add_mask(masked_1, split_pt) # Mask the preceding word
    output = modern_bert(masked_2)
    particles = output[0]

    for option in particles:
        particle = option["token_str"][1:]
        variant.append(particle.join(sentence.split(split_pt)))
    
    return variant


def add_mask(sentence : str, to_mask : str) -> str:
    """
    Given a sentence and a word to mask, substitutes word with [MASK] for BERT.
    """
    parts : str = sentence.split(to_mask)
    new_sentence = "[MASK]".join(parts)

    return new_sentence


def add_masks(sentences : list[str], to_mask : str) -> str:
    """
    Given a list of sentences and a word to mask, substitutes first instance of
    the word in each sentence with [MASK]
    """
    masked : list[str] = []

    for sentence in sentences:
        parts : list[str] = sentence.split(" ")
        ids : list[int] = get_ids(parts, to_mask)

        # for i in ids:
        if (to_mask in parts):
            temp = parts
            temp[ids[0]] = "[MASK]"
            masked.append(" ".join(temp))
        
    return masked


def extract_word(sentence : str, substr : str) -> str:
    """
    Given a sentence and substring to find, searches and returns the whole word;
    accounting for cases where BERT gives half a word as a token.
    """
    words : list[str] = sentence.split(" ")

    for word in words:
        if substr.strip().lower() in word.strip().lower():
            return word
    
    # IF SUBSTR CANNOT BE FOUND
    return " "


def get_score(original : str, alt : str) -> float:
    """
    Given two sentences (original and alternative), compares the semantic 
    similarity and returns the score.
    """
    results = bertscore.compute(predictions = [alt], references = [original], 
                                lang = "en")
    
    return round(results["precision"][0], 3)


def get_suggestions(masked_sentence : str, model : str = "MODERN") -> list[tuple]:
    """
    Given a sentence (masked & formatted for BERT), finds alternative sentences
    using the relevant model of BERT.
    """
    suggestions : list[tuple] = []

    if (model.upper().strip() == "BIO"):
        output = bio_bert(masked_sentence)
    else:
        output = modern_bert(masked_sentence)

    for item in output:
        word = item["token_str"]    # -> suggested word
        sequence = item["sequence"] # -> used to score the suggested word
        suggestions.append((word, sequence))

    return suggestions
    

def word_search(ngsl : list[str], sentence : str, original : Word, model : str = "MODERN") -> list[Word]:
    """
    Given the original sentence and word to find alternatives for, uses specified
    model of BERT to find and score alternatives.
    """
    alternatives : list[str] = []
    
    # Format sentence to be read by BERT
    bert_format = filters.bert_format(sentence)
    masked = add_mask(bert_format, original.word)

    # Find possible variations of sentence to increase possibilities
    variations = [masked]
    particle_mask = mask_particle(bert_format, original.word)
    variations = add_masks(particle_mask, original.word)
    # Do a second parse to make sure BERT hasn't affected the original sentence:

    for variant in variations:
        initial_alts = get_suggestions(variant, model)
        # Make sure word hasn't been added already
        current_words = [added.word for added in alternatives] 

        for alt in initial_alts:
            if (alt[0] not in current_words):
                # Extract the word only from sentence
                word : str = extract_word(alt[1], alt[0])

                # DO A SECOND CHECK:
                if (word not in current_words):

                    # FILTER :
                    formatted = filters.valid_format(word)
                    simple = filters.is_simple(original.word, word, ngsl)
                    type_match = filters.same_type(original.type, word)
                    not_same = (word.lower() != original.word.lower())

                    if (formatted and simple and type_match and not_same):
                        new_word = Word(word, original.type) # Create Word obj

                        # Find score based on sequence generated by BERT:
                        b_score : float = get_score(sentence, alt[1])
                        new_word.set_b_score(b_score)
                        alternatives.append(new_word) # Add to alternatives
    
    return alternatives


if __name__ == "__main__":
    ngsl = filters.get_freq().keys()
    sentence : str = "The girl had an abrasion on her knee."
    word : Word = Word("abrasion", "noun")

    results = word_search(ngsl, sentence, word, "BIO")
    for item in results:
        print(item.get_str())
