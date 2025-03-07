import csv
import sys

def count_results(suggestions : list[str]) -> tuple:
    idx : int = 0

    top_5 : list[str] = []
    i : int = 0
    v : int = 0
    g : int = 0

    while idx < len(suggestions):
        current = suggestions[idx][-1]

        # (1) log the top five tags:
        if (idx < 5):
            top_5.append(current.upper())
        
        if (current.upper() == "I"):
            i += 1
        elif (current.upper() == "V"):
            v += 1
        elif (current.upper() == "G"):
            g += 1
        

        idx += 1
    
    return (top_5, i, v, g)


def parse_stats(file : str) -> dict:
    """
    Given the filepath to a .csv file, parses through and counts / stores the tags
    for each word.
    """
    print(f"> Parsing statistics from {file}")
    results : dict = {}

    with open(file, "r") as data:
        reader = csv.DictReader(data)

        for row in reader:
            word : str = row["WORD"]
            suggestions : list[str] = row["SUGGESTIONS"].split(",")
            initial_stats : tuple = count_results(suggestions)

            to_log : dict = {}
            to_log["top5"] = initial_stats[0]
            to_log["invalid_count"] = initial_stats[1]
            to_log["valid_count"] = initial_stats[2]
            to_log["good_count"] = initial_stats[3]

            if (word[-1] == "M"):
                to_log["tag"] = "M"
            elif (word[-1] == "N"):
                to_log["tag"] = "N"
            else:
                to_log["tag"] = "-"

            results[word] = to_log
        
    print("> Statistics parsed")
    
    return results


def overall_stats(results : dict, method : str):
    """
    Records data in results dictionary to a stats.csv file; all results
    are appeneded to the same file for convenience.
    """
    total = len(results)

    no_good_suggestions : int = 0
    top5_has_good : int = 0
    top5_has_valid : int = 0
    top1_is_good : int = 0
    top1_is_valid : int = 0

    multi_phrase : int = 0
    multi_phrase_words: list[str] = []
    no_replacement : int = 0
    no_replacement_words : list[str] = []

    for word in results.keys():
        stats = results[word]
        top5 = stats["top5"]
        good = stats["good_count"]
        tag = stats["tag"]

        # (1) Contains no good suggestions
        if (good == 0):
            no_good_suggestions += 1
        if (top5[0] == "G"):
            top1_is_good += 1
        if (top5[0] == "V"):
            top1_is_valid += 1
        if ("G" in top5):
            top5_has_good += 1
        if ("V" in top5):
            top5_has_valid += 1
        
        if (tag == "M"):
            multi_phrase += 1
            multi_phrase_words.append(word[:-2])
        if (tag == "N"):
            no_replacement += 1
            no_replacement_words.append(word[:-2])

    
    # FORMAT RESULTS:
    with open("./output/stats.txt", "a") as log:
        log.write(f"[RECORDING RESULTS FOR {method} SEARCH]\n")
        log.write(f"Overall, {total} words were identified as complex and had alternative words suggested.\n")
        log.write(f"{no_good_suggestions} / {total} words had no good suggestions at all.\n")
        log.write(f"{top5_has_good} / {total} words had a good suggestion in the Top 5.\n")
        log.write(f"{top1_is_good} / {total} words had a good suggestion as the highest suggestion.\n")
        log.write(f"{top5_has_valid} / {total} words had a valid suggestion in the Top 5.\n")
        log.write(f"{top1_is_valid} / {total} words had a valid suggestion as the highest suggestion.\n")
        log.write(f"\n")
        
        log.write(f"{multi_phrase} words are better considered as multi-word phrases:\n")
        for word in multi_phrase_words:
            log.write(f"- {word}\n")
        log.write(f"\n")
        
        log.write(f"{no_replacement} words don't have an alternative:\n")
        for word in no_replacement_words:
            log.write(f"- {word}\n")

        log.write(f"----------------------------------------------------------------------------------------------------\n")

def get_search(filename : str) -> str:
    parts_1 = filename.split("/")
    parts_2 = parts_1[2].split("-")

    return "-".join([parts_2[0], parts_2[1]])


if __name__ == "__main__":
    # Update file name manually
    filename : str = "./output/WORDNET-NONE-2025-02-27 20:33:16.088429.csv"
    method : str = get_search(filename)
    results = parse_stats(filename)
    overall_stats(results, method)
