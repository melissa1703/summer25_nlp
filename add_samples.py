import csv
import sys

def get_samples(samples : str = "./datafiles/samples.csv") -> list :
    """
    Parses all sample sentences and returns them as a list.
    """
    sentences : list = []

    with open(samples, "r") as data:
        for line in data:
            stripped : str = line.strip()
            if (len(stripped) > 0):
                sentences.append(stripped)

    return sentences

def add_sample_sentences(new : str, to : str = "./datafiles/samples.csv"):
    """
    Given a filename, writes the contents of the files to samples.csv to be used
    with algorithm.
    """
    sample_sentences : list[str] = []

    # PARSE GIVEN PASSAGE TO CONVERT INTO SENTENCES
    with open(new, "r") as data:
        for line in data: 
            # (1) strip of whitespace
            stripped = line.strip()
            
            if (len(stripped) == 0):
                continue    # Skip any blank lines
            
            # (2) iterate through and determine where periods are
            i : int = 0
            start : int = 0
            max_idx : int = len(stripped) - 1
        
            while (i < len(stripped)):
                if (stripped[i] == "."):
                    # Account for the END of file
                    if (i == max_idx):
                        new_str = f"{stripped[start:].strip()}"
                        sample_sentences.append(new_str)

                    # ELSE: make sure this is the END of a sentence:
                    elif (stripped[i + 1].isspace()):
                        # Make sure no brackets ; accounting for (i.e. ...)
                        if (stripped[i - 4] != "("):
                            # Extract this segment of the sentence + format
                            new_str = f"{stripped[start : i].strip()}."
                            sample_sentences.append(new_str) # Add to list

                            # Increment start point for next sentence (skip '.' and space)
                            start = i + 2
    
                i += 1 # MAKE SURE TO INCREMENT...

    # WRITE TO CSV FILE:
    with open(to, "a") as samples:
        for sentence in sample_sentences:
            if (len(sentence) > 0):     # Make sure to skip blank lines
                samples.write(f"{sentence}\n")

if __name__ == "__main__":
    print("> Enter filepath of text to add: ")
    filepath : str = input("> ").strip()
    print("> Add to default file? [Y/N]")
    selection : str = input("> ").upper().strip()

    if (selection == "Y"):
        add_sample_sentences(filepath)
    elif (selection == "N"):
        print("> Enter destination filepath for sample sentences: ")
        filepath_2 : str = input("> ").strip()
        add_sample_sentences(filepath, filepath_2)
    else:
        print("> Invalid option entered, exiting program.")