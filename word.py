class Word:
    word : str
    type : str
    bert_score : float
    glove_score : float

    def __init__(self, word : str, type : str):
        self.word = word.lower()
        self.type = type.upper()
        self.bert_score = -1    # higher val = better -> default -ve value
        self.glove_score = 99 # lower val = better -> default high value 

    # SETTERS & GETTERS
    def get_word(self) -> str:
        return self.word
    
    def get_type(self) -> str:
        return self.type

    def get_b_score(self) -> float:
        return self.bert_score
    
    def get_g_score(self) -> float:
        return self.glove_score
    
    def set_word(self, word : str):
        self.word = word.lower()
    
    def set_type(self, type : str):
        self.type = type
    
    def set_b_score(self, bert_score : float):
        self.bert_score = bert_score
    
    def set_g_score(self, glove_score : float):
        self.glove_score = glove_score

    def get_str(self) -> str:
        str_rep = f"> {self.word.lower()} <{self.type.upper()}> || B: {self.bert_score} | G: {self.glove_score}"

        return str_rep
    
    # BASIC FUNCTIONALITIES
    def __str__(self):
        str_rep = f"> {self.word.lower()} <{self.type.upper()}> || B: {self.bert_score} | G: {self.glove_score}"

        return str_rep

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.word.lower() == other.word.lower()
        else:
            return False
    
    # NEED TO HASH TO USE AS DICTIONARY KEY
    def __hash__(self):
        return hash(self.word) 
    

if __name__ == "__main__":
    word_1 = Word("test", "NOUN")
    word_2 = Word("apple", "NOUN")

    library = {word_1 : "lalala", word_2 : "hehehe"}
    print(library)

    

