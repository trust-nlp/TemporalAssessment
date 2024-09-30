from nltk.tokenize import RegexpTokenizer
import re

tok = RegexpTokenizer(r'\w+')  # alphanumeric tokenization

def clean_and_tokenize(doc, stopwords=None):
    global tok
    # If the 'stopwords' argument is not provided, set it to an empty set.
    if stopwords is None:
        stopwords = set()
    # replace URLs in the doc string with the word "url"
    doc = re.sub(r"https?:\S+", "url", doc)
    # remove newlines and tabs 
    doc = doc.replace('\n', ' ')
    doc = doc.replace('\t', ' ')
    # replace date
    doc = re.sub(r"(\d+)+(\-)+(\d+)+(\-)+(\d+)", "date", doc)
    # remove all serialization eg 1. 1) or 1.1
    doc = re.sub(r"(\d+)+(\.|\))+(\d+)", "", doc)
    doc = re.sub(r"(\d+)+(\.|\))", "", doc)
    doc = re.sub(r"\b(\w+)( \1\b)+", r"\1", doc)  # removing consecutive duplicate words
    doc = re.sub(r"(\b)(d|[dD])\.?(r|[rR])\.?(\b)", " ", doc)  # remove Dr abbreviation (the name of Dr is hidded so we just drop Dr. as well)
    doc = re.sub(r"([^A-Za-z0-9\s](\s)){2,}", " ", doc)  # remove consecutive punctuations
    doc = re.sub(r'\.+', '..', doc) #replaces two or more consecutive ellipsis with just two (..).
    doc = re.sub(r'!+', '!', doc) #replaces two or more consecutive ! with just one 
    doc = re.sub(r'\*+', ' ', doc) #replaces two or more consecutive * with just space
    doc = re.sub(r'_+', ' ', doc) #replaces two or more consecutive underscore_
    doc = re.sub(r',+', ',', doc) #replaces two or more consecutive commas,
    # all lowercase
    doc = doc.lower()
    doc = [item.strip() for item in tok.tokenize(doc)
           if len(item.strip()) > 1 and item not in stopwords
           ]  # tokenize
    return doc

def fast_clean(doc, stopwords=None):
    global tok
    
    # If the 'stopwords' argument is not provided, set it to an empty set.
    if stopwords is None:
        stopwords = set()

    # replace URLs in the doc string with the word "url"
    doc = re.sub(r"https?:\S+", "url", doc)
    # remove newlines and tabs 
    doc = doc.replace('\n', ' ')
    doc = doc.replace('\t', ' ')
    
    doc = re.sub(r"\b(\w+)( \1\b)+", r"\1", doc)  # removing consecutive duplicate words
    doc = re.sub(r"(\b)(d|[dD])\.?(r|[rR])\.?(\b)", " ", doc)  # remove Dr abbreviation (I guess it's because the name of Dr is hidded so we just drop Dr. as well)
    doc = re.sub(r"([^A-Za-z0-9\s](\s)){2,}", " ", doc)  # remove consecutive punctuations

    
    doc = re.sub(r'\.+', '..', doc) #replaces two or more consecutive ellipsis with just two (..).
    doc = re.sub(r'!+', '!', doc) #replaces two or more consecutive ! with just one 
    doc = re.sub(r'\*+', ' ', doc) #replaces two or more consecutive * with just space
    doc = re.sub(r'_+', ' ', doc) #replaces two or more consecutive underscore_
    doc = re.sub(r',+', ',', doc) #replaces two or more consecutive commas,

    # all lowercase
    doc = doc.lower()

    doc = [item.strip() for item in tok.tokenize(doc)
           if len(item.strip()) > 1 and item not in stopwords
           ]  # tokenize

    return doc
    


