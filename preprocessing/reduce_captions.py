import csv
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem.porter import *


def get_captions(filename):
    """
        filename : path to a csv file of captions
        returns the captions as a dictionary
        from index to caption as a string.
    """
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)
        lines = [(int(idx), caption) for idx, caption in lines]
    
    
    return dict(lines)

def clean_captions(captions, stopwords=set()):
    """
        captions : dict from index to caption as a string
        returns a dict with the values replaced with lists of words
    """
    new_captions = {}
    stemmer = PorterStemmer()
    for idx, caption in captions.items():
        # Strip ad split the caption
        new_caption = []
        for s in caption.split():
            s = s.strip('\'".,?!;')
            if s == "" or s in stopwords:
                continue
            s = stemmer.stem(s)
            new_caption.append(s)

        new_captions[idx] = new_caption
    return new_captions

def get_text_corpus(filename):
    """
        filename : path to a csv file of captions
        returns the captions as a large string
    """
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)
    s = ''
    for line in lines:
        s += line[1]
    with open('tmp.txt', 'w') as f2:
        f2.write(s)
    
    return s


def get_vocab(captions):
    """
        captions : dict from index to caption as a string
        returns the non-stopword vocabulary of captions as a set
    """
    vocab = set()
    for _, caption in captions.items():
        vocab |= set(caption)
    return vocab



def main():
    caption_file = '../data/captions.csv'
    captions = get_captions(caption_file)
    # Vocab with temming: 9k
    # Vocab with stemming: 6.5k
    captions = clean_captions(captions)
    vocab = get_vocab(captions)
    get_text_corpus(caption_file)



if __name__ == '__main__':
    main()