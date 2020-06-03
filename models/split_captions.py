from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

def split_all_captions(filename):
    doc_1 = []
    doc_2 = []
    doc_3 = []
    doc_4 = []
    doc_5 = []

    count = 1
    with open(filename, 'r') as f:
        for line in f:
            if count % 5 == 1:
                doc_1.append(line)
            if count % 5 == 2:
                doc_2.append(line)
            if count % 5 == 3:
                doc_3.append(line)
            if count % 5 == 4:
                doc_4.append(line)
            if count % 5 == 5:
                doc_5.append(line)
            count += 1
    f.close()

    with open(".../data/captions_1.txt", "w") as f:
        f.writelines([line for line in doc_1])
    f.close()

    with open(".../data/captions_2.txt", "w") as f:
        f.writelines([line for line in doc_2])
    f.close()

    with open(".../data/captions_3.txt", "w") as f:
        f.writelines([line for line in doc_3])
    f.close()

    with open(".../data/captions_4.txt", "w") as f:
        f.writelines([line for line in doc_4])
    f.close()

    with open(".../data/captions_5.txt", "w") as f:
        f.writelines([line for line in doc_1])
    f.close()

def create_nouns(filename):
    words = []
    LemmaT = WordNetLemmatizer()
    count = 0
    with open(filename, 'r') as f:
        for line in f:
            words.append(line[:len(line)-1])
            count += 1

    nouns = []
    for word in words:
        if len(wn.synsets(word, 'n')) > 0 :
            nouns.append(word)
    
    with open(".../data/caption_nouns_vgg16.txt", "w") as f:
        f.writelines([LemmaT.lemmatize(line)+"\n" for line in nouns])
    f.close()

create_nouns(".../data/captions_lookup.txt")
