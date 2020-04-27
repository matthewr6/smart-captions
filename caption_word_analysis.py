import pandas as pd
import pickle
from nltk.corpus import stopwords

# read in the data
training = pd.read_csv("sources/train.tsv", sep='\t', names = ["description", "url"])

# load the .pickle file of words if it has been generated
try:
    f = open("sources/words.pickle", 'rb')
    words = pickle.load(f)
    f.close()
except:
    print(training["description"])
    words = {}
    for line in training["description"]:

        in_str = list(dict.fromkeys(line.split(" ")))
        for word in in_str:
            try:
                words[word] += 1
            except:
                words[word] = 1
    pickle.dump(words, open("sources/words.pickle", "wb"))
print("Full words collected")

# load the .pickle file of words filtered for stopwords if it has been generated
try:
    f = open("sources/filtered_words.pickle", 'rb')
    filtered_words = pickle.load(f)
    f.close()
except:
    words = {k: v for k, v in sorted(words.items(), key=lambda item: item[1], reverse=True)}

    filtered_words = {}
    for word in words.keys():
        if word not in set(stopwords.words('english')) and word.isalnum():
            filtered_words[word] = words[word]

    pickle.dump(filtered_words, open("sources/filtered_words.pickle", "wb"))
print("Filtered words collected")

# write the words to a text file and collect the top n
n = 100
file = open("sources/filtered_words.txt", "w")
num = 1
top_words = []
with open('sources/word_subset.txt', 'w') as f:
    for word in filtered_words.keys():
        output = "#" + str(num) + ": " + str(word) + ", " + str(filtered_words[word]) + "\n"
        try:
           file.write(output)
           f.write(word + '\n')
           top_words.append(word)
        except:
            continue
        num += 1
        if num > n:
            break
z = set(z)

file.close()
print("Filtered words output and top {} words found".format(n))

# write all the captions containing at least one of the top N words to a .csv file
count = 0
captions = []
stopword_set = set(stopwords.words('english'))
import re
for caption in training["description"]:
    cap_words = set(filter(None, re.sub(r'[^a-zA-Z ]', '', caption).split(" ")))
    cap_words -= stopword_set 
    # if cap_words & set(z):
    # print(cap_words)
    if cap_words.issubset(z):
        count+=1
        captions.append(caption)

print("Percent of captions containing all top {} words: ".format(n) + str(count/len(training["description"])))

# new_training = training[training['description'].isin(captions)]
# new_training.to_csv('sources/filtered_captions.csv', header=False)




