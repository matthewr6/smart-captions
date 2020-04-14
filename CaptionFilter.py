import pandas as pd
import pickle
from nltk.corpus import stopwords

# read in the data
training = pd.read_csv("Train_GCC-training.tsv", sep='\t', names = ["description", "url"])

# load the .pickle file of words if it has been generated
try:
    f = open("words.pickle", 'rb')
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
    pickle.dump(words, open("words.pickle", "wb"))
print("Full words collected")

# load the .pickle file of words filtered for stopwords if it has been generated
try:
    f = open("filtered_words.pickle", 'rb')
    filtered_words = pickle.load(f)
    f.close()
except:
    words = {k: v for k, v in sorted(words.items(), key=lambda item: item[1], reverse=True)}

    filtered_words = {}
    for word in words.keys():
        if word not in set(stopwords.words('english')) and word.isalnum():
            filtered_words[word] = words[word]

    pickle.dump(filtered_words, open("filtered_words.pickle", "wb"))
print("Filtered words collected")

# write the words to a text file and collect the top 50
file = open("filtered_words.txt", "w")
num = 1
x = []
y = []
z = []
for word in filtered_words.keys():
    output = "#" + str(num) + ": " + str(word) + ", " + str(filtered_words[word]) + "\n"
    try:
       file.write(output)
       x.append(num)
       y.append(filtered_words[word])
       z.append(word)
    except:
        continue
    num += 1
    if num > 50:
        break

file.close()
print("Filtered words output and top 50 words found")


# write all the captions containing at least one of the top 50 words to a .csv file
count = 0
captions = []
for caption in training["description"]:
    cap_words = list(dict.fromkeys(caption.split(" ")))
    if (set(cap_words) & set(z)):
        count+=1
        captions.append(caption)

print("Percent of captions containing a top 50 word: " + str(count/len(training["description"])))

new_training = training[training['description'].isin(captions)]
new_training.to_csv('filtered_captions.csv', header=False)




