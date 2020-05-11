import csv

with open('../flickr8k_data/Flickr8k.lemma.token.txt', 'r') as flickr_txt, open('../data/flickr_captions.csv', 'w') as captionfile:
    reader = csv.reader(flickr_txt, delimiter='\t')
    writer = csv.writer(captionfile)
    for row in reader:
        imgname = row[0][:-2]
        caption = row[1].lower()
        writer.writerow([imgname, caption])
