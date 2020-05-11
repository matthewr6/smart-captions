import collections

def get_nonrare_words():
    counter = collections.defaultdict(int)
    with open('../data/captions.txt', 'r') as f:
        seqs = f.readlines()
    seqs = [l.split(',')[1:] for l in seqs]
    for seq in seqs:
        for word_idx in seq:
            counter[word_idx] += 1
    true_counter = collections.defaultdict(int)
    for word_idx in counter:
        if counter[word_idx] >= 10:
            true_counter[word_idx] += 1
    return true_counter.keys()

words = []
with open('../data/captions_lookup.txt', 'r') as f:
    words = f.readlines()

nonrare_words = get_nonrare_words()
# VOCAB_SIZE = len(words) + 1
VOCAB_SIZE = len(nonrare_words) + 1
int_to_word = {}
for idx, word in enumerate(words):
    int_to_word[idx] = word[:-1]

def seqs_to_captions(seqs):
    captions = []
    for seq in seqs:
        caption = ''
        for word_idx in seq:
            if int(word_idx) == 0:
                caption += ' ' + 'STOP'
                # break
            else:
                caption += ' ' + int_to_word[int(word_idx) - 1]
        captions.append(caption)
    return captions

base = ord('a') - 1
def letterseqs_to_captions(seqs):
    captions = []
    for seq in seqs:
        caption = ''
        for char_idx in seq:
            if char_idx == 0:
                # caption += '0'
                break
            if char_idx == 28:
                caption += ' '
            else:
                caption += chr(char_idx + base)
        captions.append(caption)
    return captions