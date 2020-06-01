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
with open('../data/captions_lookup_withstop.txt', 'r') as f:
    words = f.readlines()

nonrare_words = get_nonrare_words()
# nonrare_words = words

NUM_SIGNAL_TOKENS = 3
VOCAB_SIZE = len(nonrare_words) + NUM_SIGNAL_TOKENS
print('Vocab size = {}'.format(VOCAB_SIZE))
int_to_word = {}
for idx, word in enumerate(words):
    int_to_word[idx] = word[:-1]

# MAX_SEQ_LEN = 35
MAX_SEQ_LEN = 50

START_TOKEN = 1
STOP_TOKEN = 2

def seqs_to_captions(seqs):
    captions = []
    for seq in seqs:
        caption = ''
        for word_idx in seq:
            caption += ' '
            true_idx = int(word_idx)
            if true_idx == 1 or true_idx == 0:
                continue
            elif true_idx == 2:
                break
            else:
                caption += int_to_word[true_idx - NUM_SIGNAL_TOKENS]
        captions.append(caption.strip())
    return captions
