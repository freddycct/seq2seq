import re
import numpy as np

remove_chars = re.compile(r'([\(\)\'\"{}\[\]\*\-/])')
punctuation = re.compile(r'([!,.?:;@#$%&]+)')
def filter_text(text):
    # returns the text without the <EOS> token
    text = remove_chars.sub(' ', text)
    text = punctuation.sub(r' \1 ', text) # replaces the punctuation so that there is a space seperating it from the word
    text = text.lower().strip(' \t\n') # replaces big caps with small caps
    return text

white_spaces = re.compile(r'[ \n\r\t]+')
def get_vocab(file, vocab_count={}):
    with open(file, 'r', encoding='utf-8', errors='ignore') as fid:
        for line in fid:
            if len(line) == 0:
                continue
            tokens = white_spaces.split(filter_text(line))
            for token in tokens:
                if len(token) > 0:
                    if token in vocab_count:
                        vocab_count[token] += 1
                    else:
                        vocab_count[token] = 1
    return vocab_count

def text_2_indices(word2idx, text):
    # return the list of indices representing this text including the <EOS> token at the end...
    tokens = white_spaces.split(filter_text(text))
    indices = []
    unk_index = word2idx.get('<UNK>')
    indices = [ word2idx.get(token, unk_index) for token in tokens ]
    indices.append(word2idx['<EOS>'])
    return np.array(indices)

def get_unified_vocab(enc_input_file, dec_input_file, percentile=80):
    vocab_count = get_vocab(enc_input_file) # this returns a dictionary
    vocab_count = get_vocab(dec_input_file, vocab_count) # this returns a dictionary

    word_distribution = np.array( [ v for v in vocab_count.values() ] )
    min_count = np.percentile(word_distribution, percentile)
    vocab = []
    for k,v in vocab_count.items():
        if v >= min_count:
            vocab.append(k)
    vocab.sort()

    vocab.append('<UNK>') # token representing a word unseen in the training set, reserved for rare words
    vocab.append('<EOS>') # token representing the End-of-Sentence
    vocab.append('<PAD>') # token representing the padding for use in bucketing RNN of different lengths

    word2idx = { w:i for i,w in enumerate(vocab) }
    idx2word = [ w for w in vocab ]

    return word2idx, idx2word

def get_data_label(enc_input_file, dec_input_file, word2idx):
    enc_input = []
    with open(enc_input_file, 'r', encoding='utf-8', errors='ignore') as fid:
        for line in fid:
            indices = text_2_indices(word2idx, line)
            enc_input.append(indices)

    dec_input = []
    with open(dec_input_file, 'r', encoding='utf-8', errors='ignore') as fid:
        for line in fid:
            indices = text_2_indices(word2idx, line)
            dec_input.append(indices)
    return np.array( list(zip(enc_input, dec_input)) )
