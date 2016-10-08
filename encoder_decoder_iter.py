import mxnet as mx
import numpy as np

from collections import namedtuple
from sklearn.cluster import KMeans

EncDecBucketKey = namedtuple('EncDecBucketKey', ['enc_len', 'dec_len'])

class EncoderDecoderBatch(object):
    def __init__(self, all_data, all_mask, all_label, init_states, bucket_key):
        # provide data, essential assignment
        # need to optimize this list creation!!!
        self.data = [ mx.nd.array(all_data), mx.nd.array(all_mask) ] + [ mx.nd.zeros(x[1]) for x in init_states ]

        # provide label, essential assignment
        self.label = [ mx.nd.array(all_label) ]

        self.init_states = init_states

        # bucket_key is essential for this databatch
        self.bucket_key = bucket_key

        #all_data.shape is (x,y,z)
        self.batch_size = all_data.shape[0]

    # this two properties are essential too!
    @property
    def provide_data(self):
        return [
            ('data', (self.batch_size, self.bucket_key.enc_len + self.bucket_key.dec_len)),
            ('mask', (self.batch_size, self.bucket_key.enc_len + self.bucket_key.dec_len))
        ] + self.init_states

    @property
    def provide_label(self):
        return [('label', (self.batch_size, self.bucket_key.dec_len))]


def synchronize_batch_size(train_iter, test_iter):
    batch_size = min(train_iter.batch_size, test_iter.batch_size)
    train_iter.batch_size = batch_size
    test_iter.batch_size = batch_size
    train_iter.generate_init_states()
    test_iter.generate_init_states()


# now define the bucketing, padding and batching SequenceIterator...
class EncoderDecoderIter(mx.io.DataIter):
    def __init__(self, data_label, word2idx, idx2word, num_hidden, num_layers,
                 init_states_function, batch_size=1, num_buckets=10, shuffle=False, rev=False):

        super(EncoderDecoderIter, self).__init__() # calling DataIter.__init__()

        # data is a numpy array of 3 dimensions, (#, timesteps, vector_dim)
        # let's say you have 2 sequences, #1 has len 5, dimension 10
        self.data_label = data_label #numpy multi-dimensional array

        # data_label[i] is the ith sequence


        self.word2idx = word2idx
        self.idx2word = idx2word

        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.num_buckets = num_buckets

        # arrange the data so that

        # now we need to find the buckets based on the input data...
        self.buckets, self.buckets_count, self.assignments = self.generate_buckets()
        # buckets are a tuple of the encoder/decoder length

        self.batch_size = min(np.min(self.buckets_count), batch_size)
        self.init_states_function = init_states_function
        self.pad_label = word2idx['<PAD>']
        self.shuffle = shuffle
        self.rev = rev # reverse the encoder input
        self.reset()
        self.generate_init_states()

    def generate_init_states(self):
        self.init_states = self.init_states_function(self.num_layers, self.num_hidden, self.batch_size)

    def generate_buckets(self):
        enc_dec_data = []
        for data, label in self.data_label:
            enc_len = len(data) - 1 # minue one because of the <EOS>
            dec_len = len(label)
            enc_dec_data.append((enc_len, dec_len))

        enc_dec_data = np.array(enc_dec_data)

        kmeans = KMeans(n_clusters = self.num_buckets, random_state = 1) # use clustering to decide the buckets
        assignments = kmeans.fit_predict(enc_dec_data) # get the assignments

        # get the max of every cluster
        buckets = np.array([np.max( enc_dec_data[assignments==i], axis=0 ) for i in range(self.num_buckets)])

        # get # of sequences in each bucket... then assign the batch size as the minimum(minimum(bucketsize), batchsize)
        buckets_count = np.array( [ enc_dec_data[assignments==i].shape[0] for i in range(self.num_buckets) ] )

        return buckets, buckets_count, assignments

    @property
    def default_bucket_key(self):
        enc_len, dec_len = np.max(self.buckets, axis=0)
        return EncDecBucketKey(enc_len = enc_len, dec_len = dec_len)

    @property
    def provide_data(self): # this is necessary when specifying custom DataIter
        # length of data variable is length of encoder + length of decoder
        enc_dec_bucket_key = self.default_bucket_key

        return [
            ('data', (self.batch_size, enc_dec_bucket_key.enc_len + enc_dec_bucket_key.dec_len)),
            ('mask', (self.batch_size, enc_dec_bucket_key.enc_len + enc_dec_bucket_key.dec_len))
        ] + self.init_states
    #
    @property
    def provide_label(self): # this is necessary when specifying custom DataIter
        # length of label variable is only the length of decoder
        enc_dec_bucket_key = self.default_bucket_key
        return [('label', (self.batch_size, enc_dec_bucket_key.dec_len))]

    # for custom DataIter, we must implement this class as an iterable and return a DataBatch
    def __iter__(self): # this is necessary to convert this class into an iterable
        return self

    def __next__(self):
        if self.iter_next():
            # suppose to get self.cursor:self.cursor + self.batch_size
            batch = self.data_label[self.assignments == self.cur_permute_bucket]\
                [ self.in_bucket_permutation[self.cursor:self.cursor+self.batch_size] ]

            # get size of this bucket
            enc_len, dec_len = self.buckets[self.cur_permute_bucket] # this enc_len already deducted the <EOS>
            # total length of rnn sequence is enc_len+dec_len

            all_data  = np.full((self.batch_size, enc_len+dec_len), self.pad_label, dtype=float)
            all_label = np.full((self.batch_size, dec_len), self.pad_label, dtype=float)
            all_mask = np.zeros((self.batch_size, enc_len+dec_len), dtype=float)

            for i, (data, label) in enumerate(batch):
                if self.rev:
                    # reverse the input except for the <EOS> at end of input
                    # according to Ilya Sutskever et al. Sequence to Sequence Learning with Neural Networks
                    data[:-1] = np.flipud(data[:-1])

                enc_input = np.concatenate((data, label[:-1])) # data <EOS> label
                z = enc_len - data.shape[0] + 1
                all_data[i, z:enc_len + label.shape[0]] = enc_input
                all_mask[i, z:enc_len + label.shape[0]] = 1.0
                all_label[i, :label.shape[0]] = label

            return EncoderDecoderBatch(all_data, all_mask, all_label, self.init_states, EncDecBucketKey(enc_len=enc_len, dec_len=dec_len))
        else:
            raise StopIteration

    def iter_next(self):
        self.cursor += self.batch_size
        if self.cursor < self.buckets_count[self.cur_permute_bucket]:
            if self.cursor + self.batch_size > self.buckets_count[self.cur_permute_bucket]:
                # it is going to overflow the bucket
                self.cursor -= self.cursor + self.batch_size - self.buckets_count[self.cur_permute_bucket]
            return True
        else:
            self.cur_bucket += 1
            if self.cur_bucket < self.num_buckets:
                self.cursor = 0
                self.cur_permute_bucket = self.bucket_permutation[self.cur_bucket]
                if self.shuffle:
                    self.in_bucket_permutation = np.random.permutation(self.buckets_count[self.cur_permute_bucket])
                else:
                    self.in_bucket_permutation = np.array(range(self.buckets_count[self.cur_permute_bucket]))
                return True
            else:
                return False

    def reset(self): # for iterable
        self.cursor = -self.batch_size
        self.cur_bucket = 0

        if self.shuffle:
            self.bucket_permutation = np.random.permutation(self.num_buckets)
        else:
            self.bucket_permutation = np.array(range(self.num_buckets))

        self.cur_permute_bucket = self.bucket_permutation[self.cur_bucket]
        if self.shuffle:
            self.in_bucket_permutation = np.random.permutation(self.buckets_count[self.cur_permute_bucket])
        else:
            self.in_bucket_permutation = np.array(range(self.buckets_count[self.cur_permute_bucket]))
