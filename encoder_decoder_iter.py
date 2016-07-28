import mxnet as mx
import numpy as np

from collections import namedtuple
from sklearn.cluster import KMeans

class EncoderDecoderBatch(object):
    def __init__(self, all_data, all_label, init_states, bucket_key):
        self.pad = 0 # at this point i do not know what is this for...

        #all_data.shape is (x,y,z)
        self.batch_size = all_data.shape[0]

        # provide data, essential assignment
        self.data = [ mx.nd.array(all_data) ]

        # essential assignment
        self.provide_data = [('data', (self.batch_size, bucket_key))]
        for x in init_states:
            self.data.append(mx.nd.zeros(x[1])) # x[1] is the shape of the initial data
            self.provide_data.append(x)

        # provide label, essential assignment
        self.label = [ mx.nd.array(all_label) ]
        self.provide_label = [ ('label', (self.batch_size, bucket_key)) ]

        self.init_states = init_states

        # bucket_key is essential for this databatch
        self.bucket_key = bucket_key

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

        # data is a numpy array of 3 dimensions, #, timesteps, vector_dim
        self.data_label = data_label

        self.word2idx = word2idx
        self.idx2word = idx2word

        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.num_buckets = num_buckets

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
        enc_dec_data = [ len(data)+len(label)-1 for data, label in self.data_label ]
        enc_dec_data = np.reshape(np.array(enc_dec_data), (-1, 1))

        kmeans = KMeans(n_clusters=self.num_buckets, random_state=1) # use clustering to decide the buckets
        assignments = kmeans.fit_predict(enc_dec_data) # get the assignments

        # get the max of every cluster
        buckets = np.array([np.amax(enc_dec_data[assignments==i]) for i in range(self.num_buckets)])

        # get # of sequences in each bucket... then assign the batch size as the minimum(minimum(bucketsize), batchsize)
        buckets_count = np.array([enc_dec_data[assignments==i].shape[0] for i in range(self.num_buckets)])

        return buckets, buckets_count, assignments

    @property
    def default_bucket_key(self):
        return np.amax(self.buckets)

    @property
    def provide_data(self): # this is necessary when specifying custom DataIter
        # length of data variable is length of encoder + length of decoder
        bucket_key = self.default_bucket_key
        return [('data', (self.batch_size, bucket_key))] + self.init_states
    #
    @property
    def provide_label(self): # this is necessary when specifying custom DataIter
        # length of label variable is only the length of decoder
        bucket_key = self.default_bucket_key
        return [('label', (self.batch_size, bucket_key))]

    # for custom DataIter, we must implement this class as an iterable and return a DataBatch
    def __iter__(self): # this is necessary to convert this class into an iterable
        return self

    def __next__(self):
        if self.iter_next():
            # suppose to get self.cursor:self.cursor + self.batch_size
            batch = self.data_label[self.assignments == self.cur_permute_bucket]\
                [self.in_bucket_permutation[self.cursor:self.cursor+self.batch_size]]

            # get size of this bucket
            seqlen = self.buckets[self.cur_permute_bucket] # this seqlen already deducted the <EOS>

            all_data = np.full((self.batch_size, seqlen), self.pad_label, dtype=float)
            all_label = np.full((self.batch_size, seqlen), self.pad_label, dtype=float)

            for i, (data, label) in enumerate(batch):
                if self.rev:
                    # reverse the input except for the <EOS> at end of input
                    # according to Ilya Sutskever et al. Sequence to Sequence Learning with Neural Networks
                    # there is a reason for this... which you should ask freddy
                    data[:-1] = np.flipud(data[:-1])

                all_data[i, :data.shape[0]] = data
                all_data[i, data.shape[0]:data.shape[0]+label.shape[0]-1] = label[:-1]
                all_label[i, data.shape[0]-1:data.shape[0]-1+label.shape[0]] = label

            return EncoderDecoderBatch(all_data, all_label, self.init_states, seqlen)
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
