import mxnet as mx
import numpy as np

import re, os, sys, argparse, logging

from lstm import LSTMState, LSTMParam, init_lstm, \
    get_lstm_init_states, lstm_cell, lstm_unroll, get_lstm_sym_generator
from encoder_decoder_iter import EncoderDecoderBatch, EncoderDecoderIter, synchronize_batch_size
from text_io import get_unified_vocab, get_data_label

def perplexity(label, pred, ignore_label):
    label = label.T.reshape((-1,))
    loss = 0.
    for i in range(pred.shape[0]):
        if label[i] == ignore_label:
            break
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / label.size)

def main(argv):
    parser = argparse.ArgumentParser(description="Encoder Decoder Model for Summarization")

    parser.add_argument('--num-buckets', default=1, type=int,
        help='number of buckets for clustering sequences of different length')
    parser.add_argument('--num-layers', default=1, type=int,
        help='number of layers for the LSTM recurrent neural network')
    parser.add_argument('--num-hidden', default=10, type=int,
        help='number of hidden units in the neural network')
    parser.add_argument('--batch-size', default=2, type=int,
        help='batch size for each databatch')
    parser.add_argument('--iterations', default=1, type=int,
        help='number of iterations (epoch)')
    parser.add_argument('--expt-name', default='test', type=str,
        help='the experiment name, this is also the prefix for the parameters file')
    parser.add_argument('--params-dir', default='params', type=str,
        help='the directory to store the parameters of the training')
    parser.add_argument('--gpus', type=str,
        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('enc_train_input', type=str,
        help='the file name of the encoder input for training')
    parser.add_argument('dec_train_input', type=str,
        help='the file name of the decoder input for training')
    parser.add_argument('--enc-test-input', type=str,
        help='the file name of the encoder input for testing')
    parser.add_argument('--dec-test-input', type=str,
        help='the file name of the decoder input for testing')
    parser.add_argument('--shuffle', default=False, type=bool,
        help='whether to shuffle the training set')
    parser.add_argument('--seed', default=1, type=int,
        help='seed for the numpy random number genenerator')
    parser.add_argument('--reverse-input', default=False, type=bool,
        help='reverse the input of the encoder')
    parser.add_argument('--dropout', default=0.0, type=float,
        help='dropout is the probability to ignore the neuron outputs')
    parser.add_argument('--top-words', default=80, type=int,
        help='the top percentile of word count to retain in the vocabulary')
    parser.add_argument('--log-freq', default=100, type=int,
        help='the frequency to printout the training verbose information')
    parser.add_argument('--lr', default=0.01, type=float,
        help='learning rate of the stochastic gradient descent')

    args = parser.parse_args()
    print(args)

    num_buckets = args.num_buckets
    num_layers  = args.num_layers
    num_hidden  = args.num_hidden
    batch_size  = args.batch_size
    iterations  = args.iterations
    expt_name   = args.expt_name
    params_dir  = args.params_dir
    seed        = args.seed
    shuffle     = args.shuffle
    rev         = args.reverse_input
    dropout     = args.dropout
    top_words   = args.top_words
    log_freq    = args.log_freq
    context     = mx.cpu() if args.gpus is None else [ mx.gpu(int(i)) for i in args.gpus.split(',') ]
    enc_train_input = args.enc_train_input
    dec_train_input = args.dec_train_input
    enc_test_input  = args.enc_test_input
    dec_test_input  = args.dec_test_input
    lr = args.lr

    np.random.seed(seed)

    word2idx, idx2word = get_unified_vocab(enc_train_input, dec_train_input, top_words)
    train_data_label = get_data_label(enc_train_input, dec_train_input, word2idx)

    if enc_test_input is not None and dec_test_input is not None:
        test_data_label = get_data_label(enc_test_input, dec_test_input, word2idx)

    if iterations > 0:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        train_iter = EncoderDecoderIter(train_data_label, word2idx, idx2word,
            num_hidden, num_layers, get_lstm_init_states, batch_size=batch_size,
            num_buckets=num_buckets, shuffle=shuffle, rev=rev)

        if 'test_data_label' in locals():
        # if enc_test_input is not None and dec_test_input is not None:
            test_iter = EncoderDecoderIter(test_data_label, word2idx, idx2word,
                num_hidden, num_layers, get_lstm_init_states, batch_size=batch_size,
                num_buckets=num_buckets, shuffle=shuffle, rev=rev)

            # the reason why we synchronize them is because the they screw up otherwise
            synchronize_batch_size(train_iter, test_iter)

        batch_size = train_iter.batch_size
        print('batch_size:', batch_size)
        # load parameters file if exists!!!

        model_args = {}
        if os.path.isfile('%s/%s-symbol.json' % (params_dir, expt_name)):
            filelist = os.listdir(params_dir) # get list of params file
            paramfilelist = []
            for f in filelist:
                if f.startswith('%s-' % expt_name) and f.endswith('.params'):
                    paramfilelist.append( int(re.split(r'[-.]', f)[1]) )
            last_iteration = max(paramfilelist)
            print('loading pretrained model %s/%s at epoch %d' % (params_dir, expt_name, last_iteration))
            tmp = mx.model.FeedForward.load('%s/%s' % (params_dir, expt_name), last_iteration)
            model_args.update({
                'arg_params' : tmp.arg_params,
                'aux_params' : tmp.aux_params,
                'begin_epoch' : tmp.begin_epoch
            })

        num_labels = len(word2idx)
        model = mx.model.FeedForward(
            ctx           = context, # uses all the available CPU in the machine
            symbol        = get_lstm_sym_generator(num_layers, num_hidden, num_labels, dropout),
            num_epoch     = iterations,
            learning_rate = lr,
            momentum      = 0.0,
            wd            = 0.00001,
            initializer   = mx.init.Xavier(factor_type="in", magnitude=2.34),
            **model_args
        )

        if not os.path.exists(params_dir):
            os.makedirs(params_dir)

        model.fit(
            X = train_iter,
            eval_data = test_iter if 'test_iter' in locals() else None,
            eval_metric = mx.metric.np(perplexity, use_ignore=True, ignore_label=num_labels),
            batch_end_callback = [ mx.callback.Speedometer(batch_size, frequent=log_freq) ],
            epoch_end_callback = [ mx.callback.do_checkpoint( '%s/%s' % (params_dir, expt_name) ) ]
        )
    # if iterations > 0:
# def main(argv):

if __name__ == "__main__":
    main(sys.argv)
