import mxnet as mx
import numpy as np

from collections import namedtuple

LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias", "h2h_weight", "h2h_bias"])

def lstm_cell(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0.):
    """LSTM Cell symbol"""
    if dropout > 0.:
        indata = mx.sym.Dropout(data=indata, p=dropout)

    i2h = mx.sym.FullyConnected(
        data=indata,
        weight=param.i2h_weight,
        bias=param.i2h_bias,
        num_hidden=num_hidden * 4,
        name="t%d_l%d_i2h" % (seqidx, layeridx)
    )

    h2h = mx.sym.FullyConnected(
        data=prev_state.h,
        weight=param.h2h_weight,
        bias=param.h2h_bias,
        num_hidden=num_hidden * 4,
        name="t%d_l%d_h2h" % (seqidx, layeridx)
    )

    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4, name="t%d_l%d_slice" % (seqidx, layeridx))

    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")

    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")

    return LSTMState(c=next_c, h=next_h)

def init_lstm(num_layer, prefix=''):
    param_cells = []
    last_states = []
    for i in range(num_layer):
        param_cells.append(
            LSTMParam(
                i2h_weight=mx.sym.Variable("%s_l%d_i2h_weight" % (prefix, i)),
                i2h_bias=mx.sym.Variable("%s_l%d_i2h_bias" % (prefix, i)),
                h2h_weight=mx.sym.Variable("%s_l%d_h2h_weight" % (prefix, i)),
                h2h_bias=mx.sym.Variable("%s_l%d_h2h_bias" % (prefix, i))
            )
        )
        last_states.append(
            LSTMState(
                c=mx.sym.Variable("l%d_init_c" % i),
                h=mx.sym.Variable("l%d_init_h" % i)
            )
        )
    return param_cells, last_states

def lstm_unroll(num_layer, seqlen, num_hidden, num_labels, dropout=0.0):
    cls_weight   = mx.sym.Variable("cls_weight")
    cls_bias     = mx.sym.Variable("cls_bias")
    embed_weight = mx.sym.Variable("embed_weight")

    param_cells, last_states = init_lstm(num_layer)
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')

    embed = mx.sym.Embedding(
        data=data, # the idx to the embedding
        input_dim=num_labels, # the number of rows for embed_weight
        weight=embed_weight,  # the matrix representing the idx2vec
        output_dim=num_hidden, # the number of cols for embed_weight
        name='embed'
    )

    wordvec = mx.sym.SliceChannel(data=embed, num_outputs=seqlen, squeeze_axis=1)

    hidden_all = []
    for seqidx in range(seqlen):
        hidden = wordvec[seqidx]
        # stack LSTM
        for i in range(num_layer):
            dp = 0.0 if i == 0 else dropout
            next_state = lstm_cell(
                num_hidden,
                indata=hidden,
                prev_state=last_states[i],
                param=param_cells[i],
                seqidx=seqidx,
                layeridx=i,
                dropout=dp
            )
            hidden = next_state.h
            last_states[i] = next_state
        # decoder
        if dropout > 0.0:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)

        hidden_all.append(hidden)

    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    pred = mx.sym.FullyConnected(
        data=hidden_concat,
        num_hidden=num_labels, # num_labels is the index of <PAD> that means this layer will predict 0, 1, ..., num_labels-1
        weight=cls_weight,
        bias=cls_bias,
        name='pred'
    )

    label = mx.sym.transpose(data=label) # e.g. if shape is (1,M) it becomes (M,1)
    label = mx.sym.Reshape(data=label, shape=(-1,)) # if shape is (M,1) it becomes (M,)
    output = mx.sym.SoftmaxOutput(
        data=pred,
        label=label,
        name='t%d_softmax' % seqidx,
        use_ignore=True,
        ignore_label=num_labels # ignore the index of <PAD>
    ) # output becomes (num_labels, M)
    return output

def get_lstm_sym_generator(num_layers, num_hidden, num_labels, dropout=0.0):
    def generate_lstm_sym(seqlen):
        return lstm_unroll(num_layers, seqlen, num_hidden, num_labels, dropout)
    return generate_lstm_sym

def get_lstm_init_states(num_layers, num_dim, batch_size=1):
    init_h = [('l%d_init_h' % i, (batch_size, num_dim)) for i in range(num_layers)]
    init_c = [('l%d_init_c' % i, (batch_size, num_dim)) for i in range(num_layers)]
    init_states = init_h + init_c
    return init_states
