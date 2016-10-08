import mxnet as mx
import numpy as np

from collections import namedtuple

LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias", "h2h_weight", "h2h_bias"])

def lstm_cell_no_mask(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0.):
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

def lstm_cell(num_hidden, indata, mask, prev_state, param, seqidx, layeridx, dropout=0.):
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

    # mask out the output
    next_c = mx.sym.element_mask(next_c, mask)
    next_h = mx.sym.element_mask(next_h, mask)

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

def lstm_unroll(num_lstm_layer, enc_len, dec_len, num_hidden, num_labels, dropout=0.0):
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    embed_weight=mx.sym.Variable("embed_weight")

    enc_param_cells, last_states = init_lstm(num_lstm_layer, prefix='enc')
    dec_param_cells, _ = init_lstm(num_lstm_layer, prefix='dec')

    data  = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    mask  = mx.sym.Variable('mask')

    # (batch, time, vec) so axis 1 is the time step

    embed = mx.sym.Embedding(
        data=data, input_dim=num_labels,
        weight=embed_weight, output_dim=num_hidden, name='embed'
    )

    # num_hidden = 10
    # data is a sequence of index (0,1,2,3)
    # embedding -> 10 x 4 matrix...
    # each column is the word embedding vector

    wordvec = mx.sym.SliceChannel(data=embed, num_outputs=enc_len + dec_len, squeeze_axis=1)
    maskvec   = mx.sym.SliceChannel(data=mask,  num_outputs=enc_len + dec_len, squeeze_axis=1)

    # numpy array is (2,3) => (2,3)
    # numpy array is (1,3) => (3,)

    hidden_all = []
    for seqidx in range(enc_len + dec_len):
        hidden = wordvec[seqidx]
        mask_in = maskvec[seqidx]

        # stack LSTM
        for i in range(num_lstm_layer):
            dp = 0.0 if i == 0 else dropout

            # encoder RNN
            next_state = lstm_cell(
                num_hidden,
                indata     = hidden,
                mask       = mask_in,
                prev_state = last_states[i],
                param      = enc_param_cells[i] if seqidx < enc_len else dec_param_cells[i],
                seqidx     = seqidx,
                layeridx   = i,
                dropout    = dp
            )

            hidden = next_state.h
            last_states[i] = next_state

        if dropout > 0.0:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)

        if(seqidx >= enc_len):
            hidden_all.append(hidden)

    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    pred = mx.sym.FullyConnected(
        data=hidden_concat,
        num_hidden=num_labels, # num_labels is the index of <PAD> that means this layer will predict 0, 1, ..., num_labels-1
        weight=cls_weight,
        bias=cls_bias,
        name='pred'
    )

    # hidden comes from lstm... output of 1 lstm is a vector...
    # hidden_concat -> every column is the vector of each lstm at each time step

    # softmax = e^(- w' * x) / \sum e^(- w' * x) <-- richard thinks this is softmax
    # softmax = e^(-x) / \sum_i e^(-x_i)

    # output of 1 lstm (top) is a vector of H dimensions...
    # SoftmaxOutput => e^(x) / (1+e^(x)), x is a scalar
    # output of lstm is a vector

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
    def generate_lstm_sym(bucketkey):
        return lstm_unroll(num_layers, bucketkey.enc_len, bucketkey.dec_len, num_hidden, num_labels, dropout)
    return generate_lstm_sym

def get_lstm_init_states(num_layers, num_dim, batch_size=1):
    init_h = [('l%d_init_h' % i, (batch_size, num_dim)) for i in range(num_layers)]
    init_c = [('l%d_init_c' % i, (batch_size, num_dim)) for i in range(num_layers)]
    init_states = init_h + init_c
    return init_states
