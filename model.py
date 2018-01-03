import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import os
import numpy as np
import scipy.io
import theano
import theano.tensor as T
import codecs
import cPickle

#from theano.compile.debugmode import DebugMode

floatX = theano.config.floatX
theano.config.exception_verbosity='high'
#theano.config.optimizer='fast_compile'


class Model(object):
    """
    Network architecture.
    """
    def __init__(self, parameters=None, models_path=None, model_name=None, model_path=None):
        """
        Initialize the model. We either provide the parameters and a path where
        we store the models, or the location of a trained model.
        """
        if model_path is None:
            assert parameters and models_path
            # Create a name based on the parameters
            self.parameters = parameters
            self.name = model_name
            # Model location
            model_path = os.path.join(models_path, self.name)
            self.model_path = model_path
            self.parameters_path = os.path.join(model_path, 'parameters.pkl')
            self.mappings_path = os.path.join(model_path, 'mappings.pkl')
            # Create directory for the model if it does not exist
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            # Save the parameters to disk
            with open(self.parameters_path, 'wb') as f:
                self.parameters = cPickle.dump(parameters, f)
        else:
            assert parameters is None and models_path is None
            # Model location
            self.model_path = model_path
            self.parameters_path = os.path.join(model_path, 'parameters.pkl')
            self.mappings_path = os.path.join(model_path, 'mappings.pkl')
            # Load the parameters and the mappings from disk
            with open(self.parameters_path, 'rb') as f:
                self.parameters = cPickle.load(f)
            self.reload_mappings()
        self.components = {}

    def save_mappings(self, id_to_word, id_to_slb, id_to_char, id_to_pos, id_to_tag):
        """
        We need to save the mappings if we want to use the model later.
        """
        self.id_to_word = id_to_word
        self.id_to_slb = id_to_slb
        self.id_to_char = id_to_char
        self.id_to_pos = id_to_pos
        self.id_to_tag = id_to_tag
        with open(self.mappings_path, 'wb') as f:
            mappings = {
                'id_to_word': self.id_to_word,
                'id_to_slb': self.id_to_slb,
                'id_to_char': self.id_to_char,
                'id_to_pos': self.id_to_pos,
                'id_to_tag': self.id_to_tag,
            }
            cPickle.dump(mappings, f)

    def reload_mappings(self):
        """
        Load mappings from disk.
        """
        with open(self.mappings_path, 'rb') as f:
            mappings = cPickle.load(f)
        self.id_to_word = mappings['id_to_word']
        self.id_to_slb = mappings['id_to_slb']
        self.id_to_char = mappings['id_to_char']
        self.id_to_pos = mappings['id_to_pos']
        self.id_to_tag = mappings['id_to_tag']

    def add_component(self, param):
        """
        Add a new parameter to the network.
        """
        if param.name in self.components:
            raise Exception('The network already has a parameter "%s"!'
                            % param.name)
        self.components[param.name] = param

    def save(self):
        """
        Write components values to disk.
        """
        for name, param in self.components.items():
            param_path = os.path.join(self.model_path, "%s.mat" % name)
            if hasattr(param, 'params'):
                param_values = {p.name: p.get_value() for p in param.params}
            else:
                param_values = {name: param.get_value()}
            scipy.io.savemat(param_path, param_values)

    def reload(self):
        """
        Load components values from disk.
        """
        for name, param in self.components.items():
            param_path = os.path.join(self.model_path, "%s.mat" % name)
            param_values = scipy.io.loadmat(param_path)
            if hasattr(param, 'params'):
                for p in param.params:
                    set_values(p.name, p, param_values[p.name])
            else:
                set_values(name, param, param_values[name])

    def build(self,
              dropout,
              char_dim, char_lstm_dim, char_bidirect,
              slb_dim, slb_lstm_dim, slb_bidirect,
              word_dim, word_lstm_dim, word_bidirect,
              lr_method, pre_emb, crf,
              pos_dim, lexicon_dim,
              training=True,
              **kwargs
              ):
        """
        Build the network.
        """
        # Training parameters
        n_words = len(self.id_to_word)
        n_tags = len(self.id_to_tag)

        # Number of features
        if slb_dim:
            n_slbs = len(self.id_to_slb)
        if char_dim:
            n_chars = len(self.id_to_char)
        if pos_dim:
            n_pos = len(self.id_to_pos)+2
        if lexicon_dim:
            n_lex = lexicon_dim

        # Network variables
        is_train = T.iscalar('is_train')
        word_ids = T.ivector(name='word_ids')
        if slb_dim:
            slb_for_ids = T.imatrix(name='slb_for_ids')
        if slb_lstm_dim:
            slb_rev_ids = T.imatrix(name='slb_rev_ids')
            if slb_bidirect:
                slb_pos_ids = T.ivector(name='slb_pos_ids')
        if char_dim:
            char_for_ids = T.imatrix(name='char_for_ids')
        if char_lstm_dim:
            char_rev_ids = T.imatrix(name='char_rev_ids')
            if char_bidirect:
                char_pos_ids = T.ivector(name='char_pos_ids')
        if pos_dim:
            pos_ids = T.ivector(name='pos_ids')
        if lexicon_dim:
            lex_ids = T.fmatrix(name='lex_ids')
        tag_ids = T.ivector(name='tag_ids')

        # Sentence length
        s_len = (word_ids if word_dim else char_pos_ids).shape[0]

        # Final input (all word features)
        input_dim = 0
        inputs = []

        #
        # Word inputs
        #
        if word_dim:
            input_dim += word_dim
            word_layer = EmbeddingLayer(n_words, word_dim, name='word_layer')
            word_input = word_layer.link(word_ids)
            inputs.append(word_input)

            # Initialize with pretrained embeddings
            if pre_emb and training:
                new_weights = word_layer.embeddings.get_value()
                print 'Loading pretrained embeddings...'
                pretrained = {}
                emb_invalid = 0
                for i, line in enumerate(codecs.open(pre_emb, 'r', 'utf-8')):
                    line = line.rstrip().split()
                    if len(line) == word_dim + 1:
                        pretrained[line[0]] = np.array([float(x) for x in line[1:]]).astype(np.float32)
                    else:
                        emb_invalid += 1
                # if emb_invalid > 0:
                #     print 'WARNING: %i invalid lines' % emb_invalid
                # Lookup table initialization
                for i in xrange(n_words):
                    word = self.id_to_word[i]
                    if word in pretrained:
                        # print word
                        new_weights[i] = pretrained[word]
                word_layer.embeddings.set_value(new_weights)

        #
        # Syllable inputs
        #
        if slb_dim:
            slb_layer = EmbeddingLayer(n_slbs, slb_dim, name='slb_layer')
            if slb_lstm_dim:
                input_dim += slb_lstm_dim
                slb_lstm_for = LSTM(slb_dim, slb_lstm_dim, with_batch=True, name='slb_lstm_for')
                slb_lstm_rev = LSTM(slb_dim, slb_lstm_dim, with_batch=True, name='slb_lstm_rev')
                slb_lstm_for.link(slb_layer.link(slb_for_ids))
                slb_lstm_rev.link(slb_layer.link(slb_rev_ids))
                slb_for_input = slb_lstm_for.h.dimshuffle((1, 0, 2))[T.arange(s_len), slb_pos_ids]
                slb_rev_input = slb_lstm_rev.h.dimshuffle((1, 0, 2))[T.arange(s_len), slb_pos_ids]
                inputs.append(slb_for_input)
                if slb_bidirect:
                    inputs.append(slb_rev_input)
                    input_dim += slb_lstm_dim

        #
        # Chars inputs
        #
        if char_dim:
            char_layer = EmbeddingLayer(n_chars, char_dim, name='char_layer')
            if char_lstm_dim:
                input_dim += char_lstm_dim
                char_lstm_for = LSTM(char_dim, char_lstm_dim, with_batch=True, name='char_lstm_for')
                char_lstm_rev = LSTM(char_dim, char_lstm_dim, with_batch=True, name='char_lstm_rev')
                char_lstm_for.link(char_layer.link(char_for_ids))
                char_lstm_rev.link(char_layer.link(char_rev_ids))
                char_for_input = char_lstm_for.h.dimshuffle((1, 0, 2))[T.arange(s_len), char_pos_ids]
                char_rev_input = char_lstm_rev.h.dimshuffle((1, 0, 2))[T.arange(s_len), char_pos_ids]
                inputs.append(char_for_input)
                if char_bidirect:
                    inputs.append(char_rev_input)
                    input_dim += char_lstm_dim

        #
        # PoS & Lexicon feature
        #
        if pos_dim:
            input_dim += pos_dim
            pos_layer = EmbeddingLayer(n_pos, pos_dim, name='pos_layer')
            inputs.append(pos_layer.link(pos_ids))

        if lexicon_dim:
            input_dim += lexicon_dim
            lex_layer = HiddenLayer(n_lex, lexicon_dim, name='lex_layer', activation=None)
            inputs.append(lex_layer.link(lex_ids))

        # Prepare final input
        if len(inputs) != 1:
            inputs = T.concatenate(inputs, axis=1)
        else:
            inputs = inputs[0]

        #
        # Dropout on final input
        #
        if dropout:
            dropout_layer = DropoutLayer(p=dropout)
            input_train = dropout_layer.link(inputs)
            input_test = (1 - dropout) * inputs
            inputs = T.switch(T.neq(is_train, 0), input_train, input_test)

        # LSTM for words
        word_lstm_for = LSTM(input_dim, word_lstm_dim, with_batch=False, name='word_lstm_for')
        word_lstm_rev = LSTM(input_dim, word_lstm_dim, with_batch=False, name='word_lstm_rev')
        word_lstm_for.link(inputs)
        word_lstm_rev.link(inputs[::-1, :])
        word_for_output = word_lstm_for.h
        word_rev_output = word_lstm_rev.h[::-1, :]
        if word_bidirect:
            final_output = T.concatenate([word_for_output, word_rev_output], axis=1)
            tanh_layer = HiddenLayer(2 * word_lstm_dim, word_lstm_dim, name='tanh_layer', activation='tanh')
            final_output = tanh_layer.link(final_output)
        else:
            final_output = word_for_output

        # Sentence to Named Entity tags - Score
        final_layer = HiddenLayer(word_lstm_dim, n_tags, name='final_layer',
                                  activation=(None if crf else 'softmax'))
        tags_scores = final_layer.link(final_output)

        # No CRF
        if not crf:
            cost = T.nnet.categorical_crossentropy(tags_scores, tag_ids).mean()
        # CRF
        else:
            transitions = shared((n_tags + 2, n_tags + 2), 'transitions')
            small = -1000
            b_s = np.array([[small] * n_tags + [0, small]]).astype(np.float32)
            e_s = np.array([[small] * n_tags + [small, 0]]).astype(np.float32)
            observations = T.concatenate([tags_scores, small * T.ones((s_len, 2))], axis=1)
            observations = T.concatenate([b_s, observations, e_s], axis=0)

            # Score from tags
            real_path_score = tags_scores[T.arange(s_len), tag_ids].sum()

            # Score from transitions
            b_id = theano.shared(value=np.array([n_tags], dtype=np.int32))
            e_id = theano.shared(value=np.array([n_tags + 1], dtype=np.int32))
            padded_tags_ids = T.concatenate([b_id, tag_ids, e_id], axis=0)
            real_path_score += transitions[
                padded_tags_ids[T.arange(s_len + 1)],
                padded_tags_ids[T.arange(s_len + 1) + 1]
            ].sum()

            all_paths_scores = forward(observations, transitions)
            cost = - (real_path_score - all_paths_scores)

        # Network parameters
        params = []
        if word_dim:
            self.add_component(word_layer)
            params.extend(word_layer.params)
        if slb_dim:
            self.add_component(slb_layer)
            params.extend(slb_layer.params)
            if slb_lstm_dim:
                self.add_component(slb_lstm_for)
                params.extend(slb_lstm_for.params)
                if slb_bidirect:
                    self.add_component(slb_lstm_rev)
                    params.extend(slb_lstm_rev.params)
        if char_dim:
            self.add_component(char_layer)
            params.extend(char_layer.params)
            if char_lstm_dim:
                self.add_component(char_lstm_for)
                params.extend(char_lstm_for.params)
                if char_bidirect:
                    self.add_component(char_lstm_rev)
                    params.extend(char_lstm_rev.params)
        self.add_component(word_lstm_for)
        params.extend(word_lstm_for.params)
        if word_bidirect:
            self.add_component(word_lstm_rev)
            params.extend(word_lstm_rev.params)
        if pos_dim:
            self.add_component(pos_layer)
            params.extend(pos_layer.params)
        if lexicon_dim:
            self.add_component(lex_layer)
            params.extend(lex_layer.params)
        self.add_component(final_layer)
        params.extend(final_layer.params)
        if crf:
            self.add_component(transitions)
            params.append(transitions)
        if word_bidirect:
            self.add_component(tanh_layer)
            params.extend(tanh_layer.params)

        # Prepare train and eval inputs
        eval_inputs = []
        if word_dim:
            eval_inputs.append(word_ids)
        if slb_dim:
            eval_inputs.append(slb_for_ids)
            if slb_lstm_dim:
                if slb_bidirect:
                    eval_inputs.append(slb_rev_ids)
                eval_inputs.append(slb_pos_ids)
        if char_dim:
            eval_inputs.append(char_for_ids)
            if char_lstm_dim:
                if char_bidirect:
                    eval_inputs.append(char_rev_ids)
                eval_inputs.append(char_pos_ids)
        if pos_dim:
            eval_inputs.append(pos_ids)
        if lexicon_dim:
            eval_inputs.append(lex_ids)
        train_inputs = eval_inputs + [tag_ids]

        # Parse optimization method parameters
        if "-" in lr_method:
            lr_method_name = lr_method[:lr_method.find('-')]
            lr_method_parameters = {}
            for x in lr_method[lr_method.find('-') + 1:].split('-'):
                split = x.split('_')
                assert len(split) == 2
                lr_method_parameters[split[0]] = float(split[1])
        else:
            lr_method_name = lr_method
            lr_method_parameters = {}

        # Compile training function
        print 'Compiling...'
        if training:
            updates = Optimization(clip=5.0).get_updates(lr_method_name, cost, params, **lr_method_parameters)
            f_train = theano.function(
                inputs=train_inputs,
                outputs=cost,
                updates=updates,
                givens=({is_train: np.cast['int32'](1)} if dropout else {})
            )
        else:
            f_train = None

        # Compile evaluation function
        if not crf:
            f_eval = theano.function(
                inputs=eval_inputs,
                outputs=tags_scores,
                givens=({is_train: np.cast['int32'](0)} if dropout else {})
            )
        else:
            f_eval = theano.function(
                inputs=eval_inputs,
                outputs=forward(observations, transitions, viterbi=True,
                                return_alpha=False, return_best_sequence=True),
                givens=({is_train: np.cast['int32'](0)} if dropout else {})
                #, mode=DebugMode(check_isfinite=True)
            )

        return f_train, f_eval

class HiddenLayer(object):
    """
    Hidden layer with or without bias.
    Input: tensor of dimension (dims*, input_dim)
    Output: tensor of dimension (dims*, output_dim)
    """
    def  __init__(self, input_dim, output_dim, bias=True, activation='sigmoid',
                 name='hidden_layer'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.name = name
        if activation is None:
            self.activation = None
        elif activation == 'tanh':
            self.activation = T.tanh
        elif activation == 'sigmoid':
            self.activation = T.nnet.sigmoid
        elif activation == 'softmax':
            self.activation = T.nnet.softmax
        else:
            raise Exception("Unknown activation function: " % activation)

        # Initialize weights and bias
        self.weights = shared((input_dim, output_dim), name + '__weights')
        self.bias = shared((output_dim,), name + '__bias')

        # Define parameters
        if self.bias:
            self.params = [self.weights, self.bias]
        else:
            self.params = [self.weights]

    def link(self, input):
        """
        The input has to be a tensor with the right
        most dimension equal to input_dim.
        """
        self.input = input
        self.linear_output = T.dot(self.input, self.weights)
        if self.bias:
            self.linear_output = self.linear_output + self.bias
        if self.activation is None:
            self.output = self.linear_output
        else:
            self.output = self.activation(self.linear_output)
        return self.output


class EmbeddingLayer(object):
    """
    Embedding layer: word embeddings representations
    Input: tensor of dimension (dim*) with values in range(0, input_dim)
    Output: tensor of dimension (dim*, output_dim)
    """

    def __init__(self, input_dim, output_dim, name='embedding_layer'):
        """
        Typically, input_dim is the vocabulary size,
        and output_dim the embedding dimension.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name

        # Randomly generate weights
        self.embeddings = shared((input_dim, output_dim),
                                 self.name + '__embeddings')

        # Define parameters
        self.params = [self.embeddings]

    def link(self, input):
        """
        Return the embeddings of the given indexes.
        Input: tensor of shape (dim*)
        Output: tensor of shape (dim*, output_dim)
        """
        self.input = input
        self.output = self.embeddings[self.input]
        return self.output


class DropoutLayer(object):
    """
    Dropout layer. Randomly set to 0 values of the input
    with probability p.
    """
    def __init__(self, p=0.5, name='dropout_layer'):
        """
        p has to be between 0 and 1 (1 excluded).
        p is the probability of dropping out a unit, so
        setting p to 0 is equivalent to have an identity layer.
        """
        assert 0. <= p < 1.
        self.p = p
        self.rng = T.shared_randomstreams.RandomStreams(seed=123456)
        self.name = name

    def link(self, input):
        """
        Dropout link: we just apply mask to the input.
        """
        if self.p > 0:
            mask = self.rng.binomial(n=1, p=1-self.p, size=input.shape,
                                     dtype=theano.config.floatX)
            self.output = input * mask
        else:
            self.output = input

        return self.output


class LSTM(object):
    """
    Long short-term memory (LSTM). Can be used with or without batches.
    Without batches:
        Input: matrix of dimension (sequence_length, input_dim)
        Output: vector of dimension (output_dim)
    With batches:
        Input: tensor3 of dimension (batch_size, sequence_length, input_dim)
        Output: matrix of dimension (batch_size, output_dim)
    """
    def __init__(self, input_dim, hidden_dim, with_batch=True, name='LSTM'):
        """
        Initialize neural network.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.with_batch = with_batch
        self.name = name

        # Input gate weights
        self.w_xi = shared((input_dim, hidden_dim), name + '__w_xi')
        self.w_hi = shared((hidden_dim, hidden_dim), name + '__w_hi')
        self.w_ci = shared((hidden_dim, hidden_dim), name + '__w_ci')

        # Forget gate weights
        # self.w_xf = shared((input_dim, hidden_dim), name + '__w_xf')
        # self.w_hf = shared((hidden_dim, hidden_dim), name + '__w_hf')
        # self.w_cf = shared((hidden_dim, hidden_dim), name + '__w_cf')

        # Output gate weights
        self.w_xo = shared((input_dim, hidden_dim), name + '__w_xo')
        self.w_ho = shared((hidden_dim, hidden_dim), name + '__w_ho')
        self.w_co = shared((hidden_dim, hidden_dim), name + '__w_co')

        # Cell weights
        self.w_xc = shared((input_dim, hidden_dim), name + '__w_xc')
        self.w_hc = shared((hidden_dim, hidden_dim), name + '__w_hc')

        # Initialize the bias vectors, c_0 and h_0 to zero vectors
        self.b_i = shared((hidden_dim,), name + '__b_i')
        # self.b_f = shared((hidden_dim,), name + '__b_f')
        self.b_c = shared((hidden_dim,), name + '__b_c')
        self.b_o = shared((hidden_dim,), name + '__b_o')
        self.c_0 = shared((hidden_dim,), name + '__c_0')
        self.h_0 = shared((hidden_dim,), name + '__h_0')

        # Define parameters
        self.params = [self.w_xi, self.w_hi, self.w_ci,
                       # self.w_xf, self.w_hf, self.w_cf,
                       self.w_xo, self.w_ho, self.w_co,
                       self.w_xc, self.w_hc,
                       self.b_i, self.b_c, self.b_o,  # self.b_f,
                       self.c_0, self.h_0]

    def link(self, input):
        """
        Propagate the input through the network and return the last hidden
        vector. The whole sequence is also accessible via self.h, but
        where self.h of shape (sequence_length, batch_size, output_dim)
        """
        def recurrence(x_t, c_tm1, h_tm1):
            i_t = T.nnet.sigmoid(T.dot(x_t, self.w_xi) +
                                 T.dot(h_tm1, self.w_hi) +
                                 T.dot(c_tm1, self.w_ci) +
                                 self.b_i)
            # f_t = T.nnet.sigmoid(T.dot(x_t, self.w_xf) +
            #                      T.dot(h_tm1, self.w_hf) +
            #                      T.dot(c_tm1, self.w_cf) +
            #                      self.b_f)
            c_t = ((1 - i_t) * c_tm1 + i_t * T.tanh(T.dot(x_t, self.w_xc) +
                   T.dot(h_tm1, self.w_hc) + self.b_c))
            o_t = T.nnet.sigmoid(T.dot(x_t, self.w_xo) +
                                 T.dot(h_tm1, self.w_ho) +
                                 T.dot(c_t, self.w_co) +
                                 self.b_o)
            h_t = o_t * T.tanh(c_t)
            return [c_t, h_t]

        # If we use batches, we have to permute the first and second dimension.
        if self.with_batch:
            self.input = input.dimshuffle(1, 0, 2)
            outputs_info = [T.alloc(x, self.input.shape[1], self.hidden_dim)
                            for x in [self.c_0, self.h_0]]
        else:
            self.input = input
            outputs_info = [self.c_0, self.h_0]

        [_, h], _ = theano.scan(
            fn=recurrence,
            sequences=self.input,
            outputs_info=outputs_info,
            n_steps=self.input.shape[0]
        )
        self.h = h
        self.output = h[-1]

        return self.output


def log_sum_exp(x, axis=None):
    """
    Sum probabilities in the log-space.
    """
    xmax = x.max(axis=axis, keepdims=True)
    xmax_ = x.max(axis=axis)
    return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))


def forward(observations, transitions, viterbi=False,
            return_alpha=False, return_best_sequence=False):
    """
    Takes as input:
        - observations, sequence of shape (n_steps, n_classes)
        - transitions, sequence of shape (n_classes, n_classes)
    Probabilities must be given in the log space.
    Compute alpha, matrix of size (n_steps, n_classes), such that
    alpha[i, j] represents one of these 2 values:
        - the probability that the real path at node i ends in j
        - the maximum probability of a path finishing in j at node i (Viterbi)
    Returns one of these 2 values:
        - alpha
        - the final probability, which can be:
            - the sum of the probabilities of all paths
            - the probability of the best path (Viterbi)
    """
    assert not return_best_sequence or (viterbi and not return_alpha)

    def recurrence(obs, previous, transitions):
        previous = previous.dimshuffle(0, 'x')
        obs = obs.dimshuffle('x', 0)
        if viterbi:
            scores = previous + obs + transitions
            out = scores.max(axis=0)
            if return_best_sequence:
                out2 = scores.argmax(axis=0)
                return out, out2
            else:
                return out
        else:
            return log_sum_exp(previous + obs + transitions, axis=0)

    initial = observations[0]
    alpha, _ = theano.scan(
        fn=recurrence,
        outputs_info=(initial, None) if return_best_sequence else initial,
        sequences=[observations[1:]],
        non_sequences=transitions
    )

    if return_alpha:
        return alpha
    elif return_best_sequence:
        sequence, _ = theano.scan(
            fn=lambda beta_i, previous: beta_i[previous],
            outputs_info=T.cast(T.argmax(alpha[0][-1]), 'int32'),
            sequences=T.cast(alpha[1][::-1], 'int32')
        )
        sequence = T.concatenate([sequence[::-1], [T.argmax(alpha[0][-1])]])
        return sequence
    else:
        if viterbi:
            return alpha[-1].max(axis=0)
        else:
            return log_sum_exp(alpha[-1], axis=0)

class Optimization:

    def __init__(self, clip=None):
        """
        Initialization
        """
        self.clip = clip

    def get_gradients(self, cost, params):
        """
        Compute the gradients, and clip them if required.
        """
        if self.clip is None:
            return T.grad(cost, params)
        else:
            assert self.clip > 0
            return T.grad(
                theano.gradient.grad_clip(cost, -1 * self.clip, self.clip),
                params
            )

    def get_updates(self, method, cost, params, *args, **kwargs):
        """
        Compute the updates for different optimizers.
        """
        if method == 'sgd':
            updates = self.sgd(cost, params, **kwargs)
        elif method == 'sgdmomentum':
            updates = self.sgdmomentum(cost, params, **kwargs)
        elif method == 'adagrad':
            updates = self.adagrad(cost, params, **kwargs)
        elif method == 'adadelta':
            updates = self.adadelta(cost, params, **kwargs)
        elif method == 'adam':
            updates = self.adam(cost, params, **kwargs)
        elif method == 'rmsprop':
            updates = self.rmsprop(cost, params, **kwargs)
        else:
            raise("Not implemented learning method: %s" % method)
        return updates

    def sgd(self, cost, params, lr=0.01):
        """
        Stochatic gradient descent.
        """
        lr = theano.shared(np.float32(lr).astype(floatX))

        gradients = self.get_gradients(cost, params)

        updates = []
        for p, g in zip(params, gradients):
            updates.append((p, p - lr * g))

        return updates

    def sgdmomentum(self, cost, params, lr=0.01, momentum=0.9):
        """
        Stochatic gradient descent with momentum. Momentum has to be in [0, 1)
        """
        # Check that the momentum is a correct value
        assert 0 <= momentum < 1

        lr = theano.shared(np.float32(lr).astype(floatX))
        momentum = theano.shared(np.float32(momentum).astype(floatX))

        gradients = self.get_gradients(cost, params)
        velocities = [theano.shared(np.zeros_like(param.get_value(borrow=True)).astype(floatX)) for param in params]

        updates = []
        for param, gradient, velocity in zip(params, gradients, velocities):
            new_velocity = momentum * velocity - lr * gradient
            updates.append((velocity, new_velocity))
            updates.append((param, param + new_velocity))
        return updates

    def adagrad(self, cost, params, lr=1.0, epsilon=1e-6):
        """
        Adagrad. Based on http://www.ark.cs.cmu.edu/cdyer/adagrad.pdf
        """
        lr = theano.shared(np.float32(lr).astype(floatX))
        epsilon = theano.shared(np.float32(epsilon).astype(floatX))

        gradients = self.get_gradients(cost, params)
        gsums = [theano.shared(np.zeros_like(param.get_value(borrow=True)).astype(floatX)) for param in params]

        updates = []
        for param, gradient, gsum in zip(params, gradients, gsums):
            new_gsum = gsum + gradient ** 2.
            updates.append((gsum, new_gsum))
            updates.append((param, param - lr * gradient / (T.sqrt(gsum + epsilon))))
        return updates

    def adadelta(self, cost, params, rho=0.95, epsilon=1e-6):
        """
        Adadelta. Based on:
        http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf
        """
        rho = theano.shared(np.float32(rho).astype(floatX))
        epsilon = theano.shared(np.float32(epsilon).astype(floatX))

        gradients = self.get_gradients(cost, params)
        accu_gradients = [theano.shared(np.zeros_like(param.get_value(borrow=True)).astype(floatX)) for param in params]
        accu_deltas = [theano.shared(np.zeros_like(param.get_value(borrow=True)).astype(floatX)) for param in params]

        updates = []
        for param, gradient, accu_gradient, accu_delta in zip(params, gradients, accu_gradients, accu_deltas):
            new_accu_gradient = rho * accu_gradient + (1. - rho) * gradient ** 2.
            delta_x = - T.sqrt((accu_delta + epsilon) / (new_accu_gradient + epsilon)) * gradient
            new_accu_delta = rho * accu_delta + (1. - rho) * delta_x ** 2.
            updates.append((accu_gradient, new_accu_gradient))
            updates.append((accu_delta, new_accu_delta))
            updates.append((param, param + delta_x))
        return updates

    def adam(self, cost, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Adam. Based on http://arxiv.org/pdf/1412.6980v4.pdf
        """
        updates = []
        gradients = self.get_gradients(cost, params)

        t = theano.shared(np.float32(1.).astype(floatX))

        for param, gradient in zip(params, gradients):
            value = param.get_value(borrow=True)
            m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
            v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)

            m = beta1 * m_prev + (1. - beta1) * gradient
            v = beta2 * v_prev + (1. - beta2) * gradient ** 2.
            m_hat = m / (1. - beta1 ** t)
            v_hat = v / (1. - beta2 ** t)
            theta = param - (lr * m_hat) / (T.sqrt(v_hat) + epsilon)

            updates.append((m_prev, m))
            updates.append((v_prev, v))
            updates.append((param, theta))

        updates.append((t, t + 1.))
        return updates

    def rmsprop(self, cost, params, lr=0.001, rho=0.9, eps=1e-6):
        """
        RMSProp.
        """
        lr = theano.shared(np.float32(lr).astype(floatX))

        gradients = self.get_gradients(cost, params)
        accumulators = [theano.shared(np.zeros_like(p.get_value()).astype(np.float32)) for p in params]

        updates = []

        for param, gradient, accumulator in zip(params, gradients, accumulators):
            new_accumulator = rho * accumulator + (1 - rho) * gradient ** 2
            updates.append((accumulator, new_accumulator))

            new_param = param - lr * gradient / T.sqrt(new_accumulator + eps)
            updates.append((param, new_param))

        return updates

def set_values(name, param, pretrained):
    """
    Initialize a network parameter with pretrained values.
    We check that sizes are compatible.
    """
    param_value = param.get_value()
    if pretrained.size != param_value.size:
        raise Exception(
            "Size mismatch for parameter %s. Expected %i, found %i."
            % (name, param_value.size, pretrained.size)
        )
    param.set_value(np.reshape(
        pretrained, param_value.shape
    ).astype(np.float32))


def shared(shape, name):
    """
    Create a shared object of a numpy array.
    """
    if len(shape) == 1:
        value = np.zeros(shape)  # bias are initialized with zeros
    else:
        drange = np.sqrt(6. / (np.sum(shape)))
        value = drange * np.random.uniform(low=-1.0, high=1.0, size=shape)
    return theano.shared(value=value.astype(theano.config.floatX), name=name)
