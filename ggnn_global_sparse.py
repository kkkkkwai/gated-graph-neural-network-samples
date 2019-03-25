#!/usr/bin/env/python
"""
Usage:
    chem_tensorflow_sparse.py [options]

Options:
    -h --help                Show this screen.
    --config-file FILE       Hyperparameter configuration file path (in JSON format).
    --config CONFIG          Hyperparameter configuration dictionary (in JSON format).
    --log_dir DIR            Log dir name.
    --data_dir DIR           Data dir name.
    --restore FILE           File to restore weights from.
    --freeze-graph-model     Freeze weights of graph model components.
"""
import os
from typing import List, Tuple, Dict, Sequence, Any

from docopt import docopt
from collections import defaultdict, namedtuple
import numpy as np
import tensorflow as tf
import sys, traceback
import pdb
import re

from ggnn_global import ProgramModel
from utils import glorot_init, SMALL_NUMBER, EDGE_TYPE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

GGNNWeights = namedtuple('GGNNWeights', ['edge_weights',
                                         'edge_biases',
                                         'edge_type_attention_weights',
                                         'rnn_cells', ])


class SparseGGNNChemModel(ProgramModel):
    def __init__(self, args):
        super().__init__(args)

    @classmethod
    def default_params(cls):
        params = dict(super().default_params())
        params.update({
            'batch_size': 10,
            'use_edge_bias': False,
            'use_propagation_attention': False,
            'use_edge_msg_avg_aggregation': True,
            'residual_connections': {  # For layer i, specify list of layers whose output is added as an input
                '2': [0]
            },

            'layer_timesteps': [4, 4, 2],  # number of layers & propagation steps per layer

            'graph_rnn_cell': 'GRU',  # GRU, CudnnCompatibleGRUCell, or RNN
            'graph_rnn_activation': 'tanh',  # tanh, ReLU
            'graph_state_dropout_keep_prob': 1.,
            'task_sample_ratios': 1.0,
            'edge_weight_dropout_keep_prob': .8,

            'keep_letter_only': False
        })
        return params

    def prepare_specific_graph_model(self) -> None:
        word_dim = self.params['word_embedding_size']
        type_dim = self.params['type_embedding_size']
        h_dim = self.params['hidden_size']
        self.placeholders['initial_word_ids'] = tf.placeholder(tf.int32, [None, self.params['max_node_length']],
                                                               name='word_ids')
        self.placeholders['initial_type_ids'] = tf.placeholder(tf.int32, [None, self.params['max_node_length']],
                                                               name='type_ids')
        self.placeholders['candidates'] = tf.placeholder(tf.float32, [None, 1], name='candidates')
        self.placeholders['slots'] = tf.placeholder(tf.int32, [None], name='slots')
        self.placeholders['num_candidates_per_graph'] = tf.placeholder(tf.int32, [self.params['batch_size']],
                                                                       name='num_candidates')
        self.placeholders['adjacency_lists'] = [tf.placeholder(tf.int32, [None, 2], name='adjacency_e%s' % e)
                                                for e in range(self.num_edge_types)]
        self.placeholders['num_incoming_edges_per_type'] = tf.placeholder(tf.float32, [None, self.num_edge_types],
                                                                          name='num_incoming_edges_per_type')
        self.placeholders['graph_nodes_list'] = tf.placeholder(tf.int32, [None], name='graph_nodes_list')
        self.placeholders['graph_state_keep_prob'] = tf.placeholder(tf.float32, None, name='graph_state_keep_prob')
        self.placeholders['edge_weight_dropout_keep_prob'] = tf.placeholder(tf.float32, None,
                                                                            name='edge_weight_dropout_keep_prob')

        activation_name = self.params['graph_rnn_activation'].lower()
        if activation_name == 'tanh':
            activation_fun = tf.nn.tanh
        elif activation_name == 'relu':
            activation_fun = tf.nn.relu
        else:
            raise Exception("Unknown activation function type '%s'." % activation_name)

        # create embeddings
        with tf.variable_scope('embedding_layers'):
            self.word_embedding = tf.Variable(glorot_init([len(self.vocabs) + 2, word_dim]), name='word_embed')
            self.type_embedding = tf.Variable(glorot_init([len(self.type_hierarchy[0]['types']) + 2, type_dim]),
                                              name='type_embed')
            self.init_node_weights = tf.Variable(glorot_init([word_dim + type_dim + 1, h_dim]), name='embed_layer_w')
            self.init_node_bias = tf.Variable(np.zeros([h_dim]), name='embed_layer_b', dtype=tf.float32)

        # Generate per-layer values for edge weights, biases and gated units:
        self.weights = {}  # Used by super-class to place generic things
        self.gnn_weights = GGNNWeights([], [], [], [])
        for layer_idx in range(len(self.params['layer_timesteps'])):
            with tf.variable_scope('gnn_layer_%i' % layer_idx):
                edge_weights = tf.Variable(glorot_init([self.num_edge_types * h_dim, h_dim]),
                                           name='gnn_edge_weights_%i' % layer_idx)
                edge_weights = tf.reshape(edge_weights, [self.num_edge_types, h_dim, h_dim])
                edge_weights = tf.nn.dropout(edge_weights, keep_prob=self.placeholders['edge_weight_dropout_keep_prob'])
                self.gnn_weights.edge_weights.append(edge_weights)

                if self.params['use_propagation_attention']:
                    self.gnn_weights.edge_type_attention_weights.append(
                        tf.Variable(np.ones([self.num_edge_types], dtype=np.float32),
                                    name='edge_type_attention_weights_%i' % layer_idx))

                if self.params['use_edge_bias']:
                    self.gnn_weights.edge_biases.append(
                        tf.Variable(np.zeros([self.num_edge_types, h_dim], dtype=np.float32),
                                    name='gnn_edge_biases_%i' % layer_idx))

                cell_type = self.params['graph_rnn_cell'].lower()
                if cell_type == 'gru':
                    cell = tf.nn.rnn_cell.GRUCell(h_dim, activation=activation_fun)
                elif cell_type == 'cudnncompatiblegrucell':
                    assert (activation_name == 'tanh')
                    import tensorflow.contrib.cudnn_rnn as cudnn_rnn
                    cell = cudnn_rnn.CudnnCompatibleGRUCell(h_dim)
                elif cell_type == 'rnn':
                    cell = tf.nn.rnn_cell.BasicRNNCell(h_dim, activation=activation_fun)
                else:
                    raise Exception("Unknown RNN cell type '%s'." % cell_type)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                     state_keep_prob=self.placeholders['graph_state_keep_prob'])
                self.gnn_weights.rnn_cells.append(cell)

    def compute_final_node_representations(self) -> tf.Tensor:

        # get the embeddings and compute initial node representation
        with tf.variable_scope('embedding_layers'):

            word_embed = tf.nn.embedding_lookup(params=self.word_embedding, ids=self.placeholders['initial_word_ids'])
            word_mask = tf.cast(tf.not_equal(self.placeholders['initial_word_ids'], len(self.vocabs) + 1), tf.float32)
            dividend = tf.reduce_sum(word_mask, axis=1) + SMALL_NUMBER
            word_mask = tf.tile(tf.expand_dims(word_mask, -1), [1, 1, self.params['word_embedding_size']])
            dividend = tf.tile(tf.expand_dims(dividend, -1), [1, self.params['word_embedding_size']])
            word_embed = tf.reduce_sum(word_embed * word_mask, axis=1) / dividend

            # get a mask and expand its dimension

            type_embed = tf.nn.embedding_lookup(params=self.type_embedding, ids=self.placeholders['initial_type_ids'])
            type_mask = tf.cast(
                tf.equal(self.placeholders['initial_type_ids'], len(self.type_hierarchy[0]['types']) + 1), tf.float32)
            type_mask = tf.tile(tf.expand_dims(type_mask, -1), [1, 1, self.params['type_embedding_size']])
            type_embed = tf.reduce_max(type_embed - type_mask * 10000, axis=1)

            initial_node_representation = tf.concat([word_embed, type_embed, self.placeholders['candidates']], axis=1)
            initial_node_representation = tf.matmul(initial_node_representation,
                                                    self.init_node_weights) + self.init_node_bias

        node_states_per_layer = []  # one entry per layer (final state of that layer), shape: number of nodes in batch v x D
        node_states_per_layer.append(initial_node_representation)
        num_nodes = tf.shape(self.placeholders['initial_word_ids'], out_type=tf.int32)[0]

        message_targets = []  # list of tensors of message targets of shape [E]
        message_edge_types = []  # list of tensors of edge type of shape [E]
        for edge_type_idx, adjacency_list_for_edge_type in enumerate(self.placeholders['adjacency_lists']):
            edge_targets = adjacency_list_for_edge_type[:, 1]
            message_targets.append(edge_targets)
            message_edge_types.append(tf.ones_like(edge_targets, dtype=tf.int32) * edge_type_idx)
        message_targets = tf.concat(message_targets, axis=0)  # Shape [M]
        message_edge_types = tf.concat(message_edge_types, axis=0)  # Shape [M]

        for (layer_idx, num_timesteps) in enumerate(self.params['layer_timesteps']):
            with tf.variable_scope('gnn_layer_%i' % layer_idx):
                # Used shape abbreviations:
                #   V ~ number of nodes
                #   D ~ state dimension
                #   E ~ number of edges of current type
                #   M ~ number of messages (sum of all E)

                # Extract residual messages, if any:
                layer_residual_connections = self.params['residual_connections'].get(str(layer_idx))
                if layer_residual_connections is None:
                    layer_residual_states = []
                else:
                    layer_residual_states = [node_states_per_layer[residual_layer_idx]
                                             for residual_layer_idx in layer_residual_connections]

                if self.params['use_propagation_attention']:
                    message_edge_type_factors = tf.nn.embedding_lookup(
                        params=self.gnn_weights.edge_type_attention_weights[layer_idx],
                        ids=message_edge_types)  # Shape [M]

                # Record new states for this layer. Initialised to last state, but will be updated below:
                node_states_per_layer.append(node_states_per_layer[-1])
                for step in range(num_timesteps):
                    with tf.variable_scope('timestep_%i' % step):
                        messages = []  # list of tensors of messages of shape [E, D]
                        message_source_states = []  # list of tensors of edge source states of shape [E, D]

                        # Collect incoming messages per edge type
                        for edge_type_idx, adjacency_list_for_edge_type in enumerate(
                                self.placeholders['adjacency_lists']):
                            edge_sources = adjacency_list_for_edge_type[:, 0]
                            edge_source_states = tf.nn.embedding_lookup(params=node_states_per_layer[-1],
                                                                        ids=edge_sources)  # Shape [E, D]
                            all_messages_for_edge_type = tf.matmul(edge_source_states,
                                                                   self.gnn_weights.edge_weights[layer_idx][
                                                                       edge_type_idx])  # Shape [E, D]
                            messages.append(all_messages_for_edge_type)
                            message_source_states.append(edge_source_states)

                        messages = tf.concat(messages, axis=0)  # Shape [M, D]

                        if self.params['use_propagation_attention']:
                            message_source_states = tf.concat(message_source_states, axis=0)  # Shape [M, D]
                            message_target_states = tf.nn.embedding_lookup(params=node_states_per_layer[-1],
                                                                           ids=message_targets)  # Shape [M, D]
                            message_attention_scores = tf.einsum('mi,mi->m', message_source_states,
                                                                 message_target_states)  # Shape [M]
                            message_attention_scores = message_attention_scores * message_edge_type_factors

                            # The following is softmax-ing over the incoming messages per node.
                            # As the number of incoming varies, we can't just use tf.softmax. Reimplement with logsumexp trick:
                            # Step (1): Obtain shift constant as max of messages going into a node
                            message_attention_score_max_per_target = tf.unsorted_segment_max(
                                data=message_attention_scores,
                                segment_ids=message_targets,
                                num_segments=num_nodes)  # Shape [V]
                            # Step (2): Distribute max out to the corresponding messages again, and shift scores:
                            message_attention_score_max_per_message = tf.gather(
                                params=message_attention_score_max_per_target,
                                indices=message_targets)  # Shape [M]
                            message_attention_scores -= message_attention_score_max_per_message
                            # Step (3): Exp, sum up per target, compute exp(score) / exp(sum) as attention prob:
                            message_attention_scores_exped = tf.exp(message_attention_scores)  # Shape [M]
                            message_attention_score_sum_per_target = tf.unsorted_segment_sum(
                                data=message_attention_scores_exped,
                                segment_ids=message_targets,
                                num_segments=num_nodes)  # Shape [V]
                            message_attention_normalisation_sum_per_message = tf.gather(
                                params=message_attention_score_sum_per_target,
                                indices=message_targets)  # Shape [M]
                            message_attention = message_attention_scores_exped / (
                                    message_attention_normalisation_sum_per_message + SMALL_NUMBER)  # Shape [M]
                            # Step (4): Weigh messages using the attention prob:
                            messages = messages * tf.expand_dims(message_attention, -1)

                        incoming_messages = tf.unsorted_segment_sum(data=messages,
                                                                    segment_ids=message_targets,
                                                                    num_segments=num_nodes)  # Shape [V, D]

                        if self.params['use_edge_bias']:
                            incoming_messages += tf.matmul(self.placeholders['num_incoming_edges_per_type'],
                                                           self.gnn_weights.edge_biases[layer_idx])  # Shape [V, D]

                        if self.params['use_edge_msg_avg_aggregation']:
                            num_incoming_edges = tf.reduce_sum(self.placeholders['num_incoming_edges_per_type'],
                                                               keep_dims=True, axis=-1)  # Shape [V, 1]
                            incoming_messages /= num_incoming_edges + SMALL_NUMBER

                        incoming_information = tf.concat(layer_residual_states + [incoming_messages],
                                                         axis=-1)  # Shape [V, D*(1 + num of residual connections)]

                        # pass updated vertex features into RNN cell
                        node_states_per_layer[-1] = self.gnn_weights.rnn_cells[layer_idx](incoming_information,
                                                                                          node_states_per_layer[-1])[
                            1]  # Shape [V, D]

        return node_states_per_layer[-1]

    def regression(self, last_h, regression_transform):
        # last_h: [v x h]
        candidate_ids = tf.squeeze(tf.where(tf.greater(tf.squeeze(self.placeholders['candidates']), 0)))
        slot_ids = self.placeholders['slots']
        context = tf.gather(params=last_h, indices=slot_ids)
        usage = tf.gather(params=last_h, indices=candidate_ids)
        # segments = tf.gather(params=self.placeholders['graph_nodes_list'], indices=candidate_ids)

        final_node_states = tf.concat([context, usage], axis=1)
        outputs = tf.squeeze(regression_transform(final_node_states))

        targets = self.placeholders['target_values']
        output_splits = tf.split(value=outputs, num_or_size_splits=self.placeholders['num_candidates_per_graph'])
        target_splits = tf.split(value=targets, num_or_size_splits=self.placeholders['num_candidates_per_graph'])
        assert len(output_splits) == len(target_splits)

        _loss = []
        _accuracy = []
        for graph_idx, out in enumerate(output_splits):
            _loss.append(tf.losses.softmax_cross_entropy(tf.expand_dims(target_splits[graph_idx], axis=0), tf.expand_dims(out, axis=0)))
            _accuracy.append(tf.cast(tf.equal(tf.argmax(out, axis=0), tf.argmax(target_splits[graph_idx], axis=0)),
                                     tf.float32))
        loss = tf.reduce_mean(tf.stack(_loss))
        accuracy = tf.reduce_mean(tf.stack(_accuracy))

        return accuracy, loss  # [g]

    # ----- Data preprocessing and chunking into minibatches:
    def process_raw_graphs(self, raw_data: Sequence[Any], is_training: bool, _filename) -> Any:
        processed_graphs = []
        print('{} has {} graphs'.format(_filename, len(raw_data)), end='\r')
        for g_idx, d in enumerate(raw_data):
            slot = d.get('SlotDummyNode')
            context_graph = d.get('ContextGraph')
            symbol_candidates = d.get('SymbolCandidates')
            # skip incomplete graph
            if slot is None or context_graph is None or symbol_candidates is None:
                continue
            node_labels = context_graph.get('NodeLabels')
            edges = context_graph.get('Edges')
            node_types = context_graph.get('NodeTypes')
            # if not (node_labels and edges and node_types):
            if node_labels is None or edges is None or node_types is None:
                continue

            num_nodes = len(node_labels)
            (adjacency_lists, num_incoming_edge_per_type) = self.__graph_to_adjacency_lists(edges)
            node_words, node_types = self.__node_preprocess(node_labels,
                                                            node_types, num_nodes)
            candidates, target = self.__candidate_preprocess(symbol_candidates, num_nodes)

            assert num_nodes == len(node_words)
            assert num_nodes == len(node_types)
            processed_graphs.append({"adjacency_lists": adjacency_lists,
                                     "num_incoming_edge_per_type": num_incoming_edge_per_type,
                                     "node_labels": node_words,
                                     "node_types": node_types,
                                     "slot": slot,
                                     "candidates": candidates,
                                     "num_candidates": len(symbol_candidates),
                                     "target": target})
            if is_training:
                np.random.shuffle(processed_graphs)

        return processed_graphs

    def __graph_to_adjacency_lists(self, graph) -> Tuple[Dict[int, Any], Dict[int, Dict[int, int]]]:
        adj_lists = defaultdict(list)
        num_incoming_edges_dicts_per_type = defaultdict(lambda: defaultdict(lambda: 0))
        for e_id, e_type in enumerate(EDGE_TYPE):
            if e_type in graph:
                # TODO: simplify the process
                for src, dest in graph[e_type]:
                    fwd_edge_type = e_id  # Make edges start from 0
                    adj_lists[fwd_edge_type].append((src, dest))
                    num_incoming_edges_dicts_per_type[fwd_edge_type][dest] += 1
                    if self.params['tie_fwd_bkwd']:
                        adj_lists[fwd_edge_type].append((dest, src))
                        num_incoming_edges_dicts_per_type[fwd_edge_type][src] += 1

        final_adj_lists = {e: sorted(lm) for e, lm in adj_lists.items()}

        # Add backward edges as an additional edge type that goes backwards:
        if not (self.params['tie_fwd_bkwd']):
            for (edge_type, edges) in adj_lists.items():
                bwd_edge_type = self.num_original_edge_types + edge_type
                final_adj_lists[bwd_edge_type] = sorted((y, x) for (x, y) in edges)
                for (x, y) in edges:
                    num_incoming_edges_dicts_per_type[bwd_edge_type][y] += 1

        return final_adj_lists, num_incoming_edges_dicts_per_type

    # extract word and type information from the node
    def __node_preprocess(self, labels, types, num_nodes):
        # TODO: handle negative ids
        node_words = np.full([num_nodes, self.params['max_node_length']], fill_value=len(self.vocabs) + 1)
        node_types = np.full([num_nodes, self.params['max_node_length']],
                             fill_value=len(self.type_hierarchy[0]['types']) + 1)
        # labels.pop(str(slot), None)
        for k, v in labels.items():
            # split tokens and get index
            if (v.find('_') is not -1) or (v.find(' ') is not -1):
                # word_list = v.split('_'
                word_list = list(filter(None, re.split(r'\s|_', v)))
            else:
                word_list = list(filter(None, re.split(r'([A-Z][a-z]*)', v)))

            # slice if exceed maximum length
            if len(word_list) > self.params['max_node_length']:
                word_list = word_list[:self.params['max_node_length']]

            offset = 0
            for idx, w in enumerate(word_list):
                w = w.lower()
                if self.params['keep_letter_only']:
                    w = re.sub('[^a-zA-Z]+', '', w)
                if w:
                    # assert w in self.vocabs
                    if w in self.vocabs:
                        node_words[int(k)][idx - offset] = self.vocabs.index(w)
                    else:
                        node_words[int(k)][idx - offset] = len(self.vocabs)
                else:
                    offset += 1

            # get type index
            if k in types and types[k] in self.type_hierarchy[0]['types']:
                type_idx = self.type_hierarchy[0]['types'].index(types[k])
                node_types[int(k)][0] = type_idx
                for idx, t in enumerate(self.type_hierarchy[0]['outgoingEdges'][type_idx]):
                    if idx + 1 < self.params['max_node_length']:
                        node_types[int(k)][idx + 1] = t
            else:
                node_types[int(k)][0] = len(self.type_hierarchy[0]['types'])

        return node_words, node_types

    # extract candidate information
    def __candidate_preprocess(self, raw_candidates, num_nodes):
        target = None
        candidates = np.zeros([num_nodes, 1], dtype=np.int32)
        for idx, c in enumerate(raw_candidates):
            if c['IsCorrect']:
                target = idx
            candidates[c['SymbolDummyNode']][0] = 1
        return candidates, target

    def make_minibatch_iterator(self, data: Any, is_training: bool):
        """Create minibatches by flattening adjacency matrices into a single adjacency matrix with
        multiple disconnected components."""
        if is_training:
            np.random.shuffle(data)
        # Pack until we cannot fit more graphs in the batch
        state_dropout_keep_prob = self.params['graph_state_dropout_keep_prob'] if is_training else 1.
        edge_weights_dropout_keep_prob = self.params['edge_weight_dropout_keep_prob'] if is_training else 1.
        num_graphs = 0
        while len(data) - num_graphs >= self.params['batch_size']:
            num_graphs_in_batch = 0
            batch_node_labels = []
            batch_node_types = []
            batch_candidates = []
            batch_slot = []
            batch_num_candidates = []
            batch_target_task_values = []
            # batch_target_task_mask = []
            batch_adjacency_lists = [[] for _ in range(self.num_edge_types)]
            batch_num_incoming_edges_per_type = []
            batch_graph_nodes_list = []
            node_offset = 0

            while num_graphs < len(data) and num_graphs_in_batch < self.params['batch_size']:
                cur_graph = data[num_graphs]
                num_nodes_in_graph = len(cur_graph['node_labels'])
                num_candidates_in_graph = cur_graph['num_candidates']

                # pad the word and type ids
                for l in cur_graph['node_labels']:
                    padded_word_ids = np.pad(l, (0, self.params['max_node_length'] - len(l)), 'constant',
                                             constant_values=len(self.vocabs) + 1)
                    batch_node_labels.append(padded_word_ids)
                for l in cur_graph['node_types']:
                    padded_type_ids = np.pad(l, (0, self.params['max_node_length'] - len(l)), 'constant',
                                             constant_values=len(self.type_hierarchy[0]['types']) + 1)
                    batch_node_types.append(padded_type_ids)
                # batch_node_labels.extend(cur_graph['node_labels'])
                # batch_node_types.extend(cur_graph['node_types'])
                batch_candidates.append(cur_graph['candidates'])
                batch_slot.append(
                    np.full(shape=[num_candidates_in_graph], fill_value=cur_graph['slot'] + node_offset,
                            dtype=np.int32))
                batch_num_candidates.append(num_candidates_in_graph)
                batch_graph_nodes_list.append(
                    np.full(shape=[num_nodes_in_graph], fill_value=num_graphs_in_batch, dtype=np.int32))
                for i in range(self.num_edge_types):
                    if i in cur_graph['adjacency_lists']:
                        batch_adjacency_lists[i].append(np.asarray(cur_graph['adjacency_lists'][i]) + node_offset)

                # Turn counters for incoming edges into np array:
                num_incoming_edges_per_type = np.zeros((num_nodes_in_graph, self.num_edge_types))
                for (e_type, num_incoming_edges_per_type_dict) in cur_graph['num_incoming_edge_per_type'].items():
                    for (node_id, edge_count) in num_incoming_edges_per_type_dict.items():
                        num_incoming_edges_per_type[node_id, e_type] = edge_count
                batch_num_incoming_edges_per_type.append(num_incoming_edges_per_type)

                target_one_hot = np.zeros([num_candidates_in_graph], dtype=np.int32)
                if cur_graph['target'] is None:  # This is one of the examples we didn't sample...
                    batch_target_task_values.append(target_one_hot)
                    # batch_target_task_mask.append(0.)
                else:
                    target_one_hot[cur_graph['target']] = 1
                    batch_target_task_values.append(target_one_hot)
                    # batch_target_task_mask.append(1.)
                num_graphs += 1
                num_graphs_in_batch += 1
                node_offset += num_nodes_in_graph
            batch_feed_dict = {
                self.placeholders['initial_word_ids']: np.reshape(np.concatenate(batch_node_labels),
                                                                  [node_offset, self.params['max_node_length']]),
                self.placeholders['initial_type_ids']: np.reshape(np.concatenate(batch_node_types),
                                                                  [node_offset, self.params['max_node_length']]),
                self.placeholders['candidates']: np.reshape(np.concatenate(batch_candidates), [node_offset, 1]),
                self.placeholders['slots']: np.concatenate(batch_slot),
                self.placeholders['num_candidates_per_graph']: np.array(batch_num_candidates, dtype=np.int32),
                self.placeholders['num_incoming_edges_per_type']: np.concatenate(batch_num_incoming_edges_per_type,
                                                                                 axis=0),
                self.placeholders['graph_nodes_list']: np.concatenate(batch_graph_nodes_list),
                self.placeholders['target_values']: np.concatenate(batch_target_task_values, axis=0),
                # self.placeholders['target_mask']: np.array(batch_target_task_mask),
                self.placeholders['num_graphs']: num_graphs_in_batch,
                self.placeholders['graph_state_keep_prob']: state_dropout_keep_prob,
                self.placeholders['edge_weight_dropout_keep_prob']: edge_weights_dropout_keep_prob
            }

            # Merge adjacency lists and information about incoming nodes:
            for i in range(self.num_edge_types):
                if len(batch_adjacency_lists[i]) > 0:
                    adj_list = np.concatenate(batch_adjacency_lists[i])
                else:
                    adj_list = np.zeros((0, 2), dtype=np.int32)
                batch_feed_dict[self.placeholders['adjacency_lists'][i]] = adj_list

            yield batch_feed_dict


def main():
    args = docopt(__doc__)
    try:
        model = SparseGGNNChemModel(args)
        model.train()
    except:
        typ, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


if __name__ == "__main__":
    main()
