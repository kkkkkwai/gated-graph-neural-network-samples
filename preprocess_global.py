import os
import json
import re
import numpy as np
from typing import List, Tuple, Dict, Sequence, Any
from collections import defaultdict, namedtuple
from utils import EDGE_TYPE, seenProjects, unseenProjects

keep_only_letter = False


tie_fwd_bkwd = False
num_edge_types = len(EDGE_TYPE)
max_node_length = 25

data_path = os.path.expanduser('~/project-data/graph-dataset')
# project_name = 'ninject'
# data_path = os.path.expanduser('~/Downloads/graph-dataset')

vocabs = []
types = []


def collect_vocab(raw_data, _filename):
    print('Collecting vocab from file {}...'.format(_filename), end='\r')
    for d in raw_data:
        for k, v in d['ContextGraph']['NodeLabels'].items():
            # split tokens and get index
            # node_words.append([])
            if (v.find('_') is not -1) or (v.find(' ') is not -1) or (v.find('-') is not -1):
                # word_list = v.split('_'
                word_list = list(filter(None, re.split(r'\s|_|-', v)))
            else:
                word_list = list(filter(None, re.split(r'([A-Z][a-z]*)', v)))
            # slice if exceed maximum length
            if len(word_list) > max_node_length:
                word_list = word_list[:max_node_length]

            for w in word_list:
                w = w.lower()
                if keep_only_letter:
                    w = re.sub('[^a-zA-Z]+', '', w)
                if w and (w not in vocabs):
                    vocabs.append(w)


def collect_types(raw_data, _filename):
    print('Collecting types from file {}...'.format(_filename), end='\r')
    types_local = raw_data['types']
    for t in types_local:
        if t not in types:
            types.append(t)


def build_type_hierarchy(raw_data, _filename):
    print('Building type hierarcy in file {}...'.format(_filename), end='\r')
    types_local = raw_data['types']
    outgoing_edges_local = raw_data['outgoingEdges']
    for tidx, oge in enumerate(outgoing_edges_local):
        # get one by one type name and its existing outgoing edges
        type_name = types_local[tidx]
        existing_type_edges = outgoing_edges[types.index(type_name)]
        # decide one by one whether the outgoing edges have been added. if not, append
        for t in oge:
            t_name = types_local[t]
            t_idx_global = types.index(t_name)
            if t_idx_global not in existing_type_edges:
                existing_type_edges.append(t_idx_global)


def load_vocab_and_type(raw_data, dump_to, idx):
    processed_graphs = []
    for graph_idx, d in enumerate(raw_data):
        if graph_idx % 50 == 0:
            print('Processing {} Graphs in file {}...'.format(graph_idx, idx), end='\r')
        slot = d['SlotDummyNode']
        num_nodes = len(d['ContextGraph']['NodeLabels'])
        (adjacency_lists, num_incoming_edge_per_type) = __graph_to_adjacency_lists(d['ContextGraph']['Edges'])
        node_words, node_types = __node_preprocess(d['ContextGraph']['NodeLabels'],
                                                   d['ContextGraph']['NodeTypes'], num_nodes)
        candidates, target = __candidate_preprocess(d['SymbolCandidates'], num_nodes)

        assert num_nodes == len(node_words)
        assert num_nodes == len(node_types)
        assert num_nodes == len(candidates)
        processed_graphs.append({"adjacency_lists": adjacency_lists,
                                 "num_incoming_edge_per_type": num_incoming_edge_per_type,
                                 "node_labels": node_words,
                                 "node_types": node_types,
                                 "slot": slot,
                                 "candidates": candidates,
                                 "num_candidates": len(d['SymbolCandidates']),
                                 "target": target})

    print('Dumping json file {}...'.format(idx))
    with open(os.path.join(dump_to, project_name + '_global_{}.json'.format(idx)), 'w') as df_:
        json.dump(processed_graphs, df_)


def __graph_to_adjacency_lists(graph):
    adj_lists = defaultdict(list)
    num_incoming_edges_dicts_per_type = defaultdict(lambda: defaultdict(lambda: 0))
    for e_id, e_type in enumerate(EDGE_TYPE):
        if e_type in graph:
            # TODO: simplify the process
            for src, dest in graph[e_type]:
                fwd_edge_type = e_id  # Make edges start from 0
                adj_lists[fwd_edge_type].append((src, dest))
                num_incoming_edges_dicts_per_type[fwd_edge_type][dest] += 1
                if tie_fwd_bkwd:
                    adj_lists[fwd_edge_type].append((dest, src))
                    num_incoming_edges_dicts_per_type[fwd_edge_type][src] += 1

    final_adj_lists = {e: sorted(lm) for e, lm in adj_lists.items()}

    # Add backward edges as an additional edge type that goes backwards:
    if not tie_fwd_bkwd:
        for (edge_type, edges) in adj_lists.items():
            bwd_edge_type = num_edge_types + edge_type
            final_adj_lists[bwd_edge_type] = sorted((y, x) for (x, y) in edges)
            for (x, y) in edges:
                num_incoming_edges_dicts_per_type[bwd_edge_type][y] += 1

    return final_adj_lists, num_incoming_edges_dicts_per_type


def __node_preprocess(labels, ntypes, num_nodes):
    # TODO: handle negative ids
    # node_words = np.full([num_nodes, max_node_length], fill_value=len(vocabs))
    # node_types = np.full([num_nodes, max_node_length], fill_value=len(type_hierarchy[0]['types']) + 1)
    # labels.pop(str(slot), None)
    node_words = []
    node_types = []
    for k, v in labels.items():
        word_ids = []
        type_ids = []
        # split tokens and get index
        # node_words.append([])
        if (v.find('_') is not -1) or (v.find(' ') is not -1) or (v.find('-') is not -1):
            # word_list = v.split('_'
            word_list = list(filter(None, re.split(r'\s|_|-', v)))
        else:
            word_list = list(filter(None, re.split(r'([A-Z][a-z]*)', v)))
        # slice if exceed maximum length
        if len(word_list) > max_node_length:
            word_list = word_list[:max_node_length]

        for word_idx, w in enumerate(word_list):
            w = w.lower()
            if keep_only_letter:
                w = re.sub('[^a-zA-Z]+', '', w)
            if w:
                if w in vocabs:
                # node_words[int(k)][word_idx] = vocabs.index(w)
                    word_ids.append(vocabs.index(w))
                else:
                    word_ids.append(len(vocabs))
        if not word_ids:
            word_ids.append(len(vocabs))

        # get type index
        if k in ntypes and ntypes[k] in types:
            type_idx = types.index(ntypes[k])
            # node_types[int(k)][0] = type_idx
            type_ids.append(type_idx)
            for og_idx, t in enumerate(outgoing_edges[type_idx]):
                if og_idx + 1 < max_node_length:
                    # node_types[int(k)][og_idx + 1] = t
                    type_ids.append(t)
        else:
            type_ids.append(len(types))
            # node_types[int(k)][0] = len(type_hierarchy[0]['types'])
        node_words.append(word_ids)
        node_types.append(type_ids)

    return node_words, node_types


# extract candidate information
def __candidate_preprocess(raw_candidates, num_nodes):
    target = None
    candidates = np.zeros([num_nodes, 1], dtype=np.int32)
    for can_idx, c in enumerate(raw_candidates):
        if c['IsCorrect']:
            target = can_idx
        candidates[c['SymbolDummyNode']][0] = 1
    return candidates.tolist(), target


# Collect vocab and types from seen projects
for project_name in seenProjects:
    data_path_local = os.path.join(data_path, project_name)
    train_data, valid_data = os.path.join(data_path_local, 'graphs-train'), os.path.join(data_path_local, 'graphs-valid')
    hierarchy_path = os.path.join(data_path_local, '{}-typehierarchy.json'.format(project_name))
    # Collect vocab from seen projects
    for j in [train_data]:
        for idx, filename in enumerate(os.listdir(j)):
            if not filename.endswith('json'):
                with open(os.path.join(j, filename), 'r') as f:
                    data = json.load(f)
                    collect_vocab(data, filename)

    with open(hierarchy_path, 'r') as f:
        data = json.load(f)
        collect_types(data, '{}-typehierarchy.json'.format(project_name))

outgoing_edges = [[] for i in range(len(types))]

# Build global type hierarchy
for project_name in seenProjects:
    data_path_local = os.path.join(data_path, project_name)
    hierarchy_path = os.path.join(data_path_local, '{}-typehierarchy.json'.format(project_name))
    with open(hierarchy_path, 'r') as f:
        data = json.load(f)
        build_type_hierarchy(data, '{}-typehierarchy.json'.format(project_name))

vocabs_json = {'vocabulary': vocabs}
print('Saving {} vocabs to {}...'.format(len(vocabs), str(data_path)))
with open(os.path.join(data_path, 'vocabulary.json'), 'w') as df:
    json.dump(vocabs_json, df)

types_json = {
    'types': types,
    'outgoingEdges': outgoing_edges
}
print('Saving types hierarchy to {}'.format(str(data_path)))
with open(os.path.join(data_path, 'typehierarchy.json'), 'w') as df:
    json.dump(types_json, df)

# for project_name in seenProjects:
#     data_path_local = os.path.join(data_path, project_name)
#     train_data, valid_data = os.path.join(data_path_local, 'graphs-train'), os.path.join(data_path_local, 'graphs-valid')
#     for i in [train_data, valid_data]:
#         offset = 0
#         for idx, filename in enumerate(os.listdir(i)):
#             if not filename.endswith('json'):
#                 with open(os.path.join(i, filename), 'r') as f:
#                     data = json.load(f)
#                     load_vocab_and_type(data, i, idx - offset)
#             else:
#                 offset = offset + 1
#
# for project_name in unseenProjects:
#     data_path_local = os.path.join(data_path, project_name, 'graphs')
#     offset = 0
#     for idx, filename in enumerate(os.listdir(data_path_local)):
#         if not filename.endswith('json'):
#             with open(os.path.join(data_path_local, filename), 'r') as f:
#                 data = json.load(f)
#                 load_vocab_and_type(data, data_path_local, idx - offset)
#         else:
#             offset = offset + 1
