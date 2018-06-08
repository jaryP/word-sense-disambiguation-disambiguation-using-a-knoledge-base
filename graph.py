import networkx as nx
import tqdm
import os
from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score
from codice import utils
import copy

def createGraph(semantic_relationships, graph_file = None):
    '''
    Build the supervised graph G=(V, E) using networkx library, where V is the set of synsets and E the set of semantic
    correlation between synsets (vertexes)
    :param semantic_relationships: the relationship from which build the graph
    :param graph_file: where to save the file (to avoid creating the graph each time)
    :return: return the graph
    '''

    if graph_file is not None and os.path.isfile(graph_file):
        return nx.read_multiline_adjlist(graph_file)

    G = nx.Graph()

    keys = list(semantic_relationships.keys())

    for lemma in tqdm.tqdm(semantic_relationships.keys()):
        G.add_node(lemma)
        for relationship, nodes in semantic_relationships[lemma].items():
            for node in nodes:
                if node in keys:
                    G.add_edge(lemma, node, v = relationship, weight=1.0)

    if graph_file is not None:
        nx.write_multiline_adjlist(G,graph_file)

    return G


def createDocumentsGraph(train, graph_file = None, sem_rel = None, sem_graph = None):
    """

    :param train:
    :param graph_file:
    :param sem_rel:
    :param sem_graph:
    :return:
    """
    if graph_file is not None and os.path.isfile(graph_file):
        return nx.read_multiline_adjlist(graph_file)

    G = nx.DiGraph()

    if sem_graph is not None:
        G = nx.compose(G, sem_graph.to_directed())

    for doc in train.keys():
        for sentence in train[doc]:
            lemmas = [l for l in sentence if isinstance(l, utils.instance)]

            for i in range(1,len(lemmas)):

                lemma = lemmas[i]
                lemma_key = lemma.lemma + '_' + lemma.pos
                prev_lemma = lemmas[i-1]

                prev_lemma_key = prev_lemma.lemma + '_' + prev_lemma.pos

                if i == 1:
                    G.add_node(prev_lemma_key)

                G.add_node(lemma_key)
                G.add_edge(prev_lemma_key, lemma_key)

                if sem_rel is not None:

                    for r, nodes in sem_rel[lemma.instance].items():
                        for node in nodes:
                            if node not in G.nodes:
                                continue
                            G.add_edge(lemma_key, node)

                    if i == 1:
                        for r, nodes in sem_rel[prev_lemma.instance].items():
                            for node in nodes:
                                if node not in G.nodes:
                                    continue
                                G.add_edge(prev_lemma_key, node)

    if graph_file is not None:
        nx.write_multiline_adjlist(G,graph_file)

    return G


def extendGraph(G, synsets_ditionary, document_graph=False):
    """
    Extend the graph with new synsets
    :param G: the graph to extend
    :param synsets_ditionary: the dictionary used to extend the graph
    :param document_graph: if the graph is a document ones
    :return: return a copy of G extended using relations in synsets_ditionary
    """
    TG = G.copy()

    for k, v in synsets_ditionary.items():
        if document_graph:
            TG.add_node(k)
        TG.add_nodes_from(list(v.keys()))
    for k, v in synsets_ditionary.items():

        for vertex, relationship in v.items():
            if len(relationship) == 0:
                continue
            if document_graph:
                TG.add_edge(k, vertex)

            for _, synsets in relationship.items():
                for s in synsets:
                    if TG.has_node(s):
                        TG.add_edge(vertex, s, weight=1.0)
                        if document_graph:
                            TG.add_edge(s, vertex)
                            TG.add_edge(k, vertex)
    return TG


def getWeightCoOc(corpus, synsets_file, win_size=10):
    '''
    Given a corpus of text the function build and return new edges weighted using co-occurrence matrix
    :param corpus: the corpus of text
    :param synsets_file: the file from which get the synset associated to the corpus
    :param win_size: the size to caculate co occurrence value
    :return: a list of triple [(V1, V2, w),...] where W = coOcMatrix[V1][V2]
    '''

    _, synsets = utils.getSynsetsDictionary(synsets_file)
    mapping = {}

    for s in synsets:
        mapping.update({s: len(mapping)})

    inverse_mapping = {v: k for k, v in mapping.items()}

    matrix = np.zeros((len(mapping), len(mapping)))

    for d, sentence in corpus.items():
        _, synsets = utils.getDocumentsLemmas(sentence)
        for i in range(len(synsets)):
            to_iter = np.arange(max(0, i - win_size), min(len(synsets), i + win_size + 1 ))
            for j in to_iter:
                if j == i:
                    continue
                matrix[mapping[synsets[i]]][mapping[synsets[j]]] += 1

    edges = list()

    for i in range(len(mapping)):
        for j in range(i+1, len(mapping)):
            v = matrix[i][j]
            if v == 0:
                continue
            edges.append(
                (inverse_mapping[i], inverse_mapping[j], v)
            )

    return edges


def staticPagerankPrediction(G, eval_set, eval_synsets_dictionary, pagerank_algo='static'):
    """
    This function return the score associated to each eval test using the pagerank algorithm. The graph is extended with
    all the test data and then pagerank vector is used to predict most likely synset for each lemma. Another approach
    consists in assign an initial probability only to synsets appearing in the eval corpus.
    :param G: the graph
    :param eval_set: the dictionary containing {eval_dataset: {document: [lemmas...]}}
    :param eval_synsets_dictionary: dictionary containing synsets and associations used to extend the graph
    :param pagerank_algo: static: each node with same probability, mass=only eval nodes have initial probability
    :return: f1 score for each eval stored in a dictionary
    """
    TG = extendGraph(G, eval_synsets_dictionary)

    if pagerank_algo == 'mass':
        dizionario = {}
        for _, vertex in eval_synsets_dictionary.items():
            dizionario.update({k: 1 for k in vertex.keys()})
        pr = nx.pagerank_scipy(TG, personalization=dizionario)
    else:
        pr = nx.pagerank_scipy(TG)

    results = {}

    for eval_set_name in eval_set.keys():
        pre = []
        all = []
        for sentence in eval_set[eval_set_name].values():
            lemmas, synsets = utils.getDocumentsLemmas(sentence, True)
            all.extend(synsets)
            for l in lemmas:
                max_prob = 0
                best_syn = 0
                for synsets in eval_synsets_dictionary[l].keys():
                    rank = pr[synsets]
                    if rank > max_prob:
                        max_prob = rank
                        best_syn = synsets
                pre.append(best_syn)

        f1 = f1_score(pre, all, average='micro')
        results[eval_set_name] = f1

    return results


def documentPagerankPrediction(G, eval_set, eval_synsets_dictionary):
    '''
    This function return the score associated to each eval test using the pagerank algorithm. The graph is extended each
    time with a single document then pagerank vector, calculated with probabily on the lemma present in the document,
    is used to predict most likely synset for each lemma.
    :param G: the graph
    :param eval_set: the dictionary containing {eval_dataset: {document: [lemmas...]}}
    :param eval_synsets_dictionary: dictionary containing synsets and associations used to extend the graph
    :return: f1 score for each eval stored in a dictionary
    '''

    results = dict()

    for eval_set_name in eval_set.keys():
        pre = []
        all = []

        for sentence in eval_set[eval_set_name].values():
            lemmas, synsets = utils.getDocumentsLemmas(sentence, True)
            all.extend(synsets)

            dizionario = {}

            near = set()
            to_add = {}

            for l in lemmas:
                near.update(eval_synsets_dictionary[l].keys())
                to_add.update({l: eval_synsets_dictionary[l]})
            TG = extendGraph(G, to_add, document_graph=False)

            for n in near:
                dizionario.update({n: 1})

            pr = nx.pagerank_scipy(TG, personalization=dizionario)

            for l in lemmas:
                max_prob = 0
                best_syn = 0
                for synsets in eval_synsets_dictionary[l].keys():
                    rank = pr[synsets]
                    if rank > max_prob:
                        max_prob = rank
                        best_syn = synsets
                pre.append(best_syn)

        f1 = f1_score(pre, all, average='micro')
        results[eval_set_name] = f1

    return results

def graphPathsPrediction(G, test_set, test_synsets_ditionary, cut=6):

    results = dict()
    for eval_set_name in test_set.keys():
        pre = []
        all = []

        for sentence in test_set[eval_set_name].values():
            lemmas, synsets = utils.getDocumentsLemmas(sentence, True)

            words_path = {}
            used_path = defaultdict(int)

            ln_lemmas = len(lemmas)

            to_add = {}
            diz = {}

            for l in lemmas:
                to_add.update({l: test_synsets_ditionary[l]})
                diz.update({s: 1 for s in test_synsets_ditionary[l].keys()})
            TG = extendGraph(G, to_add, document_graph=False)

            ln_saved = ln_lemmas*(2.0/3.0)

            for i in tqdm.tqdm(range(ln_lemmas)):

                curr_lemma = lemmas[i]
                all.append(synsets[i])
                dicz = copy.deepcopy(diz)

                nodes = set()

                for s in test_synsets_ditionary[curr_lemma].keys():
                    dicz.update({s: 0})
                    if s not in words_path:

                        if len(words_path) >= ln_saved:

                            less_used = min(used_path, key=used_path.get)
                            words_path.pop(less_used)
                            used_path.pop(less_used)

                        words_path[s] = nx.single_source_dijkstra_path(TG, s, cutoff=cut).keys()
                    used_path[s] += 1

                    nodes.update(words_path[s])

                sub_TG = TG.subgraph(nodes)

                sum = 0
                for s in dicz.values():
                    sum += s

                best_syn = 0
                probs = {}

                try:                                                                                                                                                                                                                                                                                
                    if sum == 0:
                        probs = nx.pagerank_scipy(sub_TG, max_iter=200)
                    else:
                        probs = nx.pagerank_scipy(sub_TG, max_iter=200, personalization=dicz)

                    max_prob = -1
                    for n in test_synsets_ditionary[curr_lemma].keys():
                        rank = probs[n]
                        if rank > max_prob:
                            max_prob = rank
                            best_syn = n

                    assert (best_syn != 0)
                    pre.append(best_syn)

                except nx.exception.PowerIterationFailedConvergence:
                    best_syn = list(test_synsets_ditionary[curr_lemma].keys())[0]


        print(eval_set_name, f1_score(pre, all, average='micro'))
        f1 = f1_score(pre, all, average='micro')
        results[eval_set_name] = f1

    return results


def graphPathTest(G, test_set, test_synsets_ditionary, file, cut=2):

    pre = []
    all = []
    for doc in test_set:

        words_path = {}
        diz = {}
        to_add = {}

        for lemma in doc:
            lemma, pos, truth = lemma.rsplit('_')
            all.append(truth)
            d = test_synsets_ditionary[lemma+'_'+pos]
            to_add.update({lemma+'_'+pos: d})
            diz.update({s: 1 for s in d.keys()})

        TG = extendGraph(G, to_add)

        for lemma in doc:

            lemma, pos, t = lemma.rsplit('_')
            curr_lemma = lemma+'_'+pos
            dicz = copy.deepcopy(diz)

            nodes = set()

            for s in test_synsets_ditionary[curr_lemma].keys():
                dicz.update({s: 0})
                if s not in words_path:
                    words_path[s] = nx.single_source_dijkstra_path(TG, s, cutoff=cut).keys()
                nodes.update(words_path[s])

            sub_TG = TG.subgraph(nodes)

            sum = 0
            for s in dicz.values():
                sum += s

            best_syn = ''
            probs = {}

            try:
                if sum == 0:
                    probs = nx.pagerank_scipy(sub_TG, max_iter=200)
                else:
                    probs = nx.pagerank_scipy(sub_TG, max_iter=200, personalization=dicz)

                max_prob = -1
                for n in test_synsets_ditionary[curr_lemma].keys():
                    rank = probs[n]
                    if rank > max_prob:
                        max_prob = rank
                        best_syn = n

            except nx.exception.PowerIterationFailedConvergence:
                best_syn = list(test_synsets_ditionary[curr_lemma].keys())[0]

            pre.append(best_syn)

    with open(file, 'w+') as f:
        for i in range(len(all)):
            f.write(all[i]+'\t'+pre[i]+'\n')
