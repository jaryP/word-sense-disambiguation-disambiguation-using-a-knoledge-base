import networkx as nx
import matplotlib.pyplot as plt
import tqdm
import os
from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score
from codice import utils
from itertools import islice



def createGraph(semantic_relationships, graph_file = None):

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
    :param G:
    :param synsets_ditionary:
    :param document_graph:
    :return:
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


def plotGraph(G, with_labels = True):
    pos = nx.spring_layout(G)
    nx.draw(G, pos=pos, with_labels=with_labels)
    # if with_labels:
    #     labels = list({e: G.get_edge_data(e[0], e[1])['v'] for e in G.edges()}.values())
    #     nx.draw_networkx_edge_labels(G, pos=pos, labels=labels)
    plt.show()


def getWeightCoOc(corpus, synsets_file, win_size=10):

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


# def applOcMatrix(G, matrix, mapping):


def staticPagerankPrediction(G, test_set, test_synsets_ditionary, dumping = 0.85, pagerank_algo = 'static'):
    """

    :param G:
    :param test_set:
    :param test_synsets_ditionary:
    :param dumping:
    :param pagerank_algo:
    :return:
    """
    TG = extendGraph(G, test_synsets_ditionary)

    if pagerank_algo == 'mass':
        dizionario = {}
        for _, vertex in test_synsets_ditionary.items():
            dizionario.update({k: 1 for k in vertex.keys()})
        pr = nx.pagerank_scipy(TG, alpha=dumping, personalization=dizionario)
    else:
        pr = nx.pagerank_scipy(TG, alpha=dumping)

    results = {}

    for eval_set in test_set.keys():
        pre = []
        all = []

        for sentence in test_set[eval_set].values():
            lemmas, synsets = utils.getDocumentsLemmas(sentence, True)
            all.extend(synsets)
            for l in lemmas:
                max_prob = 0
                best_syn = 0
                for synsets in test_synsets_ditionary[l].keys():
                    rank = pr[synsets]
                    if rank > max_prob:
                        max_prob = rank
                        best_syn = synsets
                pre.append(best_syn)

        f1 = f1_score(pre, all, average='micro')
        results[eval_set] = f1

    return results


def documentPagerankPrediction(G, test_set, test_synsets_ditionary, dumping = 0.85):

    results = dict()

    for eval_set in test_set.keys():
        pre = []
        all = []

        for sentence in test_set[eval_set].values():
            lemmas, synsets = utils.getDocumentsLemmas(sentence, True)
            all.extend(synsets)

            dizionario = {}

            near = set()
            to_add = {}

            for l in lemmas:
                near.update(test_synsets_ditionary[l].keys())
                to_add.update({l: test_synsets_ditionary[l]})
            TG = extendGraph(G, to_add, document_graph=False)

            for n in near:
                dizionario.update({n: 1})

            pr = nx.pagerank_scipy(TG, alpha=dumping, personalization=dizionario)

            for l in lemmas:
                max_prob = 0
                best_syn = 0
                for synsets in test_synsets_ditionary[l].keys():
                    rank = pr[synsets]
                    if rank > max_prob:
                        max_prob = rank
                        best_syn = synsets
                pre.append(best_syn)

        f1 = f1_score(pre, all, average='micro')
        results[eval_set] = f1

    return results


def documentPagerankPrediction1(G, test_set, test_synsets_ditionary, dumping = 0.85):

    results = dict()

    for eval_set in test_set.keys():
        pre = []
        all = []
        for sentence in test_set[eval_set].values():
            lemmas, synsets = utils.getDocumentsLemmas(sentence, True)
            all.extend(synsets)

            dizionario = {}

            near = set()
            to_add = {}

            for l in lemmas:
                near.update(test_synsets_ditionary[l].keys())
                to_add.update({l: test_synsets_ditionary[l]})
            TG = extendGraph(G, to_add, document_graph=False)

            for n in near:
                dizionario.update({n: 1})

            for l in tqdm.tqdm(lemmas):

                diz = dizionario.copy()
                lemma_syns = list(test_synsets_ditionary[l].keys())

                for syns in lemma_syns:
                    diz.update({syns: 0})

                pr = nx.pagerank_scipy(TG, alpha=dumping, personalization=diz)

            # for l in lemmas:
                max_prob = 0
                best_syn = 0
                for s in lemma_syns:
                    rank = pr[s]
                    if rank > max_prob:
                        max_prob = rank
                        best_syn = s
                pre.append(best_syn)
        f1 = f1_score(pre, all, average='micro')
        results[eval_set] = f1

    return results


def dynamicPagerankPrediction1(G, test_set, test_synsets_ditionary, dumping = 0.85, contex=-1, wtw = False):

    results = dict()

    for eval_set in test_set.keys():
        pre = []
        all = []
        if eval_set != 'semeval2007':
            continue
        for sentence in test_set[eval_set].values():

            lemmas, synsets = utils.getDocumentsLemmas(sentence)
            # all.extend(synsets)

            ln_lemmas = len(lemmas)

            to_add = {}

            # for l in lemmas:
            #     to_add[l] = test_synsets_ditionary[l]

            # TG = extendGraph(G, to_add)

            # diz = {}
            # g_nodes = G.nodes
            # for n in g_nodes:
            #     # if n not in TG.nodes:
            #     diz.update({n: 1})

            # pr = nx.pagerank_scipy(TG, personalization= diz)

                # d = {k:1 for k, v in to_add.keys()}

            for i in tqdm.tqdm(range(ln_lemmas)):
                truth = list(test_synsets_ditionary[lemmas[i]].keys())
                to_iter = np.arange(max(0, i-contex), min(ln_lemmas, i+contex+1))
                near = set()
                all.append(synsets[i])
                # for l in lemmas:
                #     to_add[l]

                for window in to_iter:
                    l = lemmas[window]
                    keys = test_synsets_ditionary[l].keys()
                    near.update(keys)
                    to_add[l] = test_synsets_ditionary[l]

                TG = extendGraph(G, to_add)
                dizionario = {k:1 for k in near}
                dizionario.update({k: 0 for k in truth})

                # for n in near:
                #     dizionario.update({n: 1})

                pr = nx.pagerank_scipy(TG, alpha=dumping, personalization=dizionario)

                max_prob = 0
                best_syn = 0
                for s in truth:
                    rank = pr[s]
                    if rank > max_prob:
                        max_prob = rank
                        best_syn = s
                pre.append(best_syn)
            # print(f1_score(pre, all, average='micro'))

        f1 = f1_score(pre, all, average='micro')
        results[eval_set] = f1

    return results


def graphDegreePrediction1(G, test_set, test_synsets_ditionary, dumping = 0.85):

    results = dict()

    for eval_set in test_set.keys():
        pre = []
        all = []

        if eval_set != 'semeval2007':
            continue
        for sentence in test_set[eval_set].values():
            lemmas, synsets = utils.getDocumentsLemmas(sentence)
            all.extend(synsets)

            dizionario = {}

            near = set()
            to_add = {}

            for l in lemmas:
                near.update(test_synsets_ditionary[l].keys())
                to_add.update({l: test_synsets_ditionary[l]})
            TG = extendGraph(G, to_add, document_graph=False)
            vals = {}

            # for n in near:
            #     dizionario.update({n: 1})
            # probs = nx.closeness_centrality(TG)

            for l in tqdm.tqdm(lemmas):

                diz = dizionario.copy()
                lemma_syns = list(test_synsets_ditionary[l].keys())

                for syns in lemma_syns:
                    diz.update({syns: 0})

                # pr = nx.pagerank_scipy(TG, alpha=dumping, personalization=diz)
                # probs = TG.closeness_centrality()

            # for l in lemmas:
                max_prob = 0
                best_syn = 0
                for s in lemma_syns:
                    if s in vals:
                        rank = vals.get(s)
                    else:
                        rank = nx.closeness_centrality(TG, s)
                        vals[s] = rank

                    if rank > max_prob:
                        max_prob = rank
                        best_syn = s

                pre.append(best_syn)
        f1 = f1_score(pre, all, average='micro')
        results[eval_set] = f1

    return results


def graphBFSprediction(G, test_set, test_synsets_ditionary, cut=6, degree_heuristic = False):

    results = dict()
    G = G.to_directed()
    # TG = extendGraph(G, test_synsets_ditionary, document_graph=False)

    for eval_set in test_set.keys():
        pre = []
        all = []

        for sentence in test_set[eval_set].values():

            lemmas, synsets = utils.getDocumentsLemmas(sentence, True)

            ln_lemmas = len(lemmas)

            to_add = {}
            diz = {}

            for l in lemmas:
                to_add.update({l: test_synsets_ditionary[l]})
                diz.update({s:1 for s in test_synsets_ditionary[l].keys()})
            TG = extendGraph(G, to_add, document_graph=False)

            for i in tqdm.tqdm(range(ln_lemmas)):

                curr_lemma = lemmas[i]
                all.append(synsets[i])
                dicz = diz.copy()
                nodes = set()

                for s in test_synsets_ditionary[lemmas[i]].keys():
                    dicz.update({s: 0})
                    nodes.update(nx.single_source_shortest_path(TG, s, cutoff=cut).keys())

                sub_TG = TG.subgraph(nodes)
                # probs = sub_TG.degree()

                probs = nx.pagerank_scipy(TG, personalization=dicz)

                ln_nodes = len(sub_TG.nodes)

                max_prob = -1
                best_syn = 0
                for n in test_synsets_ditionary[curr_lemma].keys():
                    rank = probs[n]  # /ln_nodes
                    if rank > max_prob:
                        max_prob = rank
                        best_syn = n
                assert (best_syn != 0)
                pre.append(best_syn)

                # print(f1_score(pre, all, average='micro'))
        f1 = f1_score(pre, all, average='micro')
        results[eval_set] = f1
    return results


def graphDegreePrediction(G, test_set, test_synsets_ditionary, contex = 5, degree_heuristic = False):

    results = dict()
    G = G.to_directed()
    # TG = extendGraph(G, test_synsets_ditionary, document_graph=True)

    for eval_set in test_set.keys():
        pre = []
        all = []

        if eval_set != 'semeval2007':
            continue

        for sentence in test_set[eval_set].values():

            lemmas, synsets = utils.getDocumentsLemmas(sentence)

            ln_lemmas = len(lemmas)

            to_add = {}

            for l in lemmas:
                to_add.update({l: test_synsets_ditionary[l]})
            TG = extendGraph(G, to_add, document_graph=False)

            for i in tqdm.tqdm(range(ln_lemmas)):

                to_iter = np.arange(max(0, i - contex), min(ln_lemmas, i + contex + 1))
                curr_lemma = lemmas[i]
                to_reach = set()
                all.append(synsets[i])

                for j in to_iter:
                    syns = test_synsets_ditionary[lemmas[j]]
                    for s in syns:
                        neigh = TG.neighbors(s)
                        to_reach.update(islice(neigh, 1, None))


                nodes = set()
                # for node in to_reach:
                #     for k in test_synsets_ditionary[curr_lemma].keys():
                #         try:
                #             path = nx.shortest_path(TG, k, node)
                #             nodes.update(path)
                #         except Exception as e:
                #             nodes.update(k)
                #             pass



                sub_TG = nx.subgraph(TG, nodes)

                if degree_heuristic:
                    probs = sub_TG.degree()
                else:
                    d = {n: 1 for n in sub_TG.nodes}
                    d.update({n:0 for n in test_synsets_ditionary[curr_lemma].keys()})
                    probs = nx.pagerank_scipy(TG)

                ln_nodes = len(sub_TG.nodes)

                max_prob = -1
                best_syn = 0
                for n in test_synsets_ditionary[curr_lemma].keys():
                    rank = probs[n]#/ln_nodes
                    if rank > max_prob:
                        max_prob = rank
                        best_syn = n
                assert (best_syn != 0)
                pre.append(best_syn)

                print(f1_score(pre, all, average='micro'))
        f1 = f1_score(pre, all, average='micro')
        results[eval_set] = f1
    return results


def contexGraphPrediction(G, test_set, test_synsets_ditionary, dumping = 0.85, contex=1, d_level = 0):

    assert(contex > -1)

    results = dict()
    TG = None

    if d_level == 0:
        TG = extendGraph(G, test_synsets_ditionary, document_graph=False)

    for eval_set in test_set.keys():
        pre = []
        true_syns = []
        if eval_set != 'semeval2007':
            continue
        print(eval_set)

        for sentence in test_set[eval_set].values():
            lemmas, synsets = utils.getDocumentsLemmas(sentence)

            ln_lemmas = len(lemmas)

            if d_level == 1:
                to_add = {}

                for l in lemmas:
                    to_add.update({l: test_synsets_ditionary[l]})
                TG = extendGraph(G, to_add, document_graph=False)

            for i in tqdm.tqdm(range(ln_lemmas)):

                associated_synset = list(test_synsets_ditionary[lemmas[i]].keys())

                true_syns.append(synsets[i])
                to_iter = np.arange(max(0, i - contex), min(ln_lemmas, i + contex + 1))

                if d_level == 2:
                    to_add = {}
                    for j in to_iter:
                        to_add.update({lemmas[j]: test_synsets_ditionary[lemmas[j]]})
                    TG = extendGraph(G, to_add, document_graph=False)

                assert(TG is not None)

                dizionario = {}
                for j in to_iter:
                    keys = test_synsets_ditionary[lemmas[j]].keys()
                    dizionario.update({k: 1 for k in keys})

                dizionario.update({k: 0 for k in associated_synset})

                pr = nx.pagerank_scipy(TG, personalization=dizionario)

                max_prob = 0
                best_syn = 0
                for w in test_synsets_ditionary[lemmas[i]].keys():
                    rank = pr[w]
                    if rank > max_prob:
                        max_prob = rank
                        best_syn = w
                pre.append(best_syn)
            print(f1_score(pre, true_syns, average='micro'))

        f1 = f1_score(pre, true_syns, average='micro')
        results[eval_set] = f1

    return results

def contexGraphPredictionVoting(G, test_set, test_synsets_ditionary, dumping = 0.85, contex=1, d_level = 1):

    assert(contex > -1)

    results = dict()
    TG = None


    # if d_level == 0:
    #     TG = extendGraph(G, test_synsets_ditionary, document_graph=False)

    for eval_set in test_set.keys():

        if eval_set != 'semeval2007':
            continue
        pre = []
        true_syns = []

        print(eval_set)

        for sentence in test_set[eval_set].values():

            voting = defaultdict(dict)

            lemmas, synsets = utils.getDocumentsLemmas(sentence)

            ln_lemmas = len(lemmas)

            if d_level == 1:
                to_add = {}

                for l in lemmas:
                    to_add.update({l: test_synsets_ditionary[l]})
                TG = extendGraph(G, to_add, document_graph=False)

            for i in tqdm.tqdm(range(ln_lemmas)):

                associated_synset = list(test_synsets_ditionary[lemmas[i]].keys())

                true_syns.append(synsets[i])
                to_iter = np.arange(max(0, i - contex), min(ln_lemmas, i + contex + 1))

                # if d_level == 2:
                #     to_add = {}
                #     for j in to_iter:
                #         to_add.update({lemmas[j]: test_synsets_ditionary[lemmas[j]]})
                #     TG = extendGraph(G, to_add, document_graph=False)

                assert(TG is not None)

                dizionario = {}
                for j in to_iter:
                    keys = test_synsets_ditionary[lemmas[j]].keys()
                    dizionario.update({k: 1 for k in keys})

                dizionario.update({k: 0 for k in associated_synset})

                pr = nx.pagerank_scipy(TG, personalization=dizionario)

                max_prob = 0
                best_syn = 0
                # for j in to_iter:
                l = lemmas[i]
                for w in test_synsets_ditionary[l].keys():
                    rank = pr[w]
                    if rank > max_prob:
                        max_prob = rank
                        best_syn = w
                # pre.append(best_syn)
                # print(lemmas[i], best_syn)
                if best_syn in voting[l]:
                    voting[l].update({ best_syn : voting[l][best_syn]+1})
                else:
                    voting[l].update({best_syn: 1})

            for l in lemmas:
                d = voting[l]
                print(d)
                max_key = max(d, key=lambda k: d[k])
                pre.append(max_key)
            print(pre, all)
        print(f1_score(pre, true_syns, average='micro'))

        print(voting)
        exit()
        f1 = f1_score(pre, true_syns, average='micro')
        results[eval_set] = f1

    return results

if __name__ == '__main__':

    d = utils.getTrainDataset('../semcor.data.xml', '../semcor.gold.key.bnids.txt')
    getWeightCoOc(d, '../semcor.gold.key.bnids.txt')
    # _, synsets = utils.getSynsetsDictionary('../semcor.gold.key.bnids.txt')
    # print(d.keys())
    # print(d)
    exit()
    synG = createGraph(semantic_relationships={}, graph_file='train_graph.adjlist')
    # relationships, _ = utils.getAssociatedSynsets(file='../data_train.json', testset=None, limit=0)
    # G = createDocumentsGraph(d, sem_rel=relationships, sem_graph=synG)
    # G = createGraph(semantic_relationships=[], graph_file='train_graph.adjlist')

    # print(len(G.nodes), len(G.edges))
    testset = utils.getEvalDataset('../ALL.data.xml', '../ALL.gold.key.bnids.txt')
    relationships, _ = utils.getAssociatedSynsets(file='../data_eval_WN.json', testset=testset, limit=0)

    # graphDegreePrediction(synG, testset, relationships, contex=10)
    #
    # exit()
    #
    # for i in [5, 10, 15, 20]:
    #     r = contexGraphPrediction(synG, testset, relationships, dumping=0.85, contex=i, d_level=0)
    #     print(i, 0, r)
    # for i in [0,1,2]:
    #     print(i, contexGraphPrediction(synG, testset, relationships, dumping=0.85, contex=30, d_level=i))
    # exit()
    #
    # r = contexGraphPredictionVoting(synG, testset, relationships, dumping=0.85, contex=10, d_level=1)
    # print(r)
    # exit()
    # r = contexGraphPrediction(synG, testset, relationships, dumping=0.85, contex=20, d_level=2)
    # print(r)
    # exit()

    for i in [40, 45,  50 ,55 ]:
        r = contexGraphPrediction(synG, testset, relationships, contex=i, d_level=0)
        print(i, 0, r)

    # for i in [5, 10, 15, 20,25,30]:
    #     r = contexGraphPrediction(synG, testset, relationships, contex=i, d_level=1)
    #     print(i, 1,  r)

    # for i in [35, 40, 45, 50]:
    #     r = contexGraphPrediction(synG, testset, relationships, contex=i, d_level=2)
    #     print(i, 2, r)

    # n = list(G.neighbors('bn:00083181v'))
    # n.extend(['bn:00083181v'])
    # n = list(G.neighbors('long_ADJ'))
    # n.extend(['long_ADJ'])
    # n.extend(list(G.neighbors('be_VERB')))
    # n.extend(['be_VERB'])
    # # for path in nx.all_shortest_paths(G, source='bn:00083181v', target='bn:00106124a'):
    # #     print(path)
    # H = G.subgraph(n)
    # # for path in nx.all_shortest_paths(G, source='long_ADJ', target='objective_NOUN'):
    # #     print(path)
    # # print(n)
    # print(H.nodes)
    # # cycls_3 = [c for c in nx.simple_cycles(H) if len(c) == 3]
    # # print(cycls_3)
    # # color = nx.get_node_attributes(G, 'subgraph')
    # # print(G.node['bn:00083181v']['subgraph'])
    # plotGraph(H, with_labels=True)