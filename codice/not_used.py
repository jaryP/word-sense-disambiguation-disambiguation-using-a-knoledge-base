
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

             for l in lemmas:
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
        for sentence in test_set[eval_set].values():

            lemmas, synsets = utils.getDocumentsLemmas(sentence)

            ln_lemmas = len(lemmas)

            to_add = {}

            for i in tqdm.tqdm(range(ln_lemmas)):
                truth = list(test_synsets_ditionary[lemmas[i]].keys())
                to_iter = np.arange(max(0, i-contex), min(ln_lemmas, i+contex+1))
                near = set()
                all.append(synsets[i])

                for window in to_iter:
                    l = lemmas[window]
                    keys = test_synsets_ditionary[l].keys()
                    near.update(keys)
                    to_add[l] = test_synsets_ditionary[l]

                TG = extendGraph(G, to_add)
                dizionario = {k:1 for k in near}
                dizionario.update({k: 0 for k in truth})

                pr = nx.pagerank_scipy(TG, alpha=dumping, personalization=dizionario)

                max_prob = 0
                best_syn = 0
                for s in truth:
                    rank = pr[s]
                    if rank > max_prob:
                        max_prob = rank
                        best_syn = s
                pre.append(best_syn)
             print(f1_score(pre, all, average='micro'))

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

            for l in tqdm.tqdm(lemmas):

                diz = dizionario.copy()
                lemma_syns = list(test_synsets_ditionary[l].keys())

                for syns in lemma_syns:
                    diz.update({syns: 0})

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
