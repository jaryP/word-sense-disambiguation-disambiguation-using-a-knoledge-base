import xml.etree.ElementTree as ET
import urllib.request as urllib2
import json
from collections import defaultdict
import os
import tqdm
from sklearn.metrics import f1_score
from itertools import islice


class instance:
    def __init__(self, value, lemma, pos, key):
        self.value = value
        self.lemma = lemma
        self.pos = pos
        self.instance = key

    def __str__(self):
        return "value: {}, lemma: {}, pos: {}, instance {}".format(self.value, self.lemma, self.pos, self.instance)

    def __hash__(self):
        return hash(self.instance)

    def __eq__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return self.instance == other.instance


class wf:
    def __init__(self, value, lemma, pos):
        self.value = value
        self.lemma = lemma
        self.pos = pos

    def __str__(self):
        return "value: {}, lemma: {}, pos: {}".format(self.value, self.lemma, self.pos)


def getSynsetsDictionary(file):
    """

    :param file:
    :return: 
    """
    keyDict = dict()
    synsets = set()
    with open(file, 'r') as f:
        for line in f:
            s = line.split()
            keyDict[s[0]] = s[1]
            synsets.add(s[1])
    return keyDict, synsets


def getTrainDataset(corpus, keysfile):

    root = ET.parse(corpus).getroot()

    keydict, _ = getSynsetsDictionary(keysfile)

    documents = defaultdict(list)
    for neighbor in root.iter('sentence'):
        sentid = neighbor.get('id').split('.')
        sent = []
        for a in neighbor:
            if a.tag == 'wf':
                sent.append(wf(a.text, a.get('lemma'), a.get('pos')))
            else:
                sent.append(instance(a.text, a.get('lemma'), a.get('pos'), keydict[a.get('id')]))
        documents[sentid[0]].append(sent)

    return dict(documents)


def getEvalDataset(corpus, keysfile):

    root = ET.parse(corpus).getroot()

    keydict, _ = getSynsetsDictionary(keysfile)

    datasets = defaultdict(dict)

    for text in root.iter('text'):
        sentid = text.get('id').split('.')
        sent = []
        for neighbor in text.iter('sentence'):
            for a in neighbor:
                if a.tag == 'wf':
                    sent.append(wf(a.text, a.get('lemma'), a.get('pos')))
                else:
                    sent.append(instance(a.text, a.get('lemma'),a.get('pos'), keydict[a.get('id')]))
        datasets[sentid[0]].update({sentid[1]:sent})

            # datasets[sentid[0]] = documents

    return datasets


def getDocumentsLemmas(document):
    lemmas = []
    synsets = []
    for w in document:
        if isinstance(w,instance):
            lemmas.append(w.lemma+'_'+w.pos)
            synsets.append(w.instance)
    return lemmas, synsets


def getAssociatedSynsetsBabelnet(lemma, postag, key):
    ur = 'https://babelnet.io/v5/getSynsetIds?lemma={}&searchLang={}&pos={}&source=WN&key={}'.format(lemma, 'EN', postag, key)
    request = urllib2.Request(ur)
    response = urllib2.urlopen(request)

    data = response.read()
    encoding = response.info().get_content_charset('utf-8')
    JSON_object = json.loads(data.decode(encoding))

    ids = []

    if 'message' in JSON_object:
        print( JSON_object)
        return -1

    for result in JSON_object:
        ids.append(result['id'])

    return ids


def getSemanticRelatioshipBabelnet(sid, key, allowedId = []):

    ur = 'https://babelnet.io/v5/getOutgoingEdges?id=' + sid + '&key=' + key
    request = urllib2.Request(ur)
    response = urllib2.urlopen(request)

    data = response.read()
    encoding = response.info().get_content_charset('utf-8')
    JSON_object = json.loads(data.decode(encoding))

    d = defaultdict(list)

    if 'message' in JSON_object:
        print(JSON_object)
        return -1

    for result in JSON_object:
        language = result.get('language')
        if language != 'EN':
            continue
        pointer = result['pointer']
        target = str(result.get('target'))
        if len(allowedId) != 0 and target not in allowedId:
            continue
        short = str(pointer.get('shortName'))
        d[short].append(target)

    return dict(d)


def getSemanticRelationships(file, keyFile, limit = -1):

    babelkey = '32796f83-09b8-4c0f-8190-f57069a8f3cf'

    d = dict()
    if os.path.isfile(file):
        with open(file) as fp:
            data = fp.read()

            d = json.loads(data)

    keys = list(d.keys())

    if limit == 0:
        return d, keys

    _, synset = getSynsetsDictionary(keyFile)

    com = []
    for syn in synset:
        if syn not in keys:
            com.append(syn)
    i = 0
    for sis in tqdm.tqdm(com):
        try:
            v = getSemanticRelatioshipBabelnet(sis, babelkey)
        except Exception as e:
            print('\nError:', end=' ')
            print(e)
            continue
        if v == -1:
            continue
        d[sis] = v
        i+=1
        if i % 20 == 0:
            with open(file, 'w') as fp:
                data = json.dumps(d)
                fp.write(data)
        if i >= limit and limit != -1:
            break
    with open(file, 'w') as fp:
        data = json.dumps(d)
        fp.write(data)

    return d, list(d.keys())


def getAssociatedSynsets(file, testset=None, semantic_rel_know=dict(), limit=-1):

    babelkey = '32796f83-09b8-4c0f-8190-f57069a8f3cf'

    d = dict()
    if os.path.isfile(file):
        with open(file) as fp:
            data = fp.read()
            d = json.loads(data)

    if limit == 0:
        return d, list(d.keys())

    assert (testset is not None)

    data = set()
    for key in testset.keys():
        for doc, sentences in testset[key].items():
            for w in sentences:
                if isinstance(w, instance):
                    tosearch_string = w.lemma.lower()+'_'+w.pos
                    if tosearch_string not in d:
                        data.add((w.lemma.lower(), w.pos, tosearch_string))
    i = 0
    for lemma in tqdm.tqdm(data):
        try:
            v = getAssociatedSynsetsBabelnet(lemma=lemma[0], postag=lemma[1], key=babelkey)
        except Exception as e:
            print('\nError:', end=' ')
            print(e)
            continue
        if v == -1:
            continue
        connections = {}
        for synset in v:
                connections[synset] = semantic_rel_know.get(synset, getSemanticRelatioshipBabelnet(synset, babelkey))
        d[lemma[2]] = connections
        i += 1
        if i % 20 == 0:
            with open(file, 'w') as fp:
                fp.write(json.dumps(d))
        if i >= limit and limit != -1:
            break

    with open(file, 'w') as fp:
        fp.write(json.dumps(d))

    return d, list(d.keys())


def calculateScores(dataset, predictions):

    res = dict.fromkeys(dataset.keys(), (0, 0))

    for eval_set in dataset.keys():
        pre = []
        all = []
        for sentence in dataset[eval_set].values():
            for word in sentence:
                if isinstance(word, instance):
                    tosearch_string = word.lemma.lower()+'_'+word.pos
                    pre.append(predictions[tosearch_string])
                    all.append(word.instance)
        res[eval_set] = f1_score(pre,all,average='micro')

    return res



if __name__ == '__main__':
    pass
    # s, _ = getSemanticRelationships(file='../data_train_prova.json', keyFile= '../semcor.gold.key.bnids.txt', limit=0)
    # testset = getEvalDataset('../ALL.data.xml', '../ALL.gold.key.bnids.txt')
    # print(len(testset['senseval2']['d000']))
    # print(len(getDocumentsLemmas(testset['senseval2']['d000']))+len(getDocumentsLemmas(testset['senseval2']['d001']))+len(getDocumentsLemmas(testset['senseval2']['d002'])))
    # # print(type(testset))
    # d, k = getAssociatedSynsets(file='../data_eval_WN.json', testset=testset, limit=-1, semantic_rel_know=s)
    # print(len(d))
    # # print(saveSemanticRelationships(file='../data_test.json', keyFile= ['../ALL.gold.key.bnids.txt'], limit=-1))
    d = getTrainDataset('../semcor.data.xml', '../semcor.gold.key.bnids.txt')
    print(len(d['d000']))
    # # # a,b =getSynsetsDictionary( '../semcor.gold.key.bnids.txt')
    # # print(list(d.keys()))
    # # print(len(d[list(d.keys())[0]]))
    # # for key in d.keys():
    # #     i = 0
    # #     for doc in d[key]:
    # #         for w in doc:
    # #             if isinstance(w, instance):
    # #                 i+=1
    # #     print(key, i)
    # # for s in d['d000']:
    # #     for w in s:
    # #         print(w,end='-')
    # #     print()
    # # print(getAssociatedSynsetsBabelnet('apple','NOUN','32796f83-09b8-4c0f-8190-f57069a8f3cf'))
    # # if os.path.isfile('../data_eval.json'):
    # #     with open('../data_eval.json') as fp:
    # #         data = fp.read()
    # #         d = json.loads(data)
    # # print(len(d))
    # # d.pop('test_VERB')
    # # print(len(d))
    # # with open('../data_eval.json', 'w') as fp:
    # #     fp.write(json.dumps(d))
    # testset = getDataset('../ALL.data.xml', '../ALL.gold.key.bnids.txt')
    # from collections import defaultdict
    # a = {}
    # # a = defaultdict(lambda: 0, testset.keys())
    # a = dict.fromkeys(testset.keys(), (0,0))
    # # print(dict(testset.keys()))
    # print(len(testset['senseval2']))
    # a['senseval2'] = 'ciao'
    # print(a)
    # a = [1, 2, 3, 4, 5, 6, 7, 8]
    # for i in window1(a,3):
    #     print(i)
    #

