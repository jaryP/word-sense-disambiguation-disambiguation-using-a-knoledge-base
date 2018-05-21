from codice import utils
from codice import graph
from collections import  Counter
import time
import argparse
import networkx as nx

# parser = argparse.ArgumentParser(description='Jary Pomponi, NLP HW2: WSD using an approach based on graphs')
#
# parser.add_argument('--train_text',  nargs='?', default='../semcor.data.xml',
#                     help='Train text data set')
# parser.add_argument('--train_text_keys',  nargs='?', default='../semcor.gold.key.bnids.txt',
#                     help='Train keys data set')
# parser.add_argument('--graph_path',  nargs='?', default=None,
#                     help='The path from wich load the graph or save it if the graph does not exists')
# parser.add_argument('--train_dict', nargs='?', default='../data_train.json',
#                     help='The already created dictionary of train dataset')
# parser.add_argument('--test_dict', nargs='?', default='../data_train.json',
#                     help='The already created dictionary of train dataset')

train = utils.getTrainDataset(corpus='../semcor.data.xml', keysfile='../semcor.gold.key.bnids.txt')
train_rel, lemmas = utils.getSemanticRelationships(file='../data_train.json', keyFile= '../semcor.gold.key.bnids.txt', limit=0)

testset = utils.getEvalDataset('../ALL.data.xml', '../ALL.gold.key.bnids.txt')
relationships, _ = utils.getAssociatedSynsets(file='../data_eval_WN.json', testset=None, limit=0)

G = graph.createGraph(semantic_relationships=train_rel, graph_file='train_graph.adjlist')

print('Normal Graph')
predictions = graph.staticPagerankPrediction(G, testset, test_synsets_ditionary=relationships, pagerank_algo='static')
print('Static predictions:', predictions)

predictions = graph.staticPagerankPrediction(G, testset, test_synsets_ditionary=relationships, pagerank_algo='mass')
print('Static mass predictions:', predictions)

predictions_documents = graph.documentPagerankPrediction(G, testset, relationships)
print('Documets prediction: ', predictions_documents)


coG = G.copy()
coG.add_weighted_edges_from(graph.getWeightCoOc(corpus=train, synsets_file='../semcor.gold.key.bnids.txt', win_size=10))


print('coOcc Graph')
predictions = graph.staticPagerankPrediction(coG, testset, test_synsets_ditionary=relationships, pagerank_algo='static')
print('Static predictions:', predictions)

predictions = graph.staticPagerankPrediction(coG, testset, test_synsets_ditionary=relationships, pagerank_algo='mass')
print('Static mass predictions:', predictions)

predictions_documents = graph.documentPagerankPrediction(coG, testset, relationships)
print('Documets prediction: ', predictions_documents)

predictions = graph.graphBFSprediction(coG, testset, test_synsets_ditionary=relationships, cut=6, degree_heuristic = False)
print('BFS prediction:', predictions)


# nasari, co occurrence, dependeny parse