from code import utils
from code import graph
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

# train = utils.getTrainDataset(corpus='../semcor.data.xml', keysfile='../semcor.gold.key.bnids.txt')
train_rel, lemmas = utils.getSemanticRelationships(file='../data_train.json', keyFile= '../semcor.gold.key.bnids.txt', limit=0)
# print(len(relationships), len(lemmas))

testset = utils.getEvalDataset('../ALL.data.xml', '../ALL.gold.key.bnids.txt')
relationships, _ = utils.getAssociatedSynsets(file='../data_eval_WN.json', testset=None, limit=0)

G = graph.createGraph(semantic_relationships=train_rel, graph_file='train_graph.adjlist')
# cycls_3 = [c for c in nx.cycle_basis(G) if len(c)==3]
#
# print(cycls_3)
# print(G.has_edge(cycls_3[0][0], cycls_3[0][-1]))
#
# val = dict()
# for c in cycls_3:
#     coppia = (c[0],c[1])
#     coppia1 = (c[1],c[0])
#
#     if coppia in val:
#         val[coppia]+=1
#     else:
#         val[coppia] = 1
#
# for c in cycls_3:
#     coppia1 = (c[1], c[0])
#
#     if coppia1 in val:
#         print('DENTRO')
# # for c in val.keys():
# #     print(c, val[c])
#
#
# exit()

# predictions = graph.graphBFSprediction(G, testset, test_synsets_ditionary=relationships)
# print('Graph BFS predictions:', predictions)
# exit()
#
# predictions = graph.graphDegreePrediction(G, testset, test_synsets_ditionary=relationships, contex = 4)
# print('Graph degree predictions:', predictions)
# exit()
predictions = graph.staticPagerankPrediction(G, testset, test_synsets_ditionary=relationships, pagerank_algo='static')
print('Static predictions:', predictions)

predictions = graph.staticPagerankPrediction(G, testset, test_synsets_ditionary=relationships, pagerank_algo='mass')
print('Static mass predictions:', predictions)

predictions_documents = graph.documentPagerankPrediction(G, testset, relationships)
print('Documets prediction: ', predictions_documents)

# predictions_documents = graph.documentPagerankPrediction1(G, testset, relationships)
# print('Documets prediction1: ', predictions_documents)
#
# # predictions_dyn = graph.dynamicPagerankPrediction(G, testset, relationships, contex=20)
# print('Dynamic predictions:', predictions_dyn)

# graph.dynamicPagerankPrediction(G,testset,relationships,contex=0)
# print(predictions)
# print(Counter(predictions.values()))
# # print(time.time() - s)
# # print(relationships['make_VERB'])
# print(utils.calculateScores(testset, predictions))

# graph.plotGraph(G.subgraph(neig))
# nasari, co occurrence, dependeny parse

# bn:00029424n