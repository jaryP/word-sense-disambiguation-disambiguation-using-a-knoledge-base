from uu import test

from codice import utils
from codice import graph

train = utils.getTrainDataset(corpus='../semcor.data.xml', keysfile='../semcor.gold.key.bnids.txt')
train_rel, lemmas = utils.getSemanticRelationships(file='../data_train.json', keyFile= '../semcor.gold.key.bnids.txt', limit=0)

testset = utils.getEvalDataset('../ALL.data.xml', '../ALL.gold.key.bnids.txt')
relationships, _ = utils.getAssociatedSynsets(file='../data_eval_WN.json', testset=None, limit=0)

G = graph.createGraph(semantic_relationships=train_rel, graph_file='train_graph.adjlist')

print('Normal Graph')
predictions = graph.staticPagerankPrediction(G, testset, eval_synsets_dictionary=dict(relationships), pagerank_algo='static')
print('Static predictions:', predictions)
#
# predictions = graph.staticPagerankPrediction(G, testset, test_synsets_ditionary=relationships, pagerank_algo='mass')
# print('Static mass predictions:', predictions)
#
# predictions_documents = graph.documentPagerankPrediction(G, testset, relationships)
# print('Documets prediction: ', predictions_documents)

# predictions = graph.graphPathsPrediction(G, testset, test_synsets_ditionary=relationships, cut=2)
# print('BFS prediction:', predictions)


# coG = G.copy()
# coG.add_weighted_edges_from(graph.getWeightCoOc(corpus=train, synsets_file='../semcor.gold.key.bnids.txt', win_size=10))


print('coOcc Graph')
# predictions = graph.staticPagerankPrediction(coG, testset, test_synsets_ditionary=relationships, pagerank_algo='static')
# print('Static predictions:', predictions)
#
# predictions = graph.staticPagerankPrediction(coG, testset, test_synsets_ditionary=relationships, pagerank_algo='mass')
# print('Static mass predictions:', predictions)
#
# predictions_documents = graph.documentPagerankPrediction(coG, testset, relationships)
# print('Documets prediction: ', predictions_documents)

# predictions = graph.graphPathsPrediction(coG, testset, test_synsets_ditionary=relationships, cut=2)
# print('BFS prediction:', predictions)

for i in range(1,10):
    coG = G.copy()
    coG.add_weighted_edges_from(
        graph.getWeightCoOc(corpus=train, synsets_file='../semcor.gold.key.bnids.txt', win_size=i))
    print(i, graph.documentPagerankPrediction(coG, testset, eval_synsets_dictionary=relationships))

# z = {**train_rel, **relationships}
# v = utils.getTestDataset('../test_data.txt', '../test_set.json', ret = True, semantic_rel_know=z)
# graph.graphPathTest(G, v[0], v[1], '1544973_test_answer.txt', 1)

# nasari, co occurrence, dependeny parse


''' 
senseval2 2282
646
961
716

sensewval3 1850
716
598
536

semeval07 455 2 h circa
111
150
194

semeval13 1644 5h circa

semeval15 1022 3.5h circa

'''