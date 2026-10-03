import streamlit as st

st.title("About & methodology")

st.markdown("""
This demo accompanies a thesis comparing two ways of turning credit card transactions into a graph
for fraud detection. It uses Kaggle's simulated *Credit Card Transactions Fraud Detection* dataset
(`fraudTrain.csv`): all 7,506 fraudulent transactions plus a random 20% of genuine ones, 265,340 in total.

### Two graph representations
- **Bipartite**: customers and merchants are nodes. Every customer–merchant pair that transacted is one
  edge, weighted by the summed amount and labelled fraud if any of its transactions was fraud
  (1,676 nodes, 201,725 edges).
- **Tripartite**: every transaction also becomes its own node, linked to its customer and its merchant
  (267,016 nodes, 530,680 edges). Each transaction keeps its own label instead of being merged with others.

### From graph to prediction
1. **Node2Vec** runs short weighted random walks over the graph (10 walks of length 10 per node) and
   trains a Word2Vec skip-gram model on them. Each node gets a 64-dimensional vector, and nodes in
   similar neighbourhoods get similar vectors. Fraud labels are never used in this step.
2. **Edge operators** combine the two endpoint vectors of an edge into one: Hadamard (product),
   Average, L1 (|difference|) or L2 (squared difference).
3. A **Random Forest** classifies edges as fraud or genuine. It is trained on a 1:1 undersampled
   training set, because without rebalancing it predicted almost everything as genuine. It is
   evaluated on an untouched test set with the natural ~3% fraud rate.

In this demo, a transaction's score is the probability for its edge (bipartite), or the mean over its
two edges (tripartite).

### What the thesis found
- **Precision is low everywhere: 0.03 to 0.09.** Even the best models flag many genuine transactions
  for each fraud they catch, while recall is 0.53 to 0.73.
- **The tripartite graph wins with Hadamard and Average** (ROC-AUC 0.805–0.824, PR-AUC 0.27–0.28
  against a 0.028 baseline) **but loses with L1 and L2**, where it is close to chance (ROC-AUC 0.563).
  The bipartite graph is more consistent across operators (ROC-AUC 0.659–0.748).
- Neither topology is uniformly better, and unsupervised K-Means barely beats random (lift 1.04–1.52).

### What this demo cannot do
Node2Vec is **transductive**: it only has vectors for nodes it was trained on. A brand-new customer or
merchant has no embedding, so this demo substitutes the average embedding of that kind of entity and
labels the result *low confidence* (or *unavailable* if both are new). In the tripartite graph, every
new transaction is itself a new node; its vector is estimated from the same pair's earlier
transactions (nodes with exactly the same neighbours) or, failing that, from the customer's and
merchant's other transactions, and labelled *approximated*. An inductive method (e.g. GraphSAGE)
would be needed to embed new nodes properly.

The amount you enter is not a classifier input: it only influenced the embeddings, through the edge
weights of the random walks.

### Two models
- **Thesis Model**: the exact configuration in the thesis (100 trees, default settings).
- **Enhanced Model (tuned)**: same embeddings, same split and same training set, but the Random Forest
  hyperparameters were chosen by `GridSearchCV` (5-fold CV, scored by average precision, i.e. PR-AUC).
  Its test-set numbers are on the **Model Card** page next to the thesis numbers.
""")
