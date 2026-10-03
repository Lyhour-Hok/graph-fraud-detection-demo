"""Published numbers from the thesis, copied verbatim (Chapters 3 and 4).

These are what the Model Card shows as "published in the thesis". The models
served by the website are a local re-run of the same code; their own measured
metrics are stored next to these in manifest.json.
"""

# Table 3.7: Characteristics of the bipartite and tripartite graphs
GRAPH_STATS = {
    "bipartite":  {"customers": 983, "merchants": 693, "transactions": 0,
                   "nodes": 1676, "edges": 201725, "fraud_edges": 7391},
    "tripartite": {"customers": 983, "merchants": 693, "transactions": 265340,
                   "nodes": 267016, "edges": 530680, "fraud_edges": 15012},
}

# Table 3.11: Train/test split sizes
SPLIT = {
    "bipartite":  {"train": 141207, "train_fraud": 5174,  "test": 60518,  "test_fraud": 2217},
    "tripartite": {"train": 371476, "train_fraud": 10508, "test": 159204, "test_fraud": 4504},
}

# Table 4.1: Random Forest results (test set), with Table 4.2 confusion-matrix counts
RF_RESULTS = [
    {"graph": "bipartite",  "operator": "hadamard", "precision": 0.078, "recall": 0.681, "f1": 0.140, "roc_auc": 0.748, "pr_auc": 0.158, "TP": 1510, "FP": 17784, "FN": 707,  "TN": 40517},
    {"graph": "bipartite",  "operator": "average",  "precision": 0.079, "recall": 0.666, "f1": 0.141, "roc_auc": 0.736, "pr_auc": 0.105, "TP": 1476, "FP": 17236, "FN": 741,  "TN": 41065},
    {"graph": "bipartite",  "operator": "l1",       "precision": 0.058, "recall": 0.606, "f1": 0.106, "roc_auc": 0.659, "pr_auc": 0.078, "TP": 1343, "FP": 21815, "FN": 874,  "TN": 36486},
    {"graph": "bipartite",  "operator": "l2",       "precision": 0.058, "recall": 0.604, "f1": 0.106, "roc_auc": 0.659, "pr_auc": 0.078, "TP": 1339, "FP": 21719, "FN": 878,  "TN": 36582},
    {"graph": "tripartite", "operator": "hadamard", "precision": 0.086, "recall": 0.721, "f1": 0.154, "roc_auc": 0.805, "pr_auc": 0.280, "TP": 3246, "FP": 34463, "FN": 1258, "TN": 120237},
    {"graph": "tripartite", "operator": "average",  "precision": 0.089, "recall": 0.734, "f1": 0.159, "roc_auc": 0.824, "pr_auc": 0.266, "TP": 3306, "FP": 33816, "FN": 1198, "TN": 120884},
    {"graph": "tripartite", "operator": "l1",       "precision": 0.034, "recall": 0.526, "f1": 0.064, "roc_auc": 0.563, "pr_auc": 0.035, "TP": 2370, "FP": 66832, "FN": 2134, "TN": 87868},
    {"graph": "tripartite", "operator": "l2",       "precision": 0.034, "recall": 0.526, "f1": 0.064, "roc_auc": 0.563, "pr_auc": 0.035, "TP": 2368, "FP": 67069, "FN": 2136, "TN": 87631},
]
PR_AUC_BASELINE = {"bipartite": 0.037, "tripartite": 0.028}

# Table 4.3: K-Means results (k = 2)
KMEANS_RESULTS = [
    {"graph": "bipartite",  "operator": "hadamard", "precision": 0.039, "recall": 0.567, "f1": 0.072, "lift": 1.056, "ARI": -0.001},
    {"graph": "bipartite",  "operator": "average",  "precision": 0.044, "recall": 0.431, "f1": 0.080, "lift": 1.204, "ARI": 0.006},
    {"graph": "bipartite",  "operator": "l1",       "precision": 0.055, "recall": 0.292, "f1": 0.092, "lift": 1.496, "ARI": 0.024},
    {"graph": "bipartite",  "operator": "l2",       "precision": 0.056, "recall": 0.194, "f1": 0.086, "lift": 1.516, "ARI": 0.026},
    {"graph": "tripartite", "operator": "hadamard", "precision": 0.038, "recall": 0.365, "f1": 0.069, "lift": 1.341, "ARI": 0.011},
    {"graph": "tripartite", "operator": "average",  "precision": 0.032, "recall": 0.646, "f1": 0.060, "lift": 1.117, "ARI": -0.002},
    {"graph": "tripartite", "operator": "l1",       "precision": 0.030, "recall": 0.552, "f1": 0.057, "lift": 1.058, "ARI": -0.000},
    {"graph": "tripartite", "operator": "l2",       "precision": 0.029, "recall": 0.863, "f1": 0.057, "lift": 1.038, "ARI": -0.007},
]

# Table 4.4: Random Forest, tripartite graph, transaction-grouped split
GROUPED_RESULTS = [
    {"operator": "hadamard", "precision": 0.080, "recall": 0.717, "f1": 0.143, "roc_auc": 0.798, "pr_auc": 0.262},
    {"operator": "average",  "precision": 0.082, "recall": 0.725, "f1": 0.147, "roc_auc": 0.814, "pr_auc": 0.232},
    {"operator": "l1",       "precision": 0.031, "recall": 0.560, "f1": 0.059, "roc_auc": 0.549, "pr_auc": 0.033},
    {"operator": "l2",       "precision": 0.031, "recall": 0.561, "f1": 0.059, "roc_auc": 0.548, "pr_auc": 0.033},
]
