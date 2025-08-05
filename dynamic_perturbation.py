import numpy as np
import pandas as pd
from collections import Counter

df2 = pd.read_csv("./demo_data/hesc2/regulatory_t2.csv")
df3 = pd.read_csv("./demo_data/hesc2/regulatory_t3.csv")
df4 = pd.read_csv("./demo_data/hesc2/regulatory_t4.csv")
df5 = pd.read_csv("./demo_data/hesc2/regulatory_t5.csv")
df6 = pd.read_csv("./demo_data/hesc2/regulatory_t6.csv")


result2= [(row[0], row[1]) for row in df2.values]
result3= [(row[0], row[1]) for row in df3.values]
result4= [(row[0], row[1]) for row in df4.values]
result5= [(row[0], row[1]) for row in df5.values]
result6= [(row[0], row[1]) for row in df6.values]

network_list = [result2,result3,result4,result5,result6]

g2 = list(set(list(df2["TF"].values)+list(df2["Target"].values)))
g3 = list(set(list(df2["TF"].values)+list(df2["Target"].values)))
g4 = list(set(list(df2["TF"].values)+list(df2["Target"].values)))
g5 = list(set(list(df2["TF"].values)+list(df2["Target"].values)))
g6 = list(set(list(df2["TF"].values)+list(df2["Target"].values)))

all_genes  = list(set(g2+g3+g4+g5+g6))
#all_genes = all_genes[:500]
print(all_genes)
gene_len = len(all_genes)
print(len(all_genes))



# Function for calculating network entropy (based on node occurrence rating)
# def calculate_entropy(network):
#     #node_count = len(set(sum(network, ())))  #
#
#     #
#     node_probabilities = {}
#     for edge in network:
#         for node in edge:
#             node_probabilities[node] = node_probabilities.get(node, 0) + 1
#
#     node_probabilities = {k: v / len(network) for k, v in node_probabilities.items()}
#
#     # #
#     # node_freq = Counter(sum(network, ()))
#     # node_probabilities = {node: count / len(network) for node, count in node_freq.items()}
#     #print("???")
#
#     #
#     entropy = -sum(p * np.log(p) for p in node_probabilities.values())
#     return entropy



## Calculation based on degree distribution

from collections import Counter
import math


def calculate_entropy(network):
    all_nodes = [node for edge in network for node in edge]
    node_degrees = dict(Counter(all_nodes))
    total_nodes = len(node_degrees)
    degree_counts = dict(Counter(node_degrees.values()))
    entropy = 0
    for degree in degree_counts.values():
        #degree_count = list(node_degrees.values()).count(degree)
        probability = degree / total_nodes
        entropy -= probability * math.log2(probability)

    return entropy


def remove_gene_and_edges(network, gene):
    modified_network = [edge for edge in network if gene not in edge]
    return modified_network





entropy_list = [calculate_entropy(network) for network in network_list]

mean_entropy = np.mean(entropy_list)
std_entropy = np.std(entropy_list)

gene_scores = {}
for gene in all_genes:  # all_genes
    print("!!")
    perturbation_values = []
    for i, network in enumerate(network_list):
        modified_network = remove_gene_and_edges(network, gene)

        modified_entropy = calculate_entropy(modified_network)
        perturbation_values.append(modified_entropy)

    mean_perturbation = np.mean(perturbation_values)
    std_perturbation = np.std(perturbation_values)


    R = abs(mean_perturbation - mean_entropy) * abs(std_perturbation - std_entropy)
    #R = abs(std_perturbation - std_entropy)
    #R = abs(mean_perturbation - mean_entropy)
    gene_scores[gene] = R

    print("done!")

# Sort according to the disturbance assessment
sorted_genes = sorted(gene_scores.items(), key=lambda x: x[1], reverse=True)
print(sorted_genes[:20])
df = pd.DataFrame(sorted_genes,columns=['gene','R_score'])

df.to_csv("key_gene2.csv",index=False)
