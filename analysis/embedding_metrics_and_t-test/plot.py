import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


metrics = ['cosine_distance', 'manhattan_distance', 'euclidean_distance']


# Load the JSON data from the file
with open('/home/weisi/TemporalAssessment/analysis/t-test/1_t-test-results.json', 'r') as file:
    data = json.load(file)

# Extracting the keys that contain the temporal comparisons
#keys = [key for key in data.keys() if '_' in key]
keys=['T1_T1', 'T1_T2', 'T1_T3', 'T1_T4', 'T2_T2', 'T2_T3', 'T2_T4', 'T3_T3', 'T3_T4', 'T4_T4']

# Create an empty matrix for the distances
matrix_size = 4
# Assuming it's a square matrix
#cosine_similarity_matrix = np.zeros((matrix_size, matrix_size))
cosine_distance_matrix = np.full((matrix_size, matrix_size), np.nan)
manhattan_distance_matrix = np.full((matrix_size, matrix_size), np.nan)
euclidean_distance_matrix = np.full((matrix_size, matrix_size), np.nan)

# Filling the matrices with the corresponding values
for key in keys:
    i=int(key[1:].split('_T')[1])
    j=int(key[1:].split('_T')[0])
    cosine_distance_matrix[i-1, j-1] = data[key]['metrics']['cosine_distance']
    manhattan_distance_matrix[i-1, j-1] = data[key]['metrics']['manhattan_distance']
    euclidean_distance_matrix[i-1, j-1] = data[key]['metrics']['euclidean_distance']
print(cosine_distance_matrix)
print(manhattan_distance_matrix)
print(euclidean_distance_matrix)
# Define a function to plot a heatmap from a matrix
def plot_heatmap(matrix, title):
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap='coolwarm', cbar=False)
    plt.title(title)
    plt.xlabel('Temporal Domain')
    plt.ylabel('Temporal Domain')
    plt.show()
    plt.savefig(f'{title}.png')
    plt.close()

# Plotting the heatmaps for each distance measure
#plot_heatmap(cosine_similarity_matrix, 'Cosine Similarity')
plot_heatmap(cosine_distance_matrix, 'Cosine Distance of BioASQ')
plot_heatmap(manhattan_distance_matrix, 'Manhattan Distance of BioASQ')
plot_heatmap(euclidean_distance_matrix, 'Euclidean Distance of BioASQ')
