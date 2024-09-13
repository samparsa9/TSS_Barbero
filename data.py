import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#some methods for visualization

#take one sequence and display a heat map of the base pairs
def heatmap_single_seq(sequence):
    plt.figure(figsize=(15, 5))
    sns.heatmap(sequence.T, cmap="Blues", cbar=False, xticklabels=False, yticklabels=['A', 'C', 'G', 'T'])
    plt.title('One-Hot Encoded Sequence (Sample 0)')
    plt.ylabel('Nucleotide')
    plt.xlabel('Position in Sequence')
    plt.show()

#get distribution at a certain position of one hot encodings
def distribution(pos_sequences, neg_sequences, position):
    pos_nucleotide_counts = np.sum(pos_sequences[:, position, :], axis=0)
    neg_nucleotide_counts = np.sum(neg_sequences[:, position, :], axis=0)

    nucleotides = ['A', 'C', 'G', 'T']

    df = pd.DataFrame({
        'Nucleotide': nucleotides,
        'Positive Counts': pos_nucleotide_counts,
        'Negative Counts': neg_nucleotide_counts
    })

    plt.figure(figsize=(12, 6))

    #positive sequences
    plt.subplot(1, 2, 1)
    sns.barplot(x='Nucleotide', y='Positive Counts', data=df)
    plt.title('Nucleotide Distribution at Position 500 (Positive Sequences)')

    #negative sequences
    plt.subplot(1, 2, 2)
    sns.barplot(x='Nucleotide', y='Negative Counts', data=df)
    plt.title('Nucleotide Distribution at Position 500 (Negative Sequences)')

    plt.tight_layout()
    plt.show()

def y_train_distribution(y_train):
    plt.figure(figsize=(10, 6))

    # Plot the labels: '1' for positive and '0' for negative
    plt.plot(y_train, label='Label (1=Positive, 0=Negative)', linestyle='-', marker='o', markersize=2, color='b')

    # Add labels and title
    plt.title('Distribution of Positive (1) and Negative (0) Sequences in y_train')
    plt.xlabel('Index of Sequence')
    plt.ylabel('Label (1=Positive, 0=Negative)')
    plt.show()


with open('X_train_TSS_1to1.pkl', 'rb') as f:
    x_train = pickle.load(f)

with open('y_train_TSS_1to1.pkl', 'rb') as f:
    y_train = pickle.load(f)


#print(f"x_train shape: {x_train.shape}")
#print(f"x_train shape: {y_train.shape}")


positive_sequences = x_train[y_train == 1]
negative_sequences = x_train[y_train == 0]

#print(positive_sequences.shape)
#print(negative_sequences.shape)

#distribution(positive_sequences, negative_sequences, 900)

y_train_distribution(y_train)
