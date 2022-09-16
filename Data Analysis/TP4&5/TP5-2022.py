import matplotlib.pyplot as plt
import numpy as np
from random import randrange
from scipy import stats
import pandas as pd
import seaborn as sns
from itertools import combinations
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.manifold import LocallyLinearEmbedding, MDS, Isomap, TSNE



#B digits
digits = pd.read_csv("digits.csv",header=None,index_col=False)
print(digits.head())
print(digits.shape)
nb_samples, nb_features = digits.shape
nb_features -= 1
digit_class=digits.iloc[:, 64]
dig_list=np.unique(digit_class.values)
digits=digits.iloc[:, 0:63]
pca = PCA(n_components=2)
digits_pca = pca.fit_transform(digits)
print(dig_list)

#Show the PCA results
fig, ax = plt.subplots();
for dig in dig_list:
    ax.scatter(digits_pca[digit_class==dig, 0], digits_pca[digit_class==dig, 1],label=dig);
ax.legend();    
ax.set_title('PCA space', fontsize=12);

#TSNE
model = TSNE(n_components=2,perplexity=100, random_state=0)
digits_tsne = model.fit_transform(digits)
#Show the TSNE results
fig, ax = plt.subplots();
for dig in dig_list:
    ax.scatter(digits_tsne[digit_class==dig, 0], digits_tsne[digit_class==dig, 1],label=dig);
ax.legend();    
ax.set_title('TSNE space', fontsize=12);



#C
#1. Load the Golub data
df = pd.read_csv("golub_data.csv").T
nb_samples, nb_features = df.shape
print("{0} samples and {1} features:".format(nb_samples, nb_features))
print(df[:5])

#2. Open the associated labels
df_labels = pd.read_csv("golub_class2.csv", header = None)
nb_samples, _ = df_labels.shape
print("{0} samples:".format(nb_samples))
print(df[:5])
lab_names = np.unique(df_labels.values[:, 1])

#3. PCA on the Golub data

#Get the feature names
feat_names = df.columns.tolist()

#Center and reduce the Golub data
scaler = StandardScaler()
dfn = df.copy()
dfn[feat_names] = scaler.fit_transform(df[feat_names])

#pca
pca = PCA(n_components=2, random_state=0)
df_new = pca.fit_transform(df[feat_names])

#Ratio of explained variance
pc1 = int(round(pca.explained_variance_ratio_[0] * 100))
pc2 = int(round(pca.explained_variance_ratio_[1] * 100))

#Show the result
fig, ax = plt.subplots()
for lab_name in lab_names:
    ax.scatter(df_new[df_labels.values[:, 1]==lab_name, 0],
               df_new[df_labels.values[:, 1]==lab_name, 1], label=lab_name)
ax.legend()
ax.set_xlabel("PC1 ({0}% of explained variance)".format(pc1, fontsize=12))
ax.set_ylabel("PC2 ({0}% of explained variance)".format(pc2, fontsize=12))
ax.set_title('PCA space', fontsize=12);

#4. MDS

mds = MDS(random_state=0)
df_new = mds.fit_transform(dfn[feat_names])

#Show the result
fig, ax = plt.subplots()
for lab_name in lab_names:
    ax.scatter(df_new[df_labels.values[:, 1]==lab_name, 0],
               df_new[df_labels.values[:, 1]==lab_name, 1], label=lab_name)
ax.legend()
ax.set_xlabel('x\'', fontsize=12);
ax.set_ylabel('y\'', fontsize=12);
ax.set_title('MDS space', fontsize=12);


  
#5. Apply the Isomap dimension reduction technique

nb_lines = 2
nb_cols = 3

fig, ax = plt.subplots(nb_lines, nb_cols, figsize=(15, 15))
line_ids, col_ids = np.meshgrid(range(0, nb_lines), 
                                 range(0, nb_cols))
fig.suptitle('Isomap space', fontsize=12)

for n_neighbor, line_id, col_id in zip([4, 8, 10, 13, 16, 20],
                                       line_ids.flatten(), col_ids.flatten()):
    model = Isomap(n_neighbors=n_neighbor)
    df_new = model.fit_transform(df[feat_names])
    for lab_name in lab_names:
        ax[line_id, col_id].scatter(df_new[df_labels.values[:, 1]==lab_name, 0],
                                    df_new[df_labels.values[:, 1]==lab_name, 1], 
                                    label=lab_name)
    ax[line_id, col_id].set_title("{0} neighbors".format(n_neighbor))  
    
 
#6. LLE with different neighbors sizes
nb_lines = 2
nb_cols = 3

fig, ax = plt.subplots(nb_lines, nb_cols, figsize=(15, 15))
line_ids, col_ids = np.meshgrid(range(0, nb_lines), 
                                 range(0, nb_cols))
fig.suptitle('LLE space', fontsize=12)

for n_neighbor, line_id, col_id in zip([3, 5, 8, 10, 12, 15],
                                       line_ids.flatten(), col_ids.flatten()):
    model = LocallyLinearEmbedding(n_neighbors=n_neighbor, random_state=0)
    df_new = model.fit_transform(df[feat_names])
    for lab_name in lab_names:
        ax[line_id, col_id].scatter(df_new[df_labels.values[:, 1]==lab_name, 0],                             
                                    df_new[df_labels.values[:, 1]==lab_name, 1], 
                                    label=lab_name)
    ax[line_id, col_id].set_title("{0} neighbors".format(n_neighbor))    
    
    
#7. T-SNE with different perplexities
nb_lines = 2
nb_cols = 3

fig, ax = plt.subplots(nb_lines, nb_cols, figsize=(15, 15))
line_ids, col_ids = np.meshgrid(range(0, nb_lines), 
                                 range(0, nb_cols))
fig.suptitle('T-SNE space', fontsize=12)

for perp, line_id, col_id in zip([2, 2.5, 3.25, 3.5, 3.75, 4 ],
                                       line_ids.flatten(), col_ids.flatten()):
    model = TSNE(n_components=2,perplexity=perp, random_state=0)
    df_new2 = model.fit_transform(df[feat_names])
    for lab_name in lab_names:
        ax[line_id, col_id].scatter(df_new2[df_labels.values[:, 1]==lab_name, 0],                             
                                    df_new2[df_labels.values[:, 1]==lab_name, 1], 
                                    label=lab_name)
    ax[line_id, col_id].set_title("Perplexity={0}".format(perp))       


    
#Alon

#1. Load the Alon data
df = pd.read_csv("alon.csv", sep=";")
nb_samples, nb_features = df.shape
print("{0} samples and {1} features:".format(nb_samples, nb_features))
print(df[:5])

#2. Open the associated labels
df_labels = pd.read_csv("alon_class.csv")
nb_samples, _ = df_labels.shape
print("{0} samples:".format(nb_samples))
lab_names = np.unique(df_labels.values)
print(lab_names)

#3. PCA on the Alon data

# Get the feature names
feat_names = df.columns.tolist()

# Center and reduce the Alon data
scaler = StandardScaler()
dfn = df.copy()
dfn[feat_names] = scaler.fit_transform(dfn[feat_names])

# Pca
pca = PCA(n_components=2, random_state=0)
df_new = pca.fit_transform(df[feat_names])

#Ratio of explained variance
pc1 = int(round(pca.explained_variance_ratio_[0] * 100))
pc2 = int(round(pca.explained_variance_ratio_[1] * 100))
# Show the result
fig, ax = plt.subplots()
for lab_name in lab_names:
   ax.scatter(df_new[df_labels.values[:, 0]==lab_name, 0],
              df_new[df_labels.values[:, 0]==lab_name, 1], label=lab_name)
#ax.legend()
ax.set_xlabel("PC1 ({0}% of explained variance)".format(pc1, fontsize=12))
ax.set_ylabel("PC2 ({0}% of explained variance)".format(pc2, fontsize=12))
ax.set_title('PCA space', fontsize=12);

#4. MDS

mds = MDS(random_state=0)
df_new = mds.fit_transform(df[feat_names])

#Show the result
fig, ax = plt.subplots()
for lab_name in lab_names:
   ax.scatter(df_new[df_labels.values[:, 0]==lab_name, 0],
              df_new[df_labels.values[:, 0]==lab_name, 1], label=lab_name)
ax.legend()
ax.set_title('MDS space', fontsize=12);
ax.set_xlabel('x\'', fontsize=12);
ax.set_ylabel('y\'', fontsize=12);
ax.set_title('MDS space', fontsize=12);

#5. LLE with different neihbors sizes
nb_lines = 2
nb_cols = 3

fig, ax = plt.subplots(nb_lines, nb_cols, figsize=(15, 15))
line_ids, col_ids = np.meshgrid(range(0, nb_lines), 
                                range(0, nb_cols))
fig.suptitle('LLE space', fontsize=12)

for n_neighbor, line_id, col_id in zip([3, 5, 8, 10, 12, 15],
                                      line_ids.flatten(), col_ids.flatten()):
   model = LocallyLinearEmbedding(n_neighbors=n_neighbor, random_state=0)
   df_new = model.fit_transform(df[feat_names])
   for lab_name in lab_names:
       ax[line_id, col_id].scatter(df_new[df_labels.values[:, 0]==lab_name, 0],
                                   df_new[df_labels.values[:, 0]==lab_name, 1], 
                                   label=lab_name)
   ax[line_id, col_id].set_title("{0} neighbors".format(n_neighbor))
   
   
#6. Apply the Isomap dimension reduction technique

nb_lines = 2
nb_cols = 3

fig, ax = plt.subplots(nb_lines, nb_cols, figsize=(15, 15))
line_ids, col_ids = np.meshgrid(range(0, nb_lines), 
                                range(0, nb_cols))
fig.suptitle('Isomap space', fontsize=12)

for n_neighbor, line_id, col_id in zip([4, 8, 10, 13, 16, 20],
                                      line_ids.flatten(), col_ids.flatten()):
   model = Isomap(n_neighbors=n_neighbor)
   df_new = model.fit_transform(df[feat_names])
   for lab_name in lab_names:
       ax[line_id, col_id].scatter(df_new[df_labels.values[:, 0]==lab_name, 0],
                                   df_new[df_labels.values[:, 0]==lab_name, 1], 
                                   label=lab_name)
   ax[line_id, col_id].set_title("{0} neighbors".format(n_neighbor))
        
#7. T-SNE with different perplexities
nb_lines = 2
nb_cols = 3

fig, ax = plt.subplots(nb_lines, nb_cols, figsize=(15, 15))
line_ids, col_ids = np.meshgrid(range(0, nb_lines), 
                                 range(0, nb_cols))
fig.suptitle('T-SNE space', fontsize=12)

for perp, line_id, col_id in zip([2, 2.5, 3.25, 3.5, 3.75, 4 ],
                                       line_ids.flatten(), col_ids.flatten()):
    model = TSNE(n_components=2,perplexity=perp, random_state=0)
    df_new2 = model.fit_transform(df[feat_names])
    for lab_name in lab_names:
        ax[line_id, col_id].scatter(df_new2[df_labels.values[:, 1]==lab_name, 0],                             
                                    df_new2[df_labels.values[:, 1]==lab_name, 1], 
                                    label=lab_name)
    ax[line_id, col_id].set_title("Perplexity={0}".format(perp))         