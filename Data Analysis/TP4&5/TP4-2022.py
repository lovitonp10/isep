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



#1. Load the iris dataset
df = pd.read_csv("iris.csv")
nb_samples, nb_features = df.shape
nb_features -= 1
print("{0} samples and {1} features:".format(nb_samples, nb_features))
print(df[:5])

#2 associate each class to a color
feat_names = df.columns.tolist()
feat_names.remove("Class")
print(feat_names)

species = np.unique(df["Class"].values)

#Store the specy colors in a dictionary
colors = ['navy', 'turquoise', 'darkorange']
color_dict = {}
for color, specy in zip(colors, species):
    color_dict[specy] = color
print(color_dict)    

#3 Center and reduce the data
scaler = StandardScaler()
dfn = df.copy()
dfn[feat_names] = scaler.fit_transform(dfn[feat_names])

#4 PCA 
pca = PCA(n_components=4)
df_new = pca.fit_transform(dfn[feat_names])

#Explained variance ratio
pc1 = int(round(pca.explained_variance_ratio_[0] * 100))
pc2 = int(round(pca.explained_variance_ratio_[1] * 100))
print(pc1)
print(pc2)

#Show the results
fig, ax = plt.subplots();
for specy in species:
    ax.scatter(df_new[df["Class"]==specy, 0], df_new[df["Class"]==specy, 1],
               c=color_dict[specy], label=specy);

ax.legend();
ax.set_xlabel("PC1 ({0}% of explained variance)".format(pc1, fontsize=12));
ax.set_ylabel("PC2 ({0}% of explained variance)".format(pc2, fontsize=12));
ax.set_title('PCA space', fontsize=12);

#5 correlation circles

eigval = pca.explained_variance_
print("Scikit eigenvalues: {0}".format(eigval))

#Corrected eigenvalue
eigval = (nb_samples - 1) / nb_samples * eigval
print("Scikit corrected eigenvalues: {0}".format(eigval))

#eigen value square root
sqrt_eigval = np.sqrt(eigval)

#correlation of variables with axes
corvar = np.zeros((nb_features, nb_features))

for k in range(nb_features):
    corvar[:, k] = pca.components_[k, :] * sqrt_eigval[k]
    
#Draw the correlation circle    
fig, ax = plt.subplots()
an = np.linspace(0, 2 * np.pi, 100)
ax.plot(np.cos(an), np.sin(an), 'b', linewidth=0.5) 
# Add a unit circle for scale
#<c> Break the line of code above </c>
for i in range(0, corvar.shape[0]):
    ax.arrow(0,
             0,  # Start the arrow at the origin
             corvar[i, 0],  #PC1
             corvar[i, 1], #PC2
             head_width=0.1,
             head_length=0,
             color = 'r')    
    ax.text(corvar[i, 0]* 1.15, corvar[i, 1] * 1.15,
    feat_names[i], color = 'k', ha = 'center',
    va = 'center')

ax.axis('equal')
ax.set_xlabel("DIM1", fontsize=11)
ax.set_ylabel("DIM2", fontsize=11)
ax.set_title('Correlation circle');



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

