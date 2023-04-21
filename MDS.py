# CIS 569 Project 3 MDS Code for Visualizing the positions of high dimensional data
# Utilizes Project 2 (code here before MDS code) clustering before performing MDS
# Ben Pfeffer, Andrew Anctil, Rui Zhou
# CIS 569 - Professor Yuchou Chang

###########
### OLD ###
###########
# Import required libraries
import os
import glob
import json

# Initialize data storage
names = []
texts = []

# Iterate through files, if not correct file (.txt extension) then skip, otherwise add to correct location in dictionary
for infile in glob.glob(os.path.join("./dataset", '*')):
    if(infile.endswith(".txt")):
        continue
    review_file = open(infile,'r').read()
    subKey = infile.split("/")[-1]
    mainKey = subKey.split("_")[0]
    names.append(subKey)
    texts.append(review_file)


import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download("wordnet")
nltk.download("omw-1.4")
from num2words import num2words
from sklearn.feature_extraction.text import TfidfVectorizer



from sklearn.cluster import KMeans
import pandas as pd
import collections


# Get English stopwords
stopwords = nltk.corpus.stopwords.words('english')

# Process data and make list of list of words
textFull = []
for text in texts:
    currText = text.lower()
    tokens = word_tokenize(currText)
    
    # Remove stopwords
    no_stop = [w for w in tokens if w not in stopwords]
    
    # Remove punctuation
    no_punct = [w for w in no_stop if w.isalnum()]
    
    # Remove single characters
    no_single = [w for w in no_punct if len(w)>1]
    
    # Lemmatize (stemming, but more meaningful and more computationally expensive)
    # Initialize wordnet lemmatizer
    wnl = WordNetLemmatizer()

    # Perform lemmatization
    lemmatized = [wnl.lemmatize(w, pos="v") for w in no_single]
    
    # Append to list of list of words
    textFull.append(lemmatized)



# Make list of all documents for use in tf-idf matrix
textListFull = []
for textList in textFull:
    textListFull.append(" ".join(textList))


# Create TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Generate matrix of word vectors
tfidf_matrix = vectorizer.fit_transform(textListFull)

# Get the shape, matches 111 documents given as data
print(tfidf_matrix.shape)


# Store the features for insight later
features = vectorizer.get_feature_names()


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# Cluster the tf-idf matrix with the decided 4 clusters
num_clusters = 4
km = KMeans(n_clusters=num_clusters, random_state=1)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()


# Store the clusters as a dataframe
clusterDf = pd.DataFrame()
clusterDf["File"] = names
clusterDf["Text"] = texts
clusterDf["Cluster"] = clusters


import re
# Create a function to tokenize by sentence and word, and limit results to alphanumeric tokens
def tokenize(text):
    # Tokenize by sentence and word
    word_tokens = [word.lower() for sentence in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sentence)]
    final = []
    for val in word_tokens:
        # Remove words based on regex
        if re.search('[a-zA-Z0-9]', val):
            final.append(val)
    return final


# Get a list -> dataframe of all words used for reference in insight
total_words = []
for i in textListFull:
    words = tokenize(i)
    total_words.extend(words)
words = pd.DataFrame({'words': total_words}, index = total_words)

# Display the top terms per cluster and the titles of the documents in each cluster
print("Top terms per cluster:")
print()
# Sort cluster centers by proximity to centroid
ordered_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    print("Cluster", i, "words:")
    
    for idx in ordered_centroids[i, :10]: # Best 10 words per cluster
        print(words.loc[features[idx].split(' ')].values.tolist()[0][0], end=", ")
    print()
    print()
    
    print("Cluster", i, "titles:")
    for title in clusterDf.set_index("Cluster").loc[i]['File'].values.tolist():
        print(title, end=", ")
    print()
    print()

# Labels: threats, source information, communications, reports

# Convert from cluster dataframe to dictionary
clustDict = {}
for i in range(len(clusterDf)):
    cNum = str(clusterDf.iloc[i].Cluster)
    doc = clusterDf.iloc[i].File
    text = clusterDf.iloc[i].Text
    try:
        clustDict[cNum][doc] = text
    except:
        clustDict[cNum] = {}
        clustDict[cNum][doc] = text
clustDict = dict(collections.OrderedDict(sorted(clustDict.items())))

# Export dictionary to json file
with open('ClusterData.json', 'w') as fp:
    json.dump(clustDict, fp)




###########
### NEW ###
###########



# MDS Section
# Using cosine distance of tf-idf matrix, rescaling, and plotting results in plotly here, then saving to file


# Import similarity metric
from sklearn.metrics.pairwise import cosine_similarity
# Convert cosine similiarity to distance
dis = 1 - cosine_similarity(tfidf_matrix, tfidf_matrix)


# Import the library to perform MDS from
from sklearn import manifold

# Initialize MDS into 3 components for a 3d plot
mds = manifold.MDS(
    n_components=3,
    max_iter=3000,
    eps=1e-9,
    random_state=1,
    dissimilarity="precomputed",
    n_jobs=1
)

# Fit the model on the distance data
pos = mds.fit(dis).embedding_



# Import plotting
import matplotlib.pyplot as plt

# Plot 2 dimensions in 2d
plt.scatter([i[0] for i in pos], [i[1] for i in pos])


# Import plotly for 3d plotting
import plotly.express as px

# Create a DataFrame containing the 3 dimensional positions and the cluster number
df = pd.DataFrame()
df["P1"] = [i[0] for i in pos]
df["P2"] = [i[1] for i in pos]
df["P3"] = [i[2] for i in pos]
df["Cluster"] = clusters

# Plot a 3d scatterplot of the positions, and color by cluster
fig = px.scatter_3d(df, x='P1', y='P2', z='P3', color="Cluster")
fig.show()



# Import PCA library and numpy
from sklearn.decomposition import PCA
import numpy as np

# Assistance in rescaling from: https://scikit-learn.org/stable/auto_examples/manifold/plot_mds.html
# Rescale the data
pos *= np.sqrt((dis**2).sum()) / np.sqrt((pos**2).sum())

# Rotate the data
clf = PCA(n_components=3)
X_true = clf.fit_transform(dis)

# Transform the data based on the rescaling
pos = clf.fit_transform(pos)

# Create a new dataframe with the new positions, as well as the document name
df = pd.DataFrame()
df["P1"] = [i[0] for i in pos]
df["P2"] = [i[1] for i in pos]
df["P3"] = [i[2] for i in pos]
df["Cluster"] = clusters
df["ID"] = names

# Plot the rescaled data in the same way
fig = px.scatter_3d(df, x='P1', y='P2', z='P3', color="Cluster")
fig.show()

# Save the data containing 3d position, cluster, and document name to a csv file
df.to_csv("MDS.csv")




