# -*- coding: utf-8 -*-

"""
TextHeatmapper.py

Writen by Peter Broadwell (broadwell@library.ucla.edu)
Based on Steve Tjoa's answer to this Stack Overflow post:
http://stackoverflow.com/questions/2982929/plotting-results-of-hierarchical-clustering-ontop-of-a-matrix-of-data-in-python
and Brandon Rose's "Document Clustering with Python":
http://brandonrose.org/clustering

USAGE: python TextHeatmapper.py [documents_file] [labels_file]

Please see the README.md file for input file specifications, errata, and
other important tips.

"""

import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.metrics import adjusted_rand_score
#from sklearn.metrics.pairwise import linear_kernel
#from sklearn.pipeline import make_pipeline
#from sklearn.preprocessing import Normalizer
#from sklearn.decomposition import TruncatedSVD
#from sklearn.metrics import pairwise_distances
from matplotlib import font_manager
fontP = font_manager.FontProperties()
fontP.set_size(2.5)
# Use the below for Chinese text
#fontP = font_manager.FontProperties(fname = 'C:\\Windows\\Fonts\\simsun.ttc')
#fontP.set_family('SimSun')
import matplotlib as mpl
mpl.rc('font', family='Arial')
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import codecs
import pandas as pd
import mpld3
import numpy as np
import palettable
import pprint
from scipy.cluster.hierarchy import ward, dendrogram, linkage
import gensim
#from gensim import corpora, models, similarities

import os
# Set working directory (Windows only)
#os.chdir("//ad/home/h/broadwell/libdata/clustering/")
#cwd = os.getcwd()
#print("current folder: " + cwd)

import sys
# Get the names of documents file and the labels file from the
# command line
docsfile = sys.argv[1]
labelsfile = sys.argv[2]

# Set this to "False" to recompute all data structures and save them to
# cache files. Otherwise, the script will attempt to load them
useCachedfiles = True 

# Function to compute the Shannon entropy of a string
def stringEntropy(instr):
  tokenFreqs = {}
  instrArray = instr.split(' ')

  totalTokens = len(instrArray)

  for token in instrArray:
    if (token in tokenFreqs):
      tokenFreqs[token] += 1
    else:
      tokenFreqs[token] = 1

  entropy = 0

  for token in tokenFreqs:
    p_x = float(tokenFreqs[token])/totalTokens
    if p_x > 0:
      entropy += - p_x*math.log(p_x, 2)
  
  return entropy

# Load the texts. Format is just one document per line. Final string
# normalization can be done here.
documents = [line.rstrip('\n') for line in codecs.open(docsfile, 'r', 'utf-8')]
print("number of documents read: " + str(len(documents)))

# There should be one label per document (line) in the documents file.
# It's best to replace spaces in the labels with underscores.
labels = [line.rstrip('\n').replace(' ', '_') for line in codecs.open(labelsfile, 'r', 'utf-8')]
print("number of labels: " + str(len(labels)))

# Attempt to shorten the labels so that the clustering plot might be
# more readable (usually only marginally effective)
collections = []
for label in labels:
  labelArray = label.split('_')
  collections.append(labelArray[0] + '_' + labelArray[-1])

# Used when drawing tick marks on the plot axes
nBundles = len(labels)

# Usually we can't show every label on the Y axis, so this skips some of them
labelsToSkip = 10
axisLabels = labels[0::labelsToSkip]

# Create a corpus structure for use with LDA
doc_corpus = []
for doc in documents:
  word_array = list(gensim.utils.tokenize(doc))
  doc_corpus.append(word_array)

kt_dict = gensim.corpora.Dictionary(doc_corpus)

dict_corpus = [kt_dict.doc2bow(text) for text in doc_corpus]

print("running LDA")
if (os.path.isfile('LDAmodel.pkl') and (useCachedfiles == True)):
  model = gensim.models.LdaModel.load("LDAmodel.pkl")
else:
  model = gensim.models.ldamodel.LdaModel(corpus=dict_corpus, id2word=kt_dict, num_topics=100)
  model.save("LDAmodel.pkl")
if (os.path.isfile('LDAsims.pkl') and (useCachedfiles == True)):
  lda_index = gensim.similarities.MatrixSimilarity.load("bundledADL_lda.sim")
else:
  lda_index = gensim.similarities.MatrixSimilarity(model[dict_corpus])
  lda_index.save("LDAsims.sim")

# This is the topic similarity matrix
lda_sim = None

for d1 in dict_corpus:
  vec_lda = model[d1]
  sims = lda_index[vec_lda]
  #print("length of simliarity array: " + str(len(sims)))
  
  if (lda_sim is None):
    lda_sim = np.array([sims])
  else:
    lda_sim = np.append(lda_sim, [sims], 0)
  
print("LDA matrix shape: " + str(lda_sim.shape))

print("plotting LDA similary heatmap")
fig = plt.figure(figsize=(8,8))

ldadendro = fig.add_axes([0.1,0.9,0.89,0.09])
Y = linkage(lda_sim, method='ward')
Z1 = dendrogram(Y, orientation="top", p=8, truncate_mode="level", no_labels=True)
ldadendro.set_xticks([])
ldadendro.set_yticks([])
#ldadendro.invert_xaxis()

ldamatrix = fig.add_axes([0.1,0.01,0.89,0.89])
im = ldamatrix.matshow(lda_sim, aspect='auto', origin='lower', cmap=plt.get_cmap('YlGnBu'))
ticks = np.arange(0, nBundles, 10)
ldamatrix.set_xticks(ticks, minor=False)
ldamatrix.set_yticks(ticks, minor=False)
ldamatrix.set_xticklabels([], minor=False, fontproperties=fontP)
ldamatrix.set_yticklabels(axisLabels, minor=False, fontproperties=fontP)

# Plot colorbar.
axcolor = fig.add_axes([0.01,0.01,0.02,0.89])
cbar = plt.colorbar(im, cax=axcolor)
cbar.ax.tick_params(axis='y', labelsize=4, pad=.25)

fig.savefig('LDA_heatmap.png', dpi=450)

plt.close()

# Compute relative entropies of documents 

print("running entropy calculations")

# This usually doesn't take very long, so it's not cached
docEntropies = []
for doc in documents:
  docEntropies.append(stringEntropy(doc))

A = np.array(docEntropies)
meanEntropy = np.mean(docEntropies)

diffEntropies = []
for ent in docEntropies:
  diffEntropies.append(ent - meanEntropy)
  
dA = np.array(diffEntropies)

print("mean document entropy: " + str(np.mean(docEntropies)))
print("median document entropy: " + str(np.median(docEntropies)))

# Just in case we're curious
fullEntropy = stringEntropy(' '.join(documents))
print("full corpus entropy: " + str(fullEntropy))

diff = np.subtract.outer(A, A)
entropyMatrix = np.abs(diff)

diffdiff = np.subtract.outer(dA, dA)
entropyDiffMatrix = np.abs(diffdiff)
print("entropy matrix shape: " + str(entropyDiffMatrix.shape))

print("plotting entropy dendro")
fig = plt.figure(figsize=(8,8))

endendro = fig.add_axes([0.15,0.89,0.84,0.1])
Y = linkage(diff, method='ward')
Z1 = dendrogram(Y, orientation="top", p=8, truncate_mode="level", no_labels=True)
endendro.set_xticks([])
endendro.set_yticks([])
endendro.invert_xaxis()

print("plotting entropy line graph. Matrix shape: ")
y = np.array(range(nBundles))
print(A.shape)
print(y.shape)
axline = fig.add_axes([0.01,0.01,0.05,0.89])
ticks = np.arange(0, nBundles, 10)
xticks = np.arange(y.min(), y.max(), .5)
axline.xaxis.set_ticks_position('top')
axline.xaxis.set_label_position('top')
axline.set_xticks(xticks, minor=False)
axline.set_yticks(ticks, minor=False)
fontP.set_size(3)
axline.set_xticklabels(xticks, minor=False, fontproperties=fontP)
axline.tick_params(axis='x', which='major', pad=.25)
axline.set_yticklabels([], minor=False, fontproperties=fontP)
axline.plot(A, y)
fontP.set_size(2.5)

print("plotting entropy matrix")
enmatrix = fig.add_axes([0.15,0.01,0.84,0.89])
im = enmatrix.matshow(entropyMatrix, aspect='auto', origin='lower', cmap=plt.get_cmap('YlGnBu_r'))
ticks = np.arange(0, nBundles, 10)
enmatrix.set_xticks(ticks, minor=False)
enmatrix.set_yticks(ticks, minor=False)
enmatrix.set_xticklabels([], minor=False, fontproperties=fontP)
enmatrix.set_yticklabels(axisLabels, minor=False, fontproperties=fontP)

# Plot colorbar.
axcolor = fig.add_axes([0.06,0.01,0.02,0.89])
cbar = plt.colorbar(im, cax=axcolor)
cbar.ax.tick_params(which='y', labelsize=4, pad=.25)

fig.savefig('entropy_heatmap.png', dpi=450)
plt.close()

print("computing term vectors")

def tokenize(x):
  return x.split(' ')

if (os.path.isfile('docVectors.pkl') and (useCachedfiles == True)):
  vectorizer = joblib.load('docVectors.pkl')
else:
  vectorizer = TfidfVectorizer(norm=None, use_idf=True, max_features=200000, min_df=.05, max_df=.95, ngram_range=(1, 3), tokenizer=tokenize)
  joblib.dump(vectorizer, 'docVectors.pkl')

print("calculating TF-IDF matrix")
# Calculating TF-IDF matrix
if (os.path.isfile('termMatrix.pkl') and (useCachedfiles == True)):
  tfidf_matrix = joblib.load('termMatrix.pkl')
else:
  tfidf_matrix = vectorizer.fit_transform(documents)
  joblib.dump(tfidf_matrix, 'termMatrix.pkl')

  print("shape of term matrix: " + str(tfidf_matrix.shape))
# This fails with pickled data
#terms = vectorizer.get_feature_names()

print("calculating document term similarity distances")

if (os.path.isfile('termDistances.pkl') and (useCachedfiles == True)):
  dist = joblib.load('termDistances.pkl')
else:
  dist = 1 - cosine_similarity(tfidf_matrix)
  joblib.dump(dist, 'termDistances.pkl')

fig = plt.figure(figsize=(8,8))
# Try vanilla dendrogram first

print("plotting term similarity dendrogram")

axdendro = fig.add_axes([0.1,0.9,0.89,0.09])
Y = linkage(dist, method='centroid')
Z1 = dendrogram(Y, orientation="top", p=8, truncate_mode="level", no_labels=True)
axdendro.set_xticks([])
axdendro.set_yticks([])
#axdendro.invert_xaxis()

print("plotting document term distances")
axmatrix = fig.add_axes([0.1,0.01,0.89,0.89])
im = axmatrix.matshow(dist, aspect='auto', origin='lower', cmap=plt.get_cmap('YlGnBu_r'))
ticks = np.arange(0, nBundles, 10)
axmatrix.set_xticks(ticks, minor=False)
axmatrix.set_yticks(ticks, minor=False)
axmatrix.set_xticklabels([], minor=False, fontproperties=fontP)
axmatrix.set_yticklabels(axisLabels, minor=False, fontproperties=fontP)

# Plot colorbar.
axcolor = fig.add_axes([0.01,0.01,0.02,0.89])
cbar = plt.colorbar(im, cax=axcolor)
cbar.ax.tick_params(axis='y', labelsize=4, pad=.25) 

fig.savefig('words_heatmap.png', dpi=450)
plt.close()

# NOTE: The rest of the code plots k-means clusters, in HTML and
# as an image. Stop the script here if you don't want to do this
#sys.exit()

print("calculating k-means for clustering")

# Number of centroids
true_k = 30 

if (os.path.isfile('docClusters.pkl') and (useCachedfiles == True)):
  km = joblib.load('docClusters.pkl')
else:
  km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
  km.fit(tfidf_matrix)
  joblib.dump(km, 'docClusters.pkl')

clusters = km.labels_.tolist()

#print("top terms per cluster:")
#order_centroids = km.cluster_centers_.argsort()[:, ::-1]
#terms = vectorizer.get_feature_names()
#for i in range(true_k):
#    print "cluster %d:" % i,
#    for ind in order_centroids[i, :10]:
#        print ' %s' % terms[ind],
#    print

docs = { 'label': labels, 'cluster': clusters, 'collection': collections }
frame = pd.DataFrame(docs, index = [clusters] , columns = ['label', 'collection', 'cluster'])

print("cluster characteristics:")
pprint.pprint(frame['cluster'].value_counts())

print("reducing clusters to two dimensions")

if (os.path.isfile('MDS.pkl') and (useCachedfiles == True)):
  pos = joblib.load('MDS.pkl')
else:
  MDS()
  mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
  pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
  joblib.dump(pos, 'MDS.pkl')

xs, ys = pos[:, 0], pos[:, 1]

print("plotting cluster graph")

df = pd.DataFrame(dict(x=xs, y=ys, label=collections, title=labels)) 
groups = df.groupby('label')

# set up plot
fig, ax = plt.subplots(figsize=(8, 7)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

entityIDs = {}
entityCount = 0
entityColors = {}

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:

    thisAnth="docs"
    entityName = name.split('_')[-1].strip().replace(u'　','')
    if (entityName in entityIDs):
      entityID = entityIDs[entityName]
    else:
      entityCount += 1
      entityIDs[entityName] = entityCount
      entityID = entityCount

    print "thisAnth is " + thisAnth

    if (entityName in entityColors):
      entityColor = entityColors[entityName]
    else:
      if (thisAnth == u'docs'):
        entityColor = palettable.colorbrewer.qualitative.Set3_12.hex_colors[entityID % 12]
      elif (thisAnth == u'other_docs'):
        #entityColor = palettable.colorbrewer.sequential.Greys_9.hex_colors[entityID % 9]
        entityColor = '#252525' 
      entityColors[entityName] = entityColor
    
    print "color for " + entityName + " is " + str(entityColor)

    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=entityName, color=entityColor, 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
    
#ax.legend(numpoints=1, prop=fontP)  #show legend with only 1 point
ax.legend().set_visible(False)

#add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8, fontproperties=fontP)  
    
#plt.show()

plt.savefig('similarityClusters.png', dpi=450)
plt.close()

print("exporting similarity figure to HTML")

#define custom toolbar location
class TopToolbar(mpld3.plugins.PluginBase):
    """Plugin for moving toolbar to top of figure"""

    JAVASCRIPT = """
    mpld3.register_plugin("toptoolbar", TopToolbar);
    TopToolbar.prototype = Object.create(mpld3.Plugin.prototype);
    TopToolbar.prototype.constructor = TopToolbar;
    function TopToolbar(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    TopToolbar.prototype.draw = function(){
      // the toolbar svg doesn't exist
      // yet, so first draw it
      this.fig.toolbar.draw();

      // then change the y position to be
      // at the top of the figure
      this.fig.toolbar.toolbar.attr("x", 250);
      this.fig.toolbar.toolbar.attr("y", 50);

      // then remove the draw function,
      // so that it is not called again
      this.fig.toolbar.draw = function() {}
    }
    """
    def __init__(self):
        self.dict_ = {"type": "toptoolbar"}

#define custom css to format the font and to remove the axis labeling
css = """
text.mpld3-text, div.mpld3-tooltip {
  font-family:Arial, Helvetica, sans-serif;
}

g.mpld3-xaxis, g.mpld3-yaxis {
display: none; }

svg.mpld3-figure {
margin-left: -200px;}
"""

# Plot 
fig, ax = plt.subplots(figsize=(14,10)) #set plot size
ax.margins(0.02) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    entityName = name.split('_')[-1]

    points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=18, 
                     label=name, mec='none', 
                     color=entityColors[entityName])
    ax.set_aspect('auto')
    labels = [i for i in group.title]
    
    #set tooltip using points, labels and the already defined 'css'
    tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels,
                                       voffset=10, hoffset=10, css=css)
    #connect tooltip to fig
    mpld3.plugins.connect(fig, tooltip, TopToolbar())    
    
    #set tick marks as blank
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    
    #set axis as blank
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

#ax.legend(numpoints=1) #show legend with only one dot
ax.legend().set_visible(False)

mpld3.display() #show the plot

#uncomment the below to export to html
html = mpld3.fig_to_html(fig)

htmlfile = codecs.open("similarityClusters.html", "w", "utf-8")
htmlfile.write(html)
#print(html)

#sys.exit()
