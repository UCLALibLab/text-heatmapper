# text-heatmapper

TextHeatmapper.py

Writen by Peter Broadwell (broadwell@library.ucla.edu), based on Steve Tjoa's answer to [this Stack Overflow post] (http://stackoverflow.com/questions/2982929/plotting-results-of-hierarchical-clustering-ontop-of-a-matrix-of-data-in-python)
and Brandon Rose's ["Document Clustering with Python"] (http://brandonrose.org/clustering) tutorial.

  USAGE: python TextHeatmapper.py [documents_file] [labels_file]

## INPUT

The format of the input documents file is *one line per document.* Ideally,
texts should be stripped of all punctuation and coverted to lowercase,
with a single space separating all words, and normalized to avoid orthographic
variation between texts (e.g., ö=>ø). The results are likely to be more
consistent if all of the input documents are roughly the same length. A
document * document matrix that is larger than can be held in memory (~4000
documents for a machine with 8GB of RAM) may cause thrashing. Therefore, 
judicious "chunking" and/or "bundling" of the input texts is recommended.

The format of the labels file is *one line per document label.* For some
of the visualizations, it's preferable if the labels are not too long
(~20 characters) and contain sections separated by underscores, e.g.,
  author_title_year
  or
  collection_volume_number
The heatmaps likely will be more meaningful if the documents are in a logical
sequence, e.g., chronological and/or grouped by author.

## REQUIREMENTS

This script relies upon several Python libraries. Most are standard libraries 
that should be available via package management systems like pip. The most 
important among them are sklearn, gensim, matplotlib/pyplot, pandas, and 
numpy; some of the more obscure packages are mpld3 and palettable.

## OUTPUT
When successful, this script produces the following files in the
main working directory:

* words_heatmap.png: heatmap + dendrogram plot of 1/2/3-gram document term similarity using middle 98% of frequencies, TF-IDF weighting, cosine similarity

* LDA_heatmap.png: heatmap + dendrogram plot of document topic similarities, computed by training a 100-topic model on the documents with LDA, then inferring the likelihood of each document given every other document

* entropy_heatmap.png: heatmap + dendrogram + line graph plot of the absolute (vertical line graph) and relative (heatmap) Shannon entropies of the documents

* similarityClusters.png: output of k-means clustering with multidimensional scaling of the term similarity matrix computed for the words_heatmap

* similarityClusters.html: interactive HTML version of the term similarity clustering plot (only works on Mac)

## OTHER NOTES

* It's possible to override previously cached results by setting the 
useCachedFiles variable to "False" or just manually deleting all .pkl files
in the working directory.
* The working directory may need to be set below in order for the script to
run successfully in Windows.
* Also, the final portion of the script is only able to generate the
interactive HTML cluster file on Mac OS X, not Windows. I have no idea why.
