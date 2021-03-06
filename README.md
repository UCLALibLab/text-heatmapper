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

* similarityClusters.html: interactive HTML version of the term similarity clustering plot

## OTHER NOTES

* It's possible to override previously cached results by setting the 
useCachedFiles variable to "False" or just manually deleting all .pkl files
in the working directory.
* The working directory may need to be set in the code in order for the script 
to run successfully in Windows.

## SAMPLE SIMILARITY MATRICES

**Note:** Labels files have one matrix colum/row label per line and should be in UTF-8 encoding. Matrices should have values only between 0 and 1 and should be tab-delimited, **but the sample files below are actually comma-delimited** to make them more usable with other tools.

Nijūichidaishū - 21 Japanese imperial anthologies of *waka* poetry, 905-1439
* ~38,000 poems, binned into roughy 10-poem bundles (matrix is 3331x3331)
* upper-left diagonal: 1/2/3-gram term cosine similarity; bottom-right diagonal: 100-topic LDA topic cosine similarity
* Matrix: http://babylon.library.ucla.edu/~broadwell/waka/merged_waka.tar.gz -- 48 MB compressed, 120 MB expanded
* Labels: http://babylon.library.ucla.edu/~broadwell/waka/dist_labels.txt -- labeled and ordered by anthology + genre

The Evald Tang Kristensen archive of Danish folklore, ca. 1860-1920
* 31,088 stories, binned into roughly 15-story bundles (matrix is 1869x1869)
* upper-left diagonal: 1/2/3-gram term cosine similarity; bottom-right diagonal: 100-topic LDA topic cosine similarity
* Matrix: http://etkspace.scandinavian.ucla.edu/~broadwell/etksim/etk_merged.tar.gz -- 17 MB compressed, 52 MB expanded
* Labels: http://etkspace.scandinavian.ucla.edu/~broadwell/etksim/dist_labels.txt -- labeled and ordered by collection + volume + genre

Arkiv for Dansk litteratur (Danish literature archive), 1170-1925 (most between 1500-1925)
* 1691 bundles of a much larger number of texts in a mix of genres: novels, poetry, stage works
* Matrix is symmetrical, using 1/2/3-gram term cosine similarity
* Matrix: http://babylon.library.ucla.edu/~broadwell/adl_sim/adl_sim.tar.gz -- 18 MB compressed, 43 MB expanded
* Labels: http://babylon.library.ucla.edu/~broadwell/adl_sim/dist_labels.txt -- labeled and ordered by date + author + work

Manga cover images from the Manga Cover Database (https://mcd.iosphe.re/)
* 8,337 cover images
* Matrix is symmetrical, using cosine similarity of 2,048 features from penultimate layer of Inception neural network
* Matrix: http://babylon.library.ucla.edu/~broadwell/manga_sim/manga_sim.tar.gz  -- 20 MB compressed, 278 MB expanded
* Labels: http://babylon.library.ucla.edu/~broadwell/manga_sim/dist_labels.txt -- labeled by title, ordered by audience/genre, essentially women-men-girls-boys

K-pop video frames from top ~212 most-viewed videos on YouTube
* 5,844 images
* Matrix is symmetrical, using cosine similarity of 2,048 features from penultimate layer of Inception neural network
* Matrix: http://babylon.library.ucla.edu/~broadwell/kpop/kpop_sim.tar.gz -- 10 MB compressed, 137 MB expanded
* Labels: http://babylon.library.ucla.edu/~broadwell/kpop/kpop_labels.txt -- labeled by YouTube ID, ordered by gender + year posted
