# birds

## summarize_bird_stuff.py

Python code to summarize morphospace/ niche diversity, tissue availability, and proportion of specie threatened/endangered across all bird species by order and by family. 

The script first does a discriminant analysis of principal components using scikit-learn using morphospace and beak shape PCs from [Pigot et al. 2020](https://www.nature.com/articles/s41559-019-1070-4). It also does a classification using 80% test data and will spit out classification accuracy to stdout. Typically ~65% for classifying to order from morpho/beak PCs. 

Then it parses at order/ family level using data from [a bird taxonomy database](http://datazone.birdlife.org/species/taxonomy) joined with the Pigot et al. dataset and a large [VertNet query database](http://vertnet.org/) containing tissue accessions for bird species to calculate various neat things and make some plots. 


## parse_vertnet_query.py

This script parses a large tab-delimited database of all tissues accessioned in VertNet. To access the file of all bird tissues in VertNet, go here: https://osf.io/d9szb/. It was too large to host on GitHub... 

The goal of this script is to select a minimum number of museums required to find X number of tissue specimens for each taxon in a given family. Settings for the family to search and how many specimens are set at the beginning of the script (open it in a text editor). 

