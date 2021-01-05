# birds

Python code to summarize morphospace/ niche diversity, tissue availability, and proportion of specie threatened/endangered across all bird species by order and by family. 

The script first does a discriminant analysis of principal components using scikit-learn using morphospace and beak shape PCs from [Pigot et al. 2020](https://www.nature.com/articles/s41559-019-1070-4). It also does a classification using 80% test data and will spit out classification accuracy to stdout. Typically ~65% for classifying to order from morpho/beak PCs. 

Then it parses at order/ family level using data from [a bird taxonomy database](http://datazone.birdlife.org/species/taxonomy) joined with the Pigot et al. dataset and a large [VertNet query database](http://vertnet.org/) containing tissue accessions for bird species to calculate various neat things and make some plots. 

