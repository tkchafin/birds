#!/usr/bin/python3

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from scipy.spatial import ConvexHull
import seaborn as sns
import matplotlib.pyplot as plt

def main():
	print('Loading taxonomy database...')
	taxonomy=pd.read_csv("~/Desktop/PRFB_Birds/HBW-BirdLife_List_of_Birds_v5.txt", sep="\t", header=0, encoding = "ISO-8859-1", low_memory=False)
	taxonomy = taxonomy[taxonomy["2020 IUCN Red List category"] != "EX"]
	taxonomy["species"]=taxonomy["Scientific name"]
	
	print('Loading VertNet queries...')
	vertnet=pd.read_csv("~/Desktop/PRFB_Birds/VertNet_query.tsv", sep="\t", header=0, low_memory=False)
	vertnet["species"]=vertnet["scientificname"]
	
	print('Loading Pigot et al 2020 data...')
	pigot=pd.read_csv("~/Desktop/PRFB_Birds/Pigot_etal_2020-SuppDat.txt", sep="\t", header=0, low_memory=False)
	
	pigot["species"]=pigot["Binomial"].str.replace("_", " ")
	
	print("Performing discriminant analysis of morphospace PCs...")
	pigot_joined=pigot.merge(taxonomy, on="species")
	morph_dapc=discriminant_analysis(pigot_joined, "Order", 
		["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9"], 
		scale=False, n_components=2, solver='svd')
	da1, da2 = zip(*morph_dapc)
	pigot_joined["morph_DA1"]=da1
	pigot_joined["morph_DA2"]=da2
	
	print("Performing discriminant analysis of beak shape PCs...")
	beak_dapc=discriminant_analysis(pigot_joined, "Order", 
		["Beak_PC1", "Beak_PC2", "Beak_PC3", "Beak_PC4"], 
		scale=False, n_components=2, solver='svd')
	da1b, da2b = zip(*beak_dapc)
	pigot_joined["beak_DA1"]=da1b
	pigot_joined["beak_DA2"]=da2b
	del pigot
	
	print("Gathering data by order...")
	orders = parseStuff(taxonomy, vertnet, pigot_joined, level='order', plot=True)
	print(orders)
	orders.to_csv("order_data.tsv", sep="\t", header=True, index=False, quoting=None)

	print("Gathering data by family...")
	fams = parseStuff(taxonomy, vertnet, pigot_joined, level='family', plot=True)
	print(fams)
	fams.to_csv("family_data.tsv", sep="\t", header=True, index=False, quoting=None)	

def parseStuff(tax, vert, pigot, level='family', plot=False):
	#summarize orders
	groups=list()
	num_species=list()
	num_te=list()
	prop_te=list()
	n_hastissue=list()
	prop_hastissue=list()
	area_morph=list()
	area_beak=list()
	num_realm=list()
	d_realm=list()
	num_trophlevel=list()
	d_trophlevel=list()
	num_trophniche=list()
	d_trophniche=list()
	num_forage=list()
	d_forage=list()
	
	grouped=None
	if level=="family":
		grouped=tax.groupby(by='Family name')
	elif level=="order":
		grouped=tax.groupby(by='Order')
	
	morph_hulls=dict()
	beak_hulls=dict()
	
	for name, group in grouped:
		print(name)
		
		#get number of species
		groups.append(name)
		ns=len(group.index)
		num_species.append(ns)
		
		#get threatened/endangered
		te=["VU", "EN", "CR"]
		nt=len(group[group["2020 IUCN Red List category"].isin(te)].index)
		num_te.append(nt)
		prop_te.append(float(nt)/float(ns))
		
		#get num and Simpson's diversity for Realms, TrophicLevel, TrophicNiche, and ForagingNiche
		joined=group.merge(pigot, on="species", how="left")
		
		num_realm.append(joined['Realm'].nunique(dropna=True))
		d_realm.append(simpsons_diversity(joined['Realm']))
		
		num_trophlevel.append(joined['TrophicLevel'].nunique(dropna=True))
		d_trophlevel.append(simpsons_diversity(joined['TrophicLevel']))
		
		num_trophniche.append(joined['TrophicNiche'].nunique(dropna=True))
		d_trophniche.append(simpsons_diversity(joined['TrophicNiche']))
		
		num_forage.append(joined['ForagingNiche'].nunique(dropna=True))
		d_forage.append(simpsons_diversity(joined['ForagingNiche']))
		
		#get convex hull around morphospace and beak shape DAPC axes
		m=["morph_DA1", "morph_DA2"]
		points=joined[m].dropna().values
		if len(points) <= 2:
			area_morph.append(0.0)
		else:
			hull=ConvexHull(points)
			area_morph.append(hull.volume)
			morph_hulls["name"]=hull
		
		mb=["beak_DA1", "beak_DA2"]
		pointsb=joined[mb].dropna().values
		if len(pointsb) <= 2:
			area_beak.append(0.0)
		else:
			hullb=ConvexHull(pointsb)
			area_beak.append(hullb.volume)
			beak_hulls["name"]=hullb
		
		#get number that have tissue
		tissues=group.merge(vert, on="species", how='inner')
		n_ht=tissues['species'].nunique(dropna=True)
		n_hastissue.append(n_ht)
		prop_hastissue.append(float(n_ht)/float(ns))
		
	ret=pd.DataFrame({'name':groups, 'n_species':num_species, 'n_te' : num_te,
	'prop_te':prop_te, 'n_realm' : num_realm, 'd_realm':d_realm,
	'n_trophlevel':num_trophlevel, 'd_trophlevel':d_trophlevel,
	'n_trophniche':num_trophniche, 'd_trophniche':d_trophniche,
	'n_forageniche':num_forage, 'd_forageniche':d_forage,
	'beak_area':area_beak, 'morph_area':area_morph, 
	'n_hasTissue':n_hastissue, 'prop_hasTissue':prop_hastissue
	})
	
	if plot:
		if level=="order":
			plotDA(pigot, hue="Order", x='morph_DA1', y='morph_DA2')
			plotDA(pigot, hue="Order", x='beak_DA1', y='beak_DA2')
		elif level=="family":
			plotDA(pigot, hue="Family name", x='morph_DA1', y='morph_DA2')
			plotDA(pigot, hue="Family name", x='beak_DA1', y='beak_DA2')
		pairPlot(ret, level)
	
	return(ret)

def pairPlot(data, prefix):
	sns.pairplot(data, vars=["prop_te", "d_realm", "d_trophlevel", "d_trophniche", "d_forageniche", "morph_area", "beak_area", "prop_hasTissue"])
	plt.savefig(str(prefix)+"_pairplot.pdf")
	plt.clf()

def plotDA(points, hue='Order', x='morph_DA1', y='morph_DA2'):
	centers=dict()
	for name,group in points.groupby(hue):
		centroid=getCentroid(points[x].to_numpy(), points[y].to_numpy())
		centers[name]=tuple(centroid)
	print(centers)
	#create a new figure
	plt.figure(figsize=(5,5))
	
	#generate custom palette
	customPalette=sns.color_palette(n_colors=len(centers.keys()))
	
	#loop through labels and plot each cluster
	for i, label in enumerate(centers.keys()):
		#add data points 
		plt.scatter(x=points.loc[points[hue]==label, x], 
			y=points.loc[points[hue]==label, y], 
			color=customPalette[i], 
			alpha=0.3)
		#add label
		# plt.annotate(label, 
		# 	xy=(centers[label][0], centers[label][1]),
		# 	horizontalalignment='center',
		# 	verticalalignment='center',
		# 	size=8, weight='bold',
		# 	color=customPalette[i]) 
	
	#sns.scatterplot(data=points, x=x, y=y, hue=hue)
	plt.savefig(str(hue)+"_"+str(x)+"_"+str(y)+".pdf")
	plt.clf()
	#plt.show()

def getCentroid(x, y):
	length = x.shape[0]
	sum_x = np.sum(x)
	sum_y = np.sum(y)
	return ((sum_x/length), (sum_y/length))

def discriminant_analysis(traitdata, label, vars, n_components=2, test_size=0.8, test_classify=True, solver='svd', scale=True, plot=False):
	variables=traitdata[vars].values
	labels=traitdata[label].values
	
	#make training and test datasets
	X_train, X_test, y_train, y_test = train_test_split(variables, labels, test_size=test_size, random_state=0)
	
	if scale:
		sc = StandardScaler()
		X_train = sc.fit_transform(X_train)
		X_test = sc.transform(X_test)
	
	#run LDA
	lda = LDA(n_components=n_components, solver=solver)
	X_train = lda.fit_transform(X_train, y_train)
	X_test = lda.transform(X_test)
	#print(len(labels))
	#print(y_train)
	
	if test_classify:
		classifier = RandomForestClassifier(max_depth=2, random_state=0)
		classifier.fit(X_train, y_train)
		y_pred = classifier.predict(X_test)
		cm = confusion_matrix(y_test, y_pred)
		#print(cm)
		print('Accuracy: ' + str(accuracy_score(y_test, y_pred)))
	
	#get coordinates for all variables
	coords_all=lda.transform(variables)
	return(coords_all)

def simpsons_diversity(series):
	counts=series.value_counts(dropna=True)
	if len(counts.index) <= 1:
		return(0.0)
	#print(counts)
	return(1-(sum(counts*(counts-1))/(sum(counts)*(sum(counts)-1))))

#Call main function
if __name__ == '__main__':
	main()





