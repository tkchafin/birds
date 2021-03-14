#!/usr/bin/python3

import sys
import os
import pandas as pd

def main():
	
	family="Corvidae"
	max_keep=5
	remove_nospecies=True
	kept=dict()
	priorities=["DMNH", "DMNS", "UCM", "LACM", "MSB"] #museums to check first
	
	#Dataframe: species museum specimen_id record_id
	print('Loading VertNet queries...')
	vertnet=pd.read_csv("~/Desktop/PRFB_Birds/VertNet_query.tsv", sep="\t", header=0, low_memory=False)
	vertnet = vertnet.loc[(vertnet['family']==family) & (vertnet['isfossil'] == 0) & (vertnet['hastissue'] == 1)]
	
	#remove samples with no epithet
	if remove_nospecies:
		bads=["Sp.", "Sp", "sp", "sp.", "Cf.", "cf.", "Cf", "cf", "NaN", "nan"]
		vertnet = vertnet.loc[~vertnet["specificepithet"].isin(bads)]
		vertnet = vertnet.dropna(subset=["specificepithet"])
	
	#subset columns 
	vertnet = vertnet[["scientificname", "family", "institutioncode", "catalognumber", "recordnumber", "preparations", "year", "country", "hastissue", "isfossil"]]
	
	#initialize dict of keeps
	kept = {key: 0 for key in set(vertnet["scientificname"])} 
	print(len(kept.keys()))
	
	#get counts of unique species per museum, sort 
	counts = vertnet.groupby('institutioncode')['scientificname'].nunique().sort_values(ascending=False).reset_index(name='count')
	prettyPrint(counts)
	
	#select top X specimens that are most recent and not skeletal? 
	samples = sampleGetOne_newFirst(vertnet, kept, priorities, counts["institutioncode"])
	

def getSamples(vertnet, kept, priorities, order, max_keep=1):
	ret=list()
	
	grouped=vertnet.groupby("institutioncode")
	
	for group in priorities:
		data=grouped.get_group(group)
		row=None
		for species, sp_data in data.groupby("scientificname"):
			sp_data = sp_data.sort_values("year", ascending=False)
			if kept[species] >= max_keep:
				continue
			else:
				best=sp_data.loc[sp_data["preparations"].contains("muscle|heart|liver|kidney", regex=True)]
				print(best.shape[0], " - ", sp_data.shape[0])
				print(best)
				

def prettyPrint(data, max_row=None, max_col=None):
	with pd.option_context('display.max_rows', max_row, 'display.max_columns', max_col):
		print(data)	


#Call main function
if __name__ == '__main__':
	main()





