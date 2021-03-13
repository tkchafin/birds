#!/usr/bin/python3

import sys
import os
import pandas as pd

def main():
	
	family="Corvidae"
	max_keep=2
	kept=dict()
	
	#Dataframe: species museum specimen_id record_id
	print('Loading VertNet queries...')
	vertnet=pd.read_csv("~/Desktop/PRFB_Birds/VertNet_query.tsv", sep="\t", header=0, low_memory=False)
	vertnet = vertnet.loc[(vertnet['family']==family) & (vertnet['isfossil'] == 0) & (vertnet['hastissue'] == 1)]
	
	prettyPrint(vertnet[["scientificname", "family", "institutionid", "catalognumber", "recordnumber", "preparations", "year", "country", "hastissue", "isfossil"]])
	
	#initialize dict of keeps
	kept = {key: 0 for key in set(vertnet["scientificname"])} 
	print(len(kept.keys()))
	#group_by museum, sort by size, iterate one-by-one 
	
	#for each species found, add to 'found' list, remove from subsequent museums
	#alternatively, could have a dict counter and remove a species once it has been found X number of times?
	
	#select top X specimens that are most recent and not skeletal? 
	
	# print('Loading taxonomy database...')
	# taxonomy=pd.read_csv("~/Desktop/PRFB_Birds/HBW-BirdLife_List_of_Birds_v5.txt", sep="\t", header=0, encoding = "ISO-8859-1", low_memory=False)
	# taxonomy = taxonomy[taxonomy["2020 IUCN Red List category"] != "EX"]
	# taxonomy["species"]=taxonomy["Scientific name"]
	

def prettyPrint(data, max_row=10, max_col=None):
	with pd.option_context('display.max_rows', max_row, 'display.max_columns', max_col):
		print(data)	


#Call main function
if __name__ == '__main__':
	main()





