#!/usr/bin/python3

import sys
import os
import pandas as pd
import numpy as np
import xlsxwriter

def main():
	
	family="Tyrannidae"
	max_keep=5
	remove_nospecies=True
	remove_hybrids=True
	remove_uncertain=True
	collapse_subspecies=True
	age_cutoff=1985
	kept=dict()
	priorities={"DMNH":3, "DMNS":3, "UCM":3, "LACM":1, "MSB":1} #museums to check first; highest number = best
	
	#Dataframe: species museum specimen_id record_id
	print('Loading VertNet queries...')
	vertnet=pd.read_csv("~/Desktop/PRFB_Birds/VertNet_query.tsv", sep="\t", header=0, low_memory=False)
	#vertnet = vertnet.loc[(vertnet['order']==family) & (vertnet['isfossil'] == 0) & (vertnet['hastissue'] == 1)]
	vertnet = vertnet.loc[(vertnet['family']==family) & (vertnet['isfossil'] == 0) & (vertnet['hastissue'] == 1)]

	
	#remove samples with no epithet
	if remove_nospecies:
		bads=["Sp.", "Sp", "sp", "sp.", "Cf.", "cf.", "Cf", "cf", "NaN", "nan"]
		vertnet = vertnet.loc[~vertnet["specificepithet"].isin(bads)]
		vertnet = vertnet.dropna(subset=["specificepithet"])
		vertnet = vertnet[~vertnet["scientificname"].str.contains("needs identification")]
	
	if remove_uncertain:
		vertnet = vertnet[~vertnet["scientificname"].str.contains(",")]
		vertnet = vertnet[~vertnet["scientificname"].str.contains("\?")]

	if remove_hybrids:
		vertnet = vertnet[~vertnet["scientificname"].str.contains(" x ")]
		vertnet = vertnet[~vertnet["scientificname"].str.contains("hybrid")]

	if age_cutoff > 0:
		vertnet = vertnet[vertnet["year"] > age_cutoff]

	#format species name 
	vertnet["scientificname"] = vertnet['genus'].map(str) + ' ' + vertnet['specificepithet'].map(str) + ' ' + vertnet['infraspecificepithet'].map(str)

	#subset columns 
	vertnet = vertnet[["scientificname", "family", "institutioncode", "catalognumber", "recordnumber", "preparations", "year", "country"]]
	vertnet["source"] = "vertnet"
	vertnet = vertnet.replace({'scientificname': r'\snan$'}, {'scientificname': ''}, regex=True)
	
	#if collapse, change that here
	if collapse_subspecies:
		vertnet["subspecies"] = vertnet["scientificname"]
		vertnet["scientificname"] = vertnet.scientificname.str.split().str.get(0) + " " + vertnet.scientificname.str.split().str.get(1)
	
	#print(vertnet)
	#initialize dict of keeps
	#kept = {key: 0 for key in set(vertnet["scientificname"])} 
	speclist = set(list(vertnet["scientificname"]))
	#print(len(kept.keys()))
	#print(len(list(vertnet["scientificname"])))
	#print(len(set(list(vertnet["scientificname"]))))
	
	#get counts of unique species per museum, sort 
	counts = vertnet.groupby('institutioncode')['scientificname'].nunique().sort_values(ascending=False).reset_index(name='count')
	#prettyPrint(counts)
	#print(counts)
	
	#select top X specimens that are most recent and not skeletal? 
	samples = getSamples(vertnet, priorities, counts["institutioncode"])
	del(vertnet)
	
	m=max(counts["count"])
	for idx, row in counts.iterrows():
		if row["institutioncode"] in priorities:
			counts.loc[counts["institutioncode"]==row["institutioncode"], "count"] = m + priorities[row["institutioncode"]]
	samples["order"]=0
	for idx, row in counts.iterrows():
		samples.loc[samples["institutioncode"]==row["institutioncode"], "order"] = row["count"]
	samples = samples.sort_values("order", ascending=True, ignore_index=True)
	samples = samples.reset_index(drop=True)
	
	#make master sheet of species presence/ absense etc 
	(summary, kept) = makeSummarySheet(samples, speclist, 1)
	#print(summary)
	#print(kept)

	
	#fill in any missing species
	taxonomy = pd.read_csv("~/Desktop/PRFB_Birds/HBW-BirdLife_List_of_Birds_v5.txt", sep="\t", header=0, encoding = "ISO-8859-1", low_memory=False)
	taxonomy = taxonomy.loc[taxonomy["Family name"]==family]
	summary = fillSummaryMissingTaxa(list(taxonomy["Scientific name"]), summary)
	summary = summary.sort_values("Species", ignore_index=True)
	#print(summary)
	#df[df['ids'].str.contains("ball")]
	
	#write excel file 
	writer = pd.ExcelWriter(str(family + '.xlsx'), engine='xlsxwriter')
	summary.to_excel(writer, sheet_name='Summary')
	for group, data in kept.groupby("institutioncode"):
		data = data.sort_values("scientificname", ignore_index=True)
		data.to_excel(writer, sheet_name=group)
	writer.save()

def fillSummaryMissingTaxa(species, df):
	#print(species)
	for spec in species:
		#print(spec)
		sp_data = df[df["Species"]==spec]
		#print(sp_data)
		if sp_data is None or sp_data.shape[0] == 0:
			df = df.append(pd.Series([spec, "NA", "NA"], index=["Species", "Institutions_sampled", "All_Institutions"]), ignore_index=True)
			#print(spec)
			#sys.exit()
	return(df)	

def makeSummarySheet(samples, speclist, max_keep=1):
	summary=list()
	kept_samples=list()
	for species in speclist:
		sp_data = samples.loc[samples["scientificname"]==species]
		alts=",".join(set(list(sp_data["institutioncode"])))
		inst=list()
		for i in range(0, max_keep):
			kept_samples.append(sp_data.iloc[i-1].copy(deep=True))
			inst.append(sp_data.iloc[i-1]["institutioncode"])
		inst=",".join(set(inst))
		summary.append([species, inst, alts])
	sum_df = pd.DataFrame(summary)
	sum_df.columns=["Species", "Institutions_sampled", "All_Institutions"]
	kept_df = pd.DataFrame(kept_samples)
	return(sum_df, kept_df)
	

def getSamples(vertnet, priorities, order):
	ret=list()
	
	for group in priorities.keys():
		
		grouped = data=vertnet.groupby("institutioncode")
		if group not in grouped.groups:
			continue
		data=vertnet.groupby("institutioncode").get_group(group)
		for species, sp_data in data.groupby("scientificname"):
			best=None
			picked=None
			sp_data = sp_data.sort_values("year", ascending=False)
			picked=sp_data.iloc[0].copy(deep=True)

			alternates=""
			idx=0
			if sp_data.shape[0] > 1:
				for i, row in sp_data.iterrows():
					idx+=1
					if idx==1:
						continue
					else:
						if idx==2:
							alternates=str(row["catalognumber"])
							if str(row["recordnumber"]) not in ["nan", "NaN", "NAN"]:
								alternates = alternates + "_" + str(row["recordnumber"])
						else:
							alternates=alternates + " / " + str(row["catalognumber"])
							if str(row["recordnumber"]) not in ["nan", "NaN", "NAN", None, np.nan]:
								alternates = alternates + "_" + str(row["recordnumber"])
			best=sp_data.loc[sp_data["preparations"].str.contains("muscle|heart|liver|kidney", regex=True)]
			if best is not None:
				if best.shape[0] > 0:
					picked=best.iloc[0].copy(deep=True)
			
			picked["alternates"]=alternates
			ret.append(picked)
	
	for group in order:
		data=vertnet.groupby("institutioncode").get_group(group)
		if group in list(priorities.keys()):
			#print("skipping",group)
			continue
		else:
			for species, sp_data in data.groupby("scientificname"):
				best=None
				picked=None
				sp_data = sp_data.sort_values("year", ascending=False, ignore_index=True)
				picked=sp_data.iloc[0].copy(deep=True)

				alternates=""
				idx=0
				if sp_data.shape[0] > 1:
					for i, row in sp_data.iterrows():
						idx+=1
						if idx==1:
							continue
						else:
							if idx==2:
								alternates=str(row["catalognumber"])
								if str(row["recordnumber"]) not in ["nan", "NaN", "NAN"]:
									alternates = alternates + "_" + str(row["recordnumber"])
							else:
								alternates=alternates + " / " + str(row["catalognumber"])
								if str(row["recordnumber"]) not in ["nan", "NaN", "NAN", None, np.nan]:
									alternates = alternates + "_" + str(row["recordnumber"])
				best=sp_data.loc[sp_data["preparations"].str.contains("muscle|heart|liver|kidney", regex=True)]
				if best is not None:
					if best.shape[0] > 0:
						picked=best.iloc[0].copy(deep=True)
				
				picked["alternates"]=alternates
				ret.append(picked)
	#return(ret, kept)
	df = pd.DataFrame(ret)
	#lessprint(df)
	return(df)


def prettyPrint(data, max_row=None, max_col=None):
	with pd.option_context('display.max_rows', max_row, 'display.max_columns', max_col):
		print(data)	


#Call main function
if __name__ == '__main__':
	main()





