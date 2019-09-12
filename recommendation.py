import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import jsonify
from pandas.io.json import json_normalize

class Recommender:
	def recommend(d, dest):
														###### Helper functions #######
		def get_destination_name_from_id(id):
			return df[df.id == id]["destination_name"].values[0]

		def get_id_from_destination_name(destination_name):
			return df[df.destination_name == destination_name]["id"].values[0]
									##################################################
		url = "http://192.168.137.1/api/show-features"
		dt = pd.read_json(url)

		df = json_normalize(dt['data']) 
		# print(df.columns)


		features = ['features.world_heritage', 'features.bike_ride', 'features.cycling', 'features.mountain_view', 'features.honeymoon', 'features.hill_station', 'features.cultural_site', 'features.beach', 'features.trek']

		def combine_features(row):
			try:
				return row['features.world_heritage'] + " " + row['features.bike_ride'] + " " + row['features.cycling']+ " " + row['features.mountain_view']+ " " + row['features.honeymoon']+ " " + row['features.hill_station']+ " " + row['features.cultural_site']+ " " + row['features.beach']+ " " + row['features.trek']
			except:
				print ("Error: ", row)

		for feature in features:
			df[feature] = df[feature].fillna(' ')

		df["combined_features"] = df.apply(combine_features,axis=1)
		df.iloc[0].combined_features
		# print("Combined Features: ", df["combined_features"].head(20))

		cv = CountVectorizer()
		count_matrix = cv.fit_transform(df["combined_features"])

		cosine_sim = cosine_similarity(count_matrix)
		#destination = "Kathmandu"

		# print("The received destination is:", dest)
		destination_index = get_id_from_destination_name(dest)
		similar_destination = list(enumerate(cosine_sim[destination_index]))

		sorted_similar_destinations = sorted(similar_destination, key= lambda x:x[1], reverse=True)

		i=0
		destination_array=[]
		for dest in sorted_similar_destinations:
			# print(get_destination_name_from_id(dest[0]))
			destination_array.append(get_destination_name_from_id(dest[0]))
			# requests.post(get_destination_name_from_id(dest[0]))
			i =  i+1
			if i>2:
				break
		return destination_array