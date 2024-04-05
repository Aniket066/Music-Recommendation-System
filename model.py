import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
# t-SNE [1] is a tool to visualize high-dimensional data. 
# It converts similarities between data points to joint probabilities.

import warnings

warnings.filterwarnings('ignore')
tracks = pd.read_csv('songs_normalize (1).csv')
print("Head: ")
print(tracks.head())
# shows numbers of rows and columns
print("Shape: ")
print(tracks.shape)


# check if there are null values in the columns of our data frame
tracks.info()
# calculating number of null values
print(tracks.isnull().sum())
# if present then dropping them
tracks.dropna(inplace=True)

# t-SNE is an algorithm that can convert high dimensional data to low dimensions
# t-SNE (t-distributed Stochastic Neighbor Embedding) is an unsupervised non-linear 
# dimensionality reduction technique for data exploration and visualizing 
# high-dimensional data.
x = tracks.iloc[:, 3:17]
model = TSNE(n_components=2, random_state=0)
# https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
tsne_data = model.fit_transform(x.head(2000))
plt.figure(figsize=(7, 7))
plt.scatter(tsne_data[:, 0], tsne_data[:, 1])
plt.show()

# shows unique songs in dataset:
print("shows unique songs in dataset: ")
print(tracks['song'].nunique(), tracks.shape)
# sorting tracks according to popularity
tracks = tracks.sort_values(by=['popularity'], ascending=False)
print(tracks.head)
print(tracks.drop_duplicates(subset=['song'], keep='first', inplace=True))

# findng coloums containing float values
floats = []
for col in tracks.columns:
    if tracks[col].dtype == 'float':
        floats.append(col)
# plotting its graph
print("Number of coloums containing float values: ")
print(len(floats))
plt.subplots(figsize=(15, 5))
for i, col in enumerate(floats):
    plt.subplot(2, 5, i + 1)
    sb.distplot(tracks[col])
plt.tight_layout()
plt.show()

# predicting popularity of songs
song_vectorizer = CountVectorizer()
song_vectorizer.fit(tracks['genre'])
tracks = tracks.sort_values(by=['popularity'], ascending=False).head(1000)

def get_similarities(song_name, data):
    # Getting vector for the input song.
    text_array1 = song_vectorizer.transform(data[data['song'] == song_name]['genre']).toarray()
    num_array1 = data[data['song'] == song_name].select_dtypes(include=np.number).to_numpy()

    # We will store similarity for each row of the dataset.
    sim = []
    for idx, row in data.iterrows():
        name = row['song']

        # Getting vector for current song.
        text_array2 = song_vectorizer.transform(data[data['song'] == name]['genre']).toarray()
        num_array2 = data[data['song'] == name].select_dtypes(include=np.number).to_numpy()

        # Calculating similarities for text as well as numeric features
        text_sim = cosine_similarity(text_array1, text_array2)[0][0]
        num_sim = cosine_similarity(num_array1, num_array2)[0][0]
        sim.append(text_sim + num_sim)

    return sim


def recommend_songs(song_name, data=tracks):
    # Base case
    if tracks[tracks['song'] == song_name].shape[0] == 1:
        print('This song is either not so popular or you\ have entered invalid_name.\n Some songs you may like:\n')

    for song1 in data.sample(n=5)['song'].values:
        print(song1)

    return
    data['similarity_factor'] = get_similarities(song_name, data)
    data.sort_values(by=['similarity_factor', 'popularity'], ascending=[False, False], inplace=True)

    # First song will be the input song itself as the similarity will be highest.



pickle.dump(tracks, open('songs.pkl', 'wb'))
pickle.dump(tracks.to_dict(), open('song_dict.pkl', 'wb'))
pickle.dump(recommend_songs('song',tracks), open('similarity.pkl', 'wb'))
