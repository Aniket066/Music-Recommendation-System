import streamlit as st
import pickle
import pandas as pd
from model import get_similarities

def recommend_songs(song_name, songs):
    songs['similarity_factor'] = get_similarities(song_name, songs)
    songs.sort_values(by=['similarity_factor', 'popularity'], ascending=[False, False], inplace=True)
    return songs['song'].head(5)

song_dict = pickle.load(open('song_dict.pkl', 'rb'))
songs = pd.DataFrame(song_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))

from PIL import Image
#opening the image
image1 = Image.open(r"C:\Users\KIIT\Desktop\download.jpeg")
st.image(image1)

#Title
st.title('Music Recommender System')
selected_song_name = st.selectbox(
    'What would you like to search for?',
    songs['song'].values
)

#Recommend button
if st.button('Recommend'):
    recommendations = recommend_songs(selected_song_name, songs)
    if songs is not None:
        for i in recommendations:
            st.write(i)

