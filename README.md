# Music-Recommendation-System
#1. Introduction
With the rapid growth of digital music platforms, users are exposed to millions of songs, making it difficult to find music that matches their preferences. A Music Recommendation System helps users discover songs similar to their interests by analyzing song features.
This project focuses on building a Music Recommendation System using Machine Learning techniques. The system recommends songs based on similarity between songs using genre and numerical audio features such as popularity, danceability, energy, tempo, etc.
The project also includes data visualization using t-SNE to visualize high-dimensional music data and a Streamlit web application to provide an interactive interface for users.

#2. Objectives of the Project
The main objectives of this project are:

To build a Music Recommendation System using Python.
To analyze song datasets and remove missing values.
To visualize high-dimensional data using t-SNE.
To calculate similarity between songs using Cosine Similarity.
To recommend top 5 similar songs.
To develop a Streamlit-based web interface.
To provide an interactive user experience.


#3. Technologies Used
Programming Language
Python
Libraries Used
Pandas — Data manipulation
NumPy — Numerical operations
Matplotlib — Data visualization
Seaborn — Statistical visualization
Scikit-learn — Machine learning algorithms
TSNE — Dimensionality reduction
CountVectorizer — Text feature extraction
Cosine Similarity — Similarity calculation
Streamlit — Web interface
Pickle — Saving model files
PIL (Python Imaging Library) — Image handling

4. Dataset Description
The dataset used in this project is:
songs_normalize.csv
It contains information about songs including:
song — Song name
artist — Artist name
genre — Music genre
popularity — Popularity score
danceability
energy
loudness
tempo
duration
acousticness
instrumentalness
valence
speechiness

These numerical features describe the audio properties of songs and are used to calculate similarity.

#5. System Architecture
The working flow of the system:
Load Dataset
Data Cleaning
Visualization
Feature Processing
Similarity Calculation
Recommendation Generation
Streamlit Interface

#6. Methodology
##Step 1: Import Libraries
The required Python libraries are imported.
Libraries used:
pandas
numpy
matplotlib
seaborn
sklearn
pickle

Warnings are ignored to improve output readability.

##Step 2: Load Dataset
The dataset is loaded using Pandas.
tracks = pd.read_csv('songs_normalize (1).csv')
The dataset structure is checked using:
head()
shape()
info()

##Step 3: Data Cleaning
Missing values are identified using:
tracks.isnull().sum()
Missing valus are removed:
tracks.dropna(inplace=True)
Duplicate songs are removed:
tracks.drop_duplicates(subset=['song'], keep='first')
Songs are sorted based on popularity.

##Step 4: Data Visualization using t-SNE
t-SNE (t-distributed Stochastic Neighbor Embedding) is used to reduce high-dimensional data into two dimensions.
model = TSNE(n_components=2, random_state=0)
tsne_data = model.fit_transform(x.head(2000))
A scatter plot is created to visualize similarity between songs.
plt.scatter(tsne_data[:, 0], tsne_data[:, 1])
This helps visualize relationships between songs in lower dimensions.

##Step 5: Distribution Visualization
Floating-point columns are identified and plotted using Seaborn.
sb.distplot(tracks[col])
This helps understand data distribution.

##Step 6: Text Feature Processing
Genres are converted into numerical vectors using CountVectorizer.
song_vectorizer = CountVectorizer()
song_vectorizer.fit(tracks['genre'])
This converts text into numerical format.

##Step 7: Similarity Calculation
Similarity between songs is calculated using:
Text similarity (genre)
Numeric similarity (audio features)

Cosine Similarity is used.

text_sim = cosine_similarity(text_array1, text_array2)
num_sim = cosine_similarity(num_array1, num_array2)

Final similarity score:
sim.append(text_sim + num_sim)

##Step 8: Recommendation System
Function:
recommend_songs(song_name, data)
Steps:

Check if song exists
Calculate similarity
Sort songs
Return top 5 recommendations

If the song is not found:

Random songs are recommended.

##Step 9: Saving Model Files

Pickle is used to store processed data.

pickle.dump(tracks, open('songs.pkl', 'wb'))
pickle.dump(tracks.to_dict(), open('song_dict.pkl', 'wb'))
pickle.dump(recommend_songs('song',tracks), open('similarity.pkl', 'wb'))

These files are later used in the web app.

##Step 10: Streamlit Web Application
The web interface is built using Streamlit.
Main components:
Image Display
st.image(image1)
Title
st.title('Music Recommender System')
Song Selection
selected_song_name = st.selectbox(...)

User selects a song from dropdown.

Recommendation Button
if st.button('Recommend'):

Top 5 similar songs are displayed.

#7. Working of the System

Step-by-step:

User opens Streamlit app.
User selects a song.
System calculates similarity.
Songs are ranked.
Top 5 songs are displayed.


The system successfully recommends similar songs.

#8. Results
The system achieved:
Successful song recommendation
Fast similarity calculation
Interactive user interface
Visual data representation

