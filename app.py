import pandas as pd
import joblib
import warnings
from warnings import filterwarnings
filterwarnings("ignore")
import streamlit as st

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("#style.css")

# Example UI
#st.title("🎬 Sample Movie Recommendation ")
#st.write("Discover your next favorite movie!")

# Example movie card
#st.markdown("""
#<div class="movie-card">
 #   <img src="https://image.tmdb.org/t/p/w200/8YFL5QQVPy3AgrEQxNYVSgiPEbe.jpg" class="movie-poster">
  #  <h3>Inception</h3>
   # <p><span class="star">★ ★ ★ ★ ☆</span></p>
    #<p>A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea...</p>
#</div>
#""", unsafe_allow_html=True)


def load_data(file_path):
    data=pd.read_csv(file_path + "/" +'movie_data_for_app.csv')
    dataframe=pd.read_csv(file_path + "/" +'movie_dataframe_for_app.csv')
    return data , dataframe

def load_models(file_path):
    sig=joblib.load(file_path + "/" +'sigmoid_kernel.pkl')
    tfv=joblib.load(file_path + "/" +'tfidf_vectorizer.pkl')
    return sig , tfv

data , dataframe =load_data(r'C:\Users\varshith d\OneDrive\Desktop\New folder\dump_obj')
sig , tfv =load_models(r'C:\Users\varshith d\OneDrive\Desktop\New folder\dump_obj')

def give_recommendations(movie_title , model,data,dataframe):
    
    indices = pd.Series(data = data.index , index= data['original_title'])
    
    idx = indices[movie_title]
    
    model_scores = list(enumerate(model[idx]))
    
    model_scores_sorted = sorted(model_scores , key= lambda x : x[1] , reverse = True)
    
    model_scores_10 = model_scores_sorted[1:11]
    
    movie_indices_10 = [i[0] for i in model_scores_10 ]
    
    return dataframe['original_title'][movie_indices_10]

import streamlit as st
st.set_page_config(page_title="Sample Movie Recommender",layout="centered")

st.title(" 🎬 Sample Movie Recommender")
st.write("FIND MOVIES SIMILAR TO YOUR FAVORITE ONE")
movie_list=data['original_title'].sort_values().tolist()
selected_movie=st.selectbox('SELECT A MOVIE :',movie_list)
if st.button('Get recommendation'):
    if selected_movie:
        recommendations = give_recommendations(selected_movie,sig,data,dataframe)

        st.subheader('Movies similar to:'+ selected_movie)
        for index , movie in enumerate(recommendations):
            st.write(str(index + 1)+". "+ movie)

st.markdown("----")
st.markdown('This app uses content based filtering')