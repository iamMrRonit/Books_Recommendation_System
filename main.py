import numpy as np
import pickle
import streamlit as st

st.header("Book Recommendation System")
a = pickle.load(open('bk.pkl', 'rb'))
b = pickle.load(open('suggest.pkl', 'rb'))
m = pickle.load(open('model.pkl', 'rb'))

name = st.text_input("Enter the book you have read.")
model=NearestNeighbors(algorithm='brute', metric='cosine') ## model

def recommend_books(book_name):
    similar_books = []
    try:
        book_id = np.where(b.index == book_name)[0][0]
        distances, suggestions = m.kneighbors(b.iloc[book_id, :].values.reshape(1, -1))
        for i in range(len(suggestions)):
            similar_books = b.index[suggestions[i]]
    except:
        st.error("Please try a different book")

    return similar_books


print(recommend_books('Timeline')[1])

if st.button("recommend"):
    similar_books = recommend_books(name)
    for i in range(1, len(similar_books)):
        st.info(similar_books[i])
