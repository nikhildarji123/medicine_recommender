import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample data
data = pd.DataFrame({
    'Medicine': ['Medicine A', 'Medicine B', 'Medicine C'],
    'Description': ['Helps with condition X', 'Effective for condition Y', 'Recommended for condition Z']
})

# Vectorize the descriptions
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['Description'])

# Calculate similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def recommend_medicine(medicine_name):
    idx = data.index[data['Medicine'] == medicine_name].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:3]  # Get top 2 similar medicines
    medicine_indices = [i[0] for i in sim_scores]
    return data['Medicine'].iloc[medicine_indices].tolist()

# Streamlit UI
st.title('Personalized Medicine Recommender')

# User input
medicine_name = st.text_input('Enter a medicine name:')

if medicine_name:
    recommendations = recommend_medicine(medicine_name)
    st.write('Recommended Medicines:')
    for rec in recommendations:
        st.write(f'- {rec}')
