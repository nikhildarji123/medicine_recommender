import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

@st.cache_data
def load_data():
    # Load your dataset
    data = pd.read_csv("your_data.csv")
    return data

def preprocess_data(data):
    # Example preprocessing
    X = data.drop('target_column', axis=1)
    y = data['target_column']
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

@st.cache_data
def get_model():
    data = load_data()
    X, y = preprocess_data(data)
    model = train_model(X, y)
    return model

st.title("Medicine Recommendation System")
st.write("This is a simple implementation of a Medicine Recommendation System using Streamlit.")

def user_input_features():
    age = st.slider('Age', 1, 100, 30)
    bmi = st.slider('BMI', 10.0, 50.0, 25.0)
    # Add other input features based on your dataset
    # Example: gender = st.selectbox('Gender', ('Male', 'Female'))
    data = {'age': age, 'bmi': bmi}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input Features')
st.write(input_df)

model = get_model()
prediction = model.predict(input_df)

st.subheader('Prediction')
st.write(prediction)

if __name__ == '__main__':
    st.run()
