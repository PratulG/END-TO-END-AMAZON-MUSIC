import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# Load the dataset
data = pd.read_csv('Musical_instruments_reviews.csv')
data = data[['reviewText', 'overall']]

# Handle missing values
data = data.dropna()  # drop missing values

# Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(data.reviewText, data.overall, test_size=0.2, random_state=0)

# Create a pipeline for bag of words with logistic regression
lr_bow_pipeline = Pipeline([
    ('cv', CountVectorizer(min_df=0, max_df=1, ngram_range=(1,3))),
    ('lr', LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=0))
])

# Fit the pipeline to the training data
lr_bow_pipeline.fit(x_train, y_train)

# Define a function for the prediction page
def predict():
    # Get the review text from the user
    review_text = st.text_input('Enter your review:')
    # Make a prediction using the pipeline
    if review_text:
        prediction = lr_bow_pipeline.predict([review_text])
        # Display the prediction
        st.write(f"Prediction: {prediction[0]}")

# Define the Streamlit app
def app():
    st.title('Musical Instruments Review Sentiment Analysis')
    # Add a sidebar menu
    menu = ['Prediction', 'Hyperparameters', 'Performance']
    choice = st.sidebar.selectbox('Select an option', menu)
    
    if choice == 'Prediction':
        predict()
    elif choice == 'Hyperparameters':
        # Use sliders to let the user customize the hyperparameters
        min_df = st.sidebar.slider('min_df', 0, 10, 1)
        max_df = st.sidebar.slider('max_df', 0.5, 1.0, 0.75)
        ngram_min = st.sidebar.slider('ngram_min', 1, 3, 1)
        ngram_max = st.sidebar.slider('ngram_max', 1, 3, 3)
        C = st.sidebar.slider('C', 0.1, 10.0, 1.0)
        # Update the pipeline with the new hyperparameters
        lr_bow_pipeline.set_params(cv__min_df=min_df,
                                   cv__max_df=max_df,
                                   cv__ngram_range=(ngram_min, ngram_max),
                                   lr__C=C)
        st.write('Updated hyperparameters:')
        st.write(f'min_df: {min_df}, max_df: {max_df}, ngram_range: ({ngram_min}, {ngram_max}), C: {C}')
    elif choice == 'Performance':
        # Make predictions on the test set
        y_pred = lr_bow_pipeline.predict(x_test)
        # Display the classification report and accuracy score
        st.write(classification_report(y_test, y_pred))
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

if __name__ == '__main__':
    app()