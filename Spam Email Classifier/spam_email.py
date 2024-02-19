import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def load_model():
    # Load the trained SVM classifier
    model = joblib.load('svm_model.joblib')
    # Load the TF-IDF vectorizer
    vectorizer = joblib.load('tfidf_vect.joblib')
    return model, vectorizer

def predict_spam(input_text, model, vectorizer):
    # Preprocess the input text
    input_vector = vectorizer.transform([input_text])
    # Make a prediction
    prediction = model.predict(input_vector)[0]
    return prediction

def main():
    # Load the model and vectorizer
    model, vectorizer = load_model()

    # Get user input
    input_text = input("Enter an email text: ")

    # Make a prediction
    prediction = predict_spam(input_text, model, vectorizer)

    # Display the prediction
    if prediction == 1:
        print("Predicted: Spam")
    else:
        print("Predicted: Ham")

if __name__ == "__main__":
    main()
