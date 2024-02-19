import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def load_model():
    # Loading the trained SVM classifier
    model = joblib.load('svm_model.joblib')
    # Loading the TF-IDF vectorizer
    vectorizer = joblib.load('tfidf_vect.joblib')
    return model, vectorizer

def predict_spam(input_text, model, vectorizer):
    input_vector = vectorizer.transform([input_text])
    prediction = model.predict(input_vector)[0]
    return prediction

def main():
    model, vectorizer = load_model()
    input_text = input("Enter an email text: ")
    prediction = predict_spam(input_text, model, vectorizer)

    if prediction == 1:
        print("Predicted: Spam")
    else:
        print("Predicted: Ham")

if __name__ == "__main__":
    main()
