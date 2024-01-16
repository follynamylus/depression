import streamlit as st
import joblib

# Load your trained model
model = joblib.load('random_forest_model.pkl')


def predict(text):
    # Perform any necessary preprocessing on the text_input if needed

    # Make a prediction using the loaded model
    prediction = model.predict([text])[0]

    return prediction

def main():
    st.title("Text Prediction App")

    # Create a text input box
    text_input = st.text_area("Enter text:", "")

    if st.button("Predict"):
        if text_input:
            prediction = predict(text_input)
            st.success(f"Prediction: {prediction}")
        else:
            st.warning("Please enter some text.")

if __name__ == '__main__':
    main()
