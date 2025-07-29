import pickle
import streamlit as st

tf_idf = pickle.load(open('tfidf_vectorizer.sav', 'rb'))
model = pickle.load(open('random_forest_model.sav', 'rb'))

st.title('News Spam Classifier')

title = st.text_input('Enter the news title:')
text = st.text_area('Enter the news text:')

content = title.strip() + " " + text.strip()
prediction = ''

if st.button('Classify'):
    if content.strip():
        try: 
            text_tfidf = tf_idf.transform([content])

            prediction = model.predict(text_tfidf)
    
            if prediction == 0:
                st.success('The news is likely NOT spam.')
            else:
                st.warning('The news is likely spam.')
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning('Please enter both the title and text of the news.')
