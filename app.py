import streamlit as st
import pandas as pd
import nltk
import random
import requests
import os
from nltk import tokenize
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import docx2txt
from PyPDF2 import PdfReader

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')


USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.97 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36',
]

def get_random_headers():
    """Returns a random header to make HTTP requests."""
    return {
        'User-Agent': random.choice(USER_AGENTS)
    }

def get_sentences(text):
    """Splits text into sentences."""
    return tokenize.sent_tokenize(text)

def get_url(sentence):
    """Fetches the first URL result for the sentence from Google Search."""
    api_key = os.getenv('SERPAPI_KEY')
    if not api_key:
        st.error("SERPAPI_KEY not found in environment variables. Please set it to use plagiarism detection.")
        return None

    params = {
        "q": sentence,
        "api_key": api_key,
        "engine": "google",
    }
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        if 'organic_results' in results:
            return results['organic_results'][0].get('link', None)
    except Exception as e:
        print(f"Error fetching URL: {e}")
    return None

def read_file(file):
    """Reads content from a file and returns it as text."""
    content = ""
    if file.type == "text/plain":
        content = file.getvalue().decode("utf-8")
    elif file.type == "application/pdf":
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            content += page.extract_text()
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        content = docx2txt.process(file)
    return content

def get_text_from_url(url):
    """Fetches and returns the textual content from a URL."""
    response = requests.get(url, headers=get_random_headers())
    soup = BeautifulSoup(response.text, 'html.parser')
    return ' '.join([p.text for p in soup.find_all('p')])

def get_similarity(text1, text2):
    """Calculates cosine similarity between two texts."""
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    return cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]

def get_similarity_between_files(file_texts):
    """Calculates cosine similarity between multiple file texts."""
    similarity_list = []
    for i in range(len(file_texts)):
        for j in range(i + 1, len(file_texts)):
            similarity = get_similarity(file_texts[i], file_texts[j])
            similarity_list.append({
                'File 1': f"File {i + 1}",
                'File 2': f"File {j + 1}",
                'Similarity': similarity
            })
    return similarity_list

def plot_similarity_graphs(df):
    """Plots similarity graphs."""
    st.write("### Similarity Graphs")
    
    # Scatter plot
    st.plotly_chart(px.scatter(df, x='File 1', y='File 2', color='Similarity', title='Similarity Scatter Plot'))

    # Line plot
    st.plotly_chart(px.line(df, x='File 1', y='File 2', color='Similarity', title='Similarity Line Chart'))
    
    # Bar plot
    st.plotly_chart(px.bar(df, x='File 1', y='Similarity', color='File 2', title='Similarity Bar Chart'))

# Streamlit App Configuration
st.set_page_config(page_title='Plagiarism Detection')
st.title('Plagiarism Detector')

option = st.radio("Select input option:", ('Enter text', 'Upload file', 'Find similarities between files'))

if option == 'Enter text':
    text = st.text_area("Enter text here", height=200)
    uploaded_files = None
elif option == 'Upload file':
    uploaded_file = st.file_uploader("Upload file (.docx, .pdf, .txt)", type=["docx", "pdf", "txt"])
    text = read_file(uploaded_file) if uploaded_file else ""
    uploaded_files = None
else:
    uploaded_files = st.file_uploader("Upload multiple files (.docx, .pdf, .txt)", type=["docx", "pdf", "txt"], accept_multiple_files=True)
    text = " ".join([read_file(file) for file in uploaded_files]) if uploaded_files else ""

if st.button('Check for plagiarism or find similarities'):
    if not text:
        st.warning("No text found for plagiarism check or finding similarities.")
    else:
        if option == 'Find similarities between files':
            # For comparing files with each other (using cosine similarity)
            if not uploaded_files:
                st.warning("Please upload files to compare.")
            else:
                file_texts = [read_file(file) for file in uploaded_files]
                similarity_list = get_similarity_between_files(file_texts)

                # Create DataFrame
                df = pd.DataFrame(similarity_list).sort_values(by='Similarity', ascending=False)

                # Plot similarity graphs
                plot_similarity_graphs(df)

                # Show the similarity table
                st.write("### Similarity Table")
                st.write(df)
        
        else:
            sentences = get_sentences(text)
            urls = [get_url(sentence) for sentence in sentences]
            similarity_list = []

            for idx, url in enumerate(urls):
                if url:  # Ensure URL exists before processing
                    text_from_url = get_text_from_url(url)
                    similarity = get_similarity(text, text_from_url)
                    similarity_list.append({
                        'Sentence': sentences[idx],
                        'File 1': sentences[idx],  # This could be the sentence itself or a filename
                        'File 2': url,  # You can also add 'URL' here or another description
                        'Similarity': similarity
                    })

            # Create DataFrame
            df = pd.DataFrame(similarity_list).sort_values(by='Similarity', ascending=False)

            # Format URL as clickable link
            df['File 2'] = df['File 2'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>' if x != "No URL found" else 'No URL found')
            
            st.markdown(df.to_html(escape=False), unsafe_allow_html=True)
