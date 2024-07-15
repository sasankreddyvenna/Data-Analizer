import os
import pandas as pd
import nltk
from textblob import TextBlob
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Set paths for stop words and master dictionary
stop_words_path = 'StopWords'
master_dictionary_path = 'MasterDictionary'

# Function to load stop words
def load_stop_words(path):
    stop_words = set()
    stop_words_files = [
        'StopWords_Auditor.txt',
        'StopWords_Currencies.txt',
        'StopWords_DatesandNumbers.txt',
        'StopWords_Generic.txt',
        'StopWords_GenericLong.txt',
        'StopWords_Geographic.txt',
        'StopWords_Names.txt'
    ]
    for file_name in stop_words_files:
        try:
            with open(os.path.join(path, file_name), 'r', encoding='utf-8') as f:
                stop_words.update(f.read().split())
        except UnicodeDecodeError:
            with open(os.path.join(path, file_name), 'r', encoding='latin-1') as f:
                stop_words.update(f.read().split())
        except FileNotFoundError:
            print(f"Warning: File '{file_name}' not found in '{path}'. Skipping.")
    return stop_words

# Load stop words and master dictionary
try:
    stop_words = load_stop_words(stop_words_path)
except FileNotFoundError:
    print(f"Error: The directory '{stop_words_path}' was not found. Please ensure it exists and contains the stop word files.")
    exit(1)

positive_words = set()
negative_words = set()
try:
    with open(os.path.join(master_dictionary_path, 'positive-words.txt'), 'r', encoding='utf-8') as f:
        positive_words.update(f.read().split())
    with open(os.path.join(master_dictionary_path, 'negative-words.txt'), 'r', encoding='utf-8') as f:
        negative_words.update(f.read().split())
except FileNotFoundError:
    print(f"Error: The directory '{master_dictionary_path}' or the dictionary files were not found. Please ensure they exist and contain the appropriate files.")
    exit(1)

# Function to extract article text from a URL using Selenium
def extract_article_text(url):
    try:
        # Set up Selenium WebDriver
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run headless Chrome
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
        
        # Navigate to the URL
        driver.get(url)
        
        # Extract the title
        title = driver.find_element(By.TAG_NAME, 'h1').text
        
        # Extract the article text
        paragraphs = driver.find_elements(By.TAG_NAME, 'p')
        article_text = ' '.join([p.text for p in paragraphs])
        
        driver.quit()
        
        return title + "\n" + article_text
    except Exception as e:
        print(f"Error extracting {url}: {e}")
        return None

# Function to clean text
def clean_text(text):
    tokens = nltk.word_tokenize(text)
    cleaned_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
    return cleaned_tokens

# Sentiment Analysis Functions
def calculate_positive_score(tokens):
    return sum(1 for word in tokens if word in positive_words)

def calculate_negative_score(tokens):
    return sum(1 for word in tokens if word in negative_words)

def calculate_polarity_score(positive_score, negative_score):
    return (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)

def calculate_subjectivity_score(positive_score, negative_score, total_words):
    return (positive_score + negative_score) / (total_words + 0.000001)

# Readability Analysis Functions
def calculate_avg_sentence_length(text):
    sentences = nltk.sent_tokenize(text)
    total_words = len(nltk.word_tokenize(text))
    return total_words / max(len(sentences), 1)  # Avoid division by zero

def calculate_complex_word_percentage(tokens):
    complex_words = [word for word in tokens if len([char for char in word if char in 'aeiou']) > 2]
    return len(complex_words) / max(len(tokens), 1)  # Avoid division by zero

def calculate_fog_index(avg_sentence_length, complex_word_percentage):
    return 0.4 * (avg_sentence_length + complex_word_percentage)

def calculate_avg_words_per_sentence(text):
    sentences = nltk.sent_tokenize(text)
    total_words = len(nltk.word_tokenize(text))
    return total_words / max(len(sentences), 1)  # Avoid division by zero

def calculate_complex_word_count(tokens):
    return len([word for word in tokens if len([char for char in word if char in 'aeiou']) > 2])

def calculate_word_count(tokens):
    return len(tokens)

def calculate_syllable_per_word(tokens):
    syllable_count = lambda word: sum([1 for char in word if char in 'aeiou'])
    return sum(syllable_count(word) for word in tokens) / max(len(tokens), 1)  # Avoid division by zero

def calculate_personal_pronouns(text):
    pronouns = re.findall(r'\b(I|we|my|ours|us)\b', text, re.I)
    return len(pronouns)

def calculate_avg_word_length(tokens):
    return sum(len(word) for word in tokens) / max(len(tokens), 1)  # Avoid division by zero

# Function to analyze text and calculate variables
def analyze_text(text):
    cleaned_tokens = clean_text(text)
    
    positive_score = calculate_positive_score(cleaned_tokens)
    negative_score = calculate_negative_score(cleaned_tokens)
    polarity_score = calculate_polarity_score(positive_score, negative_score)
    subjectivity_score = calculate_subjectivity_score(positive_score, negative_score, len(cleaned_tokens))
    
    avg_sentence_length = calculate_avg_sentence_length(text)
    complex_word_percentage = calculate_complex_word_percentage(cleaned_tokens)
    fog_index = calculate_fog_index(avg_sentence_length, complex_word_percentage)
    avg_words_per_sentence = calculate_avg_words_per_sentence(text)
    complex_word_count = calculate_complex_word_count(cleaned_tokens)
    word_count = calculate_word_count(cleaned_tokens)
    syllable_per_word = calculate_syllable_per_word(cleaned_tokens)
    personal_pronouns = calculate_personal_pronouns(text)
    avg_word_length = calculate_avg_word_length(cleaned_tokens)
    
    return {
        'POSITIVE SCORE': positive_score,
        'NEGATIVE SCORE': negative_score,
        'POLARITY SCORE': polarity_score,
        'SUBJECTIVITY SCORE': subjectivity_score,
        'AVG SENTENCE LENGTH': avg_sentence_length,
        'PERCENTAGE OF COMPLEX WORDS': complex_word_percentage,
        'FOG INDEX': fog_index,
        'AVG NUMBER OF WORDS PER SENTENCE': avg_words_per_sentence,
        'COMPLEX WORD COUNT': complex_word_count,
        'WORD COUNT': word_count,
        'SYLLABLE PER WORD': syllable_per_word,
        'PERSONAL PRONOUNS': personal_pronouns,
        'AVG WORD LENGTH': avg_word_length,
    }

# Function to process each URL
def process_url(url):
    article_text = extract_article_text(url)
    if article_text:
        analysis_results = analyze_text(article_text)
        return analysis_results
    else:
        print(f"Error: No article text extracted for URL '{url}'. Check the URL or try another URL.")
        return None

# Prompt for URL input
url = input("Enter the URL to analyze: ").strip()

# Analyze the provided URL
if url:
    analysis_results = process_url(url)
    if analysis_results:
        # Display results
        print("\nAnalysis Results:")
        for key, value in analysis_results.items():
            print(f"{key}: {value}")
        
        # Ask if user wants results in Excel
        save_to_excel = input("\nDo you want to save the analysis results to Excel? (yes/no): ").strip().lower()
        if save_to_excel == 'yes':
            output_df = pd.DataFrame.from_dict(analysis_results, orient='index', columns=['Value'])
            output_df.index.name = 'Metric'
            output_file = 'Analysis_Results.xlsx'
            output_df.to_excel(output_file)
            print(f"Analysis results saved to '{output_file}'.")
    else:
        print("Error analyzing URL.")
else:
    print("No URL provided.")