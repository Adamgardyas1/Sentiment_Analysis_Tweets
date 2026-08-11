![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3776AB?style=for-the-badge)
![TextBlob](https://img.shields.io/badge/TextBlob-000000?style=for-the-badge)

## Project Overview
This project focuses on performing **Sentiment Analysis** (Positive, Negative, Neutral) on text datasets using Natural Language Processing (NLP) techniques and Machine Learning algorithms. 

The goal is to extract meaningful insights from raw text data.

## Used Libraries
* `pandas`
* `nltk`
* `textblob`
* `wordcloud`
* `matplotlib`

## Step-by-Step Implementation

### Step 1: Data Loading & Column Selection
Loading the CSV file with predefined column names, specifying `ISO-8859-1` encoding, and selecting the target text column.

```python
import pandas as pd

# Path to the CSV file
file_path = 'training.1600000.processed.noemoticon.csv'

# Dataset columns
columns = ['target', 'id', 'date', 'flag', 'user', 'text']

# Load dataset
df = pd.read_csv(file_path, encoding='ISO-8859-1', names=columns)

# Select relevant column
df = df[['text']]
```

### Step 2: Text Cleaning & Preprocessing (NLP)
* **URL & Mention Removal:** Filtered out web links (`http/https/www`), `@user` mentions, and hashtags (`#`) using regular expressions (`re`).
* **Punctuation & Formatting:** Stripped special characters/punctuation and converted all text to lower case for normalization.
* **Tokenization & Stop Words:** Tokenized sentences into individual words using `nltk.word_tokenize` and eliminated common English stop words (`nltk.corpus.stopwords`).

```python
# Download stop words
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Apply clean_text function to 'text' column
df['cleaned_text'] = df['text'].apply(clean_text)
```
