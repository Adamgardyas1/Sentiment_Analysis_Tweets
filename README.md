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
- Loaded the CSV file with predefined column names, specifying `ISO-8859-1` encoding, and selecting the target text column.

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
- Filtered out web links (`http/https/www`), `@user` mentions, and hashtags (`#`) using regular expressions (`re`).
- Stripped special characters/punctuation and converted all text to lower case for normalization.
- Tokenized sentences into individual words using `nltk.word_tokenize` and eliminated common English stop words (`nltk.corpus.stopwords`).

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

### Step 3: Sentiment Analysis via TextBlob
- Extracted continuous polarity scores (`-1.0` to `1.0`) for cleaned text using **TextBlob**.
- Categorized polarity values into three discrete classes (`negative` for `< 0`, `neutral` for `== 0`, `positive` for `> 0`).
- Assigned the resulting classification labels to a new `sentiment` column in the dataframe.


```python
def analyze_sentiment(text):
    # Initialize TextBlob object to extract polarity score (-1.0 to 1.0)
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    # Map polarity score to discrete sentiment labels
    if polarity < 0:
        return 'negative'
    elif polarity == 0:
        return 'neutral'
    else:
        return 'positive'

# Apply sentiment analysis to cleaned text column
df['sentiment'] = df['cleaned_text'].apply(analyze_sentiment)
```

### Step 4: Output Inspection & Summary Statistics
- Displayed dataset head to verify successful column creation and data transformation.
- Evaluated global sentiment distribution across all records using `value_counts()`.
- Extracted top 5 sample tweets for each sentiment class (`negative`, `neutral`, `positive`) to inspect classification output.

```python
# Display first 5 rows of the modified dataframe
print(df.head())

# Compute total count for each sentiment category
print(df['sentiment'].value_counts())

# Extract and inspect sample tweets for each sentiment class
print("\nNegative Tweets:")
print(df[df['sentiment'] == 'negative'].head(5))

print("\nNeutral Tweets:")
print(df[df['sentiment'] == 'neutral'].head(5))

print("\nPositive Tweets:")
print(df[df['sentiment'] == 'positive'].head(5))
