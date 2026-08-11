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

1. **Dataset Loading & Formatting**
   * Loaded the **Sentiment140** dataset (`training.1600000.processed.noemoticon.csv`) using `ISO-8859-1` encoding.
   * Mapped original columns (`target`, `id`, `date`, `flag`, `user`, `text`) and extracted the raw `text` column for processing.
