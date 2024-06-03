#!/usr/bin/env python
# coding: utf-8



#Installing Required Packages
!pip install describe bertopic numpy gensim wordcloud nltk langdetect spacy textblob matplotlib seaborn transformers detoxify perspective

# Import necessary libraries
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
import pyLDAvis.gensim_models as gensimvis
import pickle
import pyLDAvis
import spacy

from wordcloud import WordCloud
from langdetect import detect
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from gensim import corpora, models
from pprint import pprint
from string import punctuation
from collections import Counter
from bertopic import BERTopic
from detoxify import Detoxify
from perspective import PerspectiveAPI
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('words')
nltk.download('wordnet')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load the male dataset and female dataset

male_data_orj = pd.read_excel('/male_data.xlsx', engine='openpyxl')
female_data_orj = pd.read_excel('/female_data.xlsx', engine='openpyxl')

# Description of data

male_data_orj.head()
female_data_orj.head(3)

male_data_orj.columns
female_data_orj.columns

male_data_orj.describe()
female_data_orj.describe()



# Data Preprocessing

# Drop columns with all NaN values
male_data_orj = male_data_orj.dropna(axis=1, how='all')
female_data_orj = female_data_orj.dropna(axis=1, how='all')


total_rows_maledata_orj = len(male_data_orj)
distinct_rows_maledata_orj = male_data_orj.nunique()

# Count of all rows in male_data and female_data
print(f"Total Rows for Male Data: {total_rows_maledata_orj}")

# Count of distinct rows in male_data
print("Distinct Count for Each Column for Male Data:")
print(distinct_rows_maledata_orj)

total_rows_femaledata_orj = len(female_data_orj)
distinct_rows_femaledata_orj = female_data_orj.nunique()

# Count of all rows in male_data and female_data
print(f"Total Rows for Female Data: {total_rows_femaledata_orj}")

# Count of distinct rows in male_data
print("Distinct Count for Each Column for Female Data:")
print(distinct_rows_femaledata_orj)

# Remove unusual comments or posts (empty comments or comments with only symbols)
male_data_orj = male_data_orj[male_data_orj['comment_text'].str.strip() != '']  # Remove empty comment
female_data_orj = female_data_orj[female_data_orj['comment_text'].str.strip() != '']  # Remove empty comments

#Converting the data into string format 
male_data_orj['comment_text'] = male_data_orj['comment_text'].astype(str)
female_data_orj['comment_text'] = female_data_orj['comment_text'].astype(str)

# Remove unusual characters from comment_text using regex
male_data_orj['comment_text'] = male_data_orj['comment_text'].apply(lambda x: re.sub(r'[^\x00-\x7F]+', '', x))
female_data_orj['comment_text'] = female_data_orj['comment_text'].apply(lambda x: re.sub(r'[^\x00-\x7F]+', '', x))
# Define a function to filter out non-English and non-alphabetic words
def clean_text(text):
    # Define a set of English words
    english_word_set = set(english_words.words())

    # Split the text into words
    words = text.split()

    # Filter out non-English words and non-alphabetic characters
    clean_words = [word for word in words if word.lower() in english_word_set and word.isalpha()]

    # Join the clean words back into a clean text
    clean_text = " ".join(clean_words)

    return clean_text

# Apply the function to the 'comment_text' column
male_data_orj['male_cleaned_comment_text'] = male_data_orj['comment_text'].apply(clean_text)
female_data_orj['female_cleaned_comment_text'] = female_data_orj['comment_text'].apply(clean_text)

print(male_data_orj.columns)
print(female_data_orj.columns)

# Display the cleaned comments
print("Male Data Cleaned Comments:")
print(male_data_orj[['comment_text', 'male_cleaned_comment_text']].head())

# Display the cleaned comments
print("Female Data Cleaned Comments:")
print(female_data_orj[['comment_text', 'female_cleaned_comment_text']].head())

# Remove unusual comments or posts (empty comments or comments with only symbols)
male_data_orj['male_cleaned_comment_text_unusual_symbols'] = male_data_orj['male_cleaned_comment_text'].apply(lambda x: re.sub(r"[^\w\s]", "", str(x)))
# Remove unusual comments or posts (empty comments or comments with only symbols)
female_data_orj['female_cleaned_comment_text_unusual_symbols'] = female_data_orj['female_cleaned_comment_text'].apply(lambda x: re.sub(r"[^\w\s]", "", str(x)))

# Function to remove URLs from text data
def remove_urls(text):
    # Define a regular expression pattern to match URLs
    url_pattern = r'https?://\S+|www\.\S+'
    without_urls = re.sub(url_pattern, '', text)
    return without_urls

print(male_data_orj['male_cleaned_comment_text_unusual_symbols'])

# Apply the function to the specific column in the DataFrame
male_data_orj['male_cleaned_comment_text_removed_urls'] = male_data_orj['male_cleaned_comment_text_unusual_symbols'].apply(remove_urls)
print(male_data_orj['male_cleaned_comment_text_removed_urls'])

# Apply the function to the specific column in the DataFrame
female_data_orj['female_cleaned_comment_text_removed_urls'] = female_data_orj['female_cleaned_comment_text_unusual_symbols'].apply(remove_urls)
print(female_data_orj['female_cleaned_comment_text_removed_urls'])
male_data_orj.drop_duplicates(inplace=True)
female_data_orj.drop_duplicates(inplace=True)
from nltk.tokenize import word_tokenize
import numpy as np

# Tokenize comments and calculate comment lengths
male_data_orj['comment_length'] = male_data_orj['comment_text'].apply(lambda x: len(word_tokenize(str(x))))
female_data_orj['comment_length'] = female_data_orj['comment_text'].apply(lambda x: len(word_tokenize(str(x))))

# Calculate average comment length
avg_comment_length_male = np.mean(male_data_orj['comment_length'])
avg_comment_length_female = np.mean(female_data_orj['comment_length'])

# Print results
print(f"Number of comments (Male): {len(male_data_orj)}")
print(f"Average comment length (Male): {avg_comment_length_male:.2f} words")

print(f"Number of comments (Female): {len(female_data_orj)}")
print(f"Average comment length (Female): {avg_comment_length_female:.2f} words")

# Tokenize comments and calculate comment lengths
male_data_orj['cleaned_comment_length'] = male_data_orj['male_cleaned_comment_text_removed_urls'].apply(lambda x: len(word_tokenize(str(x))))
female_data_orj['cleaned_comment_length'] = female_data_orj['female_cleaned_comment_text_removed_urls'].apply(lambda x: len(word_tokenize(str(x))))

# Calculate average comment length
avg_comment_length_cleaned_male = np.mean(male_data_orj['cleaned_comment_length'])
avg_comment_length_cleaned_female = np.mean(female_data_orj['cleaned_comment_length'])

# Print results
print(f"Number of comments (Cleaned_Male): {len(male_data_orj)}")
print(f"Average comment length (Cleaned_Male): {avg_comment_length_cleaned_male:.2f} words")

print(f"Number of comments (Cleaned_Female): {len(female_data_orj)}")
print(f"Average comment length (Cleaned_Female): {avg_comment_length_cleaned_female:.2f} words")


# Function to remove stop words using spaCy
def remove_stop_words(string):
    # Check if the value is NaN or a boolean (True/False)
    if pd.isna(string) or isinstance(string, bool):
        return ''

    # Tokenize the string into individual words
    doc = nlp(string)

    # Filter out stop words
    filtered_words = [token.text for token in doc if not token.is_stop]

    # Join the filtered words back into a string
    new_string = ' '.join(filtered_words)

    return new_string


male_data_orj['stop_words_removed'] = male_data_orj['cleaned_comment_length'].apply(remove_stop_words)

# Display the original and filtered comments for the first few rows in the random samples
print("stop_words_removed:")



female_data_orj['stop_words_removed'] = female_data_orj['cleaned_comment_length'].apply(remove_stop_words)

# Display the original and filtered comments for the first few rows in the random samples
print("stop_words_removed:")



# Tokenize comments and calculate comment lengths
male_data_orj['comment_length'] = male_data_orj['stop_words_removed'].apply(lambda x: len(word_tokenize(str(x))))
female_data_orj['comment_length'] = female_data_orj['stop_words_removed'].apply(lambda x: len(word_tokenize(str(x))))
# Calculate average comment length
avg_comment_length_male = np.mean(male_data_orj['comment_length'])
avg_comment_length_female = np.mean(female_data_orj['comment_length'])
# Print results
print(f"Number of comments (Male): {len(female_data_orj)}")
print(f"Average comment length (Male): {avg_comment_length_female:.2f} words")

print(f"Number of comments (Male): {len(male_data_orj)}")
print(f"Average comment length (Male): {avg_comment_length_female:.2f} words")


# Combine comments into a single text
combined_text_male_data = ' '.join(male_data['comment_text'])
combined_text_female_data = ' '.join(female_data['comment_text'])

# Basic statistics
num_comments_male = len(male_data)
avg_comment_length_male = male_data['comment_text'].str.split().apply(len).mean()
avg_comment_length_male = male_data['stop_words_removed'].str.split().apply(len).mean()

num_comments_female = len(female_data)
avg_comment_length_female = female_data['comment_text'].str.split().apply(len).mean()
avg_comment_length_female = female_data['stop_words_removed'].str.split().apply(len).mean()


print(f"Number of comments: {num_comments_male}")
print(f"Average comment length: {avg_comment_length_male:.2f} words")


print(f"Number of comments: {num_comments_female}")
print(f"Average comment length: {avg_comment_length_female:.2f} words")


male_data_after = pd.read_excel('/LAST/male_dataset.xlsx', engine='openpyxl')

female_data_after = pd.read_excel('/LAST/female_dataset.xlsx', engine='openpyxl')
    


# In[8]:


# Summary statistics before data preprocessing
print("Summary Statistics for Male Politicians dataset orjinal:")
print(orjinal_male_data.describe())
print("\nSummary Statistics for Female Politicians dataset orjinal:")
print(orjinal_female_data.describe())


# In[20]:


# Summary statistics after preprocessing
print("Summary Statistics for Male Politicians:")
print(male_data.describe())
print("\nSummary Statistics for Female Politicians:")
print(female_data.describe())


# Distribution of engagement metrics (likes, shares, comments)
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(male_data['likes'], bins=20, color='blue', kde=True)
plt.title('Distribution of Likes for Male Politicians')

plt.subplot(1, 3, 2)
sns.histplot(female_data['likes'], bins=20, color='red', kde=True)
plt.title('Distribution of Likes for Female Politicians')

plt.subplot(1, 3, 3)
sns.boxplot(x='Gender', y='comments', data=pd.concat([male_data, female_data]))
plt.title('Boxplot of Comments by Gender')

plt.tight_layout()
plt.show()


# In[19]:

# Correlation between engagement metrics
engagement_metrics = ['likes', 'shares', 'comments', 'reactions_count']

correlation_male = male_data[engagement_metrics].corr()
correlation_female = female_data[engagement_metrics].corr()

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(correlation_male, dtype=bool))

# Plotting
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(correlation_male, mask=mask, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap for Male Politicians')

plt.yticks(fontsize=8)  # Reduce text size for y-axis labels

plt.subplot(1, 2, 2)
sns.heatmap(correlation_female, mask=mask, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap for Female Politicians')

plt.yticks(fontsize=8)  # Reduce text size for y-axis labels

plt.tight_layout()
plt.show()


# Summary statistics before data preprocessing
print("Summary Statistics for Male Politicians dataset orjinal:")
print(orjinal_male_data.describe())
print("\nSummary Statistics for Female Politicians dataset orjinal:")
print(orjinal_female_data.describe())


# Summary statistics after data preprocessing
print("Summary Statistics for Male Politicians:")
print(male_data.describe())
print("\nSummary Statistics for Female Politicians:")
print(female_data.describe())

# Distribution of engagement metrics (likes, shares, comments)
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(male_data['likes'], bins=20, color='blue', kde=True)
plt.title('Distribution of Likes for Male Politicians')

plt.subplot(1, 3, 2)
sns.histplot(female_data['likes'], bins=20, color='red', kde=True)
plt.title('Distribution of Likes for Female Politicians')

plt.subplot(1, 3, 3)
sns.boxplot(x='Gender', y='comments', data=pd.concat([male_data, female_data]))
plt.title('Boxplot of Comments by Gender')

plt.tight_layout()
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud

# 1. Gender Distribution
male_gender_distribution = male_data['Gender'].value_counts()
female_gender_distribution = female_data['Gender'].value_counts()
print("Male Gender Distribution:\n", male_gender_distribution)
print("Female Gender Distribution:\n", female_gender_distribution)

# 2. Engagement Metrics
engagement_metrics = ['likes', 'shares', 'loves', 'wow', 'cares', 'sad', 'angry', 'haha', 'reactions_count', 'comments']

# Male Dataset
male_engagement_data = male_data[engagement_metrics]
male_engagement_stats = male_engagement_data.describe()

# Female Dataset
female_engagement_data = female_data[engagement_metrics]
female_engagement_stats = female_engagement_data.describe()

# Plotting
plt.figure(figsize=(12, 6))

# Male Dataset
plt.subplot(1, 2, 1)
sns.boxplot(data=male_engagement_data, orient='h')
plt.title('Engagement Metrics for Male Politicians')

# Female Dataset
plt.subplot(1, 2, 2)
sns.boxplot(data=female_engagement_data, orient='h')
plt.title('Engagement Metrics for Female Politicians')

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

# Engagement metrics
engagement_metrics = ['shares', 'likes', 'loves', 'wow', 'cares', 'sad', 'angry', 'haha', 'reactions_count', 'comments']

# Male engagement metrics
male_means = [563.22, 3884.41, 592.51, 1.64, 28.50, 52.31, 109.17, 267.15, 4935.69, 1640.10]

# Female engagement metrics
female_means = [153.86, 884.18, 168.81, 0.30, 9.63, 16.83, 47.70, 81.07, 1208.52, 523.57]

# Calculate total engagement for both male and female
total_male_engagement = sum(male_means)
total_female_engagement = sum(female_means)

# Calculate percentages
male_percentages = [100 * (mean / total_male_engagement) for mean in male_means]
female_percentages = [100 * (mean / total_female_engagement) for mean in female_means]

# Plotting
plt.figure(figsize=(12, 6))

plt.barh(engagement_metrics, male_percentages, color='blue', label='Male')
plt.barh(engagement_metrics, female_percentages, color='red', alpha=0.5, label='Female')

plt.xlabel('Percentage')
plt.title('Engagement Metrics Comparison between Male and Female Politicians (Percentage)')
plt.legend()
plt.grid(axis='x')

plt.show()

# Calculate correlation between negative comments and other features
correlation_matrix = female_data[['shares', 'likes', 'loves', 'wow', 'cares', 'sad', 'angry', 'haha', 'reactions_count', 'comments']].corr()

# Print correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Visualize correlation matrix using triangle heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Triangle Heatmap of Correlation Matrix')
plt.show()


import seaborn as sns
import numpy as np

# Calculate correlation between negative comments and other features
correlation_matrix_male = male_data[['shares', 'likes', 'loves', 'wow', 'cares', 'sad', 'angry', 'haha', 'reactions_count', 'comments']].corr()

# Print correlation matrix
print("Correlation Matrix Male:")
print(correlation_matrix_male)

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(correlation_matrix_male, dtype=bool))

# Visualize correlation matrix using triangle heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_male, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Triangle Heatmap of Correlation Matrix for Male Politicians')
plt.show()


sample_data['filtered_comments'] = sample_data['comment_text_cleaned_url'].astype(str)

# NLTK Vader Sentiment Analysis
def nltk_sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)['compound']
    return sentiment_score

# TextBlob Sentiment Analysis
def textblob_sentiment_analysis(text):
    analysis = TextBlob(text)
    sentiment_score = analysis.sentiment.polarity
    return sentiment_score

# Apply sentiment analysis to create new columns
sample_data['vader_sentiment'] = sample_data['filtered_comments'].apply(nltk_sentiment_analysis)
sample_data['textblob_sentiment'] = sample_data['filtered_comments'].apply(textblob_sentiment_analysis)

# Map sentiment scores to sentiment categories
sample_data['vader_sentiment_category'] = sample_data['vader_sentiment'].apply(lambda score: 'Positive' if score > 0 else ('Negative' if score < 0 else 'Neutral'))
sample_data['textblob_sentiment_category'] = sample_data['textblob_sentiment'].apply(lambda score: 'Positive' if score > 0 else ('Negative' if score < 0 else 'Neutral'))

# Overall sentiment for the dataset
overall_vader_sentiment = sample_data['vader_sentiment_category'].value_counts()
overall_textblob_sentiment = sample_data['textblob_sentiment_category'].value_counts()

# Print or visualize the overall sentiment for the dataset
print("Overall Vader Sentiment:")
print(overall_vader_sentiment)

print("\nOverall TextBlob Sentiment:")
print(overall_textblob_sentiment)

# Function to perform sentiment analysis using TextBlob
def textblob_sentiment(text):
    analysis = TextBlob(str(text))
    sentiment_score = analysis.sentiment.polarity
    return sentiment_score

# Function to perform sentiment analysis using NLTK's Vader
def nltk_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(str(text))['compound']
    return sentiment_score

# Apply sentiment analysis to posts and comments
female_data_orj['post_sentiment_textblob'] = female_data_orj['content'].apply(textblob_sentiment)
female_data_orj['post_sentiment_nltk'] = female_data_orj['content'].apply(nltk_sentiment)
female_data_orj['comment_sentiment_textblob'] = female_data_orj['comment_text_cleaned_url'].apply(textblob_sentiment)
female_data_orj['comment_sentiment_nltk'] = female_data_orj['comment_text_cleaned_url'].apply(nltk_sentiment)

# Calculate average sentiment scores for posts and comments
avg_post_sentiment_textblob = female_data_orj['post_sentiment_textblob'].mean()
avg_post_sentiment_nltk = female_data_orj['post_sentiment_nltk'].mean()
avg_comment_sentiment_textblob = female_data_orj['comment_sentiment_textblob'].mean()
avg_comment_sentiment_nltk = female_data_orj['comment_sentiment_nltk'].mean()

# Print the average sentiment scores
print("Average sentiment score for posts (TextBlob):", avg_post_sentiment_textblob)
print("Average sentiment score for posts (NLTK):", avg_post_sentiment_nltk)
print("Average sentiment score for comments (TextBlob):", avg_comment_sentiment_textblob)
print("Average sentiment score for comments (NLTK):", avg_comment_sentiment_nltk)

# Apply sentiment analysis to posts and comments
male_data_orj['post_sentiment_textblob'] = male_data_orj['content'].apply(textblob_sentiment)
male_data_orj['post_sentiment_nltk'] = male_data_orj['content'].apply(nltk_sentiment)
male_data_orj['comment_sentiment_textblob'] = male_data_orj['filtered_comments'].apply(textblob_sentiment)
male_data_orj['comment_sentiment_nltk'] = male_data_orj['filtered_comments'].apply(nltk_sentiment)


# Calculate average sentiment scores for posts and comments
avg_post_sentiment_textblob = male_data_orj['post_sentiment_textblob'].mean()
avg_post_sentiment_nltk = male_data_orj['post_sentiment_nltk'].mean()
avg_comment_sentiment_textblob = male_data_orj['comment_sentiment_textblob'].mean()
avg_comment_sentiment_nltk = male_data_orj['comment_sentiment_nltk'].mean()

# Print the average sentiment scores
print("Average sentiment score for posts (TextBlob):", avg_post_sentiment_textblob)
print("Average sentiment score for posts (NLTK):", avg_post_sentiment_nltk)
print("Average sentiment score for comments (TextBlob):", avg_comment_sentiment_textblob)
print("Average sentiment score for comments (NLTK):", avg_comment_sentiment_nltk)


# Convert 'Comment_Text' to string to handle any non-string values
female_data_orj['Comment_Text'] = female_data_orj['comment_text_cleaned_url'].astype(str)

# Initialize SentimentIntensityAnalyzer from NLTK
sia = SentimentIntensityAnalyzer()

# Convert 'Comment_Text' to string to handle any non-string values
male_data_orj['Comment_Text'] = male_data_orj['filtered_comments'].astype(str)

# Initialize SentimentIntensityAnalyzer from NLTK
sia = SentimentIntensityAnalyzer()

def calculate_toxicity_score(comment):
    # Check if the comment is a string
    if isinstance(comment, str):
        # Use SentimentIntensityAnalyzer to get compound sentiment score
        sentiment_score = sia.polarity_scores(comment)['compound']
        return sentiment_score
    else:
        # If comment is not a string, return NaN or handle it as needed
        return None  # Or any other handling you prefer


# Apply toxicity analysis to create a new column 'Toxicity_Score'
female_data_orj['Toxicity_Score'] = female_data_orj['comment_text_cleaned_url'].apply(calculate_toxicity_score)

# Calculate the mean toxicity score for female comments
mean_toxicity_score = female_data_orj['Toxicity_Score'].mean()

# Print the mean toxicity score
print("Average toxicity score for female comments:", mean_toxicity_score)

# Visualize the distribution of toxicity scores using a histogram
plt.figure(figsize=(10, 6))
plt.hist(female_data_orj['Toxicity_Score'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Toxicity Score')
plt.ylabel('Frequency')
plt.title('Distribution of Toxicity Scores for Female Comments')
plt.grid(True)
plt.show()


# Apply toxicity analysis to create a new column 'Toxicity_Score'
male_data_orj['Toxicity_Score'] = male_data_orj['filtered_comments'].apply(calculate_toxicity_score)

# Calculate the mean toxicity score for female comments
mean_toxicity_score = male_data_orj['Toxicity_Score'].mean()

# Print the mean toxicity score
print("Average toxicity score for male comments:", mean_toxicity_score)

# Visualize the distribution of toxicity scores using a histogram
plt.figure(figsize=(10, 6))
plt.hist(male_data_orj['Toxicity_Score'], bins=20, color='green', edgecolor='black', alpha=0.7)
plt.xlabel('Toxicity Score')
plt.ylabel('Frequency')
plt.title('Distribution of Toxicity Scores for Male Comments')
plt.grid(True)
plt.show()


# List of Politicians, their role, and gender
male_politicians_info = male_data_orj[['name', 'Responsibility', 'Gender']].drop_duplicates()
female_politicians_info = female_data_orj[['name', 'Responsibility', 'Gender']].drop_duplicates()

# Amount of posts per politician
male_posts_per_politician = male_data_orj['name'].value_counts()
female_posts_per_politician = female_data_orj['name'].value_counts()

# Amount of comments per politician
male_comments_per_politician = male_data_orj.groupby('name')['comment_text'].count()
female_comments_per_politician = female_data_orj.groupby('name')['comment_text'].count()

# Descriptive statistics for male dataset
male_data_statistics = male_data_orj.describe()

# Descriptive statistics for female dataset
female_data_statistics = female_data_orj.describe()

# Display the additional information
print("List of Politicians, their role, and their gender (Male Dataset):")
print(male_politicians_info)

print("\nAmount of Posts per Politician (Male Dataset):")
print(male_posts_per_politician)

print("\nAmount of Comments per Politician (Male Dataset):")
print(male_comments_per_politician)

print("\nDescriptive Statistics for Male Dataset:")
print(male_data_statistics)

# Repeat the same for the female dataset
print("List of Politicians, their role, and their gender (FeMale Dataset):")
print(female_politicians_info)

print("\nAmount of Posts per Politician (FeMale Dataset):")
print(male_posts_per_politician)

print("\nAmount of Comments per Politician (FeMale Dataset):")
print(female_comments_per_politician)

print("\nDescriptive Statistics for FeMale Dataset:")
print(female_data_statistics)

# Combine datasets into female and male
combined_data = pd.concat([male_data_orj, female_data_orj])

# Amount of posts and comments once combined
combined_posts_per_politician = combined_data['name'].value_counts()
combined_comments_per_politician = combined_data.groupby('name')['comment_text'].count()

# Descriptive statistics for combined dataset
combined_data_statistics = combined_data.describe()

# Display the additional information for the combined dataset
print("\nAmount of Posts per Politician (Combined Dataset):")
print(combined_posts_per_politician)

print("\nAmount of Comments per Politician (Combined Dataset):")
print(combined_comments_per_politician)

print("\nDescriptive Statistics for Combined Dataset:")
print(combined_data_statistics)


print(male_data_orj.head())
print(female_data_orj.head())

# Drop columns with all NaN values
male_data_orj = male_data_orj.dropna(axis=1, how='all')
female_data_orj = female_data_orj.dropna(axis=1, how='all')
total_rows_maledata = len(male_data_orj)
distinct_rows_maledata = male_data_orj.nunique()

# Count of all rows in male_data and female_data
print(f"Total Rows: {total_rows_maledata}")

# Count of distinct rows in male_data
print("Distinct Count for Each Column:")
print(distinct_rows_maledata)
total_rows_femaledata = len(female_data_orj)
distinct_rows_femaledata = female_data_orj.nunique()

# Count of all rows in male_data and female_data
print(f"Total Rows: {total_rows_femaledata}")

# Count of distinct rows in male_data
print("Distinct Count for Each Column:")
print(distinct_rows_femaledata)

# Remove NaN values from male and female datasets
male_data_orj.dropna(subset=['content'], inplace=True)
female_data_orj.dropna(subset=['content'], inplace=True)

# Extract content for male and female datasets
male_content = male_data_orj['content'].tolist()
female_content = female_data_orj['content'].tolist()

# Tokenize the content
male_tokens = [word_tokenize(str(content).lower()) for content in male_content]
female_tokens = [word_tokenize(str(content).lower()) for content in female_content]

# Create dictionary and corpus for male and female datasets
male_dictionary = corpora.Dictionary(male_tokens)
female_dictionary = corpora.Dictionary(female_tokens)

male_corpus = [male_dictionary.doc2bow(tokens) for tokens in male_tokens]
female_corpus = [female_dictionary.doc2bow(tokens) for tokens in female_tokens]

# Build LDA models for male and female datasets
male_lda_model = models.LdaModel(male_corpus, id2word=male_dictionary, num_topics=5, passes=10)
female_lda_model = models.LdaModel(female_corpus, id2word=female_dictionary, num_topics=5, passes=10)

# Print topics for male and female datasets
print("Male Topics:")
print(male_lda_model.print_topics())

print("\nFemale Topics:")
print(female_lda_model.print_topics())

# Separate the dataset into male and female datasets
male_data = df[df['Gender'] == 'Male']
female_data = df[df['Gender'] == 'Female']

# Extract content for male and female datasets
male_content = male_data['content'].tolist()
female_content = female_data['content'].tolist()

# Initialize BERTopic models for male and female datasets
male_model = BERTopic(language="english")
female_model = BERTopic(language="english")

# Fit the models to the content data
male_topics, male_probs = male_model.fit_transform(male_content)
female_topics, female_probs = female_model.fit_transform(female_content)

# Now you can analyze, visualize, or compare the topics and their probabilities
# For example:
print("Male Topics:", male_topics)
print("Female Topics:", female_topics)

# You can also visualize the topics
male_model.visualize_topics()
female_model.visualize_topics()

# Or compare the topic frequencies
male_topic_freq = male_model.get_topic_freq().head(5)
female_topic_freq = female_model.get_topic_freq().head(5)

print("Male Topic Frequency:\n", male_topic_freq)
print("Female Topic Frequency:\n", female_topic_freq)


# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"  # Example: "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Example usage
text = "Replace this text with your input text."
encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
output = model(**encoded_input)


# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"  # Example: "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Example usage
text = "Replace this text with your input text."
encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
output = model(**encoded_input)


# Remove commenter_name values from comment_text

# Load the English language model in spacy
nlp = spacy.load("en_core_web_sm")

# Function to remove people's names from text data
def remove_names(text):
    try:
        # Convert to string if the input is a float
        if isinstance(text, float):
            text = str(text)
        doc = nlp(text)
        return ' '.join([token.text for token in doc if token.ent_type_ != 'PERSON'])
    except Exception as e:
        print(f"An error occurred: {e}")
        return text  # Return the original value if there's an error


male_data_orj['text_column_cleaned'] = male_data_orj['comment_text'].apply(remove_names)
female_data_orj['text_column_cleaned'] = female_data_orj['comment_text'].apply(remove_names)

# Filter stop words

# Load the spaCy English language model
nlp = spacy.load('en_core_web_sm')

# Function to remove stop words using spaCy
def remove_stop_words(string):
    # Check if the value is NaN or a boolean (True/False)
    if pd.isna(string) or isinstance(string, bool):
        return ''

    # Tokenize the string into individual words
    doc = nlp(string)

    # Filter out stop words
    filtered_words = [token.text for token in doc if not token.is_stop]

    # Join the filtered words back into a string
    new_string = ' '.join(filtered_words)

    return new_string

# Randomly select 1000 records from both male and female datasets
random_sample_male = male_data_orj.sample(n=1000, random_state=42)
random_sample_female = female_data_orj.sample(n=1000, random_state=42)

# Apply the remove_stop_words function to the 'comment_text_cleaned_url' column
random_sample_male['filtered_comments'] = random_sample_male['text_column_cleaned'].apply(remove_stop_words)
random_sample_female['filtered_comments'] = random_sample_female['text_column_cleaned'].apply(remove_stop_words)

# Apply the remove_stop_words function to the entire 'text_column_cleaned' column
male_data_orj['filtered_comments'] = male_data_orj['text_column_cleaned'].apply(remove_stop_words)
female_data_orj['filtered_comments'] = female_data_orj['text_column_cleaned'].apply(remove_stop_words)

# Display the cleaned datasets
print("\nCleaned female_data:")
print(female_data_orj[['text_column_cleaned', 'filtered_comments']].head())

print("\nCleaned male_data:")
print(male_data_orj[['text_column_cleaned', 'filtered_comments']].head())

# Remove duplicate rows
male_data_orj = male_data_orj.drop_duplicates()
female_data_orj = female_data_orj.drop_duplicates()

# Remove rows with empty or unusual comments
male_data_orj = male_data_orj.dropna(subset=['filtered_comments'])  # Assuming 'comment_text' is the column with comments
female_data_orj = female_data_orj.dropna(subset=['filtered_comments'])

# Display the cleaned datasets
print("\nCleaned male_data:")
print(male_data_orj.head())
print("\nCleaned female_data:")
print(female_data_orj.head())

# Apply log transformation to text length
male_data_orj['log_comment_length'] = np.log1p(male_data_orj['filtered_comments'].str.len())
female_data_orj['log_comment_length'] = np.log1p(female_data_orj['filtered_comments'].str.len())

# Visualize the distribution of log-transformed comment_text length
plt.figure(figsize=(10, 6))
plt.hist(male_data_orj['log_comment_length'], bins=50, alpha=0.5, label='Male Data')
plt.hist(female_data_orj['log_comment_length'], bins=50, alpha=0.5, label='Female Data')
plt.title('Distribution of Log-Transformed Comment Text Length')
plt.xlabel('Log(Text Length + 1)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Exclude outliers from comment_text length
male_text_lengths = male_data_orj['filtered_comments'].str.len()
emale_text_lengths = female_data_orj['filtered_comments'].str.len()


# Define a function to exclude outliers
def exclude_outliers(lengths):
    q1 = lengths.quantile(0.25)
    q3 = lengths.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return lengths[(lengths >= lower_bound) & (lengths <= upper_bound)]

# Exclude outliers from comment_text length
male_text_lengths_filtered = exclude_outliers(male_text_lengths)
female_text_lengths_filtered = exclude_outliers(female_text_lengths)

# Visualize the distribution of comment_text length after excluding outliers
plt.figure(figsize=(10, 6))
plt.hist(male_text_lengths_filtered, bins=50, alpha=0.5, label='Male Data')
plt.hist(female_text_lengths_filtered, bins=50, alpha=0.5, label='Female Data')
plt.title('Distribution of Comment Text Length (Excluding Outliers)')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# In[49]:


# Visualize the distribution of comment_text length
plt.figure(figsize=(10, 6))
plt.hist(male_data_orj['filtered_comments'].str.len(), bins=50, alpha=0.5, label='Male Data')
plt.hist(female_data_orj['filtered_comments'].str.len(), bins=50, alpha=0.5, label='Female Data')
plt.title('Distribution of Comment Text Length')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# In[53]:

# Function to remove URLs from text data
def remove_urls(text):
    try:
        # Check if the text is not NaN and not empty
        if isinstance(text, str) and text.strip():  # Check for non-empty string
            # Use regex to remove URLs
            cleaned_text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
            return cleaned_text.strip()  # Remove leading/trailing whitespaces
        elif pd.isna(text):  # Check if the value is NaN
            return text  # Return the original NaN value
        else:
            return text  # Return the original value if it's not a string or an empty string
    except Exception as e:
        print(f"An error occurred: {e}")
        return text  # Return the original value if there's an error

# Apply the function to the specific column in the DataFrame
male_data_orj['comment_text_cleaned_url'] = male_data_orj['filtered_comments'].apply(remove_urls)

# Apply the function to the specific column in the DataFrame
female_data_orj['comment_text_cleaned_url'] = female_data_orj['filtered_comments'].apply(remove_urls)


# In[54]:
# Display the cleaned datasets
print("\nCleaned male_data:")
print(male_data_orj.head())

print("\nCleaned female_data:")
print(female_data_orj.head())


# In[55]:
print(female_data_orj.columns)
print(male_data_orj.columns)


# In[56]:

# Remove duplicate rows
male_data_orj = male_data_orj.drop_duplicates()
female_data_orj = female_data_orj.drop_duplicates()


# In[59]:

# Remove rows with empty or unusual comments
male_data_orj = male_data_orj.dropna(subset=['comment_text_cleaned_url'])  
female_data_orj = female_data_orj.dropna(subset=['comment_text_cleaned_url'])


# In[67]:

# Display the modified female dataset
print(male_data_orj['comment_text_cleaned_url'])
print(female_data_orj['comment_text_cleaned_url'])



#BASIC STATISTICS

# Amount of posts per male politician
posts_per_male_politician = male_data.groupby(['name'])['Content_Reactions_post_id'].nunique().reset_index()
posts_per_male_politician.columns = ['name', 'Posts_Count_Male']


# Amount of posts per female politician
posts_per_female_politician = female_data.groupby(['name'])['Content_Reactions_post_id'].nunique().reset_index()
posts_per_female_politician.columns = ['name', 'Posts_Count_Female']



if 'Content_Reactions_post_id' not in male_data.columns:
    print("Content_Reactions_post_id column not found. Please use the correct column name.")
else:
    # Task 2: Amount of posts per male politician
    posts_per_male_politician = male_data_orj.groupby(['name'])['Content_Reactions_post_id'].nunique().reset_index()
    posts_per_male_politician.columns = ['name', 'Posts_Count_Male']
    # Amount of posts per female politician

    # Print the result
    print(posts_per_male_politician)


   
if 'Content_Reactions_post_id' not in female_data.columns:
    print("Content_Reactions_post_id column not found. Please use the correct column name.")
else:
    # Task 2: Amount of posts per male politician
    posts_per_female_politician = female_data.groupby(['name'])['Content_Reactions_post_id'].nunique().reset_index()
    posts_per_female_politician.columns = ['name', 'Posts_Count_Female']
    # Amount of posts per female politician

    # Print the result
    print(posts_per_female_politician)


# Amount of comments per politician
comments_per_male_politician = male_data_orj.groupby(['name'])['comment_id'].nunique().reset_index()
comments_per_male_politician.columns = ['name', 'Comments_Count']

print(comments_per_male_politician)

# Amount of comments per female politician
comments_per_female_politician = female_data_orj.groupby(['name'])['comment_id'].nunique().reset_index()
comments_per_female_politician.columns = ['name', 'Comments_Count']

print(comments_per_female_politician)

# Combine datasets into female and male
male_data = male_data[male_data['Gender'] == 'Male']
female_data = female_data[female_data['Gender'] == 'Female']

# Calculate the average comment length for male_data
avg_comment_length_male = male_data['comment_text_cleaned_url'].apply(lambda x: len(x) if isinstance(x, list) else 0).mean()

# Calculate the average comment length for female_data
avg_comment_length_female = female_data['comment_text_cleaned_url'].apply(lambda x: len(x) if isinstance(x, list) else 0).mean()

# Basic statistics
num_comments_male = len(male_data)
num_comments_female = len(female_data)


# Tokenize comments and calculate comment lengths
male_data['comment_length'] = male_data['comment_text'].apply(lambda x: len(word_tokenize(str(x))))
female_data['comment_length'] = female_data['comment_text'].apply(lambda x: len(word_tokenize(str(x))))

# Calculate average comment length
avg_comment_length_male = np.mean(male_data['comment_length'])
avg_comment_length_female = np.mean(female_data['comment_length'])

# Print results
print(f"Number of comments (Male): {len(male_data)}")
print(f"Average comment length (Male): {avg_comment_length_male:.2f} words")

print(f"Number of comments (Female): {len(female_data)}")
print(f"Average comment length (Female): {avg_comment_length_female:.2f} words")


# In[85]:


# Tokenize comments and calculate comment lengths
male_data['cleaned_comment_length'] = male_data['comment_text_cleaned_url'].apply(lambda x: len(word_tokenize(str(x))))
female_data['cleaned_comment_length'] = female_data['comment_text_cleaned_url'].apply(lambda x: len(word_tokenize(str(x))))

# Calculate average comment length
avg_comment_length_cleaned_male = np.mean(male_data['cleaned_comment_length'])
avg_comment_length_cleaned_female = np.mean(female_data['cleaned_comment_length'])

# Print results
print(f"Number of comments (Cleaned_Male): {len(male_data)}")
print(f"Average comment length (Cleaned_Male): {avg_comment_length_cleaned_male:.2f} words")

print(f"Number of comments (Cleaned_Female): {len(female_data)}")
print(f"Average comment length (Cleaned_Female): {avg_comment_length_cleaned_female:.2f} words")


# In[89]:

# Word frequency analysis
word_freq_male = male_data['comment_text_cleaned_url'].str.split().explode().value_counts()
word_freq_female = female_data['comment_text_cleaned_url'].str.split().explode().value_counts()

# Exclude specific words
exclude_words = ['nt', 's','m']
word_freq_male = word_freq_male[~word_freq_male.index.isin(exclude_words)]
word_freq_female = word_freq_female[~word_freq_female.index.isin(exclude_words)]

# Plot the top N most common words for male_data before text processing
top_n = 10
plt.figure(figsize=(10, 5))
word_freq_male.head(top_n).plot(kind='bar', color='skyblue')
plt.title(f'Top {top_n} Most Common Words - Male Data (Original Text)')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.show()

# Plot the top N most common words for female_data before text processing
plt.figure(figsize=(10, 5))
word_freq_female.head(top_n).plot(kind='bar', color='pink')
plt.title(f'Top {top_n} Most Common Words - Female Data (Original Text)')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.show()


# In[90]:

# Function to generate word cloud with excluded words
def generate_wordcloud_exclude(data, title, exclude_words=[], max_words=100):
    # Exclude specific words
    data = ' '.join([word for word in data.split() if word not in exclude_words])

    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, max_words=max_words, background_color='white').generate(data)
    
    # Plot the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

# Define the words to exclude
exclude_words = ['nt', 's', 'm']

# Generate word cloud for male dataset with excluded words
generate_wordcloud_exclude(male_data_text, 'Male Dataset Word Cloud (Excluding "nt", "s", "m")', exclude_words)

# Generate word cloud for female dataset with excluded words
generate_wordcloud_exclude(female_data_text, 'Female Dataset Word Cloud (Excluding "nt", "s", "m")', exclude_words)


# SENTIMENT ANALYSIS

# In[99]:

# Convert 'comment_text' to string to handle any non-string values
male_data['comment_text_cleaned_url'] = male_data['comment_text_cleaned_url'].astype(str)
female_data['comment_text_cleaned_url'] = female_data['comment_text_cleaned_url'].astype(str)

# NLTK Sentiment Analysis
# NLTK Vader Sentiment Analysis
def nltk_sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)['compound']
    return sentiment_score

# TextBlob Sentiment Analysis
def textblob_sentiment_analysis(text):
    analysis = TextBlob(text)
    sentiment_score = analysis.sentiment.polarity
    return sentiment_score

# Apply NLTK Vader Sentiment Analysis
male_data['vader_sentiment'] = male_data['comment_text_cleaned_url'].apply(vader_analysis)
# Apply NLTK Sentiment Analysis
male_data['nltk_sentiment'] = male_data['comment_text_cleaned_url'].apply(nltk_sentiment_analysis)
# Apply TextBlob Sentiment Analysis
male_data['textblob_sentiment'] = male_data['comment_text_cleaned_url'].apply(textblob_sentiment_analysis)


# Apply sentiment analysis to create new columns
female_data['vader_sentiment'] = female_data['comment_text_cleaned_url'].apply(vader_analysis)
female_data['nltk_sentiment'] = female_data['comment_text_cleaned_url'].apply(nltk_sentiment_analysis)
female_data['textblob_sentiment'] = female_data['comment_text_cleaned_url'].apply(textblob_sentiment_analysis)

# Map sentiment scores to sentiment categories
male_data['vader_sentiment_category'] = male_data['vader_sentiment'].apply(lambda score: 'Positive' if score > 0 else ('Negative' if score < 0 else 'Neutral'))
male_data['nltk_sentiment_category'] = male_data['nltk_sentiment'].apply(lambda score: 'Positive' if score > 0 else ('Negative' if score < 0 else 'Neutral'))
male_data['textblob_sentiment_category'] = male_data['textblob_sentiment'].apply(lambda score: 'Positive' if score > 0 else ('Negative' if score < 0 else 'Neutral'))

female_data['vader_sentiment_category'] = female_data['vader_sentiment'].apply(lambda score: 'Positive' if score > 0 else ('Negative' if score < 0 else 'Neutral'))
female_data['nltk_sentiment_category'] = female_data['nltk_sentiment'].apply(lambda score: 'Positive' if score > 0 else ('Negative' if score < 0 else 'Neutral'))
female_data['textblob_sentiment_category'] = female_data['textblob_sentiment'].apply(lambda score: 'Positive' if score > 0 else ('Negative' if score < 0 else 'Neutral'))

# Compare NLTK Vader with Previous Results
nltk_counts = male_data['nltk_sentiment'].value_counts()
textblob_counts = male_data['textblob_sentiment'].value_counts()

# Overall sentiment for the male and female dataset
overall_vader_sentiment_male = male_data['vader_sentiment_category'].value_counts()
overall_nltk_sentiment_male = male_data['nltk_sentiment_category'].value_counts()
overall_textblob_sentiment_male = male_data['textblob_sentiment_category'].value_counts()

overall_vader_sentiment_female = female_data['vader_sentiment_category'].value_counts()
overall_nltk_sentiment_female = female_data['nltk_sentiment_category'].value_counts()
overall_textblob_sentiment_female = female_data['textblob_sentiment_category'].value_counts()

# Print or visualize the overall sentiment for the female dataset
print("Overall Vader Sentiment:")
print(overall_vader_sentiment_male)
print(overall_vader_sentiment_female)

print("\nOverall NLTK Sentiment:")
print(overall_nltk_sentiment_male)
print(overall_nltk_sentiment_female)

print("\nOverall TextBlob Sentiment:")
print(overall_textblob_sentiment_male)
print(overall_textblob_sentiment_female)



# Apply sentiment analysis to create new columns
male_data['nltk_sentiment_score'] = male_data['comment_text_cleaned_url'].apply(nltk_sentiment_analysis)
male_data['textblob_sentiment_score'] = male_data['comment_text_cleaned_url'].apply(textblob_sentiment_analysis)

# Map sentiment scores to sentiment categories
male_data['nltk_sentiment'] = male_data['nltk_sentiment_score'].apply(lambda score: 'Positive' if score > 0 else ('Negative' if score < 0 else 'Neutral'))
male_data['textblob_sentiment'] = male_data['textblob_sentiment_score'].apply(lambda score: 'Positive' if score > 0 else ('Negative' if score < 0 else 'Neutral'))


# Visualizations
plt.figure(figsize=(12, 6))

# NLTK Sentiment Distribution
plt.subplot(2, 2, 1)
male_data['nltk_sentiment'].value_counts().plot(kind='bar', color=['green', 'red', 'gray'])
plt.title('NLTK Sentiment Distribution')
plt.xlabel('Sentiment Category')
plt.ylabel('Frequency')

# TextBlob Sentiment Distribution
plt.subplot(2, 2, 2)
male_data['textblob_sentiment'].value_counts().plot(kind='bar', color=['green', 'red', 'gray'])
plt.title('TextBlob Sentiment Distribution')
plt.xlabel('Sentiment Category')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Exact numbers for NLTK sentiment distribution
nltk_counts = male_data['nltk_sentiment'].value_counts()

# Exact numbers for TextBlob sentiment distribution
textblob_counts = male_data['textblob_sentiment'].value_counts()

print("NLTK Sentiment Counts:")
print(nltk_counts)

print("\nTextBlob Sentiment Counts:")
print(textblob_counts)

import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# NLTK Sentiment Analysis
def nltk_sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)['compound']
    return 'Positive' if sentiment_score > 0 else ('Negative' if sentiment_score < 0 else 'Neutral')

# TextBlob Sentiment Analysis
def textblob_sentiment_analysis(text):
    analysis = TextBlob(text)
    sentiment_score = analysis.sentiment.polarity
    return 'Positive' if sentiment_score > 0 else ('Negative' if sentiment_score < 0 else 'Neutral')

# Apply sentiment analysis to create new columns
female_data['nltk_sentiment'] = female_data['comment_text_cleaned_url'].apply(nltk_sentiment_analysis)
female_data['textblob_sentiment'] = female_data['comment_text_cleaned_url'].apply(textblob_sentiment_analysis)

# Group by politician and calculate sentiment counts
politician_sentiment_counts = female_data.groupby(['name', 'nltk_sentiment', 'textblob_sentiment']).size().reset_index(name='count')

# Display the result
print(politician_sentiment_counts)


# Apply sentiment analysis to create new columns
male_data['nltk_sentiment'] = male_data['comment_text_cleaned_url'].apply(nltk_sentiment_analysis)
male_data['textblob_sentiment'] = male_data['comment_text_cleaned_url'].apply(textblob_sentiment_analysis)

# Group by politician and calculate sentiment counts
male_politician_sentiment_counts = male_data.groupby(['name', 'nltk_sentiment', 'textblob_sentiment']).size().reset_index(name='count')

# Display the result
print(male_politician_sentiment_counts)



# Set the style
sns.set(style="whitegrid")

# Create a bar plot for each politician
plt.figure(figsize=(12, 8))
sns.barplot(x="name", y="count", hue="nltk_sentiment", data=politician_sentiment_counts, palette="viridis")
plt.title('NLTK Sentiment Analysis per Politician')
plt.xlabel('Politician Name')
plt.ylabel('Comment Count')
plt.xticks(rotation=45, ha='right')
plt.show()

# Create a bar plot for each politician
plt.figure(figsize=(12, 8))
sns.barplot(x="name", y="count", hue="textblob_sentiment", data=male_politician_sentiment_counts, palette="viridis")
plt.title('TextBlob Sentiment Analysis per Politician')
plt.xlabel('Politician Name')
plt.ylabel('Comment Count')
plt.xticks(rotation=45, ha='right')
plt.show()



#  Word Frequency Calculation
def calculate_word_frequencies(data):
    vectorizer = CountVectorizer(stop_words='english')
    word_matrix = vectorizer.fit_transform(data['comment_text_cleaned_url'])
    words = vectorizer.get_feature_names_out()
    word_frequencies = word_matrix.sum(axis=0).A1
    return pd.DataFrame({'word': words, 'frequency': word_frequencies})

male_word_freq = calculate_word_frequencies(male_data)
female_word_freq = calculate_word_frequencies(female_data)

# Step 3: Calculate Differences
def calculate_chi2_test(word_freq_male, word_freq_female):
    observed = pd.concat([word_freq_male.set_index('word')['frequency'], word_freq_female.set_index('word')['frequency']], axis=1, keys=['male', 'female']).fillna(0)
    chi2, _, _, _ = chi2_contingency(observed)
    return chi2

chi2_test_statistic = calculate_chi2_test(male_word_freq, female_word_freq)

# Generate Word Clouds
def generate_word_cloud(data, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(data.set_index('word')['frequency'])
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

generate_word_cloud(male_word_freq, 'Male Politicians Word Cloud')
generate_word_cloud(female_word_freq, 'Female Politicians Word Cloud')

# Differential Word Cloud
def differential_word_cloud(data1, data2, title):
    diff_cloud_data = data1.merge(data2, on='word', suffixes=('_male', '_female'), how='outer').fillna(0)
    diff_cloud_data['difference'] = diff_cloud_data['frequency_male'] - diff_cloud_data['frequency_female']
    
    # Filter words with significant differences (you can adjust the threshold)
    significant_diff_words = diff_cloud_data[abs(diff_cloud_data['difference']) > 50]
    
    generate_word_cloud(significant_diff_words, title)

differential_word_cloud(male_word_freq, female_word_freq, 'Differential Word Cloud for Politicians')



# Map sentiment categories to numerical values
sentiment_mapping = {'Positive': 1, 'Negative': -1, 'Neutral': 0}
male_data['nltk_sentiment_score'] = male_data['nltk_sentiment'].map(sentiment_mapping)
male_data['textblob_sentiment_score'] = male_data['textblob_sentiment'].map(sentiment_mapping)
comment_text_cleaned_url
# General Correlation Analysis
general_corr = male_data[['nltk_sentiment_score', 'textblob_sentiment_score', 'shares', 'likes', 'loves', 'wow', 'cares', 'sad', 'angry', 'haha', 'reactions_count', 'comments']].corr()

# Plot General Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(general_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('General Correlation Heatmap for Male Dataset')
plt.show()



# Load stop words
nltk.download('stopwords')
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# Function to tokenize a sentence into words
def sent_to_words(sentence):
    # Additional preprocessing to handle special characters or empty comments
    if isinstance(sentence, pd.Series):
        return sentence.apply(lambda x: gensim.utils.simple_preprocess(str(x), deacc=True) if x.strip() else [])
    elif isinstance(sentence, str):
        return gensim.utils.simple_preprocess(str(sentence), deacc=True) if sentence.strip() else []
    else:
        return []

def remove_stopwords(texts):
    return [
        [word for word in simple_preprocess(str(doc)) if word not in stop_words and len(word) > 1]
        for doc in texts if doc and any(doc)  # Exclude lists with no elements
    ]

# Process male dataset
data_male = sent_to_words(male_data['comment_text_cleaned_url'])
data_words_male = list(remove_stopwords(data_male))

# Process female dataset
data_female = sent_to_words(female_data['comment_text_cleaned_url'])
data_words_female = list(remove_stopwords(data_female))

# Print first few words from both datasets for verification
print("First 10 words in male dataset:", data_words_male[:10])
print("First 10 words in female dataset:", data_words_female[:10])


# Combine both male and female datasets
all_comments = data_words_male + data_words_female

# Create a dictionary representation of the comments
id2word = corpora.Dictionary(all_comments)

# Create a corpus (bag of words) representation
corpus = [id2word.doc2bow(comment) for comment in all_comments]

# Build the LDA model
lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=5, passes=10)

# Print the topics and associated words
pprint(lda_model.print_topics())

# Apply the model to a sample comment
sample_comment = data_words_male[0]  # You can change this to any other comment
sample_bow = id2word.doc2bow(sample_comment)
print("Topic distribution for the sample comment:", lda_model[sample_bow])


# Function for toxicity analysis using Perspective API
def analyze_toxicity(comment):
    API_KEY = "AIzaSyCY9zHpkm_7yqJ0jexkFe1A1vvEEbYEdsE"
    url = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key=" + API_KEY
    data = {
        "comment": {"text": comment},
        "languages": ["en"],
        "requestedAttributes": {"TOXICITY": {}}
    }
    response = requests.post(url, json=data)
    
    try:
        # Check if the 'attributeScores' key is present in the response
        attribute_scores = response.json()["attributeScores"]
    except KeyError:
        return 0.0  # Return a default toxicity score (adjust as needed)

    # Check if the 'TOXICITY' key is present in the 'attributeScores'
    if "TOXICITY" in attribute_scores:
        toxicity_score = attribute_scores["TOXICITY"]["summaryScore"]["value"]
        return toxicity_score
    else:
        return 0.0  # Return a default toxicity score (adjust as needed)

# Filter out rows where toxicity is not None
male_data_filtered = male_data.dropna(subset=['toxicity'])

# Plot histogram for toxicity
plt.figure(figsize=(10, 6))
plt.hist(male_data_filtered['toxicity'], bins=20, color='salmon', edgecolor='black', alpha=0.7)
plt.title('Toxicity Distribution for Male Politicians Comments')
plt.xlabel('Toxicity Score')
plt.ylabel('Frequency')
plt.show()


##### SAMPLE DATA ANAYSIS

female_df = pd.read_excel('/content/sample_data/female_data.xlsx', engine='openpyxl')
# Read the male and female datasets into DataFrames

male_df = pd.read_excel('/content/sample_data/male_data.xlsx', engine='openpyxl')

male_df.columns

female_df.columns

# Drop rows with NaN values in the comment_text column
female_df = female_df.dropna(subset=["comment_text_cleaned_url"])

male_df = male_df.dropna(subset=["comment_text_cleaned_url"])

female_df.describe()

male_df.describe()

# Sample 500 rows randomly from each DataFrame
male_sample = male_df.sample(n=500, random_state=42)
female_sample = female_df.sample(n=500, random_state=42)

# Concatenate the sampled DataFrames into one
merged_df = pd.concat([male_sample, female_sample])

# Reset index
merged_df.reset_index(drop=True, inplace=True)

# Save the merged dataset to a new Excel file
merged_df.to_excel("merged_dataset.xlsx", index=False)

merged_df = pd.read_excel('/content/sample_data/merged_dataset.xlsx', engine='openpyxl')

merged_df.info()

merged_df.info()



# Extract comments and gender information for males
male_comments = merged_df[merged_df['Gender'] == 'male']['comment_text_cleaned_url'].tolist()
male_genders = ['male'] * len(male_comments)  # Generating a list of 'male' for each comment

# Extract comments and gender information for females
female_comments = merged_df[merged_df['Gender'] == 'female']['comment_text_cleaned_url'].tolist()
female_genders = ['female'] * len(female_comments)  # Generating a list of 'female' for each comment

# Extract comments and gender information for males
male_comments = merged_df[merged_df['Gender'] == 'male']['comment_text_cleaned_url'].tolist()

# Extract comments and gender information for females
female_comments = merged_df[merged_df['Gender'] == 'female']['comment_text_cleaned_url'].tolist()

# Initialize Detoxify model
model = Detoxify('unbiased')


# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"  # Example: "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Read the merged dataset into a DataFrame
#merged_df = pd.read_excel("merged_dataset.xlsx")

# Filter male comments and convert them to strings
male_comments = merged_df[merged_df['Gender'] == 'male']['comment_text_cleaned_url'].tolist()
male_comments = [str(comment) for comment in male_comments]

# Filter female comments and convert them to strings
female_comments = merged_df[merged_df['Gender'] == 'female']['comment_text_cleaned_url'].tolist()
female_comments = [str(comment) for comment in female_comments]

# Tokenize the comments using BERT tokenizer
male_tokens = [tokenizer.encode(comment, add_special_tokens=True) for comment in male_comments]
female_tokens = [tokenizer.encode(comment, add_special_tokens=True) for comment in female_comments]

import requests

API_URL = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key=YOUR_API_KEY"

def analyze_toxicity(tokens):
    toxicity_scores = []

    for token_list in tokens:
        text = ' '.join(token_list)

        payload = {
            "comment": {"text": text},
            "languages": ["en"],
            "requestedAttributes": {"TOXICITY": {}}
        }

        response = requests.post(API_URL, json=payload)
        response_json = response.json()

        toxicity_score = response_json["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
        toxicity_scores.append(toxicity_score)

    return toxicity_scores

# Analyze toxicity for male and female datasets
male_toxicity_scores = analyze_toxicity(male_tokens)
female_toxicity_scores = analyze_toxicity(female_tokens)

# Print the type and value of toxicity_scores to understand its structure
print(male_toxicity_scores)
print(female_toxicity_scores)



# Load pre-trained BERT tokenizer and model for sequence classification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

merged_df

# Assume 'comment_text' is the column containing comments and 'gender' is the column containing gender information
comments = merged_df['comment_text_cleaned_url'].tolist()
genders = merged_df['Gender'].tolist()

# Tokenize comments
tokenized_comments = tokenizer(comments, padding=True, truncation=True, return_tensors='pt')

# Perform sentiment analysis
results = []
for i in range(len(comments)):
    inputs = tokenized_comments['input_ids'][i].unsqueeze(0)
    attention_mask = tokenized_comments['attention_mask'][i].unsqueeze(0)

    # Perform inference
    outputs = model(inputs, attention_mask=attention_mask)

    # Extract predicted label (sentiment)
    _, predicted_label = torch.max(outputs.logits, dim=1)
    sentiment = 'toxic' if predicted_label == 1 else 'non-toxic'

    # Append results with comment text, gender, and sentiment
    results.append({'comment_text': comments[i], 'gender': genders[i], 'sentiment': sentiment})

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Analyze prevalence and nature of toxic comments by gender
toxic_comments_by_gender = results_df[results_df['sentiment'] == 'toxic'].groupby('gender').size()
total_comments_by_gender = results_df.groupby('gender').size()

# Calculate percentage of toxic comments by gender
toxic_comments_percentage_by_gender = (toxic_comments_by_gender / total_comments_by_gender) * 100

# Print analysis results
print("Prevalence of Toxic Comments by Gender:")
print(toxic_comments_percentage_by_gender)
print("\nNature of Toxic Comments by Gender:")
print(toxic_comments_by_gender)

# Additional analysis and visualization can be performed based on the obtained results



#Load pre-trained BERT tokenizer and model for sequence classification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')


# Assume 'comment_text' is the column containing comments and 'gender' is the column containing gender information
comments = merged_df['comment_text_cleaned_url'].tolist()
genders = merged_df['Gender'].tolist()



# Tokenize comments
tokenized_comments = tokenizer(comments, padding=True, truncation=True, return_tensors='pt')

# Perform sentiment analysis
results = []
for i in range(len(comments)):
    inputs = tokenized_comments['input_ids'][i].unsqueeze(0)
    attention_mask = tokenized_comments['attention_mask'][i].unsqueeze(0)

    # Perform inference
    outputs = model(inputs, attention_mask=attention_mask)

    # Extract predicted label (sentiment)
    _, predicted_label = torch.max(outputs.logits, dim=1)
    sentiment = 'toxic' if predicted_label == 1 else 'non-toxic'

    # Append results with comment text, gender, and sentiment
    results.append({'comment_text': comments[i], 'gender': genders[i], 'sentiment': sentiment})

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Analyze prevalence and nature of toxic comments by gender
toxic_comments_by_gender = results_df.groupby('gender')['sentiment'].value_counts(normalize=True).unstack(fill_value=0)

# Print analysis results
print("Prevalence of Toxic Comments by Gender:")
print(toxic_comments_by_gender)

# Convert float values to strings in the 'content' column
male_data['content'] = merged_df['content'].astype(str)
female_data['content'] = female_data['content'].astype(str)

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/toxic-bert-base-uncased")
model = Detoxify('google/toxic-bert-base-uncased')

# Tokenize and predict toxicity levels for male comments
male_encodings = tokenizer(male_comments, truncation=True, padding=True, return_tensors="pt")
male_toxicity_scores = model.predict(male_encodings)

# Tokenize and predict toxicity levels for female comments
female_encodings = tokenizer(female_comments, truncation=True, padding=True, return_tensors="pt")
female_toxicity_scores = model.predict(female_encodings)

# Print the type and value of male_toxicity_scores and female_toxicity_scores to understand their structure
print(type(male_toxicity_scores))
print(male_toxicity_scores)

print(type(female_toxicity_scores))
print(female_toxicity_scores)

# Predict toxicity levels for female comments
female_toxicity_scores = model.predict(female_comments)

# Print the type and value of male_toxicity_scores and female_toxicity_scores to understand their structure
print(type(male_toxicity_scores))
print(male_toxicity_scores)

print(type(female_toxicity_scores))
print(female_toxicity_scores)

# Combine comments and gender information
comments = male_comments + female_comments
genders = male_genders + female_genders

# Initialize Detoxify model
model = Detoxify('unbiased')

# Predict toxicity levels for each comment
toxicity_scores = model.predict(comments)

# Combine toxicity scores with gender information
toxicity_with_gender = list(zip(genders, toxicity_scores))

# Print the type and value of toxicity_with_gender to understand its structure
print(type(toxicity_with_gender))
print(toxicity_with_gender)

# Extract comments and gender information
comments = merged_df['comment_text_cleaned_url'].tolist()
genders = merged_df['Gender'].tolist()

# Initialize Detoxify model
model = Detoxify('unbiased')

# Predict toxicity levels for each comment
toxicity_scores = model.predict(comments)

# Combine toxicity scores with gender information
toxicity_with_gender = list(zip(genders, toxicity_scores))

# Print the type and value of toxicity_with_gender to understand its structure
print(type(toxicity_with_gender))
print(toxicity_with_gender)

pip install Detoxify

random_sample = pd.read_excel('/content/sample_data/updated_sample_dataset_with_label.xlsx', engine='openpyxl')


# Load the updated dataset with labels
updated_dataset = pd.read_excel('/content/sample_data/updated_sample_dataset_with_label.xlsx', engine='openpyxl')

# Calculate average toxicity score for each label
average_toxicity_by_label = updated_dataset.groupby('label')['toxicity_score'].mean()

# Calculate grand total average of comments
grand_total_average = updated_dataset['toxicity_score'].mean()

# Display the results
print("Average Toxicity Score by Label:")
print(average_toxicity_by_label)
print("\nGrand Total Average of Comments:")
print(grand_total_average)


updated_dataset = pd.read_excel('/content/sample_data/merged_sample_data.xlsx', engine='openpyxl')

updated_dataset.head()


# Load the updated dataset with labels
#updated_dataset = pd.read_excel('updated_biden_dataset_with_label.xlsx')

# Filter comments by label and select top 10 comments for each label
top_positive_comments = updated_dataset[updated_dataset['label'] == 'Positive'].nlargest(10, 'toxicity_score')
top_negative_comments = updated_dataset[updated_dataset['label'] == 'Negative'].nlargest(10, 'toxicity_score')
top_neutral_comments = updated_dataset[updated_dataset['label'] == 'Neutral'].nlargest(10, 'toxicity_score')

# Display top 10 comments for each label
print("Top 10 Positive Comments:")
print(top_positive_comments[['filtered_comments', 'toxicity_score']])
print("\nTop 10 Negative Comments:")
print(top_negative_comments[['filtered_comments', 'toxicity_score']])
print("\nTop 10 Neutral Comments:")
print(top_neutral_comments[['filtered_comments', 'toxicity_score']])



# Adjust pandas display settings to show full content of each comment
pd.set_option('display.max_colwidth', None)

# Load the updated dataset with labels
# updated_dataset = pd.read_excel('updated_biden_dataset_with_label.xlsx')

# Filter comments by label and select top 10 comments for each label
top_positive_comments = updated_dataset[updated_dataset['label'] == 'Positive'].nlargest(10, 'toxicity_score')
top_negative_comments = updated_dataset[updated_dataset['label'] == 'Negative'].nlargest(10, 'toxicity_score')
top_neutral_comments = updated_dataset[updated_dataset['label'] == 'Neutral'].nlargest(10, 'toxicity_score')

# Display top 10 comments for each label
print("Top 10 Positive Comments:")
print(top_positive_comments[['filtered_comments', 'toxicity_score']])
print("\nTop 10 Negative Comments:")
print(top_negative_comments[['filtered_comments', 'toxicity_score']])
print("\nTop 10 Neutral Comments:")
print(top_neutral_comments[['filtered_comments', 'toxicity_score']])



# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Load the dataset
# Assuming you have the dataset loaded as 'dataset'

# Tokenize comments to extract individual words
all_comments = ' '.join(updated_dataset['filtered_comments']).split()

# Exclude stopwords, punctuation, and lowercase all words
stop_words = set(stopwords.words('english'))
filtered_words = [word.lower() for word in all_comments if word.lower() not in stop_words and word.lower() not in punctuation and word.lower() != 'nt']

# Calculate the frequency of each word
word_counts = Counter(filtered_words)

# Create a DataFrame to store word counts and average toxicity score for each word
word_df = pd.DataFrame({'word': list(word_counts.keys()), 'frequency': list(word_counts.values())})

# Function to calculate the average toxicity score and label for each word
def calculate_word_toxicity(word):
    word_comments = updated_dataset[updated_dataset['filtered_comments'].str.contains(word)]
    average_toxicity_score = word_comments['toxicity_score'].mean()
    average_label = 'Neutral'
    if average_toxicity_score > 0.5:
        average_label = 'Negative'
    elif average_toxicity_score < 0.5:
        average_label = 'Positive'
    return average_toxicity_score, average_label

# Apply the function to each word and store the results in new columns
word_df[['average_toxicity_score', 'average_label']] = word_df['word'].apply(lambda x: pd.Series(calculate_word_toxicity(x)))

# Sort the DataFrame by frequency in descending order
word_df = word_df.sort_values(by='frequency', ascending=False)

# Separate words into positive, negative, and neutral groups based on their labels
positive_words = word_df[word_df['average_label'] == 'Positive'].head(10)
negative_words = word_df[word_df['average_label'] == 'Negative'].head(10)
neutral_words = word_df[word_df['average_label'] == 'Neutral'].head(10)

# Display the top 10 words for each label
print("Top 10 Positive Words:")
print(positive_words)

print("\nTop 10 Negative Words:")
print(negative_words)

print("\nTop 10 Neutral Words:")
print(neutral_words)



# Load the updated dataset
#updated_dataset = pd.read_excel('updated_dataset.xlsx')

# Calculate sentiment analysis scores

# Initialize SentimentIntensityAnalyzer for Vader
vader_analyzer = SentimentIntensityAnalyzer()

# Define functions for sentiment analysis using NLTK and TextBlob
def nltk_sentiment(text):
    sentiment = vader_analyzer.polarity_scores(text)
    return sentiment['compound']

def textblob_sentiment(text):
    sentiment = TextBlob(text).sentiment
    return sentiment.polarity

# Apply sentiment analysis functions to calculate sentiment scores
updated_dataset['nltk_sentiment_score'] = updated_dataset['filtered_comments'].apply(nltk_sentiment)
updated_dataset['textblob_sentiment_score'] = updated_dataset['filtered_comments'].apply(textblob_sentiment)

# Assign sentiment categories
def assign_sentiment_category(score):
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Apply the function to assign sentiment categories
updated_dataset['vader_sentiment_category'] = updated_dataset['vader_sentiment'].apply(assign_sentiment_category)
updated_dataset['nltk_sentiment_category'] = updated_dataset['nltk_sentiment_score'].apply(assign_sentiment_category)
updated_dataset['textblob_sentiment_category'] = updated_dataset['textblob_sentiment_score'].apply(assign_sentiment_category)

# Analyze and compare toxicity scores
# You can use descriptive statistics or visualization techniques to compare toxicity scores
toxicity_stats = updated_dataset.groupby(['vader_sentiment_category', 'nltk_sentiment_category', 'textblob_sentiment_category']).agg({'toxicity_score': ['mean', 'median', 'std']})

# Display the comparison results
print(toxicity_stats)

# Load the updated dataset with labels
# Assuming 'gender' column exists in the dataset
# updated_dataset = pd.read_excel('updated_dataset_with_label.xlsx')

# Calculate average toxicity score for each label and gender
average_toxicity_by_label_gender = updated_dataset.groupby(['label', 'gender'])['toxicity_score'].mean()

# Calculate grand total average of comments for each gender
grand_total_average_by_gender = updated_dataset.groupby('gender')['toxicity_score'].mean()

# Display the results
print("Average Toxicity Score by Label and Gender:")
print(average_toxicity_by_label_gender)
print("\nGrand Total Average of Comments by Gender:")
print(grand_total_average_by_gender)



# Load the dataset
sample_data = pd.read_excel('/content/sample_data/sample1.xlsx', engine='openpyxl')

sample_data.head(3)

from detoxify import Detoxify

# Initialize Detoxify model
model = Detoxify('unbiased')

# Extract comments from the 'filtered_comments' column
comments = sample_data['comment_text_latest'].astype(str).tolist()

# Extract comments from the 'filtered_comments' column and convert to a list of strings
comments = sample_data['comment_text_latest'].astype(str).tolist()

# Predict toxicity levels for each comment
toxicity_scores = model.predict(comments)

# Predict toxicity levels for each comment
toxicity_scores = model.predict(comments)

# Extract toxicity scores from the dictionary
toxicity_values = toxicity_scores['toxicity']

# Classify the comments based on toxicity scores
labels = []
for score in toxicity_values:
    if score > 0.5:
        labels.append('Negative')
    elif score < 0.2:
        labels.append('Positive')
    else:
        labels.append('Neutral')

# Add 'label' column to the dataset
sample_data['label'] = labels

# Add toxicity scores as a new column in the dataset
sample_data['toxicity_score'] = toxicity_values

# Save the updated dataset
sample_data.to_excel('updated_sample_dataset_with_label.xlsx', index=False)

sample_data.head()

# Calculate average toxicity score for each label
average_toxicity_by_label = sample_data.groupby('label')['toxicity_score'].mean()

# Calculate grand total average of comments
grand_total_average = sample_data['toxicity_score'].mean()

# Display the results
print("Average Toxicity Score by Label:")
print(average_toxicity_by_label)
print("\nGrand Total Average of Comments:")
print(grand_total_average)


# Adjust pandas display settings to show full content of each comment
pd.set_option('display.max_colwidth', None)

# Load the updated dataset with labels
# updated_dataset = pd.read_excel('updated_biden_dataset_with_label.xlsx')

# Filter comments by label and select top 10 comments for each label
top_positive_comments = sample_data[sample_data['label'] == 'Positive'].nlargest(10, 'toxicity_score')
top_negative_comments = sample_data[sample_data['label'] == 'Negative'].nlargest(10, 'toxicity_score')
top_neutral_comments = sample_data[sample_data['label'] == 'Neutral'].nlargest(10, 'toxicity_score')

# Display top 10 comments for each label
print("Top 10 Positive Comments:")
print(top_positive_comments[['filtered_comments', 'toxicity_score']])
print("\nTop 10 Negative Comments:")
print(top_negative_comments[['filtered_comments', 'toxicity_score']])
print("\nTop 10 Neutral Comments:")
print(top_neutral_comments[['filtered_comments', 'toxicity_score']])



# Tokenize comments to extract individual words
# Convert float values to strings and then join the strings
all_comments = ' '.join(str(comment) for comment in sample_data['comment_text_latest'] if not pd.isnull(comment)).split()


# Exclude stopwords, punctuation, and lowercase all words
stop_words = set(stopwords.words('english'))
filtered_words = [word.lower() for word in all_comments if word.lower() not in stop_words and word.lower() not in punctuation and word.lower() != 'nt']

# Calculate the frequency of each word
word_counts = Counter(filtered_words)

# Create a DataFrame to store word counts and average toxicity score for each word
word_df = pd.DataFrame({'word': list(word_counts.keys()), 'frequency': list(word_counts.values())})

# Function to calculate the average toxicity score and label for each word
def calculate_word_toxicity(word):
    word_comments = sample_data[sample_data['filtered_comments'].str.contains(word)]
    average_toxicity_score = word_comments['toxicity_score'].mean()
    average_label = 'Neutral'
    if average_toxicity_score > 0.5:
        average_label = 'Negative'
    elif average_toxicity_score < 0.5:
        average_label = 'Positive'
    return average_toxicity_score, average_label

# Apply the function to each word and store the results in new columns
word_df[['average_toxicity_score', 'average_label']] = word_df['word'].apply(lambda x: pd.Series(calculate_word_toxicity(x)))

# Sort the DataFrame by frequency in descending order
word_df = word_df.sort_values(by='frequency', ascending=False)

# Separate words into positive, negative, and neutral groups based on their labels
positive_words = word_df[word_df['average_label'] == 'Positive'].head(10)
negative_words = word_df[word_df['average_label'] == 'Negative'].head(10)
neutral_words = word_df[word_df['average_label'] == 'Neutral'].head(10)

# Display the top 10 words for each label
print("Top 10 Positive Words:")
print(positive_words)

print("\nTop 10 Negative Words:")
print(negative_words)

print("\nTop 10 Neutral Words:")
print(neutral_words)



# Initialize SentimentIntensityAnalyzer for Vader
vader_analyzer = SentimentIntensityAnalyzer()

# Define functions for sentiment analysis using NLTK and TextBlob
def nltk_sentiment(text):
    sentiment = vader_analyzer.polarity_scores(text)
    return sentiment['compound']

def textblob_sentiment(text):
    sentiment = TextBlob(text).sentiment
    return sentiment.polarity

# Apply sentiment analysis functions to calculate sentiment scores
# Filter out float values and apply sentiment analysis function
sample_data['nltk_sentiment_score'] = sample_data['comment_text_latest'].apply(lambda x: nltk_sentiment(x) if isinstance(x, str) else None)
sample_data['textblob_sentiment_score'] = sample_data['comment_text_latest'].apply(lambda x: textblob_sentiment(x) if isinstance(x, str) else None)

# Assign sentiment categories
def assign_sentiment_category(score):
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Apply the function to assign sentiment categories
sample_data['vader_sentiment_category'] = sample_data['vader_sentiment'].apply(assign_sentiment_category)
sample_data['nltk_sentiment_category'] = sample_data['nltk_sentiment_score'].apply(assign_sentiment_category)
sample_data['textblob_sentiment_category'] = sample_data['textblob_sentiment_score'].apply(assign_sentiment_category)

# Analyze and compare toxicity scores
# You can use descriptive statistics or visualization techniques to compare toxicity scores
toxicity_stats = sample_data.groupby(['vader_sentiment_category', 'nltk_sentiment_category', 'textblob_sentiment_category']).agg({'toxicity_score': ['mean', 'median', 'std']})

# Display the comparison results
print(toxicity_stats)



# Load the updated dataset with labels
# Assuming 'gender' column exists in the dataset
# updated_dataset = pd.read_excel('updated_dataset_with_label.xlsx')

# Calculate average toxicity score for each label and gender
average_toxicity_by_label_gender = sample_data.groupby(['label', 'Gender'])['toxicity_score'].mean()

# Calculate grand total average of comments for each gender
grand_total_average_by_gender = sample_data.groupby('Gender')['toxicity_score'].mean()

# Display the results
print("Average Toxicity Score by Label and Gender:")
print(average_toxicity_by_label_gender)
print("\nGrand Total Average of Comments by Gender:")
print(grand_total_average_by_gender)

### Perspective API

# Function for toxicity analysis using Perspective API
def analyze_toxicity(comment):
    API_KEY = "AIzaSyCY9zHpkm_7yqJ0jexkFe1A1vvEEbYEdsE"
    url = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key=" + API_KEY
    data = {
        "comment": {"text": comment},
        "languages": ["en"],
        "requestedAttributes": {"TOXICITY": {}}
    }
    response = requests.post(url, json=data)

    try:
        # Check if the 'attributeScores' key is present in the response
        attribute_scores = response.json()["attributeScores"]
    except KeyError:
        return 0.0  # Return a default toxicity score (adjust as needed)

    # Check if the 'TOXICITY' key is present in the 'attributeScores'
    if "TOXICITY" in attribute_scores:
        toxicity_score = attribute_scores["TOXICITY"]["summaryScore"]["value"]
        return toxicity_score
    else:
        return 0.0  # Return a default toxicity score (adjust as needed)

import numpy as np
import pandas as pd

# Load the dataset
# sample_data = pd.read_excel('/content/dataset_with_sentiment.xlsx', engine='openpyxl')

# Replace out-of-range float values with NaN
sample_data['comment_text_latest'] = sample_data['comment_text_latest'].replace([np.inf, -np.inf], np.nan)

# Drop rows with NaN values in the 'comment_text_latest' column
sample_data = sample_data.dropna(subset=['comment_text_latest'])


sample_data['nltk_sentiment_score'] = sample_data['comment_text_latest'].apply(lambda x: nltk_sentiment(x) if isinstance(x, str) else None)
sample_data['textblob_sentiment_score'] = sample_data['comment_text_latest'].apply(lambda x: textblob_sentiment(x) if isinstance(x, str) else None)

# Apply the analyze_toxicity function to each comment in the dataset
perspective_toxicity_scores = []
for comment in sample_data['comment_text_latest']:
    toxicity_score = analyze_toxicity(comment)
    perspective_toxicity_scores.append(toxicity_score)

# Add the perspective toxicity scores as a new column in the updated dataset
sample_data['perspective_toxicity_score'] = perspective_toxicity_scores

# Save the updated dataset with Perspective toxicity scores
sample_data.to_excel('updated_dataset_with_perspective_toxicity.xlsx', index=False)

sample_data.head()

female_df = pd.read_excel('/content/sample_data/female_data.xlsx', engine='openpyxl')

# Extract columns containing toxicity scores for each method
vader_toxicity_scores = sample_data['toxicity_score']  # Assuming 'toxicity_score' is the column name for VADER
textblob_toxicity_scores = sample_data['textblob_sentiment_score']  # Assuming 'textblob_sentiment_score' is the column name for TextBlob
nltk_toxicity_scores = sample_data['nltk_sentiment_score']  # Assuming 'nltk_sentiment_score' is the column name for NLTK

# Calculate mean toxicity scores for each method
mean_vader = vader_toxicity_scores.mean()
mean_textblob = textblob_toxicity_scores.mean()
mean_nltk = nltk_toxicity_scores.mean()

# Calculate median toxicity scores for each method
median_vader = vader_toxicity_scores.median()
median_textblob = textblob_toxicity_scores.median()
median_nltk = nltk_toxicity_scores.median()

# Calculate standard deviation of toxicity scores for each method
std_vader = vader_toxicity_scores.std()
std_textblob = textblob_toxicity_scores.std()
std_nltk = nltk_toxicity_scores.std()

# Print the statistical measures
print("VADER - Mean: {}, Median: {}, Std: {}".format(mean_vader, median_vader, std_vader))
print("TextBlob - Mean: {}, Median: {}, Std: {}".format(mean_textblob, median_textblob, std_textblob))
print("NLTK - Mean: {}, Median: {}, Std: {}".format(mean_nltk, median_nltk, std_nltk))

# Visualize the distribution of toxicity scores for each method
plt.figure(figsize=(10, 6))
plt.hist(vader_toxicity_scores, bins=20, alpha=0.5, label='VADER')
plt.hist(textblob_toxicity_scores, bins=20, alpha=0.5, label='TextBlob')
plt.hist(nltk_toxicity_scores, bins=20, alpha=0.5, label='NLTK')
plt.title('Distribution of Toxicity Scores')
plt.xlabel('Toxicity Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()

sample_data.columns


# Encode the 'Gender' column
label_encoder = LabelEncoder()
sample_data['Gender_encoded'] = label_encoder.fit_transform(sample_data['Gender'])

# Define a function to perform topic modeling using Latent Dirichlet Allocation (LDA)
def perform_topic_modeling(comments):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    term_matrix = vectorizer.fit_transform(comments)

    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(term_matrix)

    return lda, vectorizer

# Apply topic modeling on comments
lda_model, vectorizer = perform_topic_modeling(sample_data['comment_text_latest'])

# Display the topics
def display_topics(model, vectorizer, n_words=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print(", ".join([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-n_words - 1:-1]]))

display_topics(lda_model, vectorizer)

# Analyze sentiment across different topics
sample_data['topic'] = lda_model.transform(vectorizer.transform(sample_data['comment_text_latest'])).argmax(axis=1)
sentiment_by_topic = sample_data.groupby('topic')['sentiment_score'].mean()

# Display sentiment scores by topic
print("Sentiment by Topic:")
print(sentiment_by_topic)

# Analyze sentiment across different genders
sentiment_by_gender = sample_data.groupby('Gender')['sentiment_score'].mean()

# Display sentiment scores by gender
print("Sentiment by Gender:")
print(sentiment_by_gender)


# Encode the 'Gender' column
label_encoder = LabelEncoder()
sample_data['Gender_encoded'] = label_encoder.fit_transform(sample_data['Gender'])

# Define a function to perform topic modeling using Latent Dirichlet Allocation (LDA)
def perform_topic_modeling(comments):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    term_matrix = vectorizer.fit_transform(comments)

    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(term_matrix)

    return lda, vectorizer

# Apply topic modeling on comments
lda_model, vectorizer = perform_topic_modeling(sample_data['comment_text_latest'])

# Display the topics
def display_topics(model, vectorizer, n_words=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print(", ".join([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-n_words - 1:-1]]))

# Excluding 'nt' from Topics
def display_topics_excluding_nt(model, vectorizer, n_words=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print(", ".join([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-n_words - 1:-1] if vectorizer.get_feature_names_out()[i] != 'nt']))

display_topics_excluding_nt(lda_model, vectorizer)

# Analyze sentiment across different topics
sample_data['topic'] = lda_model.transform(vectorizer.transform(sample_data['comment_text_latest'])).argmax(axis=1)
sentiment_by_topic = sample_data.groupby('topic')['sentiment_score'].mean()

# Display sentiment scores by topic
print("Sentiment by Topic:")
print(sentiment_by_topic)

# Analyze sentiment across different genders
sentiment_by_gender = sample_data.groupby('Gender')['sentiment_score'].mean()

# Display sentiment scores by gender
print("Sentiment by Gender:")
print(sentiment_by_gender)

# Compare sentiment scores across genders and topics
sentiment_by_gender_topic = sample_data.groupby(['Gender', 'topic'])['sentiment_score'].mean()
print("Sentiment by Gender and Topic:")
print(sentiment_by_gender_topic)

# Association with Engagement Metrics
engagement_metrics = ['likes', 'shares', 'comments']  # Add other engagement metrics if available
engagement_correlation = sample_data[engagement_metrics + ['sentiment_score']].corr(method='pearson')

# Display correlation matrix
print("Correlation between Sentiment Score and Engagement Metrics:")
print(engagement_correlation)


# Preprocessing function
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens

# Function to extract topics/themes
def extract_topics(text):
    # Perform Part-of-Speech tagging
    doc = nlp(text)
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]

    return nouns

# Apply preprocessing and topic extraction to each comment
sample_data['comment_preprocessed'] = sample_data['comment_text_latest'].apply(preprocess_text)
sample_data['topics'] = sample_data['comment_text_latest'].apply(extract_topics)

# Gender-specific topic analysis
male_topics = []
female_topics = []

for index, row in sample_data.iterrows():
    if row['Gender'] == "male":
        male_topics.extend(row['topics'])
    elif row['Gender'] == "female":
        female_topics.extend(row['topics'])

# Counting occurrences of topics
male_topic_counts = Counter(male_topics)
female_topic_counts = Counter(female_topics)

# Printing topic disparities
print("Disparities in topics discussed in toxic comments directed towards male and female politicians:")
print("Male Politician Topics:", male_topic_counts)
print("Female Politician Topics:", female_topic_counts)

# Apply sentiment analysis and toxicity categorization to each comment
sample_data.loc[:, 'sentiment_score'] = sample_data['comment_text_latest'].apply(lambda x: sia.polarity_scores(x)['compound'])
sample_data.loc[:, 'toxicity_category'] = sample_data['comment_text_latest'].apply(categorize_toxicity)



# Function for toxicity analysis using Perspective API
def analyze_toxicity(comment):
    API_KEY = "AIzaSyCY9zHpkm_7yqJ0jexkFe1A1vvEEbYEdsE"
    url = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key=" + API_KEY
    data = {
        "comment": {"text": comment},
        "languages": ["en"],
        "requestedAttributes": {"TOXICITY": {}}
    }
    response = requests.post(url, json=data)

    try:
        # Check if the 'attributeScores' key is present in the response
        attribute_scores = response.json()["attributeScores"]
    except KeyError:
        return 0.0  # Return a default toxicity score (adjust as needed)

    # Check if the 'TOXICITY' key is present in the 'attributeScores'
    if "TOXICITY" in attribute_scores:
        toxicity_score = attribute_scores["TOXICITY"]["summaryScore"]["value"]
        return toxicity_score
    else:
        return 0.0  # Return a default toxicity score (adjust as needed)



# Define the columns containing toxicity and sentiment scores
perspective_col = 'perspective_toxicity_score'
bert_col = 'bert_sentiment_score'
detoxify_col = 'detoxify_toxicity_score'

# Visual Comparison
plt.figure(figsize=(10, 6))
sns.histplot(data[perspective_col], color='blue', kde=True, label='Perspective')
sns.histplot(data[bert_col], color='orange', kde=True, label='BERT')
sns.histplot(data[detoxify_col], color='green', kde=True, label='Detoxify')
plt.title('Comparison of Toxicity and Sentiment Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Statistical Analysis
summary_stats = data[[perspective_col, bert_col, detoxify_col]].describe()
print(summary_stats)

# Correlation Analysis
correlation_matrix = data[[perspective_col, bert_col, detoxify_col]].corr()
print(correlation_matrix)

data.describe()



# Separate comments directed at male and female politicians
male_comments = data[data['Gender'] == 'Male']['comment_text_cleaned_url']
female_comments = data[data['Gender'] == 'Female']['comment_text_cleaned_url']

# Function to calculate sentiment scores using TextBlob
def get_textblob_sentiment(text):
    analysis = TextBlob(str(text))
    return analysis.sentiment.polarity

# Function to calculate sentiment scores using VADER
def get_vader_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(str(text))
    return scores['compound']

# Calculate average sentiment scores for comments directed at male politicians
male_comments_textblob_sentiment = male_comments.apply(get_textblob_sentiment)
male_comments_vader_sentiment = male_comments.apply(get_vader_sentiment)
male_avg_textblob_sentiment = male_comments_textblob_sentiment.mean()
male_avg_vader_sentiment = male_comments_vader_sentiment.mean()

# Calculate average sentiment scores for comments directed at female politicians
female_comments_textblob_sentiment = female_comments.apply(get_textblob_sentiment)
female_comments_vader_sentiment = female_comments.apply(get_vader_sentiment)
female_avg_textblob_sentiment = female_comments_textblob_sentiment.mean()
female_avg_vader_sentiment = female_comments_vader_sentiment.mean()

# Compare the average sentiment scores between male and female politicians
print("Average sentiment scores for comments directed at male politicians:")
print(f"TextBlob: {male_avg_textblob_sentiment}")
print(f"VADER: {male_avg_vader_sentiment}")

print("\nAverage sentiment scores for comments directed at female politicians:")
print(f"TextBlob: {female_avg_textblob_sentiment}")
print(f"VADER: {female_avg_vader_sentiment}")



# Sample stop words (replace this with your actual stop words list)
stop_words = ['nt', 'like', 'good', 'people', 'know', 'president']

# Separate comments directed at male and female politicians
c = data[data['Gender'] == 'Male']['comment_text_cleaned_url']
female_comments = data[data['Gender'] == 'Female']['comment_text_cleaned_url']

# Function to calculate sentiment scores using TextBlob
def get_textblob_sentiment(text):
    analysis = TextBlob(str(text))
    return analysis.sentiment.polarity

# Function to calculate sentiment scores using VADER
def get_vader_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(str(text))
    return scores['compound']

# Calculate sentiment scores for comments directed at male politicians
male_sentiments_textblob = male_comments.apply(get_textblob_sentiment)

# Calculate sentiment scores for comments directed at female politicians
female_sentiments_textblob = female_comments.apply(get_textblob_sentiment)

# Apply LDA to identify topics for comments directed at male politicians
vectorizer_male = CountVectorizer(max_features=1000, stop_words=stop_words)
X_male = vectorizer_male.fit_transform(male_comments.astype(str))
lda_male = LatentDirichletAllocation(n_components=5, random_state=42)
lda_male.fit(X_male)

# Apply LDA to identify topics for comments directed at female politicians
vectorizer_female = CountVectorizer(max_features=1000, stop_words=stop_words)
X_female = vectorizer_female.fit_transform(female_comments.astype(str))
lda_female = LatentDirichletAllocation(n_components=5, random_state=42)
lda_female.fit(X_female)

# Get the feature names from the vectorizer
feature_names_male = vectorizer_male.get_feature_names_out()
feature_names_female = vectorizer_female.get_feature_names_out()

# Function to get top words for each topic
def get_top_words(model, feature_names, n_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        topics[topic_idx] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
    return topics

# Get top words for each topic for male politicians
male_topics = get_top_words(lda_male, feature_names_male, 10)

# Get top words for each topic for female politicians
female_topics = get_top_words(lda_female, feature_names_female, 10)

# Associate sentiment scores with discussion topics for male politicians
male_sentiments_topics_textblob = pd.DataFrame({'Sentiment_TextBlob': male_sentiments_textblob, 'Topic': lda_male.transform(X_male).argmax(axis=1)})

# Associate sentiment scores with discussion topics for female politicians
female_sentiments_topics_textblob = pd.DataFrame({'Sentiment_TextBlob': female_sentiments_textblob, 'Topic': lda_female.transform(X_female).argmax(axis=1)})

# Analyze the distribution of sentiment scores across different topics
male_sentiments_avg_textblob = male_sentiments_topics_textblob.groupby('Topic')['Sentiment_TextBlob'].mean()
female_sentiments_avg_textblob = female_sentiments_topics_textblob.groupby('Topic')['Sentiment_TextBlob'].mean()

# Print average sentiment scores and associated topics for each topic
print("Average sentiment scores and associated topics for comments directed at male politicians (TextBlob):")
for topic, sentiment_score in male_sentiments_avg_textblob.items():
    print(f"Topic {topic}: {sentiment_score}")
    print(f"Top words: {', '.join(male_topics[topic])}")
    print()

print("\nAverage sentiment scores and associated topics for comments directed at female politicians (TextBlob):")
for topic, sentiment_score in female_sentiments_avg_textblob.items():
    print(f"Topic {topic}: {sentiment_score}")
    print(f"Top words: {', '.join(female_topics[topic])}")
    print()

## Average sentiment scores and associated topics for comments directed at male and female politicians (TextBlob):

# Sample stop words (replace this with your actual stop words list)
stop_words = ['nt', 'like', 'good', 'people', 'know', 'president']

# Separate comments directed at male and female politicians
male_comments = data[data['Gender'] == 'Male']['comment_text_cleaned_url']
female_comments = data[data['Gender'] == 'Female']['comment_text_cleaned_url']

# Function to calculate sentiment scores using TextBlob
def get_textblob_sentiment(text):
    analysis = TextBlob(str(text))
    return analysis.sentiment.polarity

# Function to calculate sentiment scores using VADER
def get_vader_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(str(text))
    return scores['compound']

# Calculate sentiment scores for comments directed at male politicians
male_sentiments_textblob = male_comments.apply(get_textblob_sentiment)

# Calculate sentiment scores for comments directed at female politicians
female_sentiments_textblob = female_comments.apply(get_textblob_sentiment)

# Apply LDA to identify topics for comments directed at male politicians
vectorizer_male = CountVectorizer(max_features=1000, stop_words=stop_words)
X_male = vectorizer_male.fit_transform(male_comments.astype(str))
lda_male = LatentDirichletAllocation(n_components=5, random_state=42)
lda_male.fit(X_male)

# Apply LDA to identify topics for comments directed at female politicians
vectorizer_female = CountVectorizer(max_features=1000, stop_words=stop_words)
X_female = vectorizer_female.fit_transform(female_comments.astype(str))
lda_female = LatentDirichletAllocation(n_components=5, random_state=42)
lda_female.fit(X_female)

# Get the feature names from the vectorizer
feature_names_male = vectorizer_male.get_feature_names_out()
feature_names_female = vectorizer_female.get_feature_names_out()

# Function to get top words for each topic
def get_top_words(model, feature_names, n_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        word_counts = Counter({feature_names[i]: topic[i] for i in topic.argsort()[:-n_top_words - 1:-1]})
        words = [word for word, _ in word_counts.most_common(5)]  # Extract only the words
        topics[topic_idx+1] = ", ".join(words)
    return topics

# Define descriptive labels for topics
topic_labels = {
    1: "Politics and Governance",
    2: "Social Issues and Welfare",
    3: "Economic Policies and Trade",
    4: "Healthcare and Environment",
    5: "Foreign Affairs and Security"
}

# Get top words for each topic for male politicians
male_topics = get_top_words(lda_male, feature_names_male, 5)

# Get top words for each topic for female politicians
female_topics = get_top_words(lda_female, feature_names_female, 5)

# Associate sentiment scores with discussion topics for male politicians
male_sentiments_topics_textblob = pd.DataFrame({'Sentiment_TextBlob': male_sentiments_textblob, 'Topic': lda_male.transform(X_male).argmax(axis=1)+1})

# Associate sentiment scores with discussion topics for female politicians
female_sentiments_topics_textblob = pd.DataFrame({'Sentiment_TextBlob': female_sentiments_textblob, 'Topic': lda_female.transform(X_female).argmax(axis=1)+1})

# Analyze the distribution of sentiment scores across different topics
male_sentiments_avg_textblob = male_sentiments_topics_textblob.groupby('Topic')['Sentiment_TextBlob'].mean()
female_sentiments_avg_textblob = female_sentiments_topics_textblob.groupby('Topic')['Sentiment_TextBlob'].mean()

# Print average sentiment scores and associated topics for each topic
print("Average sentiment scores and associated topics for comments directed at male politicians (TextBlob):")
for topic, sentiment_score in male_sentiments_avg_textblob.items():
    print(f"{topic_labels[topic]}:")
    print(f"Average Sentiment: {sentiment_score}")
    print(f"Top words: {male_topics[topic]}")
    print()

print("\nAverage sentiment scores and associated topics for comments directed at female politicians (TextBlob):")
for topic, sentiment_score in female_sentiments_avg_textblob.items():
    print(f"{topic_labels[topic]}:")
    print(f"Average Sentiment: {sentiment_score}")
    print(f"Top words: {female_topics[topic]}")
    print()




data.columns

# Calculate average toxicity scores for male and female politicians

male_toxicity_scores_perspective = data[data['Gender'] == 'Male']['perspective_toxicity_score']
female_toxicity_scores_perspective = data[data['Gender'] == 'Female']['perspective_toxicity_score']

male_toxicity_scores_bert = data[data['Gender'] == 'Male']['bert_sentiment_score']
female_toxicity_scores_bert = data[data['Gender'] == 'Female']['bert_sentiment_score']

male_toxicity_scores_detoxify = data[data['Gender'] == 'Male']['detoxify_toxicity_score']
female_toxicity_scores_detoxify = data[data['Gender'] == 'Female']['detoxify_toxicity_score']

# Compare average toxicity scores between male and female politicians
print("Average Toxicity Scores by Gender:")
print("Perspective Toxicity - Male:", male_toxicity_scores_perspective)
print("Perspective Toxicity - Female:", female_toxicity_scores_perspective)
print("BERT Toxicity - Male:", male_toxicity_scores_bert)
print("BERT Toxicity - Female:", female_toxicity_scores_bert)
print("Detoxify Toxicity - Male:", male_toxicity_scores_detoxify)
print("Detoxify Toxicity - Female:", female_toxicity_scores_detoxify)

import pandas as pd

# Read the original dataset
data = pd.read_excel('original_dataset.xlsx')

# Add the toxicity score columns to the original dataset
data['Perspective_Toxicity_Male'] = male_toxicity_scores_perspective.reset_index(drop=True)
data['Perspective_Toxicity_Female'] = female_toxicity_scores_perspective.reset_index(drop=True)
data['BERT_Toxicity_Male'] = male_toxicity_scores_bert.reset_index(drop=True)
data['BERT_Toxicity_Female'] = female_toxicity_scores_bert.reset_index(drop=True)
data['Detoxify_Toxicity_Male'] = male_toxicity_scores_detoxify.reset_index(drop=True)
data['Detoxify_Toxicity_Female'] = female_toxicity_scores_detoxify.reset_index(drop=True)

# Write the updated DataFrame back to the Excel file, overwriting the original file
data.to_excel('test.xlsx', index=False)

import pandas as pd

# Store the results in a dictionary
results = {
    'Perspective_Toxicity_Male': male_toxicity_scores_perspective,
    'Perspective_Toxicity_Female': female_toxicity_scores_perspective,
    'BERT_Toxicity_Male': male_toxicity_scores_bert,
    'BERT_Toxicity_Female': female_toxicity_scores_bert,
    'Detoxify_Toxicity_Male': male_toxicity_scores_detoxify,
    'Detoxify_Toxicity_Female': female_toxicity_scores_detoxify
}

# Convert the dictionary to a DataFrame
df_results = pd.DataFrame(results)

# Write the DataFrame to an Excel file
df_results.to_excel('toxicity_scores1.xlsx', index=False)

# Calculate average toxicity scores for male and female politicians
male_toxicity_scores_perspective = male_data['perspective_toxicity_score'].mean()
female_toxicity_scores_perspective = female_data['perspective_toxicity_score'].mean()

male_toxicity_scores_bert = male_data['bert_sentiment_score'].mean()
female_toxicity_scores_bert = female_data['bert_sentiment_score'].mean()

male_toxicity_scores_detoxify = male_data['detoxify_toxicity_score'].mean()
female_toxicity_scores_detoxify = female_data['detoxify_toxicity_score'].mean()

# Compare average toxicity scores between male and female politicians
print("Average Toxicity Scores by Gender:")
print("Perspective Toxicity - Male:", male_toxicity_scores_perspective)
print("Perspective Toxicity - Female:", female_toxicity_scores_perspective)
print("BERT Toxicity - Male:", male_toxicity_scores_bert)
print("BERT Toxicity - Female:", female_toxicity_scores_bert)
print("Detoxify Toxicity - Male:", male_toxicity_scores_detoxify)
print("Detoxify Toxicity - Female:", female_toxicity_scores_detoxify)

#Average Toxicity Scores by Gender:
## Perspective Toxicity - Male: 0.0387638241051
## Perspective Toxicity - Female: 0.020848230610899996
## BERT Toxicity - Male: 0.5613226505517955
## BERT Toxicity - Female: 0.5583383863568302
## Detoxify Toxicity - Male: 0.13562367167464004
## Detoxify Toxicity - Female: 0.13985555562394

# Calculate sentiment scores for male and female politicians
male_avg_sentiment_textblob = male_data['textblob_sentiment'].mean()
female_avg_sentiment_textblob = female_data['textblob_sentiment'].mean()

male_avg_sentiment_vader = male_data['vader_sentiment'].mean()
female_avg_sentiment_vader = female_data['vader_sentiment'].mean()

# Compare sentiment scores between male and female politicians
print("Average Sentiment Scores by Gender:")
print("TextBlob Sentiment - Male:", male_avg_sentiment_textblob)
print("TextBlob Sentiment - Female:", female_avg_sentiment_textblob)
print("VADER Sentiment - Male:", male_avg_sentiment_vader)
print("VADER Sentiment - Female:", female_avg_sentiment_vader)

## Average Sentiment Scores by Gender:
## TextBlob Sentiment - Male: 0.04071051541868741
## TextBlob Sentiment - Female: 0.029332115318543888
## VADER Sentiment - Male: 0.056
## VADER Sentiment - Female: 0.021754199999999998

# Add the toxicity score columns to the original dataset
data['Perspective_Toxicity_Male'] = male_toxicity_scores_perspective.reset_index(drop=True)
data['Perspective_Toxicity_Female'] = female_toxicity_scores_perspective.reset_index(drop=True)
data['BERT_Toxicity_Male'] = male_toxicity_scores_bert.reset_index(drop=True)
data['BERT_Toxicity_Female'] = female_toxicity_scores_bert.reset_index(drop=True)
data['Detoxify_Toxicity_Male'] = male_toxicity_scores_detoxify.reset_index(drop=True)
data['Detoxify_Toxicity_Female'] = female_toxicity_scores_detoxify.reset_index(drop=True)

# Write the updated DataFrame back to the Excel file, overwriting the original file
data.to_excel('original_dataset.xlsx', index=False)

data.columns

data.head()




# Separate comments directed at male and female politicians
male_comments = comments_data[comments_data['Gender'] == 'Male']['comment_text_cleaned_url']
female_comments = comments_data[comments_data['Gender'] == 'Female']['comment_text_cleaned_url']

# Function to calculate sentiment scores using TextBlob
def get_textblob_sentiment(text):
    analysis = TextBlob(str(text))
    return analysis.sentiment.polarity

# Function to calculate sentiment scores using VADER
def get_vader_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(str(text))
    return scores['compound']

# Calculate average sentiment scores for comments directed at male politicians
male_comments_textblob_sentiment = male_comments.apply(get_textblob_sentiment)
male_comments_vader_sentiment = male_comments.apply(get_vader_sentiment)
male_avg_textblob_sentiment = male_comments_textblob_sentiment.mean()
male_avg_vader_sentiment = male_comments_vader_sentiment.mean()

# Calculate average sentiment scores for comments directed at female politicians
female_comments_textblob_sentiment = female_comments.apply(get_textblob_sentiment)
female_comments_vader_sentiment = female_comments.apply(get_vader_sentiment)
female_avg_textblob_sentiment = female_comments_textblob_sentiment.mean()
female_avg_vader_sentiment = female_comments_vader_sentiment.mean()

# Compare the average sentiment scores between male and female politicians
print("Average sentiment scores for comments directed at male politicians:")
print(f"TextBlob: {male_avg_textblob_sentiment}")
print(f"VADER: {male_avg_vader_sentiment}")

print("\nAverage sentiment scores for comments directed at female politicians:")
print(f"TextBlob: {female_avg_textblob_sentiment}")
print(f"VADER: {female_avg_vader_sentiment}")



# Sample data (replace this with your actual data)
data = {
    'Gender': ['Male', 'Female'],
    'TextBlob_Sentiment': [0.0407, 0.0293],
    'VADER_Sentiment': [0.056, 0.0218]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(10, 6))

# TextBlob Sentiment
plt.subplot(1, 2, 1)
plt.bar(df['Gender'], df['TextBlob_Sentiment'], color='skyblue')
plt.title('Average Sentiment Scores (TextBlob)')
plt.ylabel('Average Sentiment Score')
plt.ylim(0, 0.1)

# VADER Sentiment
plt.subplot(1, 2, 2)
plt.bar(df['Gender'], df['VADER_Sentiment'], color='salmon')
plt.title('Average Sentiment Scores (VADER)')
plt.ylabel('Average Sentiment Score')
plt.ylim(0, 0.1)

plt.tight_layout()
plt.show()

