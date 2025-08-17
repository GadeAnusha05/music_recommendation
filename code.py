# ------------------------------------------------
# 📌 Importing Dependencies
# ------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ------------------------------------------------
# 📌 Data Collection (Manual Download Method)
# ------------------------------------------------
# Load dataset directly after manual download from Kaggle
# Make sure you extracted the ZIP and update the path
df = pd.read_csv(r"C:\Users\gadea\Downloads\archive\spotify_millsongdata.csv")  # <-- change path if needed

print("✅ Dataset Loaded Successfully")
print("Shape:", df.shape)
print(df.head())

# ------------------------------------------------
# 📌 Initial Understanding
# ------------------------------------------------
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Top artists
top_artists = df['artist'].value_counts().head(10)
print("\nTop 10 Artists:")
print(top_artists)

# Sample 10,000 for experimentation (saves memory)
df = df.sample(10000)
df = df.drop('link', axis=1).reset_index(drop=True)

print("\nSampled Dataset Shape:", df.shape)
print(df.head())

# ------------------------------------------------
# 📌 WordCloud for Lyrics
# ------------------------------------------------
all_lyrics = " ".join(df['text'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_lyrics)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Most Common Words in Lyrics")
plt.show()

# ------------------------------------------------
# 📌 Data Preprocessing
# ------------------------------------------------
# download nltk data (only needed first time)
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Apply preprocessing to lyrics
df['cleaned_text'] = df['text'].apply(preprocess_text)
print("\n✅ Preprocessing Done. Sample:")
print(df[['song', 'cleaned_text']].head())

# ------------------------------------------------
# 📌 Vectorization with TF-IDF
# ------------------------------------------------
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_text'])

# Compute Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ------------------------------------------------
# 📌 Recommendation Function
# ------------------------------------------------
def recommend_songs(song_name, cosine_sim=cosine_sim, df=df, top_n=5):
    # Find the index of the song
    idx = df[df['song'].str.lower() == song_name.lower()].index
    if len(idx) == 0:
        return "❌ Song not found in the dataset!"
    idx = idx[0]

    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]

    # Get song indices
    song_indices = [i[0] for i in sim_scores]

    # Return top n similar songs
    return df[['artist', 'song']].iloc[song_indices]

# ------------------------------------------------
# 📌 Example Recommendation
# ------------------------------------------------
print("\n🎶 Recommendations for the song 'For The First Time':")
recommendations = recommend_songs("For The First Time")
print(recommendations)
