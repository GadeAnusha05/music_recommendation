# music_recommendation
🎵 Music Recommendation App
📌 About the Project

This project is a Music Recommendation System built using Python. It recommends songs based on the similarity of lyrics. The system uses Natural Language Processing (NLP) techniques to process lyrics and recommend relevant tracks.

🛠️ Libraries Used

pandas → for data handling

numpy → for numerical operations

scikit-learn → for TF-IDF, cosine similarity, and model building

nltk → for text preprocessing (stopwords, tokenization)

🤖 Algorithms Used

TF-IDF (Term Frequency – Inverse Document Frequency): Converts lyrics into numerical vectors.

Cosine Similarity: Measures similarity between two songs based on lyrics.

Content-Based Filtering: Recommendations are generated based on similarity of lyrics with input.

🚀 Steps in the Project

Load dataset (artist, song, lyrics).

Clean and preprocess the lyrics (remove stopwords, tokenize, lowercase).

Convert lyrics into TF-IDF vectors.

Compute similarity using cosine similarity.

Recommend top-N songs similar to the given song.

🎯 Example Output

Shows dataset info (artists, songs, lyrics).

Prints top artists.

Generates recommendations based on input.
