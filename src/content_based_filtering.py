import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def train_content_based_filtering(data):
    # Fit TF-IDF on the review text
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['reviewText'])
    
    # Compute cosine similarity
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    return cosine_sim

if __name__ == "__main__":
    data = pd.read_csv('data/processed_data.csv')
    cosine_sim = train_content_based_filtering(data)
    # Save the cosine similarity matrix
    np.save('data/content_based_filtering_similarity.npy', cosine_sim)
