import pandas as pd
import numpy as np

def generate_recommendations(user_id, data, collaborative_preds, content_sim):
    # Collaborative filtering recommendations
    user_collab_preds = collaborative_preds.loc[user_id].sort_values(ascending=False)
    
    # Get the indices of the user's interactions
    user_interactions = data[data['reviewerID'] == user_id]
    user_interactions_indices = [data.index[data['asin'] == asin].tolist()[0] for asin in user_interactions['asin']]
    
    # Content-based recommendations (average of similar items)
    content_recs = content_sim[user_interactions_indices].mean(axis=0)
    
    # Combine the two
    hybrid_recs = user_collab_preds * 0.5 + content_recs * 0.5
    hybrid_recs = hybrid_recs.sort_values(ascending=False)
    
    return hybrid_recs.head(10)

if __name__ == "__main__":
    data = pd.read_csv('data/processed_data.csv')
    collaborative_preds = pd.read_csv('data/collaborative_filtering_predictions.csv', index_col=0)
    content_sim = np.load('data/content_based_filtering_similarity.npy')
    
    user_id = 'A3SGXH7AUHU8GW'  # Example user ID
    recommendations = generate_recommendations(user_id, data, collaborative_preds, content_sim)
    print(recommendations)
