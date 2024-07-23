import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

def train_collaborative_filtering(data):
    # Create a pivot table
    user_product_matrix = data.pivot(index='reviewerID', columns='asin', values='overall').fillna(0)
    
    # Decompose the matrix
    U, sigma, Vt = svds(user_product_matrix, k=50)
    sigma = np.diag(sigma)
    
    # Reconstruct the matrix
    predicted_ratings = np.dot(np.dot(U, sigma), Vt)
    
    # Convert back to DataFrame
    preds_df = pd.DataFrame(predicted_ratings, columns=user_product_matrix.columns, index=user_product_matrix.index)
    
    return preds_df

if __name__ == "__main__":
    data = pd.read_csv('data/processed_data.csv')
    preds_df = train_collaborative_filtering(data)
    preds_df.to_csv('data/collaborative_filtering_predictions.csv')
