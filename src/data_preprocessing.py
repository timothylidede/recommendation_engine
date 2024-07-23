import pandas as pd

def preprocess_data(filepath):
    data = pd.read_csv(filepath)
    
    # Drop rows with missing values in key columns
    data.dropna(subset=['reviewerID', 'asin', 'reviewText'], inplace=True)
    
    # Remove duplicate reviews
    data.drop_duplicates(subset=['reviewerID', 'asin'], inplace=True)
    
    # Convert ratings to integers
    data['overall'] = data['overall'].astype(int)
    
    # Keep only relevant columns for recommendation
    data = data[['reviewerID', 'asin', 'overall', 'reviewText']]
    
    return data

if __name__ == "__main__":
    data = preprocess_data('data/amazon_reviews.csv')
    data.to_csv('data/processed_data.csv', index=False)
