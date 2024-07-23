from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from hybrid_model import generate_recommendations

app = Flask(__name__)

data = pd.read_csv('data/processed_data.csv')
collaborative_preds = pd.read_csv('data/collaborative_filtering_predictions.csv', index_col=0)
content_sim = np.load('data/content_based_filtering_similarity.npy')

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    recommendations = generate_recommendations(user_id, data, collaborative_preds, content_sim)
    return jsonify(recommendations.to_dict())

if __name__ == "__main__":
    app.run(debug=True)
