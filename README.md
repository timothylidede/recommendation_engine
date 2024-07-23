## Clone the repository:

git clone https://github.com/yourusername/recommendation_engine.git<br>
cd recommendation_engine<br>

## Create a virtual environment and install dependencies:

python3 -m venv venv<br>
source venv/bin/activate<br>
pip install -r requirements.txt<br>

## Data Preprocessing: Preprocess the data by running:

python src/data_preprocessing.py<br>

## Model Training: Train the collaborative filtering and content-based filtering models:

python src/collaborative_filtering.py<br>
python src/content_based_filtering.py<br>
python src/hybrid_model.py<br>

## Start the API: Run the Flask API to serve recommendations:

python src/api.py<br>

## Get Recommendations: Make a request to the API to get personalized product recommendations:

curl http://localhost:5000/recommend?user_id=<user_id><br>

## Evaluation scripts are included to monitor model performance and make necessary improvements.