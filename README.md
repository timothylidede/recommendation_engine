## Clone the repository:

git clone https://github.com/yourusername/recommendation_engine.git
cd recommendation_engine

## Create a virtual environment and install dependencies:

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

## Data Preprocessing: Preprocess the data by running:

python src/data_preprocessing.py

## Model Training: Train the collaborative filtering and content-based filtering models:

python src/collaborative_filtering.py
python src/content_based_filtering.py
python src/hybrid_model.py

## Start the API: Run the Flask API to serve recommendations:

python src/api.py

## Get Recommendations: Make a request to the API to get personalized product recommendations:

curl http://localhost:5000/recommend?user_id=<user_id>

## Evaluation scripts are included to monitor model performance and make necessary improvements.