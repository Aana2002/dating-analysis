# File: analysis/recommendation_system.py

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

class RecommendationSystem:
    def __init__(self):
        self.scaler = StandardScaler()
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        self.count_vec = CountVectorizer(max_features=500, stop_words='english')
        self.knn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
        
    def prepare_features(self, df):
        """Prepare features for recommendation"""
        # Behavioral features
        behavioral_features = df[[
            'avg_response_time',
            'total_comments',
            'comment_sentiment_mean',
            'avg_words_per_comment'
        ]].fillna(0)
        
        # Text features
        text_features = self.tfidf.fit_transform(df['clean_text'].fillna(''))
        
        # Combine features
        combined_features = np.hstack([
            self.scaler.fit_transform(behavioral_features),
            text_features.toarray()
        ])
        
        return combined_features
    
    def fit(self, features):
        """Fit the KNN model"""
        self.knn_model.fit(features)
        return self
    
    def get_recommendations(self, user_features, n_recommendations=3):
        """Get recommendations for a user"""
        distances, indices = self.knn_model.kneighbors(
            user_features.reshape(1, -1),
            n_neighbors=n_recommendations + 1
        )
        return indices[0][1:], distances[0][1:]