# File: analysis/data_processing.py

import pandas as pd
import numpy as np
from datetime import datetime
from textblob import TextBlob
import re
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class DataProcessor:
    def __init__(self):
        self.processed_data = None
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        
    def _clean_text(self, text):
        """Clean text data."""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        return text.strip()
        
    def preprocess_data(self, df):
        """Main preprocessing function."""
        try:
            print("\nStarting data preprocessing...")
            processed_df = df.copy()
            
            # Basic cleaning
            processed_df['clean_title'] = processed_df['title'].apply(self._clean_text)
            processed_df['clean_text'] = processed_df['text'].apply(self._clean_text)
            
            # Time features
            processed_df['created_utc'] = pd.to_datetime(processed_df['created_utc'])
            processed_df = self._extract_time_features(processed_df)
            
            # Text features
            processed_df = self._extract_text_features(processed_df)
            
            # Comment processing
            processed_df['processed_comments'] = processed_df['comments'].apply(self._process_comments)
            comment_features = processed_df['processed_comments'].apply(self._extract_comment_features)
            comment_features_df = pd.DataFrame(comment_features.tolist())
            processed_df = pd.concat([processed_df, comment_features_df], axis=1)
            
            # Engagement and patterns
            processed_df = self._calculate_engagement_levels(processed_df)
            processed_df = self._extract_user_patterns(processed_df)
            processed_df = self._extract_communication_features(processed_df)
            
            # Clean up
            columns_to_drop = ['comments', 'processed_comments']
            processed_df = processed_df.drop(columns=columns_to_drop)
            
            print("Data preprocessing completed!")
            self.processed_data = processed_df
            return processed_df
            
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            raise
            
    def _extract_time_features(self, df):
        """Extract time-based features."""
        df['hour_posted'] = df['created_utc'].dt.hour
        df['day_of_week'] = df['created_utc'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['time_of_day'] = df['hour_posted'].apply(self._categorize_time_of_day)
        return df
        
    def _categorize_time_of_day(self, hour):
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 22:
            return 'evening'
        else:
            return 'night'
            
    def _extract_text_features(self, df):
        """Extract text features."""
        df['word_count'] = df['clean_text'].apply(lambda x: len(str(x).split()))
        df['sentence_count'] = df['clean_text'].apply(lambda x: len(TextBlob(str(x)).sentences))
        df['avg_word_length'] = df['clean_text'].apply(self._average_word_length)
        
        # Sentiment features
        sentiment_features = df['clean_text'].apply(self._get_detailed_sentiment)
        df = pd.concat([df, pd.DataFrame(sentiment_features.tolist())], axis=1)
        return df
        
    def _average_word_length(self, text):
        words = str(text).split()
        return np.mean([len(word) for word in words]) if words else 0
        
    def _get_detailed_sentiment(self, text):
        blob = TextBlob(str(text))
        return {
            'sentiment_polarity': blob.sentiment.polarity,
            'sentiment_subjectivity': blob.sentiment.subjectivity
        }
        
    def _process_comments(self, comments):
        """Process comments."""
        processed_comments = []
        for comment in comments:
            try:
                processed_comment = {
                    'author': str(comment['author']),
                    'text': self._clean_text(comment['text']),
                    'score': comment['score'],
                    'created_utc': pd.to_datetime(comment['created_utc']),
                    'word_count': len(str(comment['text']).split()),
                    'sentiment': TextBlob(str(comment['text'])).sentiment.polarity
                }
                processed_comments.append(processed_comment)
            except Exception as e:
                continue
        return processed_comments
        
    def _extract_comment_features(self, comments):
        """Extract comment features."""
        if not comments:
            return self._get_default_comment_features()
            
        comment_lengths = [len(c['text']) for c in comments]
        word_counts = [c['word_count'] for c in comments]
        sentiments = [c['sentiment'] for c in comments]
        response_times = self._calculate_response_times(comments)
        
        return {
            'avg_comment_length': np.mean(comment_lengths),
            'total_comments': len(comments),
            'avg_words_per_comment': np.mean(word_counts),
            'comment_sentiment_mean': np.mean(sentiments),
            'avg_response_time': np.mean(response_times) if response_times else 0
        }
        
    def _get_default_comment_features(self):
        return {
            'avg_comment_length': 0,
            'total_comments': 0,
            'avg_words_per_comment': 0,
            'comment_sentiment_mean': 0,
            'avg_response_time': 0
        }
        
    def _calculate_response_times(self, comments):
        """Calculate response times."""
        times = sorted([c['created_utc'] for c in comments])
        return [(times[i] - times[i-1]).total_seconds() / 3600 
                for i in range(1, len(times))]
                
    def _calculate_engagement_levels(self, df):
        """Calculate engagement levels."""
        try:
            engagement_scores = (
                (df['total_comments'].fillna(0) / 10) +
                (1 / (1 + df['avg_response_time'].fillna(24))) +
                ((df['comment_sentiment_mean'].fillna(0) + 1) / 2)
            ) / 3
            
            df['engagement_level'] = pd.cut(
                engagement_scores,
                bins=[-float('inf'), 0.3, 0.7, float('inf')],
                labels=['low', 'medium', 'high']
            )
            return df
        except Exception as e:
            print(f"Error in engagement calculation: {str(e)}")
            df['engagement_level'] = 'medium'
            return df
            
    def _extract_user_patterns(self, df):
        """Extract user patterns."""
        try:
            author_patterns = df.groupby('author').agg({
                'total_comments': 'mean',
                'avg_response_time': 'mean',
                'comment_sentiment_mean': 'mean'
            }).reset_index()
            
            # Handle missing values
            author_patterns = author_patterns.fillna({
                'avg_response_time': author_patterns['avg_response_time'].mean(),
                'comment_sentiment_mean': 0
            })
            
            # Calculate response categories
            author_patterns['user_response_category'] = 'moderate'
            if len(author_patterns) >= 3:
                author_patterns['user_response_category'] = pd.qcut(
                    author_patterns['avg_response_time'],
                    q=3,
                    labels=['quick', 'moderate', 'slow']
                )
                
            return df.merge(
                author_patterns[['author', 'user_response_category']],
                on='author',
                how='left'
            )
        except Exception as e:
            print(f"Error in pattern extraction: {str(e)}")
            df['user_response_category'] = 'moderate'
            return df
            
    def _extract_communication_features(self, df):
        """Extract communication features."""
        df['communication_style'] = df['sentiment_polarity'].apply(
            lambda x: 'positive' if x > 0.2 else ('critical' if x < -0.2 else 'neutral')
        )
        return df