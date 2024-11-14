# File: analysis/behavior_analyzer.py

import pandas as pd
import numpy as np

class BehaviorAnalyzer:
    def analyze_user_behavior(self, df):
        """Analyze user behavioral patterns"""
        behavior_metrics = {}
        
        for user in df['author'].unique():
            user_posts = df[df['author'] == user]
            behavior_metrics[user] = {
                'avg_response_time': user_posts['avg_response_time'].fillna(0).mean(),
                'message_frequency': len(user_posts),
                'avg_message_length': user_posts['avg_words_per_comment'].fillna(0).mean(),
                'sentiment_mean': user_posts['comment_sentiment_mean'].fillna(0).mean(),
                'engagement_level': user_posts['engagement_level'].mode().iloc[0] if not user_posts['engagement_level'].empty else 'medium',
                'active_hours': user_posts['hour_posted'].nunique(),
                'weekend_activity': user_posts['is_weekend'].mean()
            }
        
        return pd.DataFrame.from_dict(behavior_metrics, orient='index')

    def get_user_profile(self, user, behavior_data):
        """Get detailed profile for a user"""
        if user not in behavior_data.index:
            return {
                'messaging_patterns': {'response_time': '0 hours', 'message_frequency': 0},
                'engagement_profile': {'engagement_level': 'medium'},
                'communication_style': 'neutral',
                'activity_pattern': {'active_hours': 0, 'weekend_activity': '0%'}
            }
        
        profile = behavior_data.loc[user]
        return {
            'messaging_patterns': {
                'response_time': f"{profile['avg_response_time']:.1f} hours",
                'message_frequency': int(profile['message_frequency'])
            },
            'engagement_profile': {
                'engagement_level': profile['engagement_level']
            },
            'communication_style': self._get_communication_style(profile),
            'activity_pattern': {
                'active_hours': int(profile['active_hours']),
                'weekend_activity': f"{profile['weekend_activity']*100:.1f}%"
            }
        }

    def _get_communication_style(self, profile):
        if profile['sentiment_mean'] > 0.2:
            return 'Positive'
        elif profile['sentiment_mean'] < -0.2:
            return 'Critical'
        return 'Neutral'