# File: main.py

import os
import pandas as pd
from datetime import datetime
import json
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

from analysis.data_processing import DataProcessor
from analysis.behavior_analyzer import BehaviorAnalyzer
from analysis.recommendation_system import RecommendationSystem
from analysis.visualizer import Visualizer

class RedditDatingAnalysis:
    def __init__(self):
        """Initialize all components."""
        try:
            # Create necessary directories
            Path("data/raw").mkdir(parents=True, exist_ok=True)
            Path("data/processed").mkdir(parents=True, exist_ok=True)
            Path("data/visualizations").mkdir(parents=True, exist_ok=True)
            
            # Initialize components
            self.data_processor = DataProcessor()
            self.behavior_analyzer = BehaviorAnalyzer()
            self.recommendation_system = RecommendationSystem()
            self.visualizer = Visualizer()
            
            # Initialize data holders
            self.processed_data = None
            self.behavior_data = None
            self.features = None
            
            print("System initialized successfully!")
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def load_latest_data(self):
        """Load the most recent data file."""
        try:
            raw_dir = 'data/raw'
            data_files = [f for f in os.listdir(raw_dir) if f.startswith('dating_posts_')]
            
            if not data_files:
                raise FileNotFoundError("No data files found. Please run data_collection.py first.")
            
            latest_file = max(data_files)
            filepath = os.path.join(raw_dir, latest_file)
            
            print(f"Loading data from {filepath}")
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def analyze_data(self):
        """Run the complete analysis pipeline."""
        try:
            # Load and process data
            print("Loading and processing data...")
            raw_data = self.load_latest_data()
            self.processed_data = self.data_processor.preprocess_data(raw_data)
            
            # Analyze behavioral patterns
            print("\nAnalyzing behavioral patterns...")
            self.behavior_data = self.behavior_analyzer.analyze_user_behavior(self.processed_data)
            
            # Prepare features for recommendation
            print("Preparing recommendation features...")
            self.features = self.recommendation_system.prepare_features(self.processed_data)
            
            # Fit recommendation model
            print("Training recommendation model...")
            self.recommendation_system.fit(self.features)
            
            # Create visualizations
            print("Generating visualization dashboard...")
            self.visualizer.create_behavior_dashboard(
                self.behavior_data,
                'data/visualizations/behavior_dashboard.png'
            )
            
            print("\nAnalysis pipeline completed!")
            print(f"Total users analyzed: {len(self.behavior_data)}")
            return True
        except Exception as e:
            print(f"\nError during analysis: {str(e)}")
            return False

    def run_interactive(self):
        """Run the interactive recommendation interface."""
        if not self.analyze_data():
            print("Error in analysis pipeline. Exiting...")
            return
            
        while True:
            try:
                print("\n=== Reddit Dating Pattern Matching System ===")
                print("1. Find matches based on your preferences")
                print("q. Quit")
                
                choice = input("\nEnter your choice: ").lower()
                
                if choice == 'q':
                    break
                    
                if choice == '1':
                    # Get user preferences
                    preferences = self.get_user_preferences()
                    
                    # Find matches
                    print("\nFinding your best matches...")
                    matches = self.find_matches(preferences)
                    
                    # Print results
                    self.print_matches(matches)
                    
                input("\nPress Enter to continue...")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error processing request: {str(e)}")

    def get_user_preferences(self):
        """Get user preferences through interactive input."""
        preferences = {}
        
        print("\n=== Tell us your preferences ===")
        
        # Communication style preference
        print("\nWhat communication style do you prefer?")
        print("1. Positive and enthusiastic")
        print("2. Balanced and neutral")
        print("3. Direct and critical")
        style_choice = input("Enter your choice (1-3): ")
        preferences['communication_style'] = {
            '1': 'positive',
            '2': 'neutral',
            '3': 'critical'
        }.get(style_choice, 'neutral')

        # Response time preference
        print("\nPreferred response time:")
        print("1. Quick (within 1 hour)")
        print("2. Moderate (within few hours)")
        print("3. Relaxed (within a day)")
        time_choice = input("Enter your choice (1-3): ")
        preferences['response_time'] = {
            '1': 1,
            '2': 6,
            '3': 24
        }.get(time_choice, 6)

        # Engagement level preference
        print("\nPreferred engagement level:")
        print("1. High (frequent, detailed interactions)")
        print("2. Medium (balanced interaction)")
        print("3. Low (minimal interaction)")
        engagement_choice = input("Enter your choice (1-3): ")
        preferences['engagement_level'] = {
            '1': 'high',
            '2': 'medium',
            '3': 'low'
        }.get(engagement_choice, 'medium')

        return preferences

    def find_matches(self, preferences, n_recommendations=5):
        """Find matches based on user preferences."""
        matches = []
        
        for user in self.behavior_data.index:
            user_data = self.behavior_data.loc[user]
            score = self.calculate_match_score(preferences, user_data)
            
            matches.append({
                'user': user,
                'score': score,
                'profile': self.behavior_analyzer.get_user_profile(user, self.behavior_data)
            })
        
        # Sort by score and get top matches
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:n_recommendations]

    def calculate_match_score(self, preferences, user_data):
        """Calculate match score based on preferences."""
        score = 0
        weights = {
            'communication_style': 35,  # 35%
            'response_time': 25,        # 25%
            'engagement_level': 25,     # 25%
            'activity': 15             # 15%
        }
        
        try:
            # Communication style match (35%)
            user_style = self.behavior_analyzer._get_communication_style(user_data)
            if user_style == preferences['communication_style']:
                score += weights['communication_style']
            elif (user_style == 'neutral' and preferences['communication_style'] != 'critical') or \
                (preferences['communication_style'] == 'neutral' and user_style != 'critical'):
                # Partial match for neutral styles
                score += weights['communication_style'] * 0.5

            # Response time match (25%)
            response_time_diff = abs(user_data['avg_response_time'] - preferences['response_time'])
            time_score = 0
            if response_time_diff <= 1:  # Within 1 hour
                time_score = 1.0
            elif response_time_diff <= 3:  # Within 3 hours
                time_score = 0.8
            elif response_time_diff <= 6:  # Within 6 hours
                time_score = 0.6
            elif response_time_diff <= 12:  # Within 12 hours
                time_score = 0.4
            else:
                time_score = max(0, 1 - (response_time_diff / 24))
            score += weights['response_time'] * time_score

            # Engagement level match (25%)
            if user_data['engagement_level'] == preferences['engagement_level']:
                score += weights['engagement_level']
            elif abs(
                {'low': 0, 'medium': 1, 'high': 2}[user_data['engagement_level']] - 
                {'low': 0, 'medium': 1, 'high': 2}[preferences['engagement_level']]
            ) == 1:
                # Partial match for adjacent engagement levels
                score += weights['engagement_level'] * 0.5

            # Activity pattern match (15%)
            user_msg_freq = user_data['message_frequency']
            expected_freq = {'low': 1, 'medium': 3, 'high': 5}[preferences['engagement_level']]
            activity_diff = abs(user_msg_freq - expected_freq)
            activity_score = max(0, 1 - (activity_diff / 5))
            score += weights['activity'] * activity_score

            # Additional penalties/bonuses
            if user_data['avg_response_time'] > 24 and preferences['response_time'] < 6:
                score *= 0.8  # 20% penalty for very slow responders when quick response is preferred
            if user_data['sentiment_mean'] < -0.5 and preferences['communication_style'] == 'positive':
                score *= 0.9  # 10% penalty for very negative sentiment when positive style is preferred

            return min(100, max(0, score))  # Ensure score is between 0 and 100

        except Exception as e:
            print(f"Warning: Error calculating match score: {str(e)}")
            return 0  # Return 0 if there's an error


    def print_matches(self, matches):
        """Print matches in a formatted way without numerical scores."""
        print("\n=== Your Best Matches ===")
        
        for i, match in enumerate(matches, 1):
            print(f"\nMatch {i}:")
            print(f"User: {match['user']}")
            
            profile = match['profile']
            
            # Show compatibility factors
            print("\nCompatibility Factors:")
            compatibility_factors = []
            
            # Communication style compatibility
            if profile['communication_style'] == 'Positive':
                compatibility_factors.append("Positive communicator")
            elif profile['communication_style'] == 'Neutral':
                compatibility_factors.append("Balanced communicator")
                
            # Response time compatibility
            response_time = float(profile['messaging_patterns']['response_time'].split()[0])
            if response_time < 1:
                compatibility_factors.append("Quick responder")
            elif response_time < 3:
                compatibility_factors.append("Timely responder")
                
            # Engagement compatibility
            if profile['engagement_profile']['engagement_level'] == 'high':
                compatibility_factors.append("Highly engaged")
            elif profile['engagement_profile']['engagement_level'] == 'medium':
                compatibility_factors.append("Consistently engaged")
                
            # Activity pattern
            weekend_activity = float(profile['activity_pattern']['weekend_activity'].strip('%'))
            if weekend_activity > 50:
                compatibility_factors.append("Active on weekends")
            
            # Print compatibility factors
            for factor in compatibility_factors:
                print(f"✓ {factor}")
            
            # Print detailed profile
            print("\nDetailed Profile:")
            print("Communication Style:")
            print(f"- Style: {profile['communication_style']}")
            
            print("\nMessaging Patterns:")
            for key, value in profile['messaging_patterns'].items():
                print(f"- {key.replace('_', ' ').title()}: {value}")
            
            print("\nEngagement Level:")
            print(f"- Level: {profile['engagement_profile']['engagement_level']}")
            
            print("\nActivity Pattern:")
            for key, value in profile['activity_pattern'].items():
                print(f"- {key.replace('_', ' ').title()}: {value}")
            
            print("-" * 50)

if __name__ == "__main__":
    try:
        print("Starting Reddit Dating Pattern Analysis...")
        start_time = datetime.now()
        
        analyzer = RedditDatingAnalysis()
        analyzer.run_interactive()
        
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()
        print(f"\nTotal runtime: {runtime:.1f} seconds")
        
    except Exception as e:
        print(f"\nCritical error: {str(e)}")
    finally:
        print("\nAnalysis complete.")