# File: data_collection.py
import praw
import pandas as pd
from datetime import datetime, timedelta
import json
import time
import os
from config.credentials import REDDIT_CREDENTIALS

class RedditDataCollector:
    def __init__(self):
        print("Initializing Reddit API connection...")
        self.reddit = praw.Reddit(
            client_id=REDDIT_CREDENTIALS['client_id'],
            client_secret=REDDIT_CREDENTIALS['client_secret'],
            user_agent=REDDIT_CREDENTIALS['user_agent']
        )
        
    def collect_data(self, subreddit='dating', num_posts=200, comments_per_post=10):
        """Collect posts and comments from the last three month"""
        print(f"\nCollecting {num_posts} posts from r/{subreddit}...")
        
        posts_data = []
        subreddit = self.reddit.subreddit(subreddit)
        
        # Calculate three month ago
        one_month_ago = datetime.utcnow() - timedelta(days=90)
        
        try:
            for post in subreddit.new(limit=num_posts*2):
                post_time = datetime.fromtimestamp(post.created_utc)
                
                if post_time < one_month_ago:
                    continue
                    
                if len(posts_data) >= num_posts:
                    break
                
                print(f"\nCollecting post {len(posts_data)+1}/{num_posts}")
                
                # Get comments
                post.comments.replace_more(limit=0)
                comments = []
                
                for comment in list(post.comments)[:comments_per_post]:
                    try:
                        comments.append({
                            'comment_id': comment.id,
                            'author': str(comment.author),
                            'text': comment.body,
                            'score': comment.score,
                            'created_utc': datetime.fromtimestamp(comment.created_utc)
                        })
                        print(f"  - Collected comment {len(comments)}/{comments_per_post}")
                    except Exception as e:
                        print(f"  - Skipped comment: {str(e)}")
                
                if len(comments) >= 5:
                    post_data = {
                        'post_id': post.id,
                        'title': post.title,
                        'text': post.selftext,
                        'author': str(post.author),
                        'created_utc': datetime.fromtimestamp(post.created_utc),
                        'score': post.score,
                        'num_comments': len(comments),
                        'comments': comments
                    }
                    posts_data.append(post_data)
                
                time.sleep(0.5)
                
            print(f"\nSuccessfully collected {len(posts_data)} posts with comments")
            return pd.DataFrame(posts_data)
            
        except Exception as e:
            print(f"Error during data collection: {str(e)}")
            return pd.DataFrame(posts_data)
    
    def save_data(self, data, filename):
        """Save collected data"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = f"data/raw/{filename}_{timestamp}.json"
        
        os.makedirs('data/raw', exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, default=str, ensure_ascii=False, indent=2)
            
        print(f"\nData saved to {filepath}")
        return filepath

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('data/raw', exist_ok=True)
    
    print("Starting Reddit data collection...")
    collector = RedditDataCollector()
    posts_df = collector.collect_data()
    collector.save_data(posts_df.to_dict('records'), 'dating_posts')
    print("Data collection completed!")