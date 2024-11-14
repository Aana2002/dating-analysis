# visualizer.py
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots  
import pandas as pd
import numpy as np
import os

class Visualizer:
    def create_behavior_dashboard(self, behavior_data, save_path, sample_size=1000):
        """Create an interactive dashboard of behavioral patterns."""
        try:
            # Optional sampling if large dataset
            if len(behavior_data) > sample_size:
                print(f"Dataset too large, sampling {sample_size} rows for visualization.")
                behavior_data = behavior_data.sample(sample_size)

            # Create the dashboard with multiple subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Response Time Distribution',
                    'Engagement Levels',
                    'Communication Styles',
                    'Message Frequency Analysis'
                ),
                specs=[
                    [{"type": "histogram"}, {"type": "pie"}],
                    [{"type": "pie"}, {"type": "scatter"}]
                ],
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )

            # 1. Response Time Distribution
            response_times = behavior_data['avg_response_time'].clip(0, 24)
            fig.add_trace(
                go.Histogram(
                    x=response_times,
                    name='Response Time',
                    nbinsx=20,
                    marker_color='#1f77b4'
                ),
                row=1, col=1
            )
            fig.update_xaxes(title_text="Hours", row=1, col=1)
            fig.update_yaxes(title_text="Number of Users", row=1, col=1)

            # 2. Engagement Level Distribution
            engagement_counts = behavior_data['engagement_level'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=engagement_counts.index,
                    values=engagement_counts.values,
                    name='Engagement',
                    marker_colors=['#2ecc71', '#f1c40f', '#e74c3c']
                ),
                row=1, col=2
            )

            # 3. Communication Styles
            comm_styles = behavior_data.apply(
                lambda x: 'Positive' if x['sentiment_mean'] > 0.2 
                else ('Negative' if x['sentiment_mean'] < -0.2 else 'Neutral'),
                axis=1
            ).value_counts()
            fig.add_trace(
                go.Pie(
                    labels=comm_styles.index,
                    values=comm_styles.values,
                    name='Communication',
                    marker_colors=['#3498db', '#95a5a6', '#e67e22']
                ),
                row=2, col=1
            )

            # 4. Message Frequency vs Response Time
            fig.add_trace(
                go.Scatter(
                    x=behavior_data['message_frequency'],
                    y=behavior_data['avg_response_time'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=behavior_data['sentiment_mean'],
                        colorscale='RdBu',
                        showscale=True
                    ),
                    name='Users'
                ),
                row=2, col=2
            )
            fig.update_xaxes(title_text="Message Frequency", row=2, col=2)
            fig.update_yaxes(title_text="Avg Response Time (hours)", row=2, col=2)

            # Update layout
            fig.update_layout(
                height=1000,
                width=1200,
                title_text="Reddit Dating Behavior Analysis Dashboard",
                showlegend=True,
                template='plotly_white'
            )

            # Save interactive HTML first
            html_path = save_path.replace('.png', '.html')
            fig.write_html(html_path)
            print(f"Interactive dashboard saved as: {html_path}")

        except Exception as e:
            print(f"Error creating dashboard: {str(e)}")
