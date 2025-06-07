import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import ruptures
from statsmodels.tsa.stattools import adfuller

# Load the data
with open('public_cases.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame([
    {
        'days': case['input']['trip_duration_days'],
        'miles': case['input']['miles_traveled'],
        'receipts': case['input']['total_receipts_amount'],
        'reimbursement': case['expected_output']
    }
    for case in data
])

# Calculate derived features
df['per_day'] = df['reimbursement'] / df['days']
df['miles_per_day'] = df['miles'] / df['days']
df['receipts_per_day'] = df['receipts'] / df['days']
df['per_mile'] = df['reimbursement'] / df['miles'].replace(0, np.nan)
df['receipts_per_mile'] = df['receipts'] / df['miles'].replace(0, np.nan)
df['reimbursement_per_receipt'] = df['reimbursement'] / df['receipts']
df['receipts_rounded'] = df['receipts'].round(2)
df['ends_in_49'] = df['receipts_rounded'].apply(lambda x: str(x).endswith('.49'))
df['ends_in_99'] = df['receipts_rounded'].apply(lambda x: str(x).endswith('.99'))

# Sort data for analysis
df = df.sort_values(['days', 'miles', 'receipts'])

# 1. Structural Break Detection
def detect_breaks(series, n_bkps=3):
    # Convert to numpy array and reshape for ruptures
    signal = series.values.reshape(-1, 1)
    # Detect breakpoints
    algo = ruptures.Pelt(model="rbf").fit(signal)
    result = algo.predict(n_bkps)
    return result

# Detect breaks in each dimension
days_means = df.groupby('days')['reimbursement'].mean()
miles_means = df.groupby('miles')['reimbursement'].mean()
receipts_means = df.groupby('receipts')['reimbursement'].mean()

days_breaks = detect_breaks(days_means)
miles_breaks = detect_breaks(miles_means)
receipts_breaks = detect_breaks(receipts_means)

# 2. Clustering Analysis
# Prepare features for clustering
features = ['days', 'miles', 'receipts']
X = df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Calculate cluster statistics
cluster_stats = df.groupby('cluster').agg({
    'reimbursement': ['mean', 'std', 'count'],
    'days': 'mean',
    'miles': 'mean',
    'receipts': 'mean'
}).round(2)

# 3. Curvature Analysis
df['reimbursement_grad'] = np.gradient(df['reimbursement'])
df['reimbursement_curvature'] = np.gradient(df['reimbursement_grad'])

# 4. Synthetic Gradient Testing
delta = 1
df['receipt_impact'] = df['reimbursement'].diff() / delta
df['miles_impact'] = df['reimbursement'].diff() / delta
df['days_impact'] = df['reimbursement'].diff() / delta

# Create visualization dashboard
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=(
        'Structural Breaks in Reimbursement by Days',
        'Structural Breaks in Reimbursement by Miles',
        'Structural Breaks in Reimbursement by Receipts',
        'Cluster Analysis (3D)',
        'Reimbursement Curvature Analysis',
        'Synthetic Gradient Impact'
    ),
    specs=[
        [{'type': 'scatter'}, {'type': 'scatter'}],
        [{'type': 'scatter'}, {'type': 'scatter3d'}],
        [{'type': 'scatter'}, {'type': 'scatter'}]
    ],
    row_heights=[0.4, 0.4, 0.2],
    vertical_spacing=0.1,
    horizontal_spacing=0.1
)

# 1. Structural Breaks - Days
days_means = df.groupby('days')['reimbursement'].mean()
fig.add_trace(
    go.Scatter(
        x=days_means.index,
        y=days_means.values,
        mode='lines+markers',
        name='Days vs Reimbursement'
    ),
    row=1, col=1
)

# Add breakpoint lines
for break_point in days_breaks:
    fig.add_vline(x=break_point, line_dash="dash", line_color="red", row=1, col=1)

# 2. Structural Breaks - Miles
miles_means = df.groupby('miles')['reimbursement'].mean()
fig.add_trace(
    go.Scatter(
        x=miles_means.index,
        y=miles_means.values,
        mode='lines+markers',
        name='Miles vs Reimbursement'
    ),
    row=1, col=2
)

# Add breakpoint lines
for break_point in miles_breaks:
    fig.add_vline(x=break_point, line_dash="dash", line_color="red", row=1, col=2)

# 3. Structural Breaks - Receipts
receipts_means = df.groupby('receipts')['reimbursement'].mean()
fig.add_trace(
    go.Scatter(
        x=receipts_means.index,
        y=receipts_means.values,
        mode='lines+markers',
        name='Receipts vs Reimbursement'
    ),
    row=2, col=1
)

# Add breakpoint lines
for break_point in receipts_breaks:
    fig.add_vline(x=break_point, line_dash="dash", line_color="red", row=2, col=1)

# 4. Cluster Analysis (3D)
fig.add_trace(
    go.Scatter3d(
        x=df['days'],
        y=df['miles'],
        z=df['receipts'],
        mode='markers',
        marker=dict(
            size=5,
            color=df['cluster'],
            colorscale='Viridis',
            showscale=True
        ),
        text=[f"Cluster: {c}<br>Days: {d}<br>Miles: {m}<br>Receipts: ${r:.2f}<br>Reimbursement: ${reimb:.2f}" 
              for c, d, m, r, reimb in zip(df['cluster'], df['days'], df['miles'], df['receipts'], df['reimbursement'])],
        hovertemplate='%{text}<extra></extra>'
    ),
    row=2, col=2
)

# 5. Curvature Analysis
fig.add_trace(
    go.Scatter(
        x=df['receipts'],
        y=df['reimbursement_curvature'],
        mode='markers',
        marker=dict(
            size=6,
            color=df['days'],
            colorscale='Plasma',
            showscale=True
        ),
        text=[f"Days: {d}<br>Receipts: ${r:.2f}<br>Curvature: {c:.2f}" 
              for d, r, c in zip(df['days'], df['receipts'], df['reimbursement_curvature'])],
        hovertemplate='%{text}<extra></extra>'
    ),
    row=3, col=1
)

# 6. Synthetic Gradient Impact
fig.add_trace(
    go.Scatter(
        x=df['receipts'],
        y=df['receipt_impact'],
        mode='markers',
        marker=dict(
            size=6,
            color=df['days'],
            colorscale='Turbo',
            showscale=True
        ),
        text=[f"Days: {d}<br>Receipts: ${r:.2f}<br>Impact: ${imp:.2f}" 
              for d, r, imp in zip(df['days'], df['receipts'], df['receipt_impact'])],
        hovertemplate='%{text}<extra></extra>'
    ),
    row=3, col=2
)

# Update layout
fig.update_layout(
    height=1500,
    width=1800,
    title_text="Structural Analysis of Reimbursement System",
    showlegend=False,
    template="plotly_white"
)

# Save the figure
fig.write_html("structural_analysis_dashboard.html")

# Print cluster statistics
print("\nCluster Statistics:")
print(cluster_stats)

# Print breakpoint analysis
print("\nBreakpoint Analysis:")
print("Days breakpoints:", days_breaks)
print("Miles breakpoints:", miles_breaks)
print("Receipts breakpoints:", receipts_breaks) 