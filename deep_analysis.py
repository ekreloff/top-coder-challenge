import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

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
df['receipts_rounded'] = df['receipts'].round(2)
df['ends_in_49'] = df['receipts_rounded'].apply(lambda x: str(x).endswith('.49'))
df['ends_in_99'] = df['receipts_rounded'].apply(lambda x: str(x).endswith('.99'))
df['ends_in_00'] = df['receipts_rounded'].apply(lambda x: str(x).endswith('.00'))

# 1. Per-Day Rate Analysis
print("\n=== Per-Day Rate Analysis ===")
# Fit exponential decay model
X_days = df['days'].values.reshape(-1, 1)
y_per_day = df['per_day'].values
exp_model = LinearRegression()
exp_model.fit(np.log(X_days), y_per_day)
print(f"Exponential decay model: y = {exp_model.coef_[0]:.2f} * ln(x) + {exp_model.intercept_:.2f}")
print(f"R² score: {r2_score(y_per_day, exp_model.predict(np.log(X_days))):.3f}")

# 2. Efficiency Bonus Analysis
print("\n=== Efficiency Bonus Analysis ===")
# Create efficiency bins
efficiency_bins = pd.cut(df['miles_per_day'], bins=[0, 50, 100, 150, 200, 250, 1000])
df['efficiency_bin'] = efficiency_bins.astype(str)  # Convert to string for plotting
efficiency_stats = df.groupby('efficiency_bin').agg({
    'reimbursement': ['mean', 'std', 'count'],
    'miles_per_day': 'mean'
}).round(2)
print("\nEfficiency bonus analysis by miles/day:")
print(efficiency_stats)

# 3. Mileage Tier Analysis
print("\n=== Mileage Tier Analysis ===")
# Create mileage bins
mileage_bins = pd.cut(df['miles'], bins=[0, 100, 300, 500, 1000, 2000, 5000])
df['mileage_bin'] = mileage_bins.astype(str)  # Convert to string for plotting
mileage_stats = df.groupby('mileage_bin').agg({
    'reimbursement': ['mean', 'std', 'count'],
    'miles': 'mean',
    'per_mile': 'mean'
}).round(2)
print("\nMileage tier analysis:")
print(mileage_stats)

# 4. Receipt Analysis
print("\n=== Receipt Analysis ===")
# Analyze receipt patterns
receipt_bins = pd.cut(df['receipts'], bins=[0, 500, 1000, 1500, 2000, 3000, 5000])
df['receipt_bin'] = receipt_bins.astype(str)  # Convert to string for plotting
receipt_stats = df.groupby('receipt_bin').agg({
    'reimbursement': ['mean', 'std', 'count'],
    'receipts': 'mean',
    'receipts_per_day': 'mean'
}).round(2)
print("\nReceipt amount analysis:")
print(receipt_stats)

# Analyze receipt endings
ending_stats = pd.DataFrame({
    'Normal': df[~(df['ends_in_49'] | df['ends_in_99'] | df['ends_in_00'])]['reimbursement'],
    'Ends .49': df[df['ends_in_49']]['reimbursement'],
    'Ends .99': df[df['ends_in_99']]['reimbursement'],
    'Ends .00': df[df['ends_in_00']]['reimbursement']
}).describe().round(2)
print("\nReceipt ending analysis:")
print(ending_stats)

# 5. Clustering Analysis
print("\n=== Clustering Analysis ===")
# Prepare data for clustering
X_cluster = df[['days', 'miles', 'receipts']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# K-means clustering
kmeans = KMeans(n_clusters=6, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Analyze clusters
cluster_stats = df.groupby('cluster').agg({
    'days': ['mean', 'std'],
    'miles': ['mean', 'std'],
    'receipts': ['mean', 'std'],
    'reimbursement': ['mean', 'std', 'count']
}).round(2)
print("\nCluster analysis:")
print(cluster_stats)

# 6. Decision Tree Analysis
print("\n=== Decision Tree Analysis ===")
X_tree = df[['days', 'miles', 'receipts']].values
y_tree = df['reimbursement'].values
tree = DecisionTreeRegressor(max_depth=5, random_state=42)
tree.fit(X_tree, y_tree)
print(f"Decision Tree R² score: {tree.score(X_tree, y_tree):.3f}")

# Create visualizations
# 1. Per-Day Rate Decay
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=df['days'],
    y=df['per_day'],
    mode='markers',
    name='Actual'
))
x_range = np.linspace(1, df['days'].max(), 100)
fig1.add_trace(go.Scatter(
    x=x_range,
    y=exp_model.predict(np.log(x_range.reshape(-1, 1))),
    mode='lines',
    name='Exponential Fit'
))
fig1.update_layout(
    title='Per-Day Rate Decay Analysis',
    xaxis_title='Trip Days',
    yaxis_title='Per-Day Rate ($)'
)
fig1.write_html('per_day_decay.html')

# 2. Efficiency Bonus Analysis
fig2 = px.box(df, x='efficiency_bin', y='reimbursement',
              title='Efficiency Bonus Analysis')
fig2.write_html('efficiency_bonus.html')

# 3. Mileage Tier Analysis
fig3 = px.box(df, x='mileage_bin', y='per_mile',
              title='Mileage Tier Analysis')
fig3.write_html('mileage_tiers.html')

# 4. Receipt Analysis
fig4 = px.box(df, x='receipt_bin', y='reimbursement',
              title='Receipt Amount Analysis')
fig4.write_html('receipt_analysis.html')

# 5. Cluster Analysis
fig5 = px.scatter_3d(df, x='days', y='miles', z='receipts',
                     color='cluster', size='reimbursement',
                     title='Cluster Analysis of Trips')
fig5.write_html('cluster_analysis.html')

print("\nAnalysis complete! Check the generated HTML files for visualizations.") 