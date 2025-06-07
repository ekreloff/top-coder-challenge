import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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

# Sort data for derivative calculations
df = df.sort_values(['days', 'miles', 'receipts'])

# Calculate marginal changes
df['marginal_per_day'] = df.groupby('days')['reimbursement'].diff()
df['marginal_per_mile'] = df.groupby('miles')['reimbursement'].diff()
df['marginal_per_receipt'] = df.groupby('receipts')['reimbursement'].diff()

# Create comprehensive visualization dashboard
fig = make_subplots(
    rows=5, cols=3,
    subplot_titles=(
        # Row 1: 3D Views
        '3D: Days vs Miles vs Reimbursement',
        '3D: Days vs Receipts vs Reimbursement',
        '3D: Miles vs Receipts vs Reimbursement',
        
        # Row 2: Basic Analysis
        'Reimbursement by Trip Days',
        'Reimbursement by Miles',
        'Reimbursement by Receipt Amount',
        
        # Row 3: Efficiency Analysis
        'Per-Day Rate by Trip Length',
        'Miles/Day Efficiency vs Reimbursement',
        'Receipt Amount vs Reimbursement',
        
        # Row 4: Advanced Analysis
        'Marginal Reimbursement per Day',
        'Marginal Reimbursement per Receipt',
        'Marginal Reimbursement per Mile',
        
        # Row 5: Interaction & Error Analysis
        'Receipts/Day vs Miles/Day Heatmap',
        'Reimbursement/Receipt vs Receipt Amount',
        'Error Distribution'
    ),
    specs=[
        [{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}],
        [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
        [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
        [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
        [{'type': 'heatmap'}, {'type': 'scatter'}, {'type': 'bar'}]
    ],
    row_heights=[0.3, 0.2, 0.2, 0.2, 0.1],
    vertical_spacing=0.05,
    horizontal_spacing=0.05
)

# Row 1: 3D Views
# 1. 3D Scatter: Days vs Miles vs Reimbursement
fig.add_trace(
    go.Scatter3d(
        x=df['days'],
        y=df['miles'],
        z=df['reimbursement'],
        mode='markers',
        marker=dict(
            size=5,
            color=df['reimbursement'],
            colorscale='Viridis',
            showscale=True
        ),
        text=[f"Days: {d}<br>Miles: {m}<br>Receipts: ${r:.2f}<br>Reimbursement: ${reimb:.2f}" 
              for d, m, r, reimb in zip(df['days'], df['miles'], df['receipts'], df['reimbursement'])],
        hovertemplate='%{text}<extra></extra>'
    ),
    row=1, col=1
)

# 2. 3D Scatter: Days vs Receipts vs Reimbursement
fig.add_trace(
    go.Scatter3d(
        x=df['days'],
        y=df['receipts'],
        z=df['reimbursement'],
        mode='markers',
        marker=dict(
            size=5,
            color=df['miles'],
            colorscale='Plasma',
            showscale=True
        ),
        text=[f"Days: {d}<br>Miles: {m}<br>Receipts: ${r:.2f}<br>Reimbursement: ${reimb:.2f}" 
              for d, m, r, reimb in zip(df['days'], df['miles'], df['receipts'], df['reimbursement'])],
        hovertemplate='%{text}<extra></extra>'
    ),
    row=1, col=2
)

# 3. 3D Scatter: Miles vs Receipts vs Reimbursement
fig.add_trace(
    go.Scatter3d(
        x=df['miles'],
        y=df['receipts'],
        z=df['reimbursement'],
        mode='markers',
        marker=dict(
            size=5,
            color=df['days'],
            colorscale='Turbo',
            showscale=True
        ),
        text=[f"Days: {d}<br>Miles: {m}<br>Receipts: ${r:.2f}<br>Reimbursement: ${reimb:.2f}" 
              for d, m, r, reimb in zip(df['days'], df['miles'], df['receipts'], df['reimbursement'])],
        hovertemplate='%{text}<extra></extra>'
    ),
    row=1, col=3
)

# Row 2: Basic Analysis
# 4. Box plot: Reimbursement by Trip Days
for day in sorted(df['days'].unique()):
    day_data = df[df['days'] == day]['reimbursement']
    fig.add_trace(
        go.Box(y=day_data, name=f"{day} days", showlegend=False),
        row=2, col=1
    )

# 5. Scatter: Miles vs Reimbursement
fig.add_trace(
    go.Scatter(
        x=df['miles'],
        y=df['reimbursement'],
        mode='markers',
        marker=dict(
            size=6,
            color=df['days'],
            colorscale='Plasma',
            showscale=True
        ),
        text=[f"Days: {d}<br>Miles: {m}<br>Receipts: ${r:.2f}<br>Reimbursement: ${reimb:.2f}" 
              for d, m, r, reimb in zip(df['days'], df['miles'], df['receipts'], df['reimbursement'])],
        hovertemplate='%{text}<extra></extra>'
    ),
    row=2, col=2
)

# 6. Scatter: Receipt Amount vs Reimbursement
fig.add_trace(
    go.Scatter(
        x=df['receipts'],
        y=df['reimbursement'],
        mode='markers',
        marker=dict(
            size=6,
            color=df['miles'],
            colorscale='Viridis',
            showscale=True
        ),
        text=[f"Days: {d}<br>Miles: {m}<br>Receipts: ${r:.2f}<br>Reimbursement: ${reimb:.2f}" 
              for d, m, r, reimb in zip(df['days'], df['miles'], df['receipts'], df['reimbursement'])],
        hovertemplate='%{text}<extra></extra>'
    ),
    row=2, col=3
)

# Row 3: Efficiency Analysis
# 7. Box plot: Per-Day Rate by Trip Length
for day in sorted(df['days'].unique()):
    day_data = df[df['days'] == day]['per_day']
    fig.add_trace(
        go.Box(y=day_data, name=f"{day} days", showlegend=False),
        row=3, col=1
    )

# 8. Scatter: Miles/Day Efficiency vs Reimbursement
fig.add_trace(
    go.Scatter(
        x=df['miles_per_day'],
        y=df['reimbursement'],
        mode='markers',
        marker=dict(
            size=6,
            color=df['days'],
            colorscale='Plasma',
            showscale=True
        ),
        text=[f"Days: {d}<br>Miles/Day: {mpd:.1f}<br>Total Miles: {m}<br>Reimbursement: ${reimb:.2f}" 
              for d, mpd, m, reimb in zip(df['days'], df['miles_per_day'], df['miles'], df['reimbursement'])],
        hovertemplate='%{text}<extra></extra>'
    ),
    row=3, col=2
)

# 9. Scatter: Receipt vs Reimbursement colored by trip days
for day in sorted(df['days'].unique()):
    day_df = df[df['days'] == day]
    fig.add_trace(
        go.Scatter(
            x=day_df['receipts'],
            y=day_df['reimbursement'],
            mode='markers',
            name=f"{day} days",
            marker=dict(size=6),
            showlegend=False
        ),
        row=3, col=3
    )

# Row 4: Advanced Analysis
# 10. Marginal Reimbursement per Day
fig.add_trace(
    go.Scatter(
        x=df['days'],
        y=df['marginal_per_day'],
        mode='markers',
        marker=dict(
            size=8,
            color=df['miles'],
            colorscale='Viridis',
            showscale=True
        ),
        text=[f"Days: {d}<br>Miles: {m}<br>Marginal: ${marg:.2f}" 
              for d, m, marg in zip(df['days'], df['miles'], df['marginal_per_day'])],
        hovertemplate='%{text}<extra></extra>'
    ),
    row=4, col=1
)

# 11. Marginal Reimbursement per Receipt
fig.add_trace(
    go.Scatter(
        x=df['receipts'],
        y=df['marginal_per_receipt'],
        mode='markers',
        marker=dict(
            size=8,
            color=df['days'],
            colorscale='Plasma',
            showscale=True
        ),
        text=[f"Receipts: ${r:.2f}<br>Days: {d}<br>Marginal: ${marg:.2f}" 
              for r, d, marg in zip(df['receipts'], df['days'], df['marginal_per_receipt'])],
        hovertemplate='%{text}<extra></extra>'
    ),
    row=4, col=2
)

# 12. Marginal Reimbursement per Mile
fig.add_trace(
    go.Scatter(
        x=df['miles'],
        y=df['marginal_per_mile'],
        mode='markers',
        marker=dict(
            size=8,
            color=df['receipts'],
            colorscale='Turbo',
            showscale=True
        ),
        text=[f"Miles: {m}<br>Receipts: ${r:.2f}<br>Marginal: ${marg:.2f}" 
              for m, r, marg in zip(df['miles'], df['receipts'], df['marginal_per_mile'])],
        hovertemplate='%{text}<extra></extra>'
    ),
    row=4, col=3
)

# Row 5: Interaction & Error Analysis
# 13. Interaction Terms Heatmap
pivot_table = df.pivot_table(
    values='reimbursement',
    index=pd.qcut(df['receipts_per_day'], 10),
    columns=pd.qcut(df['miles_per_day'], 10),
    aggfunc='mean'
)

# Convert interval index and columns to strings
pivot_table.index = [f"{x.left:.1f}-{x.right:.1f}" for x in pivot_table.index]
pivot_table.columns = [f"{x.left:.1f}-{x.right:.1f}" for x in pivot_table.columns]

fig.add_trace(
    go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale='RdYlBu',
        text=np.round(pivot_table.values, 2),
        texttemplate="%{text}",
        textfont={"size": 8},
        showscale=True
    ),
    row=5, col=1
)

# 14. Reimbursement per Receipt Dollar
fig.add_trace(
    go.Scatter(
        x=df['receipts'],
        y=df['reimbursement_per_receipt'],
        mode='markers',
        marker=dict(
            size=8,
            color=df['days'],
            colorscale='Plasma',
            showscale=True
        ),
        text=[f"Receipts: ${r:.2f}<br>Days: {d}<br>Efficiency: {eff:.2f}" 
              for r, d, eff in zip(df['receipts'], df['days'], df['reimbursement_per_receipt'])],
        hovertemplate='%{text}<extra></extra>'
    ),
    row=5, col=2
)

# 15. Error Distribution
# Calculate errors using a simple linear model
X = df[['days', 'miles', 'receipts']]
y = df['reimbursement']
model = LinearRegression()
model.fit(X, y)
df['predicted'] = model.predict(X)
df['error'] = df['reimbursement'] - df['predicted']

# Create error buckets
error_bins = [-np.inf, -100, -50, -10, -1, 0, 1, 10, 50, 100, np.inf]
error_labels = ['<-100', '-100 to -50', '-50 to -10', '-10 to -1', '-1 to 0', 
                '0 to 1', '1 to 10', '10 to 50', '50 to 100', '>100']
df['error_bucket'] = pd.cut(df['error'], bins=error_bins, labels=error_labels)

error_counts = df['error_bucket'].value_counts().sort_index()

fig.add_trace(
    go.Bar(
        x=error_counts.index,
        y=error_counts.values,
        text=error_counts.values,
        textposition='auto',
    ),
    row=5, col=3
)

# Update layout
fig.update_layout(
    height=2000,
    width=1800,
    title_text="Comprehensive Reimbursement Analysis Dashboard",
    showlegend=False,
    template="plotly_white"
)

# Save the figure
fig.write_html("comprehensive_analysis_dashboard.html") 