import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
from sklearn.linear_model import LinearRegression

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

# Create comprehensive visualization dashboard
fig = make_subplots(
    rows=4, cols=3,
    subplot_titles=(
        '3D: Days vs Miles vs Reimbursement', 
        '3D: Days vs Receipts vs Reimbursement',
        '3D: Miles vs Receipts vs Reimbursement',
        'Reimbursement by Trip Days', 
        'Reimbursement by Miles', 
        'Reimbursement by Receipt Amount',
        'Per-Day Rate by Trip Length', 
        'Miles/Day Efficiency vs Reimbursement',
        'Receipt Amount vs Reimbursement (colored by days)',
        'Reimbursement Heatmap (Days x Miles)', 
        'Per-Mile Rate Distribution',
        'Special Receipt Endings Analysis'
    ),
    specs=[
        [{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}],
        [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
        [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
        [{'type': 'heatmap'}, {'type': 'histogram'}, {'type': 'box'}]
    ],
    row_heights=[0.3, 0.25, 0.25, 0.2],
    vertical_spacing=0.08,
    horizontal_spacing=0.05
)

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
            showscale=True,
            colorbar=dict(x=0.3, len=0.25, y=0.87)
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
            showscale=True,
            colorbar=dict(x=0.65, len=0.25, y=0.87, title="Miles")
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
            showscale=True,
            colorbar=dict(x=1.0, len=0.25, y=0.87, title="Days")
        ),
        text=[f"Days: {d}<br>Miles: {m}<br>Receipts: ${r:.2f}<br>Reimbursement: ${reimb:.2f}" 
              for d, m, r, reimb in zip(df['days'], df['miles'], df['receipts'], df['reimbursement'])],
        hovertemplate='%{text}<extra></extra>'
    ),
    row=1, col=3
)

# 4. Box plot: Reimbursement by Trip Days
for day in sorted(df['days'].unique()):
    day_data = df[df['days'] == day]['reimbursement']
    fig.add_trace(
        go.Box(y=day_data, name=f"{day} days", showlegend=False),
        row=2, col=1
    )

# 5. Scatter: Miles vs Reimbursement (colored by days)
for day in sorted(df['days'].unique()):
    day_df = df[df['days'] == day]
    fig.add_trace(
        go.Scatter(
            x=day_df['miles'],
            y=day_df['reimbursement'],
            mode='markers',
            name=f"{day} days",
            marker=dict(size=6),
            showlegend=True
        ),
        row=2, col=2
    )

# 6. Scatter: Receipt Amount vs Reimbursement (colored by miles)
fig.add_trace(
    go.Scatter(
        x=df['receipts'],
        y=df['reimbursement'],
        mode='markers',
        marker=dict(
            size=6,
            color=df['miles'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Miles", x=1.02)
        ),
        text=[f"Days: {d}<br>Miles: {m}<br>Receipts: ${r:.2f}<br>Reimbursement: ${reimb:.2f}" 
              for d, m, r, reimb in zip(df['days'], df['miles'], df['receipts'], df['reimbursement'])],
        hovertemplate='%{text}<extra></extra>',
        showlegend=False
    ),
    row=2, col=3
)

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
            showscale=True,
            colorbar=dict(title="Days")
        ),
        text=[f"Days: {d}<br>Miles/Day: {mpd:.1f}<br>Total Miles: {m}<br>Reimbursement: ${reimb:.2f}" 
              for d, mpd, m, reimb in zip(df['days'], df['miles_per_day'], df['miles'], df['reimbursement'])],
        hovertemplate='%{text}<extra></extra>',
        showlegend=False
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

# 10. Heatmap: Days x Miles -> Reimbursement
pivot_table = df.pivot_table(values='reimbursement', index='days', columns='miles', aggfunc='mean')
fig.add_trace(
    go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale='RdYlBu',
        text=np.round(pivot_table.values, 2),
        texttemplate="%{text}",
        textfont={"size": 8},
        showscale=True,
        colorbar=dict(x=0.35, len=0.15, y=0.08)
    ),
    row=4, col=1
)

# 11. Histogram: Per-Mile Rate Distribution
per_mile_clean = df['per_mile'].dropna()
fig.add_trace(
    go.Histogram(
        x=per_mile_clean,
        nbinsx=30,
        showlegend=False
    ),
    row=4, col=2
)

# 12. Box plot: Special Receipt Endings Analysis
normal_receipts = df[~(df['ends_in_49'] | df['ends_in_99'])]['reimbursement']
ends_49 = df[df['ends_in_49']]['reimbursement']
ends_99 = df[df['ends_in_99']]['reimbursement']

fig.add_trace(go.Box(y=normal_receipts, name='Normal', showlegend=False), row=4, col=3)
fig.add_trace(go.Box(y=ends_49, name='Ends .49', showlegend=False), row=4, col=3)
fig.add_trace(go.Box(y=ends_99, name='Ends .99', showlegend=False), row=4, col=3)

# Update layout
fig.update_layout(
    title_text="Comprehensive Reimbursement Data Analysis Dashboard",
    height=2000,
    showlegend=True,
    legend=dict(x=1.05, y=0.5)
)

# Update axes labels
fig.update_xaxes(title_text="Days", row=1, col=1)
fig.update_yaxes(title_text="Miles", row=1, col=1)

fig.update_xaxes(title_text="Days", row=1, col=2)
fig.update_yaxes(title_text="Receipts ($)", row=1, col=2)

fig.update_xaxes(title_text="Miles", row=1, col=3)
fig.update_yaxes(title_text="Receipts ($)", row=1, col=3)

fig.update_xaxes(title_text="Trip Days", row=2, col=1)
fig.update_yaxes(title_text="Reimbursement ($)", row=2, col=1)

fig.update_xaxes(title_text="Miles Traveled", row=2, col=2)
fig.update_yaxes(title_text="Reimbursement ($)", row=2, col=2)

fig.update_xaxes(title_text="Receipt Amount ($)", row=2, col=3)
fig.update_yaxes(title_text="Reimbursement ($)", row=2, col=3)

fig.update_xaxes(title_text="Trip Days", row=3, col=1)
fig.update_yaxes(title_text="Per-Day Rate ($)", row=3, col=1)

fig.update_xaxes(title_text="Miles per Day", row=3, col=2)
fig.update_yaxes(title_text="Reimbursement ($)", row=3, col=2)

fig.update_xaxes(title_text="Receipt Amount ($)", row=3, col=3)
fig.update_yaxes(title_text="Reimbursement ($)", row=3, col=3)

fig.update_xaxes(title_text="Miles", row=4, col=1)
fig.update_yaxes(title_text="Days", row=4, col=1)

fig.update_xaxes(title_text="Per-Mile Rate ($)", row=4, col=2)
fig.update_yaxes(title_text="Frequency", row=4, col=2)

fig.update_xaxes(title_text="Receipt Ending Type", row=4, col=3)
fig.update_yaxes(title_text="Reimbursement ($)", row=4, col=3)

# Show the plot
fig.show()

# Additional analysis plots
print("\n=== Statistical Analysis ===\n")

# Check for 5-day bonus
day_stats = df.groupby('days').agg({
    'reimbursement': ['mean', 'std', 'count'],
    'per_day': 'mean'
}).round(2)
print("Average reimbursement by trip length:")
print(day_stats)

# Check for efficiency bonuses
efficiency_bins = pd.cut(df['miles_per_day'], bins=[0, 50, 100, 150, 200, 250, 1000])
efficiency_stats = df.groupby(efficiency_bins)['reimbursement'].mean()
print("\nAverage reimbursement by miles/day efficiency:")
print(efficiency_stats)

# Create individual focused plots for deeper analysis
# Plot 1: Focus on 5-day trips
five_day_df = df[df['days'] == 5]
fig_5day = px.scatter_3d(
    five_day_df, 
    x='miles', 
    y='receipts', 
    z='reimbursement',
    color='miles_per_day',
    size='reimbursement',
    title='5-Day Trips Analysis',
    labels={'miles': 'Miles', 'receipts': 'Receipts ($)', 'reimbursement': 'Reimbursement ($)'}
)
fig_5day.show()

# Plot 2: Residual analysis
# Simple linear regression to find outliers
X = df[['days', 'miles', 'receipts']].values
y = df['reimbursement'].values
model = LinearRegression().fit(X, y)
predictions = model.predict(X)
residuals = y - predictions

fig_residuals = go.Figure()
fig_residuals.add_trace(go.Scatter(
    x=predictions,
    y=residuals,
    mode='markers',
    marker=dict(
        color=df['days'],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Days")
    ),
    text=[f"Days: {d}<br>Miles: {m}<br>Receipts: ${r:.2f}<br>Actual: ${actual:.2f}<br>Predicted: ${pred:.2f}<br>Residual: ${res:.2f}" 
          for d, m, r, actual, pred, res in zip(df['days'], df['miles'], df['receipts'], 
                                                 df['reimbursement'], predictions, residuals)],
    hovertemplate='%{text}<extra></extra>'
))
fig_residuals.update_layout(
    title="Residual Analysis - Finding Outliers and Patterns",
    xaxis_title="Predicted Reimbursement ($)",
    yaxis_title="Residual ($)",
    height=600
)
fig_residuals.show()

# Save interactive HTML
fig.write_html("reimbursement_analysis_dashboard.html")
print("\nDashboard saved as 'reimbursement_analysis_dashboard.html'") 