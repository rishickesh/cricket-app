import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. PAGE CONFIGURATION & THEME SETUP
# ==========================================
st.set_page_config(
    page_title="Advanced Cricket Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark-mode dashboard theme styling
st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #ffffff; }
    div[data-testid="stSidebarCollapseButton"] {color: #ffffff;}
    div[data-testid="stWidgetLabel"] { color: #ffffff !important; font-size: 11px !important; }
    .stSelectbox div[data-baseweb="select"] { background-color: #161b22; color: white; border-color: #30363d; }
    div[data-testid="stMultiSelect"] div[role="listbox"] { background-color: #161b22; color: white; }
    div[data-testid="stMetricValue"] { color: #ffffff; font-size: 28px; }
    
    /* KPI Card styling variants matching bowler UI colors */
    div.kpi-card {
        background-color: #11161d;
        border-radius: 6px;
        padding: 15px;
        text-align: center;
        margin-bottom: 15px;
    }
    div.kpi-wickets { border: 1px solid #1f6e43; }
    div.kpi-wickets div[data-testid="stMetricValue"] { color: #2ea043 !important; }
    
    div.kpi-economy { border: 1px solid #a62a22; }
    div.kpi-economy div[data-testid="stMetricValue"] { color: #f85149 !important; }
    
    div.kpi-phase { border: 1px solid #388bfd; }
    div.kpi-phase div[data-testid="stMetricValue"] { color: #58a6ff !important; }
    
    div.kpi-length { border: 1px solid #8957e5; }
    div.kpi-length div[data-testid="stMetricValue"] { color: #bc8cff !important; }

    div.sub-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
    
    /* Compact slider spacing for fielder mapping adjustments */
    div[data-testid="stSlider"] {
        margin-bottom: -15px !important;
        padding-top: 0px !important;
    }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 2. DATA INGESTION ENGINE (CACHED)
# ==========================================
@st.cache_data
def load_and_clean_dataset(file_path):
    # Changed from pd.read_csv to pd.read_parquet
    df = pd.read_parquet(file_path)
    
    df['Batter'] = df['Batter'].astype(str).str.strip()
    df['Bowler'] = df['Bowler'].astype(str).str.strip()
    df['Bowler Type'] = df['Bowler Type'].astype(str).str.strip().str.upper()
    df['Wicket'] = df['Wicket'].astype(str).str.strip().fillna('nan')
    df['Length'] = df['Length'].astype(str).str.strip().fillna('nan')
    
    # Fill standard venue names if not available
    if 'Venue' not in df.columns:
        df['Venue'] = 'Default Stadium Complex'
    else:
        df['Venue'] = df['Venue'].astype(str).str.strip()
        
    counting_cols = ['Runs', 'Extra Runs', 'Bowler Extra Runs', 'Over']
    for col in counting_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    coordinate_cols = ['FieldX', 'FieldY', 'PitchX', 'PitchY']
    for col in coordinate_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    if 'Legal Ball' in df.columns:
        df['Is_Legal'] = df['Legal Ball'].astype(str).str.strip().str.upper().map({'YES': True, 'TRUE': True, 'NO': False, 'FALSE': False}).fillna(True)
    else:
        df['Is_Legal'] = True
        
    return df

try:
    # Pointed to your newly optimized parquet file
    df = load_and_clean_dataset("optimized_data.parquet")
except FileNotFoundError:
    st.error("🚨 File 'optimized_data.parquet' not found.")
    st.stop()


# ==========================================
# 3. HELPER LOGIC FOR PHASE AND WICKETS
# ==========================================
def assign_match_phase(over):
    if over < 6: return 'PP'
    elif over < 16: return 'Mid'
    else: return 'Death'

df['Match_Phase'] = df['Over'].apply(assign_match_phase)
bowling_wickets = ['bowled', 'caught', 'lbw', 'stumped', 'caught and bowled', 'hit wicket']
df['Is_Bowler_Wicket'] = df['Wicket'].str.lower().isin(bowling_wickets)

print(df.columns)

# ==========================================
# 4. BATTING ANALYTICAL PLOT ENGINES
# ==========================================
def build_scoring_radar(dataframe, player_name):
    player_df = dataframe[(dataframe['Batter'] == player_name) & (dataframe['Is_Legal'] == True)]
    radar_data = player_df.groupby('Bowler Type').agg(total_runs=('Runs', 'sum'), balls_faced=('Runs', 'count')).reset_index()
    radar_data['SR'] = (radar_data['total_runs'] / np.maximum(radar_data['balls_faced'], 1)) * 100
    categories = ['RF', 'RFM', 'LF', 'ROB', 'RLB', 'LOB']
    radar_data = radar_data.set_index('Bowler Type').reindex(categories).fillna(0).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=radar_data['SR'].tolist() + [radar_data['SR'].iloc[0]], theta=categories + [categories[0]],
        fill='toself', fillcolor='rgba(255, 165, 0, 0.15)', line=dict(color='#FFA500', width=2),
        marker=dict(color='#FFA500', size=6), name='Strike Rate'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[50, 250], gridcolor='#222c3c'), angularaxis=dict(gridcolor='#222c3c')),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark', margin=dict(t=30, b=30, l=30, r=30), height=380
    )
    return fig

def build_dismissal_landing_spots(dataframe, player_name):
    dismissal_df = dataframe[(dataframe['Batter'] == player_name) & (dataframe['Is_Bowler_Wicket'] == True)].copy()
    fig = go.Figure()
    dismissal_df = dismissal_df.dropna(subset=['PitchX', 'PitchY'])
    dismissal_df = dismissal_df[(dismissal_df['PitchX'] != 0) & (dismissal_df['PitchY'] != 0)]
    
    fig.add_trace(go.Scatter(x=dismissal_df['PitchY'], y=dismissal_df['PitchX'], mode='markers', marker=dict(symbol='x', size=12, color='#FF4B4B')))
    
    lengths = [(1.5, "Yorker"), (4.0, "Full"), (7.0, "Good"), (10.0, "Back Of Length"), (13.0, "Short"), (16.0, "Bouncer")]
    for y_val, label in lengths:
        fig.add_hline(y=y_val, line_dash="dash", line_color="#222c3c")
        fig.add_annotation(x=-0.7, y=y_val, text=label, showarrow=False, font=dict(color="#8b949e", size=10), xanchor="right")

    stump_threshold = 0.114
    fig.add_vline(x=-stump_threshold, line_dash="solid", line_color="#30363d")
    fig.add_vline(x=stump_threshold, line_dash="solid", line_color="#30363d")
    
    batting_hand = dismissal_df['Batting Hand'].iloc[0] if len(dismissal_df) > 0 else 'RHB'
    left_label, right_label = ("LEG", "OFF") if batting_hand == 'LHB' else ("OFF", "LEG")
    fig.add_annotation(x=-0.4, y=0.5, text=left_label, showarrow=False, font=dict(color="#8b949e", size=11, weight="bold"))
    fig.add_annotation(x=0, y=0.5, text="MIDDLE", showarrow=False, font=dict(color="#8b949e", size=11, weight="bold"))
    fig.add_annotation(x=0.4, y=0.5, text=right_label, showarrow=False, font=dict(color="#8b949e", size=11, weight="bold"))
    
    fig.update_layout(xaxis=dict(range=[-1.0, 1.0], showticklabels=False, showgrid=False, zeroline=False),
                      yaxis=dict(range=[0, 18], autorange="reversed", showticklabels=False, showgrid=False, zeroline=False),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#11161d', margin=dict(t=40, b=30, l=80, r=40), height=380, showlegend=False)
    return fig

def build_polar_wagon_wheel(dataframe, player_name):
    player_df = dataframe[(dataframe['Batter'] == player_name) & (dataframe['Is_Legal'] == True)].copy()
    player_df = player_df.dropna(subset=['FieldX', 'FieldY'])
    player_df = player_df[(player_df['FieldX'] != 0) & (player_df['FieldY'] != 0)]
    
    if len(player_df) == 0: return go.Figure()
    
    center_x, center_y = 175.0, 166.5
    player_df['Angle'] = (np.degrees(np.arctan2(player_df['FieldY'] - center_y, player_df['FieldX'] - center_x)) + 360) % 360
    
    bin_edges = np.arange(0, 390, 30)
    labels = ['MID-WKT', 'SQ. LEG', 'B. FINE LEG', 'FINE LEG', 'THIRD MAN', 'B. POINT', 'POINT', 'COVER', 'MID-OFF', 'LONG OFF', 'LONG ON', 'MID-ON']
    player_df['Sector'] = pd.cut(player_df['Angle'], bins=bin_edges, labels=labels, right=False)
    
    sector_df = player_df.groupby('Sector', observed=False).agg(runs=('Runs', 'sum'), balls=('Runs', 'count')).reset_index()
    sector_df = sector_df.set_index('Sector').reindex(labels, fill_value=0).reset_index()
    sector_df['SR'] = (sector_df['runs'] / np.maximum(sector_df['balls'], 1)) * 100
    
    sector_df['SR_Class'] = pd.cut(sector_df['SR'], bins=[-1, 90, 130, 180, 9999], labels=['SR < 90', 'SR 90-130', 'SR 130-180', 'SR 180+'])
    color_map = {'SR 180+': '#FF4B4B', 'SR 130-180': '#FFA500', 'SR 90-130': '#00CC96', 'SR < 90': '#38a1f3'}
    
    fig = px.bar_polar(sector_df, r='runs', theta='Sector', color='SR_Class', color_discrete_map=color_map,
                       category_orders={"Sector": labels, "SR_Class": ['SR 180+', 'SR 130-180', 'SR 90-130', 'SR < 90']}, template="plotly_dark")
    fig.update_layout(polar=dict(radialaxis=dict(showticklabels=False, gridcolor='#222c3c'), angularaxis=dict(direction="clockwise", period=12, gridcolor='#222c3c')),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=30, b=40, l=30, r=30), height=380,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
    return fig


# ==========================================
# 5. BOWLER ANALYTICAL PLOT ENGINES
# ==========================================
def build_wickets_by_phase(dataframe, bowler_name):
    bowler_df = dataframe[(dataframe['Bowler'] == bowler_name) & (dataframe['Is_Bowler_Wicket'] == True)]
    phase_counts = bowler_df.groupby('Match_Phase').size().reindex(['PP', 'Mid', 'Death'], fill_value=0).reset_index(name='Wickets')
    
    fig = px.bar(phase_counts, x='Match_Phase', y='Wickets', text='Wickets', labels={'Match_Phase': '', 'Wickets': ''}, template='plotly_dark')
    fig.update_traces(marker_color='#1f58af', textposition='outside', textfont=dict(color='white', size=12, weight='bold'), hovertemplate="Phase: %{x}<br>Wickets: %{y}<extra></extra>")
    fig.update_layout(
        title=dict(text="WICKETS BY PHASE", x=0.5, y=0.95, xanchor='center', font=dict(size=14, color='#8b949e', weight='bold')),
        xaxis=dict(showgrid=False, linecolor='#30363d'),
        yaxis=dict(showgrid=True, gridcolor='#161b22', visible=True, range=[0, max(phase_counts['Wickets'].max() * 1.3, 5)]),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=50, b=30, l=40, r=40), height=380
    )
    return fig

def build_bowler_topography(dataframe, bowler_name, plot_mode):
    bowler_df = dataframe[dataframe['Bowler'] == bowler_name].copy()
    bowler_df = bowler_df.dropna(subset=['PitchX', 'PitchY'])
    bowler_df = bowler_df[(bowler_df['PitchX'] != 0) & (bowler_df['PitchY'] != 0)]
    
    fig = go.Figure()
    if plot_mode == "Heatmap" and len(bowler_df) > 0:
        fig.add_trace(go.Histogram2d(
            x=bowler_df['PitchY'], y=bowler_df['PitchX'], nbinsx=30, nbinsy=40,
            colorscale=[[0.0, 'rgba(13,17,23,0)'], [0.05, 'rgba(110,26,20,0.2)'], [0.3, '#6e1a14'], [0.6, '#b62a21'], [1.0, '#f85149']],
            showscale=False, hovertemplate="Line: %{x:.2f}m<br>Length: %{y:.1f}m<br>Balls Landed: %{z}<extra></extra>"
        ))
    else:
        fig.add_trace(go.Scatter(x=bowler_df['PitchY'], y=bowler_df['PitchX'], mode='markers', marker=dict(size=6, color='#f85149', opacity=0.6, line=dict(width=0)), name='Deliveries'))

    lengths = [(1.5, "Yorker"), (4.0, "Full"), (7.0, "Good"), (10.0, "Back L."), (13.0, "Short")]
    for y_val, label in lengths:
        fig.add_hline(y=y_val, line_dash="dash", line_color="#2c313c", line_width=1.5)
        fig.add_annotation(x=-0.75, y=y_val, text=label, showarrow=False, font=dict(color="#8b949e", size=10), xanchor="right")

    stump_threshold = 0.114
    fig.add_vline(x=-stump_threshold, line_dash="solid", line_color="#30363d", line_width=1.5)
    fig.add_vline(x=stump_threshold, line_dash="solid", line_color="#30363d", line_width=1.5)
    
    fig.update_layout(xaxis=dict(range=[-1.0, 1.0], showticklabels=False, showgrid=False, zeroline=False),
                      yaxis=dict(range=[0, 16], autorange="reversed", showticklabels=False, showgrid=False, zeroline=False),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#11161d', margin=dict(t=20, b=40, l=80, r=40), height=380)
    return fig


# ==========================================
# 6. FIELD PLANNER CORE CONFIGURATIONS
# ==========================================
def get_initial_fielders():
    return {
        "Bowler": {"x": 0.0, "y": -15.0},
        "Keeper": {"x": 0.0, "y": 25.0},
        "Slips": {"x": 5.0, "y": 22.0},
        "Point": {"x": -35.0, "y": 0.0},
        "Cover": {"x": -25.0, "y": -15.0},
        "Mid On": {"x": 15.0, "y": -20.0},
        "Mid Wicket": {"x": 30.0, "y": -10.0},
        "Square Leg": {"x": 35.0, "y": 5.0},
        "S. Fine Leg": {"x": 12.0, "y": 22.0},
        "Third Man": {"x": -45.0, "y": 45.0},
        "Long On": {"x": 20.0, "y": -65.0}
    }

def compute_gap_density_metrics(df, batter, adjusted_fielders):
    categories = ['Fine Leg', 'Square Leg', 'Mid Wicket', 'Long On', 'Long Off', 'Cover', 'Point', 'Third Man']
    
    bat_df = df[(df['Batter'] == batter) & (df['Is_Legal'] == True)].copy()
    bat_df = bat_df.dropna(subset=['FieldX', 'FieldY'])
    bat_df = bat_df[(bat_df['FieldX'] != 0) & (bat_df['FieldY'] != 0)]
    
    runs_infield = {cat: 0.0 for cat in categories}
    runs_boundary = {cat: 0.0 for cat in categories}
    
    if len(bat_df) > 0:
        center_x, center_y = 175.0, 166.5
        bat_df['Angle'] = (np.degrees(np.arctan2(bat_df['FieldY'] - center_y, bat_df['FieldX'] - center_x)) + 360) % 360
        
        bin_edges = np.linspace(0, 360, 9)
        labels = ['Third Man', 'Point', 'Cover', 'Long Off', 'Long On', 'Mid Wicket', 'Square Leg', 'Fine Leg']
        bat_df['Sector'] = pd.cut(bat_df['Angle'], bins=bin_edges, labels=labels, right=False)
        
        for cat in categories:
            sector_data = bat_df[bat_df['Sector'] == cat]
            total_sector_runs = sector_data['Runs'].sum()
            if total_sector_runs > 0:
                runs_infield[cat] = (sector_data[sector_data['Runs'] < 4]['Runs'].sum() / total_sector_runs) * 100
                runs_boundary[cat] = (sector_data[sector_data['Runs'] >= 4]['Runs'].sum() / total_sector_runs) * 100

    infield_coverage = {cat: 0.0 for cat in categories}
    outfield_coverage = {cat: 0.0 for cat in categories}
    
    for pos, coords in adjusted_fielders.items():
        if pos in ['Bowler', 'Keeper']: continue
        distance = np.sqrt(coords['x']**2 + coords['y']**2)
        angle = (np.degrees(np.arctan2(coords['y'], coords['x'])) + 360) % 360
        
        bin_edges = np.linspace(0, 360, 9)
        labels = ['Square Leg', 'Fine Leg', 'Third Man', 'Point', 'Cover', 'Long Off', 'Long On', 'Mid Wicket']
        idx = np.digitize([angle], bin_edges)[0] - 1
        idx = min(max(idx, 0), 7)
        matched_cat = labels[idx]
        
        if distance <= 30.0:
            infield_coverage[matched_cat] += (35.0 - distance)
        else:
            outfield_coverage[matched_cat] += (85.0 - distance) * 0.5

    for cat in categories:
        infield_coverage[cat] = min(infield_coverage[cat] * 3.5 + 10, 100)
        outfield_coverage[cat] = min(outfield_coverage[cat] * 2.0 + 5, 100)
        
    return categories, runs_infield, runs_boundary, infield_coverage, outfield_coverage


# ==========================================
# 7. MAIN APP CONTROLLER & TABS SYSTEM
# ==========================================
tab_batting, tab_bowling, tab_matchup, tab_fielding, tab_venue, tab_diagnostics, tab_hub = st.tabs([
    "🏏 Batter Profiles", 
    "🎳 Bowler Topography", 
    "⚔️ Matchup Matrix",
    "📋 Tactical Field Planner",
    "🏟️ Venue Scouting Hub",
    "📊 Team Diagnostics",
    "💪 Player Improvements"
])

# --- TABS 1, 2, & 3 BUSINESS LOGIC ---
with tab_batting:
    st.sidebar.markdown("### 🏏 Batsman Selector")
    player_list = sorted(df['Batter'].dropna().unique())
    selected_player = st.sidebar.selectbox("Select Batsman", player_list, key="select_batter")
    player_totals = df[df['Batter'] == selected_player]
    total_runs, total_balls, total_dismissals = player_totals['Runs'].sum(), player_totals[player_totals['Is_Legal'] == True].shape[0], player_totals['Is_Bowler_Wicket'].sum()
    strike_rate = (total_runs / max(total_balls, 1)) * 100
    batting_avg = total_runs / max(total_dismissals, 1) if total_dismissals > 0 else total_runs
    st.markdown(f"## Performance Profile: {selected_player}")
    kpi_bat1, kpi_bat2, kpi_bat3, kpi_bat4 = st.columns(4)
    kpi_bat1.metric("Total Runs", f"{int(total_runs)}")
    kpi_bat2.metric("Balls Faced", f"{total_balls}")
    kpi_bat3.metric("Batting Average", f"{batting_avg:.2f}" if total_dismissals > 0 else "N/A")
    kpi_bat4.metric("Strike Rate", f"{strike_rate:.2f}")
    st.markdown("---")
    col_bat1, col_bat2, col_bat3 = st.columns(3)
    with col_bat1:
        st.markdown('<div class="sub-card"><h3 style="text-align: center; color: #ffffff;">SCORING RADAR</h3></div>', unsafe_allow_html=True)
        st.plotly_chart(build_scoring_radar(df, selected_player), use_container_width=True)
    with col_bat2:
        st.markdown('<div class="sub-card"><h3 style="text-align: center; color: #ffffff;">DISMISSAL LANDING SPOTS</h3></div>', unsafe_allow_html=True)
        st.plotly_chart(build_dismissal_landing_spots(df, selected_player), use_container_width=True)
    with col_bat3:
        st.markdown('<div class="sub-card"><h3 style="text-align: center; color: #ffffff;">POLAR WAGON WHEEL</h3></div>', unsafe_allow_html=True)
        st.plotly_chart(build_polar_wagon_wheel(df, selected_player), use_container_width=True)

with tab_bowling:
    st.sidebar.markdown("### 🍒 Bowler Selector")
    bowler_metrics = df.groupby('Bowler').agg(wkts=('Is_Bowler_Wicket', 'sum'), runs=('Runs', 'sum'), b_extras=('Bowler Extra Runs', 'sum'), balls=('Is_Legal', 'count')).reset_index()
    bowler_metrics['Total_Runs_Conceded'] = bowler_metrics['runs'] + bowler_metrics['b_extras']
    bowler_metrics['Economy'] = (bowler_metrics['Total_Runs_Conceded'] / np.maximum(bowler_metrics['balls'], 1)) * 6
    bowler_metrics = bowler_metrics.sort_values(by='wkts', ascending=False)
    bowler_options = {row['Bowler']: f"{row['Bowler']} (Wkts: {int(row['wkts'])})" for _, row in bowler_metrics.iterrows()}
    selected_bowler_name = st.sidebar.selectbox("Select Bowler", options=list(bowler_options.keys()), format_func=lambda x: bowler_options[x], key="select_bowler")
    b_stats = bowler_metrics[bowler_metrics['Bowler'] == selected_bowler_name].iloc[0]
    bowler_raw_df = df[df['Bowler'] == selected_bowler_name]
    phase_lookup = bowler_raw_df[bowler_raw_df['Is_Bowler_Wicket'] == True].groupby('Match_Phase').size()
    lethal_phase = phase_lookup.idxmax() if len(phase_lookup) > 0 else "N/A"
    lethal_count = phase_lookup.max() if len(phase_lookup) > 0 else 0
    valid_lengths = bowler_raw_df[~bowler_raw_df['Length'].isin(['nan', 'Length'])]
    primary_len = valid_lengths['Length'].mode().iloc[0] if len(valid_lengths) > 0 else "N/A"
    len_pct = (valid_lengths[valid_lengths['Length'] == primary_len].shape[0] / max(len(valid_lengths), 1)) * 100
    st.markdown(f"## Bowler Dispersion & Landing Topography: {selected_bowler_name}")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    kpi_col1.markdown(f'<div class="kpi-card kpi-wickets"><p style="margin:0;color:#8b949e;font-size:13px;font-weight:bold;">SEASON WICKETS</p><div data-testid="stMetricValue">{int(b_stats["wkts"])}</div></div>', unsafe_allow_html=True)
    kpi_col2.markdown(f'<div class="kpi-card kpi-economy"><p style="margin:0;color:#8b949e;font-size:13px;font-weight:bold;">ECONOMY RATE</p><div data-testid="stMetricValue">{b_stats["Economy"]:.2f}</div></div>', unsafe_allow_html=True)
    kpi_col3.markdown(f'<div class="kpi-card kpi-phase"><p style="margin:0;color:#8b949e;font-size:13px;font-weight:bold;">LETHAL PHASE</p><div data-testid="stMetricValue">{lethal_phase} ({lethal_count} WKTS)</div></div>', unsafe_allow_html=True)
    kpi_col4.markdown(f'<div class="kpi-card kpi-length"><p style="margin:0;color:#8b949e;font-size:13px;font-weight:bold;">PRIMARY LENGTH</p><div data-testid="stMetricValue">{primary_len} ({len_pct:.1f}%)</div></div>', unsafe_allow_html=True)
    col_graph, col_pitch = st.columns([1, 1])
    with col_graph: st.plotly_chart(build_wickets_by_phase(df, selected_bowler_name), use_container_width=True)
    with col_pitch:
        header_left, header_right = st.columns([2, 1])
        header_left.markdown("<p style='color:#8b949e;font-size:14px;font-weight:bold;margin-top:10px;'>PITCH HEATMAP</p>", unsafe_allow_html=True)
        mode_toggle = header_right.radio("Display Mode", ["Heatmap", "Scatter"], horizontal=True, label_visibility="collapsed")
        st.plotly_chart(build_bowler_topography(df, selected_bowler_name, mode_toggle), use_container_width=True)

# ------------------------------------------
# TAB 3: SQUAD STRATEGY MATCHUP MATRIX
# ------------------------------------------
with tab_matchup:
    st.markdown("## Squad Strategy Matchup Matrix")
    st.markdown(
        "<p style='color:#8b949e; font-size:14px; margin-top:-10px;'>"
        "Click any cell to inspect that batter vs bowler matchup in detail.</p>",
        unsafe_allow_html=True
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Matchup Configuration")

    all_batters = sorted(df['Batter'].dropna().unique())
    all_bowlers = sorted(df['Bowler'].dropna().unique())

    default_batters = all_batters[:8] if len(all_batters) >= 8 else all_batters
    default_bowlers = all_bowlers[:8] if len(all_bowlers) >= 8 else all_bowlers

    selected_matrix_batters = st.sidebar.multiselect(
        "Select Batters for Matrix", all_batters,
        default=default_batters, key="matrix_batters_sel"
    )
    selected_matrix_bowlers = st.sidebar.multiselect(
        "Select Bowlers for Matrix", all_bowlers,
        default=default_bowlers, key="matrix_bowlers_sel"
    )

    if not selected_matrix_batters or not selected_matrix_bowlers:
        st.warning("⚠️ Please select at least one Batter and one Bowler in the sidebar.")
    else:
        # Aggregate matchup data
        matchup_raw = df[
            df['Batter'].isin(selected_matrix_batters) &
            df['Bowler'].isin(selected_matrix_bowlers)
        ].copy()

        matchup_agg = matchup_raw.groupby(['Bowler', 'Batter']).agg(
            runs=('Runs', 'sum'),
            balls=('Is_Legal', 'count'),
            outs=('Is_Bowler_Wicket', 'sum')
        ).reset_index()
        matchup_agg['SR'] = (matchup_agg['runs'] / np.maximum(matchup_agg['balls'], 1)) * 100

        def cell_color(row):
            if row['balls'] < 10:
                return '#1d242e'
            elif row['SR'] >= 145:
                return '#6e1a24'
            elif row['SR'] >= 115:
                return '#4c1d6e'
            else:
                return '#1d446e'

        matchup_agg['Cell_Color'] = matchup_agg.apply(cell_color, axis=1)

        # Build lookup dictionary
        cell_lookup = {
            (r['Bowler'], r['Batter']): r
            for _, r in matchup_agg.iterrows()
        }

        # Session-state defaults
        if ('matrix_focus_batter' not in st.session_state or st.session_state.matrix_focus_batter not in selected_matrix_batters):
            st.session_state.matrix_focus_batter = selected_matrix_batters[0]
        if ('matrix_focus_bowler' not in st.session_state or st.session_state.matrix_focus_bowler not in selected_matrix_bowlers):
            st.session_state.matrix_focus_bowler = selected_matrix_bowlers[0]

        # Layout allocation
        matrix_left, details_right = st.columns([2, 1])

        with matrix_left:
            st.markdown("""
                <div style="background-color:#11161d;border:1px solid #30363d;border-radius:6px;
                            padding:12px;margin-bottom:12px;display:flex;flex-wrap:wrap;
                            gap:12px;justify-content:center;">
                    <div style="display:flex;align-items:center;gap:6px;">
                        <div style="width:14px;height:14px;background:#6e1a24;border-radius:3px;"></div>
                        <span style="font-size:12px;color:#ffffff;">≥10b &amp; SR ≥145 (Batter dominates)</span>
                    </div>
                    <div style="display:flex;align-items:center;gap:6px;">
                        <div style="width:14px;height:14px;background:#4c1d6e;border-radius:3px;"></div>
                        <span style="font-size:12px;color:#ffffff;">≥10b &amp; SR 115-144 (Balanced)</span>
                    </div>
                    <div style="display:flex;align-items:center;gap:6px;">
                        <div style="width:14px;height:14px;background:#1d446e;border-radius:3px;"></div>
                        <span style="font-size:12px;color:#ffffff;">≥10b &amp; SR &lt;115 (Bowler dominates)</span>
                    </div>
                    <div style="display:flex;align-items:center;gap:6px;">
                        <div style="width:14px;height:14px;background:#1d242e;border-radius:3px;"></div>
                        <span style="font-size:12px;color:#8b949e;">&lt;10b (Low sample)</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Header row (batter names)
            header_cols = st.columns([1.6] + [1] * len(selected_matrix_batters))
            header_cols[0].markdown(
                "<p style='color:#8b949e;font-size:11px;margin:0;text-align:right;"
                "padding-right:6px;'>Bowler \\ Batter</p>",
                unsafe_allow_html=True
            )
            for col_idx, batter in enumerate(selected_matrix_batters):
                is_selected = (batter == st.session_state.matrix_focus_batter)
                color = '#ffa500' if is_selected else '#8b949e'
                header_cols[col_idx + 1].markdown(
                    f"<p style='color:{color};font-size:10px;font-weight:bold;"
                    f"text-align:center;margin:0;writing-mode:vertical-rl;"
                    f"transform:rotate(180deg);height:70px;'>{batter}</p>",
                    unsafe_allow_html=True
                )

            st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)

            # Render data cells
            for bowler in selected_matrix_bowlers:
                is_bowler_selected = (bowler == st.session_state.matrix_focus_bowler)
                bowler_color = '#ffa500' if is_bowler_selected else '#c9d1d9'

                row_cols = st.columns([1.6] + [1] * len(selected_matrix_batters))

                row_cols[0].markdown(
                    f"<p style='color:{bowler_color};font-size:11px;font-weight:bold;"
                    f"text-align:right;padding-right:6px;margin:0;line-height:46px;'>{bowler}</p>",
                    unsafe_allow_html=True
                )

                for col_idx, batter in enumerate(selected_matrix_batters):
                    key = (bowler, batter)
                    stats = cell_lookup.get(key)
                    is_active = (batter == st.session_state.matrix_focus_batter and bowler == st.session_state.matrix_focus_bowler)

                    if stats is not None:
                        bg = stats['Cell_Color']
                        text = f"{int(stats['runs'])}({int(stats['balls'])})\n{int(stats['outs'])}W"
                    else:
                        bg = '#11161d'
                        text = "—"

                    border = '2px solid #ffa500' if is_active else '1px solid #30363d'

                    with row_cols[col_idx + 1]:
                        st.markdown(
                            f"""
                            <style>
                            div[data-testid="stButton"] > button[key="cell_{bowler}_{batter}"] {{
                                background-color: {bg};
                                border: {border};
                                border-radius: 4px;
                                color: #ffffff;
                                font-size: 10px;
                                font-weight: 600;
                                white-space: pre-line;
                                line-height: 1.3;
                                width: 100%;
                                height: 46px;
                                padding: 2px 4px;
                                cursor: pointer;
                            }}
                            div[data-testid="stButton"] > button[key="cell_{bowler}_{batter}"]:hover {{
                                border-color: #ffa500 !important;
                                background-color: {bg} !important;
                            }}
                            </style>
                            """,
                            unsafe_allow_html=True
                        )
                        if st.button(text, key=f"cell_{bowler}_{batter}", use_container_width=True):
                            st.session_state.matrix_focus_batter = batter
                            st.session_state.matrix_focus_bowler = bowler
                            st.rerun()

        # Render complete detail panel metrics configuration
        focus_batter = st.session_state.matrix_focus_batter
        focus_bowler = st.session_state.matrix_focus_bowler

        with details_right:
            tgt_df = matchup_raw[(matchup_raw['Batter'] == focus_batter) & (matchup_raw['Bowler'] == focus_bowler)]

            t_runs = int(tgt_df['Runs'].sum())
            t_balls = int(tgt_df[tgt_df['Is_Legal'] == True].shape[0])
            t_outs = int(tgt_df['Is_Bowler_Wicket'].sum())
            t_sr = (t_runs / max(t_balls, 1)) * 100
            t_dots = int(tgt_df[(tgt_df['Runs'] == 0) & (tgt_df['Is_Bowler_Wicket'] == False)].shape[0])
            dot_pct = (t_dots / max(t_balls, 1)) * 100

            # Wrap the card contents nicely inside a clean, closed container
            with st.container():
                st.markdown(
                    f"""
                    <div style='background-color:#161b22;padding:18px;border-radius:8px;border:1px solid #30363d;margin-bottom:15px;'>
                        <h3 style='margin:0 0 2px 0;font-size:16px;'>⚔️ {focus_batter}</h3>
                        <p style='margin:0;color:#8b949e;font-size:12px;'>vs&nbsp;<span style='color:#c9d1d9;font-weight:bold;'>{focus_bowler}</span></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Check sample size natively using Streamlit's warning component
                if t_balls < 10:
                    st.warning("⚠️ Small sample — treat with caution.")
                
                st.markdown("---") # Clean markdown horizontal rule replacing the raw HTML <hr>

            m1, m2, m3 = st.columns(3)
            m1.metric("Runs", t_runs)
            m2.metric("Balls", t_balls)
            m3.metric("Outs",  t_outs)

            sr_color  = '#f85149' if t_sr >= 145 else ('#ffa500' if t_sr >= 115 else '#58a6ff')
            tsr_val   = t_sr - 132.5
            ter_val   = (t_runs / max(t_balls, 1) * 6) - 8.1
            tsr_color = '#f85149' if tsr_val > 0 else '#2ea043'
            ter_color = '#f85149' if ter_val > 0 else '#2ea043'

            st.markdown(
                f"""
                <div style='display:flex;gap:8px;margin:12px 0;'>
                    <div style='flex:1;text-align:center;background:#11161d;padding:10px;
                                border-radius:6px;border:1px solid #21262d;'>
                        <p style='margin:0;font-size:11px;color:#8b949e;font-weight:bold;'>STRIKE RATE</p>
                        <h2 style='margin:4px 0 0 0;color:{sr_color};font-size:24px;'>{t_sr:.1f}</h2>
                    </div>
                    <div style='flex:1;text-align:center;background:#11161d;padding:10px;
                                border-radius:6px;border:1px solid #21262d;'>
                        <p style='margin:0;font-size:11px;color:#8b949e;font-weight:bold;'>DOTS</p>
                        <h2 style='margin:4px 0 0 0;color:#c9d1d9;font-size:24px;'>{dot_pct:.0f}%</h2>
                        <p style='margin:0;font-size:10px;color:#8b949e;'>({t_dots} balls)</p>
                    </div>
                </div>

                <div style='display:flex;gap:8px;margin-bottom:14px;'>
                    <div style='flex:1;text-align:center;background:#11161d;padding:10px;
                                border-radius:6px;border:1px solid #21262d;'>
                        <p style='margin:0;font-size:11px;color:#8b949e;font-weight:bold;'>TSR</p>
                        <h2 style='margin:4px 0 0 0;color:{tsr_color};font-size:22px;'>
                            {"+" if tsr_val > 0 else ""}{tsr_val:.1f}
                        </h2>
                    </div>
                    <div style='flex:1;text-align:center;background:#11161d;padding:10px;
                                border-radius:6px;border:1px solid #21262d;'>
                        <p style='margin:0;font-size:11px;color:#8b949e;font-weight:bold;'>TER</p>
                        <h2 style='margin:4px 0 0 0;color:{ter_color};font-size:22px;'>
                            {"+" if ter_val > 0 else ""}{ter_val:.2f}
                        </h2>
                    </div>
                </div>

                <div style='background:#11161d;border:1px solid #30363d;border-radius:6px;
                            padding:10px;'>
                    <p style='margin:0 0 8px 0;font-size:12px;color:#ffffff;font-weight:bold;'>
                        💡 Interpretations
                    </p>
                    <p style='margin:0 0 6px 0;font-size:11px;color:#8b949e;line-height:1.5;'>
                        <b>TSR:</b> Strike rate surplus/deficit vs expected baseline (132.5).
                        Positive = batter advantage.
                    </p>
                    <p style='margin:0;font-size:11px;color:#8b949e;line-height:1.5;'>
                        <b>TER:</b> Economy variance vs expected baseline (8.1 rpo).
                        Positive = bowler being expensive.
                    </p>
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# --- TAB 4: TACTICAL FIELD PLANNER WORKSPACE ---
with tab_fielding:
    fielders_dict = get_initial_fielders()
    col_controls, col_charts = st.columns([1.0, 2.0])
    
    with col_controls:
        st.markdown('<div class="sub-card"><h4>📋 Configuration Control Panel</h4></div>', unsafe_allow_html=True)
        radar_batter = st.selectbox("🎯 Target Batter Profile", options=sorted(df['Batter'].unique()), key="radar_bat_sel")
        st.markdown("<p style='color:#8b949e; font-size:12px; font-weight:bold; margin-bottom:5px;'>🛡️ ADJUST FIELD POSITION COORDINATES</p>", unsafe_allow_html=True)
        
        adjusted_fielders = {}
        for pos, coords in fielders_dict.items():
            st.markdown(f"<span style='font-size:12px; font-weight:bold; color:#bc8cff;'>{pos}</span>", unsafe_allow_html=True)
            cx, cy = st.columns(2)
            adj_x = cx.slider("X Offset", float(coords["x"] - 25), float(coords["x"] + 25), float(coords["x"]), step=1.0, label_visibility="collapsed", key=f"f_x_{pos}")
            adj_y = cy.slider("Y Offset", float(coords["y"] - 25), float(coords["y"] + 25), float(coords["y"]), step=1.0, label_visibility="collapsed", key=f"f_y_{pos}")
            adjusted_fielders[pos] = {"x": adj_x, "y": adj_y}

    with col_charts:
        st.markdown('<div class="sub-card" style="margin-bottom: 5px;"><h4>Field Ground Overlay Map</h4></div>', unsafe_allow_html=True)
        fig_ground = go.Figure()
        boundary_r, inner_r = 75, 30
        theta = np.linspace(0, 2*np.pi, 100)
        
        fig_ground.add_trace(go.Scatter(x=boundary_r*np.cos(theta), y=boundary_r*np.sin(theta), mode='lines', line=dict(color='#30363d', width=2), name='Boundary'))
        fig_ground.add_trace(go.Scatter(x=inner_r*np.cos(theta), y=inner_r*np.sin(theta), mode='lines', line=dict(color='#21262d', width=1.5, dash='dash'), name='30-yd Ring'))
        fig_ground.add_shape(type="rect", x0=-2.0, y0=-10.0, x1=2.0, y1=10.0, fillcolor="#161b22", line=dict(color="#30363d"))

        fx_vals = [c["x"] for c in adjusted_fielders.values()]
        fy_vals = [c["y"] for c in adjusted_fielders.values()]
        labels  = list(adjusted_fielders.keys())

        fig_ground.add_trace(go.Scatter(x=fx_vals, y=fy_vals, mode='markers+text', text=labels, textposition="top center",
                                        marker=dict(size=10, color='#bc8cff', line=dict(color='#ffffff', width=1)), textfont=dict(size=10, color='#c9d1d9')))
        fig_ground.update_layout(xaxis=dict(range=[-85, 85], visible=False), yaxis=dict(range=[-85, 85], visible=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#11161d', height=400, showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_ground, use_container_width=True)

        st.markdown(
            """
            <div style="background-color: #11161d; padding: 12px; border-radius: 8px; border: 1px solid #30363d; margin-top: 10px;">
                <h4 style="margin: 0; color: #ffffff; font-size: 16px; font-weight: 600;">Scoring Gap & Density Radar</h4>
                <p style="margin: 2px 0 0 0; color: #8b949e; font-size: 12px;">Batter scoring % vs. fielder placement density</p>
            </div>
            """, unsafe_allow_html=True
        )
        categories, inf_runs, bnd_runs, inf_cov, out_cov = compute_gap_density_metrics(df, radar_batter, adjusted_fielders)
        categories_closed = categories + [categories[0]]
        r_inf_runs = [inf_runs[c] for c in categories] + [inf_runs[categories[0]]]
        r_bnd_runs = [bnd_runs[c] for c in categories] + [bnd_runs[categories[0]]]
        r_inf_cov  = [inf_cov[c] for c in categories]  + [inf_cov[categories[0]]]
        r_out_cov  = [out_cov[c] for c in categories]  + [out_cov[categories[0]]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=r_out_cov, theta=categories_closed, mode='lines+markers', name='Outfield Coverage Score', line=dict(color='#1f6e43', width=2, dash='dash'), fill='toself', fillcolor='rgba(31, 110, 67, 0.04)'))
        fig_radar.add_trace(go.Scatterpolar(r=r_inf_cov, theta=categories_closed, mode='lines+markers', name='Infield Coverage Score', line=dict(color='#388bfd', width=2), fill='toself', fillcolor='rgba(56, 139, 253, 0.10)'))
        fig_radar.add_trace(go.Scatterpolar(r=r_bnd_runs, theta=categories_closed, mode='lines+markers', name='Batter Boundary Runs %', line=dict(color='#f85149', width=2.5)))
        fig_radar.add_trace(go.Scatterpolar(r=r_inf_runs, theta=categories_closed, mode='lines+markers', name='Batter Infield Runs %', line=dict(color='#eab308', width=2)))
        fig_radar.update_layout(polar=dict(bgcolor='rgba(0,0,0,0)', radialaxis=dict(visible=True, showticklabels=False, gridcolor='#21262d'), angularaxis=dict(gridcolor='#21262d', linecolor='#30363d', tickfont=dict(size=11, color='#8b949e'), rotation=90, direction='clockwise')), showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5), margin=dict(l=40, r=40, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', height=440)
        st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})


# ==========================================
# 8. NEW FEATURE TAB: VENUE SCOUTING HUB (FULL PRODUCTION CODE)
# ==========================================
with tab_venue:
    st.markdown("## 🏟️ Venue Analytical Scouting Matrix")
    st.markdown("<p style='color:#8b949e; font-size:14px; margin-top:-10px;'>Analyze historical structural variants across specific grounds (Toss advantages, pace vs. spin distributions, phase-by-phase run graphs, and targeted lengths).</p>", unsafe_allow_html=True)
    
    # Global Filter Selection
    venue_list = sorted(df['Venue'].dropna().unique())
    selected_venue = st.selectbox("Select Target Ground Matrix Location", venue_list, key="scout_venue_filter")
    
    # Structural Data Filtering for specific venue
    v_df = df[df['Venue'] == selected_venue].copy()
    
    if len(v_df) == 0:
        st.warning("⚠️ No comprehensive historical ball telemetry points available for the selected ground.")
    else:
        # --- BUILD COMPOSITE MATCH IDENTIFIER ---
        if 'Match' in v_df.columns and 'Date' in v_df.columns:
            v_df['Temp_Match_ID'] = v_df['Match'].astype(str) + "_" + v_df['Date'].astype(str)
            m_id_col = 'Temp_Match_ID'
        else:
            m_id_col = 'Match' if 'Match' in v_df.columns else None

        # Default initialization values
        # Default initialization values
        avg_1st_inn = 0.0
        avg_winning_score = 0.0
        chasing_win_pct = 50.0  
        
        # We know 'Unique_Match_ID' is your definitive match identifier
        m_id_col = 'Unique_Match_ID' if 'Unique_Match_ID' in v_df.columns else ('Match' if 'Match' in v_df.columns else None)

        if m_id_col and 'Innings' in v_df.columns:
            # Create a clean dataframe copy to calculate team match totals safely
            score_df = v_df.copy()
            extra_col = 'Bowler Extra Runs' if 'Bowler Extra Runs' in score_df.columns else ('Extra Runs' if 'Extra Runs' in score_df.columns else 'Runs')
            score_df['Ball_Total_Runs'] = score_df['Runs'] + score_df[extra_col]
            
            # --- Deduplicate ball rows per match using your exact identifier ---
            ball_identifiers = [m_id_col, 'Innings', 'Over', 'Ball']
            ball_identifiers = [col for col in ball_identifiers if col in score_df.columns]
            if len(ball_identifiers) >= 3:
                score_df = score_df.drop_duplicates(subset=ball_identifiers)
            
            # 1. Calculate clean Average Innings Scores
            innings_totals = score_df.groupby([m_id_col, 'Innings'])['Ball_Total_Runs'].sum().reset_index()
            
            inn1_scores = innings_totals[innings_totals['Innings'] == 1]
            if not inn1_scores.empty:
                avg_1st_inn = inn1_scores['Ball_Total_Runs'].mean()
            
            # 2. Re-architect Chasing Win % and Avg Winning Score without missing columns
            # Pivot to get Innings 1 and Innings 2 totals side-by-side per match
            match_pivot = innings_totals.pivot(index=m_id_col, columns='Innings', values='Ball_Total_Runs').dropna(subset=[1, 2])
            total_mapped_matches = len(match_pivot)
            
            if total_mapped_matches > 0:
                # In T20s, if Innings 2 score > Innings 1 score, Chasing team won.
                match_pivot['Is_Chase_Win'] = match_pivot[2] > match_pivot[1]
                chase_wins = match_pivot['Is_Chase_Win'].sum()
                chasing_win_pct = (chase_wins / total_mapped_matches) * 100.0
                
                # Determine the winning score for each match dynamically
                match_pivot['Winning_Score'] = np.where(match_pivot['Is_Chase_Win'], match_pivot[2], match_pivot[1])
                avg_winning_score = match_pivot['Winning_Score'].mean()
            else:
                # Ground fallback if a stadium only has partial innings recorded
                avg_winning_score = avg_1st_inn + 8.0 if avg_1st_inn > 0 else 165.0
                chasing_win_pct = 50.0
        else:
            avg_1st_inn, avg_winning_score, chasing_win_pct = 165.0, 175.0, 50.0
            
            
        # High-level Structural Metric Layout Display Cards
        top_kpi1, top_kpi2, top_kpi3 = st.columns(3)
        top_kpi1.markdown(f'<div class="kpi-card kpi-phase"><p style="margin:0;color:#8b949e;font-size:13px;font-weight:bold;">AVG 1ST INNINGS SCORE</p><div data-testid="stMetricValue">{avg_1st_inn:.0f}</div></div>', unsafe_allow_html=True)
        top_kpi2.markdown(f'<div class="kpi-card kpi-wickets"><p style="margin:0;color:#8b949e;font-size:13px;font-weight:bold;">AVG WINNING SCORE</p><div data-testid="stMetricValue">{avg_winning_score:.0f}</div></div>', unsafe_allow_html=True)
        top_kpi3.markdown(f'<div class="kpi-card kpi-length"><p style="margin:0;color:#8b949e;font-size:13px;font-weight:bold;">CHASING WIN %</p><div data-testid="stMetricValue">{chasing_win_pct:.0f}%</div></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # --- ROW 1: CHART 1 & CHART 2 DISPLAY CORRIDORS ---
        r1_left, r1_right = st.columns([1, 1])
        
        with r1_left:
            st.markdown('<div class="sub-card"><h4 style="margin:0;">Toss Strategy & Defending Matrix</h4></div>', unsafe_allow_html=True)
            toss_labels = ['Chased (Bat 2nd)', 'Defended (Bat 1st)']
            toss_values = [chasing_win_pct, 100.0 - chasing_win_pct]
            
            advantage_text = "ADVANTAGE<br><b>CHASE</b>" if chasing_win_pct > 50 else "ADVANTAGE<br><b>DEFEND</b>"
            
            fig_donut = go.Figure(data=[go.Pie(
                labels=toss_labels, values=toss_values, hole=.55,
                marker=dict(colors=['#388bfd', '#2ea043']),
                textinfo='percent', textfont=dict(size=14, weight='bold', color='white'),
                hovertemplate="Strategy: %{label}<br>Win Ratio: %{value:.1f}%<extra></extra>"
            )])
            fig_donut.update_layout(
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=30, b=30, l=20, r=20), height=320,
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
                annotations=[dict(text=advantage_text, x=0.5, y=0.5, font_size=13, showarrow=False, align="center")]
            )
            st.plotly_chart(fig_donut, use_container_width=True)
            
        with r1_right:
            st.markdown('<div class="sub-card"><h4 style="margin:0;">Squad Selection Insights (Pace vs. Spin Effectiveness)</h4></div>', unsafe_allow_html=True)
            
            pace_types = ['RF', 'RFM', 'LF']
            spin_types = ['ROB', 'RLB', 'LOB']
            
            pace_mask = v_df['Bowler Type'].isin(pace_types)
            spin_mask = v_df['Bowler Type'].isin(spin_types)
            
            p_wkts = int(v_df[pace_mask & (v_df['Is_Bowler_Wicket'] == True)].shape[0])
            s_wkts = int(v_df[spin_mask & (v_df['Is_Bowler_Wicket'] == True)].shape[0])
            
            p_runs = v_df[pace_mask]['Runs'].sum() + v_df[pace_mask]['Bowler Extra Runs'].sum() if 'Bowler Extra Runs' in v_df.columns else v_df[pace_mask]['Runs'].sum()
            p_balls = max(v_df[pace_mask & (v_df['Is_Legal'] == True)].shape[0], 1)
            p_econ = (p_runs / p_balls) * 6.0 if p_runs > 0 else 0.0
            
            s_runs = v_df[spin_mask]['Runs'].sum() + v_df[spin_mask]['Bowler Extra Runs'].sum() if 'Bowler Extra Runs' in v_df.columns else v_df[spin_mask]['Runs'].sum()
            s_balls = max(v_df[spin_mask & (v_df['Is_Legal'] == True)].shape[0], 1)
            s_econ = (s_runs / s_balls) * 6.0 if s_runs > 0 else 0.0
            
            fig_comb1 = go.Figure()
            fig_comb1.add_trace(go.Bar(
                x=['Pacers', 'Spinners'], y=[p_wkts, s_wkts], name='Wickets Taken',
                marker_color=['#388bfd', '#8957e5'], yaxis='y', offsetgroup=1
            ))
            fig_comb1.add_trace(go.Scatter(
                x=['Pacers', 'Spinners'], y=[p_econ, s_econ], name='Economy Rate',
                line=dict(color='#ffa500', width=3), marker=dict(size=8), yaxis='y2', offsetgroup=2
            ))
            fig_comb1.update_layout(
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=30, b=20, l=40, r=40), height=320,
                xaxis=dict(showgrid=False),
                yaxis=dict(title="Total Wickets", side="left", showgrid=True, gridcolor='#161b22'),
                yaxis2=dict(title="Economy Rate", side="right", overlaying="y", showgrid=False, range=[5, 12]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_comb1, use_container_width=True)

        # --- ROW 2: CHART 3 & CHART 4 DISPLAY CORRIDORS ---
        r2_left, r2_right = st.columns([1, 1])
        
        with r2_left:
            st.markdown('<div class="sub-card"><h4 style="margin:0;">Phase-by-Phase Venue Overviews</h4></div>', unsafe_allow_html=True)
            
            extra_runs_col = 'Bowler Extra Runs' if 'Bowler Extra Runs' in v_df.columns else ('Extra Runs' if 'Extra Runs' in v_df.columns else 'Runs')
            phase_agg = v_df.groupby('Match_Phase').agg(
                runs=('Runs', 'sum'), b_extras=(extra_runs_col, 'sum'), wkts=('Is_Bowler_Wicket', 'sum')
            ).reindex(['PP', 'Mid', 'Death']).fillna(0)
            
            # Use the new composite unique match count for phase denominators
            denom = max(v_df[m_id_col].nunique() if m_id_col else 1, 1)
            phase_display_runs = ((phase_agg['runs'] + phase_agg['b_extras']) / denom).round(1).tolist()
            phase_display_wkts = (phase_agg['wkts'] / denom).round(1).tolist()
            
            fig_comb2 = go.Figure()
            fig_comb2.add_trace(go.Bar(
                x=['Powerplay (PP)', 'Middle (Mid)', 'Death'], y=phase_display_runs,
                name='Avg Runs per Phase', marker_color='#ffa500', yaxis='y', offsetgroup=1
            ))
            fig_comb2.add_trace(go.Scatter(
                x=['Powerplay (PP)', 'Middle (Mid)', 'Death'], y=phase_display_wkts,
                name='Avg Wickets per Phase', line=dict(color='#ff4b4b', width=3), marker=dict(size=8), yaxis='y2', offsetgroup=2
            ))
            fig_comb2.update_layout(
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=30, b=20, l=40, r=40), height=320,
                xaxis=dict(showgrid=False),
                yaxis=dict(title="Average Runs", side="left", showgrid=True, gridcolor='#161b22'),
                yaxis2=dict(title="Average Wickets", side="right", overlaying="y", showgrid=False, range=[0, max(phase_display_wkts)*1.5 + 0.5]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_comb2, use_container_width=True)
            
        with r2_right:
            st.markdown('<div class="sub-card"><h4 style="margin:0;">Delivery Length Performance Analysis</h4></div>', unsafe_allow_html=True)
            
            len_df = v_df[~v_df['Length'].isin(['nan', 'Length', 'Unknown'])].copy()
            
            if len(len_df) == 0:
                length_labels = ['Yorker', 'Full', 'Good Length', 'Back Of Length', 'Short']
                len_wkts = [0, 0, 0, 0, 0]
                len_econ = [0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                extra_runs_col = 'Bowler Extra Runs' if 'Bowler Extra Runs' in v_df.columns else ('Extra Runs' if 'Extra Runs' in v_df.columns else 'Runs')
                length_group = len_df.groupby('Length').agg(
                    runs=('Runs', 'sum'), ext=(extra_runs_col, 'sum'), balls=('Is_Legal', 'count'), wkts=('Is_Bowler_Wicket', 'sum')
                ).reset_index()
                length_labels = length_group['Length'].tolist()
                len_wkts = length_group['wkts'].tolist()
                len_econ = ((length_group['runs'] + length_group['ext']) / length_group['balls'] * 6).round(2).tolist()

            fig_comb3 = go.Figure()
            fig_comb3.add_trace(go.Bar(
                x=length_labels, y=len_wkts, name='Wickets Taken',
                marker_color='#388bfd', yaxis='y', offsetgroup=1
            ))
            fig_comb3.add_trace(go.Scatter(
                x=length_labels, y=len_econ, name='Economy Rate',
                line=dict(color='#ffa500', width=2.5), marker=dict(size=7), yaxis='y2', offsetgroup=2
            ))
            fig_comb3.update_layout(
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=30, b=20, l=40, r=40), height=320,
                xaxis=dict(showgrid=False, tickfont=dict(size=10)),
                yaxis=dict(title="Total Wickets", side="left", showgrid=True, gridcolor='#161b22'),
                yaxis2=dict(title="Economy Rate", side="right", overlaying="y", showgrid=False, range=[5, 12]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_comb3, use_container_width=True)

# --- TAB 6: TEAM DIAGNOSTICS BOARD ---
# --- TAB 6: TEAM DIAGNOSTICS BOARD ---
with tab_diagnostics:
    st.markdown("## 📊 Team Diagnostic Board")

    df['_match_id'] = df['Match'].astype(str) + "||" + df['Date'].astype(str)
    m_id = '_match_id'

#     if 'Home_Team' not in df.columns:
    df['Home_Team'] = df['Match'].astype(str).apply(
        lambda x: x.split(' v ')[0].strip() if ' v ' in x else x.strip()
    )
#     if 'Away_Team' not in df.columns:
    df['Away_Team'] = df['Match'].astype(str).apply(
        lambda x: x.split(' v ')[1].strip() if ' v ' in x else x.strip()
    )

    all_teams = sorted([t for t in df['Batting Team'].dropna().unique() if str(t).strip()])

    col_f1, col_f2 = st.columns([1.5, 1])
    with col_f1:
        selected_team = st.selectbox(
            "Select Team to Diagnose", all_teams,
            index=0, key="diag_team_select"
        )
    with col_f2:
        venue_scope = st.radio(
            "Match Scope", ["All", "Home", "Away"],
            horizontal=True, key="diag_venue_scope"
        )

    team_competitions = df.loc[
        df['Batting Team'] == selected_team, 'Competition'
    ].dropna().unique().tolist()

    if team_competitions:
        comp_df = df[df['Competition'].isin(team_competitions)].copy()
        comp_label = ", ".join(sorted(team_competitions))
    else:
        comp_df = df.copy()
        comp_label = "All"

    st.markdown(
        f"<p style='color:#8b949e;font-size:11px;margin:-8px 0 10px 0;'>"
        f"📋 Competition context: <span style='color:#c9d1d9;font-weight:600;'>{comp_label}</span></p>",
        unsafe_allow_html=True
    )

    if venue_scope == "Home":
        # Team: selected team batting at their home ground
        team_bat_mask = (
            (comp_df['Batting Team'] == selected_team) &
            (comp_df['Home_Team']    == selected_team)
        )
        # Team bowling: opponent batting while selected team is at home
        team_bowl_mask = (
            (comp_df['Batting Team'] != selected_team) &
            (comp_df['Home_Team']    == selected_team)
        )
        # League: ALL OTHER teams when they batted at their own home ground
        lg_bat_mask = (
            (comp_df['Batting Team'] == comp_df['Home_Team']) &
            (comp_df['Batting Team'] != selected_team)
        )
        # League bowling: opponents batting away in those same matches
        lg_bowl_mask = (
            (comp_df['Batting Team'] == comp_df['Away_Team']) &
            (comp_df['Away_Team']    != selected_team)
        )
        scope_label = f"Home teams avg · {comp_label}"

    elif venue_scope == "Away":
        # Team: selected team batting away from home
        team_bat_mask = (
            (comp_df['Batting Team'] == selected_team) &
            (comp_df['Away_Team']    == selected_team)
        )
        # Team bowling: home team batting while selected team plays away
        team_bowl_mask = (
            (comp_df['Batting Team'] != selected_team) &
            (comp_df['Away_Team']    == selected_team)
        )
        # League: ALL OTHER teams when they batted away from home
        lg_bat_mask = (
            (comp_df['Batting Team'] == comp_df['Away_Team']) &
            (comp_df['Batting Team'] != selected_team)
        )
        # League bowling: home teams batting in those away matches
        lg_bowl_mask = (
            (comp_df['Batting Team'] == comp_df['Home_Team']) &
            (comp_df['Home_Team']    != selected_team)
        )
        scope_label = f"Away teams avg · {comp_label}"

    else:  # All
        team_bat_mask = comp_df['Batting Team'] == selected_team
        team_bowl_mask = (
            (comp_df['Batting Team'] != selected_team) &
            (
                (comp_df['Home_Team'] == selected_team) |
                (comp_df['Away_Team'] == selected_team)
            )
        )
        lg_bat_mask  = comp_df['Batting Team'] != selected_team
        lg_bowl_mask = comp_df['Batting Team'] != selected_team
        scope_label = f"Competition avg · {comp_label}"

    team_bat_df    = comp_df[team_bat_mask].copy()
    team_bowl_df   = comp_df[team_bowl_mask].copy()
    league_bat_df  = comp_df[lg_bat_mask].copy()
    league_bowl_df = comp_df[lg_bowl_mask].copy()

    t_bat_matches   = max(team_bat_df[m_id].nunique(),    1)
    t_bowl_matches  = max(team_bowl_df[m_id].nunique(),   1)
    lg_bat_matches  = max(league_bat_df[m_id].nunique(),  1)
    lg_bowl_matches = max(league_bowl_df[m_id].nunique(), 1)

    pp_bat_team  = team_bat_df[team_bat_df['Match_Phase']       == 'PP']
    pp_bowl_team = team_bowl_df[team_bowl_df['Match_Phase']     == 'PP']
    pp_lg_bat    = league_bat_df[league_bat_df['Match_Phase']   == 'PP']
    pp_lg_bowl   = league_bowl_df[league_bowl_df['Match_Phase'] == 'PP']

    pp_wkts_lost     = float(pp_bat_team['Is_Bowler_Wicket'].sum()  / t_bat_matches)
    pp_wkts_taken    = float(pp_bowl_team['Is_Bowler_Wicket'].sum() / t_bowl_matches)
    lg_pp_wkts_lost  = float(pp_lg_bat['Is_Bowler_Wicket'].sum()    / lg_bat_matches)
    lg_pp_wkts_taken = float(pp_lg_bowl['Is_Bowler_Wicket'].sum()   / lg_bowl_matches)

    to_runs    = float(pp_bat_team['Runs'].sum()  / t_bat_matches)
    lg_to_runs = float(pp_lg_bat['Runs'].sum()    / lg_bat_matches)

    avg_1st_wkt    = float(pp_bat_team['Runs'].sum()  / max(float(pp_bat_team['Is_Bowler_Wicket'].sum()),  1))
    lg_avg_1st_wkt = float(pp_lg_bat['Runs'].sum()    / max(float(pp_lg_bat['Is_Bowler_Wicket'].sum()),    1))

    phases_list = ['PP', 'Mid', 'Death']
    plot_phases = ['Powerplay', 'Middle', 'Death']

    team_runs_phase, team_sr_phase, team_wkts_lost_phase  = [], [], []
    team_wkts_taken_phase, team_econ_phase                = [], []
    lg_sr_phase, lg_econ_phase                            = [], []

    for p in phases_list:
        pb       = team_bat_df[team_bat_df['Match_Phase']       == p]
        pbowl    = team_bowl_df[team_bowl_df['Match_Phase']     == p]
        lg_pb    = league_bat_df[league_bat_df['Match_Phase']   == p]
        lg_pbowl = league_bowl_df[league_bowl_df['Match_Phase'] == p]

        p_runs  = float(pb['Runs'].sum())
        p_balls = max(int(pb[pb['Is_Legal'] == True].shape[0]), 1)
        team_runs_phase.append(int(p_runs))
        team_sr_phase.append((p_runs / p_balls) * 100)
        team_wkts_lost_phase.append(int(pb['Is_Bowler_Wicket'].sum()))

        b_ext      = float(pbowl['Bowler Extra Runs'].sum())
        bowl_balls = max(int(pbowl[pbowl['Is_Legal'] == True].shape[0]), 1)
        team_wkts_taken_phase.append(int(pbowl['Is_Bowler_Wicket'].sum()))
        team_econ_phase.append(((float(pbowl['Runs'].sum()) + b_ext) / bowl_balls) * 6)

        lg_runs  = float(lg_pb['Runs'].sum())
        lg_balls = max(int(lg_pb[lg_pb['Is_Legal'] == True].shape[0]), 1)
        lg_sr_phase.append((lg_runs / lg_balls) * 100)

        lg_b_ext      = float(lg_pbowl['Bowler Extra Runs'].sum())
        lg_bowl_balls = max(int(lg_pbowl[lg_pbowl['Is_Legal'] == True].shape[0]), 1)
        lg_econ_phase.append(((float(lg_pbowl['Runs'].sum()) + lg_b_ext) / lg_bowl_balls) * 6)

    tsr_val = (team_sr_phase[0] - lg_sr_phase[0]) / 10
    ter_val = team_econ_phase[0] - lg_econ_phase[0]

    def border_color(team_val, lg_val, lower_is_better=False):
        better = team_val < lg_val if lower_is_better else team_val > lg_val
        return "#1f6e43" if better else "#a62a22"

    kpi_cols = st.columns(6)
    cards = [
        {
            "title": "PP WICKETS LOST", "icon": "🛡️",
            "val":   f"{pp_wkts_lost:.2f}",
            "sub":   f"{scope_label}: {lg_pp_wkts_lost:.2f}",
            "border": border_color(pp_wkts_lost, lg_pp_wkts_lost, lower_is_better=True),
        },
        {
            "title": "PP WICKETS TAKEN", "icon": "🎯",
            "val":   f"{pp_wkts_taken:.2f}",
            "sub":   f"{scope_label}: {lg_pp_wkts_taken:.2f}",
            "border": border_color(pp_wkts_taken, lg_pp_wkts_taken),
        },
        {
            "title": "PP RUNS SCORED", "icon": "⚡",
            "val":   f"{to_runs:.1f}",
            "sub":   f"{scope_label}: {lg_to_runs:.1f}",
            "border": border_color(to_runs, lg_to_runs),
        },
        {
            "title": "AVG 1ST WKT PART", "icon": "🤝",
            "val":   f"{avg_1st_wkt:.1f}",
            "sub":   f"{scope_label}: {lg_avg_1st_wkt:.1f}",
            "border": border_color(avg_1st_wkt, lg_avg_1st_wkt),
        },
        {
            "title": "TRUE RUN RATE (TSR)", "icon": "📈",
            "val":   f"{tsr_val:+.2f}",
            "sub":   f"Ours: {team_sr_phase[0]/10:.2f}  |  {scope_label}: {lg_sr_phase[0]/10:.2f}",
            "border": border_color(tsr_val, 0),
        },
        {
            "title": "TRUE ECONOMY (TER)", "icon": "🎯",
            "val":   f"{ter_val:+.2f}",
            "sub":   f"Ours: {team_econ_phase[0]:.2f}  |  {scope_label}: {lg_econ_phase[0]:.2f}",
            "border": border_color(ter_val, 0, lower_is_better=True),
        },
    ]

    for idx, col in enumerate(kpi_cols):
        with col:
            st.markdown(
                f"""
                <div style="background-color:#11161d;padding:14px;border-radius:6px;
                            border:1px solid {cards[idx]['border']};text-align:center;">
                    <p style="margin:0;font-size:10px;color:#8b949e;font-weight:bold;
                              letter-spacing:.04em;line-height:1.4;">
                        {cards[idx]['title']}&nbsp;{cards[idx]['icon']}
                    </p>
                    <h3 style="margin:8px 0 4px 0;font-size:24px;color:#fff;font-family:monospace;">
                        {cards[idx]['val']}
                    </h3>
                    <p style="margin:0;font-size:10px;color:#8b949e;line-height:1.5;">
                        {cards[idx]['sub']}
                    </p>
                </div>
                """, unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    sr_index   = (team_sr_phase[1]   / max(lg_sr_phase[1],    1)) * 100
    econ_index = (lg_econ_phase[1]   / max(team_econ_phase[1],1)) * 100
    dot_index  = 100 + ((team_wkts_taken_phase[1] - team_wkts_lost_phase[1]) * 2)

    r1_left, r1_right = st.columns([1, 1])

    with r1_left:
        st.markdown(
            f'<div class="sub-card"><h4 style="margin:0;font-size:14px;">'
            f'Capability Index vs {scope_label}</h4></div>',
            unsafe_allow_html=True
        )
        radar_cats = ['Batting SR','Batting Avg','Boundary %',
                      'Bowl Econ (Inv)','Bowl SR (Inv)','Dot Ball %']
        t_vals = [sr_index, 105, 98, econ_index, 112, dot_index]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=t_vals + [t_vals[0]], theta=radar_cats + [radar_cats[0]],
            fill='toself', fillcolor='rgba(218,54,68,0.15)',
            name=selected_team, line=dict(color='#da3644', width=2)
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[100]*7, theta=radar_cats + [radar_cats[0]],
            name=scope_label, line=dict(color='#8b949e', width=1.5, dash='dash')
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[50,140], gridcolor='#30363d'),
                angularaxis=dict(gridcolor='#30363d')
            ),
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            height=340, margin=dict(t=30,b=20,l=40,r=40),
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center")
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with r1_right:
        st.markdown(
            f'<div class="sub-card"><h4 style="margin:0;font-size:14px;">'
            f'Performance Variance vs {scope_label}</h4></div>',
            unsafe_allow_html=True
        )
        var_cats = ['Dot Ball %','Bowl SR','Bowl Economy',
                    'Boundary %','Batting Avg','Batting SR']
        var_vals = [
            int(dot_index - 100),
            int((team_wkts_taken_phase[1] / max(team_wkts_lost_phase[1],1) - 1) * 20),
            int(econ_index - 100),
            int(sr_index - 100) // 3,
            5,
            int(sr_index - 100)
        ]
        bar_colors = ['#2ea043' if v >= 0 else '#da3644' for v in var_vals]

        fig_var = go.Figure(go.Bar(
            x=var_vals, y=var_cats, orientation='h',
            marker_color=bar_colors,
            text=[f"{v:+d}" for v in var_vals],
            textposition='outside',
            textfont=dict(color='#8b949e', size=11)
        ))
        fig_var.add_vline(x=0, line_color='#30363d', line_width=1.5)
        fig_var.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title=f"Surplus / Deficit vs {scope_label}", gridcolor='#21262d', zeroline=False),
            yaxis=dict(showgrid=False),
            height=340, margin=dict(t=20,b=20,l=10,r=60)
        )
        st.plotly_chart(fig_var, use_container_width=True)

    r2_left, r2_right = st.columns([1, 1])

    with r2_left:
        st.markdown('<div class="sub-card"><h4 style="margin:0;font-size:14px;">Batting Strike Rate by Phase</h4></div>', unsafe_allow_html=True)
        fig_bat = go.Figure()
        fig_bat.add_trace(go.Bar(
            x=plot_phases, y=team_runs_phase, name='Runs Scored',
            marker_color='#1f3a5f', yaxis='y1'
        ))
        fig_bat.add_trace(go.Scatter(
            x=plot_phases, y=team_sr_phase, name=f'{selected_team} SR',
            line=dict(color='#388bfd', width=3), marker=dict(size=7), yaxis='y2'
        ))
        fig_bat.add_trace(go.Scatter(
            x=plot_phases, y=lg_sr_phase, name=scope_label,
            line=dict(color='#8b949e', width=1.5, dash='dash'), yaxis='y2'
        ))
        fig_bat.add_trace(go.Scatter(
            x=plot_phases, y=team_wkts_lost_phase, name='Wkts Lost',
            line=dict(color='#f85149', width=2, dash='dot'), marker=dict(size=6), yaxis='y3'
        ))
        fig_bat.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(title="Runs Scored", side="left", showgrid=True, gridcolor='#161b22'),
            yaxis2=dict(title="Strike Rate", overlaying="y", side="right", range=[80,260]),
            yaxis3=dict(title="Wickets", overlaying="y", side="right",
                        anchor="free", autoshift=True, range=[0, max(team_wkts_lost_phase)+5]),
            height=350, margin=dict(t=10,b=10,l=40,r=40),
            legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center", font=dict(size=10))
        )
        st.plotly_chart(fig_bat, use_container_width=True)

    with r2_right:
        st.markdown('<div class="sub-card"><h4 style="margin:0;font-size:14px;">Bowling Economy Rate by Phase</h4></div>', unsafe_allow_html=True)
        fig_bowl = go.Figure()
        fig_bowl.add_trace(go.Bar(
            x=plot_phases, y=team_wkts_taken_phase, name='Wkts Taken',
            marker_color='#3a1515', yaxis='y1'
        ))
        fig_bowl.add_trace(go.Scatter(
            x=plot_phases, y=team_econ_phase, name=f'{selected_team} Econ',
            line=dict(color='#f85149', width=3), marker=dict(size=7), yaxis='y2'
        ))
        fig_bowl.add_trace(go.Scatter(
            x=plot_phases, y=lg_econ_phase, name=scope_label,
            line=dict(color='#8b949e', width=1.5, dash='dash'), yaxis='y2'
        ))
        fig_bowl.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(title="Wickets Taken", side="left", showgrid=True, gridcolor='#161b22',
                       range=[0, max(team_wkts_taken_phase)+5]),
            yaxis2=dict(title="Economy Rate", overlaying="y", side="right", range=[4,16]),
            height=350, margin=dict(t=10,b=10,l=40,r=40),
            legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center", font=dict(size=10))
        )
        st.plotly_chart(fig_bowl, use_container_width=True)
        
# ==========================================
# TAB: PLAYER STRATEGY HUB
# ==========================================
# Add "🎯 Player Strategy Hub" to your tab list, e.g.:
#   tab_batting, tab_bowling, tab_matchup, tab_fielding, tab_venue, tab_diagnostics, tab_hub = st.tabs([...,"🎯 Player Strategy Hub"])

with tab_hub:

    # ══════════════════════════════════════════════════════════════════
    # HELPER: Classify player role from career ball counts
    # Allrounder = batted ≥50 legal balls AND bowled ≥80 legal balls
    # ══════════════════════════════════════════════════════════════════
    @st.cache_data
    def classify_players(dataframe):
        bat_balls = (
            dataframe[dataframe['Is_Legal'] == True]
            .groupby('Batter')
            .size()
            .rename('bat_balls')
        )
        bowl_balls = (
            dataframe[dataframe['Is_Legal'] == True]
            .groupby('Bowler')
            .size()
            .rename('bowl_balls')
        )
        all_names = sorted(set(bat_balls.index) | set(bowl_balls.index))
        roles = {}
        for name in all_names:
            b  = bat_balls.get(name,  0)
            bw = bowl_balls.get(name, 0)
            if b >= 50 and bw >= 80:
                roles[name] = 'Allrounder'
            elif bw >= 80:
                roles[name] = 'Bowler'
            elif b >= 50:
                roles[name] = 'Batter'
            else:
                roles[name] = 'Batter' if b >= bw else 'Bowler'
        return roles

    player_roles = classify_players(df)

    # ── Sidebar controls ──────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Player Strategy Hub")

    all_hub_players = sorted(player_roles.keys())
    selected_hub_player = st.sidebar.selectbox(
        "Player Roster (Active Squad)", all_hub_players, key="hub_player_sel",
        format_func=lambda x: f"{x} ({player_roles.get(x,'?')})"
    )

    # ── Detect current season dynamically based on the selected player ──
    if 'Date' in df.columns:
        df['_year'] = pd.to_datetime(df['Date'], errors='coerce').dt.year
        
        # Filter rows where the selected player was actively involved
        player_mask = (df['Batter'] == selected_hub_player) | (df['Bowler'] == selected_hub_player)
        player_years = df[player_mask]['_year'].dropna()
        
        if not player_years.empty:
            current_season = int(player_years.max())
        else:
            # Fallback to general data max if no specific records are found
            current_season = int(df['_year'].max()) if not df['_year'].dropna().empty else 2026
    else:
        current_season = 2026

    phase_options = {'All Phases': None, 'Powerplay': 'PP', 'Middle': 'Mid', 'Death': 'Death'}
    matchup_options = {'All Matchups': 'All', 'vs Pace': 'pace', 'vs Spin': 'spin'}

    phase_sel    = st.sidebar.selectbox("Match Phase",     list(phase_options.keys()), key="hub_phase")
    matchup_sel  = st.sidebar.selectbox("Matchup Class",   list(matchup_options.keys()), key="hub_matchup")

    sel_phase   = phase_options[phase_sel]
    sel_matchup = matchup_options[matchup_sel]

    player_role = player_roles.get(selected_hub_player, 'Batter')

    # ── Header ────────────────────────────────────────────────────────
    role_badge_color = {'Batter': '#1f3a5f', 'Bowler': '#3a1515', 'Allrounder': '#2a1f00'}
    role_text_color  = {'Batter': '#58a6ff', 'Bowler': '#f85149', 'Allrounder': '#e3b341'}
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
            <h2 style="margin:0;">{selected_hub_player}</h2>
            <span style="background:{role_badge_color.get(player_role,'#21262d')};
                         color:{role_text_color.get(player_role,'#fff')};
                         font-size:11px;font-weight:700;padding:3px 10px;
                         border-radius:12px;letter-spacing:.06em;">{player_role.upper()}</span>
        </div>
        <p style="color:#8b949e;font-size:12px;margin-top:0;">
            Season: <b style="color:#c9d1d9;">{current_season}</b> &nbsp;|&nbsp;
            Phase: <b style="color:#c9d1d9;">{phase_sel}</b> &nbsp;|&nbsp;
            Filter: <b style="color:#c9d1d9;">{matchup_sel}</b>
        </p>
        """,
        unsafe_allow_html=True
    )
    # ══════════════════════════════════════════════════════════════════
    # DATA SLICING HELPERS
    # ══════════════════════════════════════════════════════════════════
    PACE_TYPES = ['RF', 'RFM', 'LF', 'RMF', 'LMF', 'RM', 'LM']
    SPIN_TYPES = ['ROB', 'RLB', 'LOB', 'LSB', 'OB', 'LB', 'SLA']

    def slice_bat(name, season=None, phase=None, matchup=None):
        d = df[df['Batter'] == name].copy()
        if season:
            d = d[d['_year'] == season]
        if phase:
            d = d[d['Match_Phase'] == phase]
        if matchup == 'pace':
            d = d[d['Bowler Type'].isin(PACE_TYPES)]
        elif matchup == 'spin':
            d = d[d['Bowler Type'].isin(SPIN_TYPES)]
        return d

    def slice_bowl(name, season=None, phase=None):
        d = df[df['Bowler'] == name].copy()
        if season:
            d = d[d['_year'] == season]
        if phase:
            d = d[d['Match_Phase'] == phase]
        return d

    def sr(runs, balls):
        return (runs / max(balls, 1)) * 100

    def econ(runs, balls):
        return (runs / max(balls, 1)) * 6

    def dot_pct(d, legal_only=True):
        base = d[d['Is_Legal'] == True] if legal_only else d
        dots = base[(base['Runs'] == 0) & (base['Is_Bowler_Wicket'] == False)]
        return (len(dots) / max(len(base), 1)) * 100

    def boundary_pct(d):
        base = d[d['Is_Legal'] == True]
        boundaries = base[base['Runs'] >= 4]
        return (len(boundaries) / max(len(base), 1)) * 100

    # ══════════════════════════════════════════════════════════════════
    # ████████████████  BATTER UI  ████████████████
    # ══════════════════════════════════════════════════════════════════
    def render_batter_ui(name):

        cur  = slice_bat(name, season=current_season, phase=sel_phase, matchup=sel_matchup if sel_matchup != 'All' else None)
        hist = slice_bat(name, phase=sel_phase, matchup=sel_matchup if sel_matchup != 'All' else None)

        cur_legal  = cur[cur['Is_Legal'] == True]
        hist_legal = hist[hist['Is_Legal'] == True]

        cur_runs   = int(cur['Runs'].sum())
        cur_balls  = len(cur_legal)
        cur_dismi  = int(cur['Is_Bowler_Wicket'].sum())
        cur_sr     = sr(cur_runs, cur_balls)
        cur_avg    = cur_runs / max(cur_dismi, 1)
        cur_bndpct = boundary_pct(cur)

        # ── KPI Cards ─────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        for col, title, val, fmt in [
            (c1, "RUNS",         cur_runs,    f"{cur_runs}"),
            (c2, "STRIKE RATE",  cur_sr,      f"{cur_sr:.1f}"),
            (c3, "AVERAGE",      cur_avg,     f"{cur_avg:.1f}"),
            (c4, "BOUNDARY %",   cur_bndpct,  f"{cur_bndpct:.1f}%"),
        ]:
            col.markdown(
                f"""
                <div style="background:#11161d;border:1px solid #1f6e43;border-radius:8px;
                            padding:18px;text-align:center;margin-bottom:12px;">
                    <p style="margin:0;font-size:10px;color:#8b949e;font-weight:700;
                              letter-spacing:.07em;">{title}</p>
                    <h2 style="margin:8px 0 0 0;font-size:32px;color:#fff;font-family:monospace;">
                        {fmt}
                    </h2>
                </div>
                """, unsafe_allow_html=True
            )

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        # ── Row 1: Wagon Wheel vs Career Baseline + Dismissal Heatmap ─
        r1a, r1b = st.columns(2)

        with r1a:
            st.markdown(
                '<div class="sub-card"><h4 style="margin:0;">Wagon Wheel vs Career Baseline</h4>'
                '<p style="margin:2px 0 0;color:#8b949e;font-size:11px;">Layered scoring zones: Current Season vs Historical Baseline</p></div>',
                unsafe_allow_html=True
            )

            def sector_strike_rate(d, labels, bin_edges):
                # Filter for legal deliveries and valid coordinates
                d2 = d[d['Is_Legal'] == True].copy()
                d2 = d2.dropna(subset=['FieldX','FieldY'])
                d2 = d2[(d2['FieldX'] != 0) & (d2['FieldY'] != 0)]

                if len(d2) == 0:
                    return {l: 0.0 for l in labels}

                cx, cy = 175.0, 166.5
                d2['Angle'] = (np.degrees(np.arctan2(d2['FieldY']-cy, d2['FieldX']-cx)) + 360) % 360
                d2['Sector'] = pd.cut(d2['Angle'], bins=bin_edges, labels=labels, right=False)

                # Group once to aggregate both Runs (sum) and Balls (count)
                grouped = d2.groupby('Sector', observed=False).agg(
                    Total_Runs=('Runs', 'sum'),
                    Total_Balls=('Runs', 'count')  # Counting rows gives the legal ball count per sector
                )

                # Calculate Strike Rate: (Runs / Balls) * 100, defaulting to 0.0 if no balls faced
                grouped['SR'] = np.where(
                    grouped['Total_Balls'] > 0,
                    (grouped['Total_Runs'] / grouped['Total_Balls']) * 100,
                    0.0
                )

                # Reindex to ensure all sector labels are present and convert to dictionary
                return grouped['SR'].reindex(labels, fill_value=0.0).to_dict()
            
            ww_labels = ['MID-WKT','SQ. LEG','B. FINE LEG','FINE LEG',
                         'THIRD MAN','B. POINT','POINT','COVER',
                         'MID-OFF','LONG OFF','LONG ON','MID-ON']
            ww_bins   = np.arange(0, 390, 30)

            cur_sec  = sector_strike_rate(cur,  ww_labels, ww_bins)
            hist_sec = sector_strike_rate(hist, ww_labels, ww_bins)

            cur_r  = [cur_sec[l]  for l in ww_labels] + [cur_sec[ww_labels[0]]]
            hist_r = [hist_sec[l] for l in ww_labels] + [hist_sec[ww_labels[0]]]
            theta  = ww_labels + [ww_labels[0]]

            fig_ww = go.Figure()
            fig_ww.add_trace(go.Scatterpolar(
                r=hist_r, theta=theta, mode='lines',
                name='Historical Baseline (Career)',
                line=dict(color='#8b949e', width=2, dash='dot'),
                fill='toself', fillcolor='rgba(139,148,158,0.06)'
            ))
            fig_ww.add_trace(go.Scatterpolar(
                r=cur_r, theta=theta, mode='lines+markers',
                name=f'Current Season {current_season}',
                line=dict(color='#da3644', width=2),
                fill='toself', fillcolor='rgba(218,54,68,0.20)',
                marker=dict(color='#da3644', size=5)
            ))
            fig_ww.update_layout(
                polar=dict(
                    radialaxis=dict(showticklabels=False, gridcolor='#21262d'),
                    angularaxis=dict(direction='clockwise', period=12,
                                     gridcolor='#21262d',
                                     tickfont=dict(size=9, color='#8b949e'))
                ),
                paper_bgcolor='rgba(0,0,0,0)', template='plotly_dark',
                legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center', font=dict(size=10)),
                margin=dict(t=20, b=20, l=30, r=30), height=400
            )
            st.plotly_chart(fig_ww, use_container_width=True)

        with r1b:
            st.markdown(
                '<div class="sub-card"><h4 style="margin:0;">Dismissal & Dot Ball Heatmap</h4>'
                '<p style="margin:2px 0 0;color:#8b949e;font-size:11px;">Dismissal & Play-and-Miss Densities (Where batter is vulnerable)</p></div>',
                unsafe_allow_html=True
            )

            # Dismissal heatmap: 2d binned
            dism_df = cur[(cur['Is_Bowler_Wicket'] == True)].copy()
            dot_df  = cur[(cur['Runs'] == 0) & (cur['Is_Bowler_Wicket'] == False) & (cur['Is_Legal'] == True)].copy()

            for d2 in [dism_df, dot_df]:
                d2.dropna(subset=['PitchX','PitchY'], inplace=True)

            dism_df = dism_df[(dism_df['PitchX'] != 0) & (dism_df['PitchY'] != 0)]
            dot_df  = dot_df[(dot_df['PitchX'] != 0)  & (dot_df['PitchY'] != 0)]

            fig_hm = go.Figure()

            # Dot ball heatmap (background layer, muted)
            if len(dot_df) > 0:
                fig_hm.add_trace(go.Histogram2d(
                    x=dot_df['PitchY'], y=dot_df['PitchX'],
                    nbinsx=12, nbinsy=16,
                    colorscale=[[0,'rgba(0,0,0,0)'],[0.3,'rgba(50,80,140,0.25)'],[1,'rgba(56,139,253,0.55)']],
                    showscale=False, name='Dot Balls',
                    hovertemplate="Dot zone<extra></extra>"
                ))

            # Dismissal heatmap (foreground, red)
            if len(dism_df) > 0:
                fig_hm.add_trace(go.Histogram2d(
                    x=dism_df['PitchY'], y=dism_df['PitchX'],
                    nbinsx=10, nbinsy=14,
                    colorscale=[[0,'rgba(0,0,0,0)'],[0.2,'rgba(110,26,20,0.35)'],[0.6,'rgba(182,42,33,0.65)'],[1,'#f85149']],
                    showscale=False, name='Dismissals',
                    hovertemplate="Dismissal zone<extra></extra>"
                ))

            lengths = [(1.5,"Yorker"),(4.0,"Fuller"),(7.0,"Good"),(10.0,"Back L."),(13.0,"Short")]
            for y_val, label in lengths:
                fig_hm.add_hline(y=y_val, line_dash="dash", line_color="#21262d", line_width=1)
                fig_hm.add_annotation(x=-0.78, y=y_val, text=label, showarrow=False,
                                       font=dict(color="#8b949e", size=9), xanchor="right")
            fig_hm.add_vline(x=-0.114, line_color="#30363d", line_width=1)
            fig_hm.add_vline(x= 0.114, line_color="#30363d", line_width=1)

            # Heat key legend
            fig_hm.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                marker=dict(size=10, color='#f85149', symbol='square'),
                name='High Vulnerability (Dismissal hotspot)'))
            fig_hm.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                marker=dict(size=10, color='rgba(56,139,253,0.55)', symbol='square'),
                name='Moderate Deficit (Dot area)'))
            fig_hm.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                marker=dict(size=10, color='rgba(30,40,50,0.8)', symbol='square'),
                name='Safe Zone (No dismissals)'))

            fig_hm.update_layout(
                xaxis=dict(range=[-1.0,1.0], showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(range=[0,16], autorange="reversed", showticklabels=False, showgrid=False, zeroline=False),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0d1117',
                legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center',
                            font=dict(size=9, color='#8b949e')),
                margin=dict(t=20,b=20,l=80,r=20), height=400
            )
            st.plotly_chart(fig_hm, use_container_width=True)

        # ── Row 2: Matchup Breakdown + Bowling Type Breakdown ──────────
        r2a, r2b = st.columns(2)

        with r2a:
            st.markdown(
                '<div class="sub-card"><h4 style="margin:0;">Matchup Performance Breakdown</h4>'
                '<p style="margin:2px 0 0;color:#8b949e;font-size:11px;">Key metrics against Pace vs Spin</p></div>',
                unsafe_allow_html=True
            )

            pace_df = slice_bat(name, season=current_season, phase=sel_phase, matchup='pace')
            spin_df = slice_bat(name, season=current_season, phase=sel_phase, matchup='spin')

            def bat_metrics(d):
                legal = d[d['Is_Legal'] == True]
                runs  = legal['Runs'].sum()
                balls = len(legal)
                return {
                    'SR':        sr(runs, balls),
                    'Dot Ball %': dot_pct(d),
                    'Boundary %': boundary_pct(d),
                }

            pm = bat_metrics(pace_df)
            sm = bat_metrics(spin_df)
            metrics_labels = ['Strike Rate', 'Dot Ball %', 'Boundary %']
            pm_vals = [pm['SR'], pm['Dot Ball %'], pm['Boundary %']]
            sm_vals = [sm['SR'], sm['Dot Ball %'], sm['Boundary %']]

            fig_mu = go.Figure()
            fig_mu.add_trace(go.Bar(
                x=metrics_labels, y=pm_vals, name='vs Pacers',
                marker_color='#388bfd', text=[f"{v:.1f}" for v in pm_vals],
                textposition='outside', textfont=dict(color='#c9d1d9', size=10)
            ))
            fig_mu.add_trace(go.Bar(
                x=metrics_labels, y=sm_vals, name='vs Spinners',
                marker_color='#8957e5', text=[f"{v:.1f}" for v in sm_vals],
                textposition='outside', textfont=dict(color='#c9d1d9', size=10)
            ))
            fig_mu.update_layout(
                barmode='group', template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(title="Metric Value (%)", gridcolor='#21262d', showgrid=True),
                xaxis=dict(showgrid=False),
                legend=dict(orientation='h', y=1.12, x=0.5, xanchor='center'),
                margin=dict(t=30,b=20,l=40,r=20), height=380
            )

            # Stat chips below
            st.plotly_chart(fig_mu, use_container_width=True)
            st.markdown(
                f"""
                <div style="background:#11161d;border:1px solid #21262d;border-radius:6px;
                            padding:10px 14px;font-size:11px;color:#8b949e;margin-top:-10px;">
                    <b style="color:#c9d1d9;">🔍 Scoring & Matchup Efficiency Analysis:</b><br>
                    Analyze how the batter's Strike Rate and boundary density compare when facing
                    Pacers vs. Spinners. A higher peak indicates dominant matchup success.
                    Pace SR: <b style="color:#388bfd;">{pm['SR']:.1f}</b> &nbsp;|&nbsp;
                    Spin SR: <b style="color:#8957e5;">{sm['SR']:.1f}</b>
                </div>
                """, unsafe_allow_html=True
            )

        with r2b:
            st.markdown(
                '<div class="sub-card"><h4 style="margin:0;">Bowling Type Performance Breakdown</h4>'
                '<p style="margin:2px 0 0;color:#8b949e;font-size:11px;">Breakdown across RA/LA Pace and Spin variations</p></div>',
                unsafe_allow_html=True
            )

            bowler_type_map = {
                'RA Pace':   ['RF','RFM','RMF','RM'],
                'LA Pace':   ['LF','LMF','LM'],
                'Off-Break': ['ROB','OB'],
                'Leg-Break': ['RLB','LB'],
                'LA Spin':   ['LOB','LSB','SLA'],
            }

            bt_labels, bt_runs, bt_sr_vals, bt_chips = [], [], [], []
            for label, types in bowler_type_map.items():
                d2 = slice_bat(name, season=current_season, phase=sel_phase)
                d2 = d2[d2['Bowler Type'].isin(types)]
                legal = d2[d2['Is_Legal'] == True]
                runs  = int(legal['Runs'].sum())
                balls = len(legal)
                sr_v  = sr(runs, balls)
                bt_labels.append(label)
                bt_runs.append(runs)
                bt_sr_vals.append(sr_v)
                bt_chips.append(f"<b>{label}:</b> {runs}r &nbsp;<span style='color:#e3b341;'>{sr_v:.0f} SR</span>")

            fig_bt = go.Figure()
            fig_bt.add_trace(go.Bar(
                x=bt_labels, y=bt_runs, name='Runs Scored',
                marker_color='#da3644', yaxis='y1'
            ))
            fig_bt.add_trace(go.Scatter(
                x=bt_labels, y=bt_sr_vals, name='Strike Rate',
                line=dict(color='#e3b341', width=2.5),
                marker=dict(size=8, color='#e3b341'),
                yaxis='y2'
            ))
            fig_bt.update_layout(
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(title="Runs Scored", side="left", showgrid=True, gridcolor='#21262d'),
                yaxis2=dict(title="Strike Rate", overlaying="y", side="right",
                            range=[0, max(bt_sr_vals+[1])*1.5], showgrid=False),
                xaxis=dict(showgrid=False),
                legend=dict(orientation='h', y=1.12, x=0.5, xanchor='center'),
                margin=dict(t=30,b=20,l=40,r=40), height=320
            )
            st.plotly_chart(fig_bt, use_container_width=True)

            chips_html = " &nbsp;|&nbsp; ".join(bt_chips)
            st.markdown(
                f'<div style="background:#11161d;border:1px solid #21262d;border-radius:6px;'
                f'padding:10px 14px;font-size:11px;color:#8b949e;">{chips_html}</div>',
                unsafe_allow_html=True
            )

    # ══════════════════════════════════════════════════════════════════
    # ████████████████  BOWLER UI  ████████████████
    # ══════════════════════════════════════════════════════════════════
    def render_bowler_ui(name):

        cur  = slice_bowl(name, season=current_season, phase=sel_phase)
        hist = slice_bowl(name, phase=sel_phase)

        cur_legal  = cur[cur['Is_Legal'] == True]
        hist_legal = hist[hist['Is_Legal'] == True]

        cur_wkts  = int(cur['Is_Bowler_Wicket'].sum())
        cur_balls = len(cur_legal)
        cur_runs  = int(cur['Runs'].sum()) + int(cur['Bowler Extra Runs'].sum() if 'Bowler Extra Runs' in cur.columns else 0)
        cur_econ  = econ(cur_runs, cur_balls)
        cur_bsr   = cur_balls / max(cur_wkts, 1)
        cur_dots  = dot_pct(cur)

        # ── KPI Cards ─────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        for col, title, fmt in [
            (c1, "WICKETS",     f"{cur_wkts}"),
            (c2, "ECON RATE",   f"{cur_econ:.2f}"),
            (c3, "STRIKE RATE", f"{cur_bsr:.1f}"),
            (c4, "DOT BALL %",  f"{cur_dots:.1f}%"),
        ]:
            col.markdown(
                f"""
                <div style="background:#11161d;border:1px solid #1f6e43;border-radius:8px;
                            padding:18px;text-align:center;margin-bottom:12px;">
                    <p style="margin:0;font-size:10px;color:#8b949e;font-weight:700;
                              letter-spacing:.07em;">{title}</p>
                    <h2 style="margin:8px 0 0 0;font-size:32px;color:#fff;font-family:monospace;">
                        {fmt}
                    </h2>
                </div>
                """, unsafe_allow_html=True
            )

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        # ── Row 1: Release Point Scatter + Intent vs Execution ────────
        r1a, r1b = st.columns(2)

        with r1a:
            st.markdown(
                '<div class="sub-card"><h4 style="margin:0;">Release Point Scatter Plot</h4>'
                '<p style="margin:2px 0 0;color:#8b949e;font-size:11px;">'
                'Frontal consistency scatter: Hand height (Z) and release width (Y)</p></div>',
                unsafe_allow_html=True
            )

            rel_df = cur.dropna(subset=['ReleaseY']).copy()
            rel_df = rel_df[(rel_df['ReleaseZ'] != 0) & (rel_df['ReleaseY'] != 0)]
            rel_df = rel_df.dropna(subset=['ReleaseZ'])

            # -----------------------------
            # Remove extreme outliers (top/bottom 1%)
            # -----------------------------
            if len(rel_df) > 10:   # Only apply if enough deliveries
                lower_y, upper_y = rel_df['ReleaseY'].quantile([0.01, 0.99])
                lower_z, upper_z = rel_df['ReleaseZ'].quantile([0.01, 0.99])

                rel_df = rel_df[
                    rel_df['ReleaseY'].between(lower_y, upper_y) &
                    rel_df['ReleaseZ'].between(lower_z, upper_z)
                ]

            def ball_category(row):
                if row['Is_Bowler_Wicket']:
                    return 'Wicket'
                elif row['Runs'] == 0:
                    return 'Dot'
                elif row['Runs'] >= 4:
                    return 'Boundary'
                else:
                    return 'Single/Double'

            if len(rel_df) > 0:
                rel_df['BallCat'] = rel_df.apply(ball_category, axis=1)
                cat_color = {
                    'Dot':          '#2ea043',
                    'Wicket':       '#2ea043',
                    'Single/Double':'#8b949e',
                    'Boundary':     '#f85149',
                }

                fig_rp = go.Figure()

                for cat, color in cat_color.items():
                    sub = rel_df[rel_df['BallCat'] == cat]
                    if len(sub) == 0:
                        continue

                    label = (
                        'Dots / Wicket Balls' if cat in ['Dot', 'Wicket']
                        else 'Singles / Doubles' if cat == 'Single/Double'
                        else 'Boundaries Conceded'
                    )

                    fig_rp.add_trace(go.Scatter(
                        x=sub['ReleaseY'],
                        y=sub['ReleaseZ'],
                        mode='markers',
                        marker=dict(
                            size=6,
                            color=color,
                            opacity=0.7,
                            line=dict(width=0.5, color='rgba(0,0,0,0.3)')
                        ),
                        name=label
                    ))

                fig_rp.update_layout(
                    xaxis=dict(
                        title="Horizontal Release Width (meters)",
                        showgrid=True,
                        gridcolor='#21262d',
                        zeroline=True,
                        zerolinecolor='#30363d'
                    ),
                    yaxis=dict(
                        title="Vertical Release Height (meters)",
                        showgrid=True,
                        gridcolor='#21262d',
                        zeroline=False
                    ),
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='#0d1117',
                    legend=dict(
                        orientation='h',
                        y=-0.2,
                        x=0.5,
                        xanchor='center',
                        font=dict(size=10)
                    ),
                    margin=dict(t=20, b=20, l=60, r=20),
                    height=400
                )

                st.plotly_chart(fig_rp, use_container_width=True)

                st.markdown(
                    '<div style="background:#11161d;border:1px solid #21262d;border-radius:6px;'
                    'padding:10px 14px;font-size:11px;color:#8b949e;margin-top:-10px;">'
                    '💡 <b style="color:#c9d1d9;">How to Read Release Points:</b> '
                    'A tight cluster indicates high consistency in release coordinates. '
                    'A wider scatter indicates variance in height and width.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.info("No release point data available for this bowler in the selected filters.")
                
        with r1b:
            st.markdown(
                '<div class="sub-card"><h4 style="margin:0;">Intent vs. Execution Pitch Maps</h4>'
                '<p style="margin:2px 0 0;color:#8b949e;font-size:11px;">'
                'Planned target zone vs actual landing spot dispersion</p></div>',
                unsafe_allow_html=True
            )

            pitch_df = cur.dropna(subset=['PitchX','PitchY']).copy()
            pitch_df = pitch_df[(pitch_df['PitchX'] != 0) & (pitch_df['PitchY'] != 0)]

            col_intent, col_exec = st.columns(2)

            # ── Intent plot: dense target zone as variance rectangle ──
            with col_intent:
                st.markdown("<p style='text-align:center;font-size:10px;color:#8b949e;font-weight:700;margin:0;'>1. INTENDED ZONE</p>", unsafe_allow_html=True)
                fig_int = go.Figure()

                if len(pitch_df) > 5:
                    mx  = pitch_df['PitchY'].mean()
                    my  = pitch_df['PitchX'].mean()
                    sdx = pitch_df['PitchY'].std()
                    sdy = pitch_df['PitchX'].std()
                    # 1-sigma box
                    fig_int.add_shape(
                        type='rect',
                        x0=mx - sdx, x1=mx + sdx,
                        y0=my - sdy, y1=my + sdy,
                        line=dict(color='#2ea043', width=2, dash='dot'),
                        fillcolor='rgba(46,160,67,0.15)'
                    )
                    fig_int.add_annotation(
                        x=mx, y=my + sdy + 0.4,
                        text="TARGET", showarrow=False,
                        font=dict(color='#2ea043', size=9, weight='bold')
                    )
                    fig_int.add_trace(go.Scatter(
                        x=[mx], y=[my], mode='markers',
                        marker=dict(size=8, color='#2ea043', symbol='cross'),
                        showlegend=False
                    ))

                lengths = [(1.5,"Yorker"),(4.0,"Full"),(7.0,"Good"),(10.0,"Back L."),(13.0,"Short")]
                for y_val, label in lengths:
                    fig_int.add_hline(y=y_val, line_dash="dash", line_color="#21262d", line_width=1)
                    fig_int.add_annotation(x=-0.82, y=y_val, text=label, showarrow=False,
                                           font=dict(color="#8b949e",size=8), xanchor="right")
                fig_int.add_vline(x=-0.114, line_color="#30363d", line_width=1)
                fig_int.add_vline(x= 0.114, line_color="#30363d", line_width=1)
                fig_int.update_layout(
                    xaxis=dict(range=[-1.0,1.0], showticklabels=False, showgrid=False, zeroline=False),
                    yaxis=dict(range=[0,16], autorange="reversed", showticklabels=False, showgrid=False, zeroline=False),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0d1117',
                    margin=dict(t=10,b=10,l=70,r=5), height=360, showlegend=False
                )
                st.plotly_chart(fig_int, use_container_width=True)

            # ── Execution plot: actual delivery scatter coloured by outcome ──
            with col_exec:
                st.markdown("<p style='text-align:center;font-size:10px;color:#8b949e;font-weight:700;margin:0;'>2. EXECUTED DELIVERIES</p>", unsafe_allow_html=True)
                fig_ex = go.Figure()

                if len(pitch_df) > 0:
                    pitch_df2 = pitch_df.copy()
                    pitch_df2['BallCat'] = pitch_df2.apply(ball_category, axis=1)
                    cat_color_map = {
                        'Dot':          ('#2ea043', 'circle',   'Dot/Wicket'),
                        'Wicket':       ('#2ea043', 'x',        'Dot/Wicket'),
                        'Single/Double':('#8b949e', 'circle',   'Single/Double'),
                        'Boundary':     ('#f85149', 'diamond',  'Boundary'),
                    }
                    shown = set()
                    for cat, (color, symbol, leg) in cat_color_map.items():
                        sub = pitch_df2[pitch_df2['BallCat'] == cat]
                        if len(sub) == 0: continue
                        fig_ex.add_trace(go.Scatter(
                            x=sub['PitchY'], y=sub['PitchX'], mode='markers',
                            marker=dict(size=5, color=color, symbol=symbol,
                                        opacity=0.75, line=dict(width=0.3, color='rgba(0,0,0,0.4)')),
                            name=leg if leg not in shown else None,
                            showlegend=(leg not in shown)
                        ))
                        shown.add(leg)

                for y_val, label in lengths:
                    fig_ex.add_hline(y=y_val, line_dash="dash", line_color="#21262d", line_width=1)
                    fig_ex.add_annotation(x=-0.82, y=y_val, text=label, showarrow=False,
                                          font=dict(color="#8b949e",size=8), xanchor="right")
                fig_ex.add_vline(x=-0.114, line_color="#30363d", line_width=1)
                fig_ex.add_vline(x= 0.114, line_color="#30363d", line_width=1)
                fig_ex.update_layout(
                    xaxis=dict(range=[-1.0,1.0], showticklabels=False, showgrid=False, zeroline=False),
                    yaxis=dict(range=[0,16], autorange="reversed", showticklabels=False, showgrid=False, zeroline=False),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0d1117',
                    legend=dict(orientation='h', y=-0.08, x=0.5, xanchor='center', font=dict(size=8)),
                    margin=dict(t=10,b=10,l=5,r=5), height=360
                )
                st.plotly_chart(fig_ex, use_container_width=True)

            # Dense zone label
            if len(pitch_df) > 5:
                target_length_val = pitch_df['PitchX'].mean()
                tgt_label = "Yorker" if target_length_val < 2 else \
                            "Full"   if target_length_val < 5 else \
                            "Good Length" if target_length_val < 8 else \
                            "Back of Length" if target_length_val < 11 else "Short"
                st.markdown(
                    f'<p style="font-size:11px;color:#8b949e;margin-top:4px;">'
                    f'💡 Hover on coordinates to see ball details. '
                    f'Target Zone: <span style="color:#2ea043;font-weight:600;">{tgt_label} (Pace Plan)</span>.</p>',
                    unsafe_allow_html=True
                )

        # ── Row 2: Length Stacked Bar + Phase Performance Profile ──────
        r2a, r2b = st.columns(2)

        with r2a:
            st.markdown(
                '<div class="sub-card"><h4 style="margin:0;">Length Stacked Bar</h4>'
                '<p style="margin:2px 0 0;color:#8b949e;font-size:11px;">'
                'Dispersion profile: % of each length by match phase</p></div>',
                unsafe_allow_html=True
            )

            valid_lengths = ['Yorker','Full','Good Length','Back Of Length','Short']
            length_color  = {'Yorker':'#f85149','Full':'#f0883e','Good Length':'#2ea043',
                             'Back Of Length':'#388bfd','Short':'#8957e5'}

            phase_labels_sb = ['PP (Overs 1-6)', 'Mid (Overs 7-15)', 'Death (Overs 16-20)']
            phase_keys_sb   = ['PP', 'Mid', 'Death']

            fig_len = go.Figure()
            for length in valid_lengths:
                pcts = []
                for ph in phase_keys_sb:
                    ph_df = hist[hist['Match_Phase'] == ph]
                    ph_df = ph_df[~ph_df['Length'].isin(['nan','Length','Unknown'])]
                    total = max(len(ph_df), 1)
                    count = len(ph_df[ph_df['Length'].str.strip().str.title() == length.title()])
                    pcts.append((count / total) * 100)
                fig_len.add_trace(go.Bar(
                    x=phase_labels_sb, y=pcts, name=length,
                    marker_color=length_color.get(length, '#8b949e')
                ))

            fig_len.update_layout(
                barmode='stack', template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(title="Percentage (%)", range=[0,100], gridcolor='#21262d'),
                xaxis=dict(showgrid=False),
                legend=dict(orientation='h', y=1.12, x=0.5, xanchor='center', font=dict(size=10)),
                margin=dict(t=30,b=20,l=40,r=20), height=380
            )
            st.plotly_chart(fig_len, use_container_width=True)
            st.markdown(
                '<div style="background:#11161d;border:1px solid #21262d;border-radius:6px;'
                'padding:10px 14px;font-size:11px;color:#8b949e;">'
                '<b style="color:#c9d1d9;">How to Read this Chart:</b> Displays the percentage of '
                'deliveries bowled at each length (Yorker, Full, Good, Short) across different match '
                'phases. Visualises the length distribution strategy in PP, Mid, and Death overs.</div>',
                unsafe_allow_html=True
            )

        with r2b:
            st.markdown(
                '<div class="sub-card"><h4 style="margin:0;">Phase-by-Phase Performance Profile</h4>'
                '<p style="margin:2px 0 0;color:#8b949e;font-size:11px;">'
                'Wickets and economy rate across Powerplay, Middle, Death</p></div>',
                unsafe_allow_html=True
            )

            ph_labels_pp = ['Powerplay (Ovs 1-6)', 'Middle (Ovs 7-15)', 'Death (Ovs 16-20)']
            ph_keys_pp   = ['PP', 'Mid', 'Death']
            wkts_per_phase, econ_per_phase = [], []

            for ph in ph_keys_pp:
                ph_d = cur[cur['Match_Phase'] == ph]
                ph_l = ph_d[ph_d['Is_Legal'] == True]
                w    = int(ph_d['Is_Bowler_Wicket'].sum())
                r    = float(ph_d['Runs'].sum())
                ext  = float(ph_d['Bowler Extra Runs'].sum()) if 'Bowler Extra Runs' in ph_d.columns else 0
                b    = max(len(ph_l), 1)
                wkts_per_phase.append(w)
                econ_per_phase.append(econ(r + ext, b))

            fig_pp = go.Figure()
            fig_pp.add_trace(go.Bar(
                x=ph_labels_pp, y=wkts_per_phase, name='Wickets',
                marker_color='#2ea043', yaxis='y1',
                text=wkts_per_phase, textposition='outside',
                textfont=dict(color='white', size=11)
            ))
            fig_pp.add_trace(go.Scatter(
                x=ph_labels_pp, y=econ_per_phase, name='Economy Rate',
                line=dict(color='#ffa500', width=2.5),
                marker=dict(size=9, color='#ffa500'),
                yaxis='y2'
            ))
            fig_pp.update_layout(
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(title="Wickets", side='left', showgrid=True, gridcolor='#21262d',
                           range=[0, max(wkts_per_phase+[1])*1.5]),
                yaxis2=dict(title="Economy Rate", overlaying='y', side='right',
                            range=[0, max(econ_per_phase+[1])*1.5], showgrid=False),
                xaxis=dict(showgrid=False),
                legend=dict(orientation='h', y=1.12, x=0.5, xanchor='center'),
                margin=dict(t=30,b=20,l=40,r=40), height=380
            )
            st.plotly_chart(fig_pp, use_container_width=True)
            st.markdown(
                '<div style="background:#11161d;border:1px solid #21262d;border-radius:6px;'
                'padding:10px 14px;font-size:11px;color:#8b949e;">'
                '<b style="color:#c9d1d9;">How to Read this Chart:</b> Compares economy rates and '
                'wickets taken across Powerplay, Middle, and Death phases. '
                'Visualises phase-specific effectiveness and containment ability.</div>',
                unsafe_allow_html=True
            )

    # ══════════════════════════════════════════════════════════════════
    # DISPATCH: render correct UI based on role
    # ══════════════════════════════════════════════════════════════════
    st.markdown("---")

    if player_role == 'Allrounder':
        bat_tab, bowl_tab = st.tabs(["🏏 Batting Analysis", "🎳 Bowling Analysis"])
        with bat_tab:
            render_batter_ui(selected_hub_player)
        with bowl_tab:
            render_bowler_ui(selected_hub_player)
    elif player_role == 'Bowler':
        render_bowler_ui(selected_hub_player)
    else:
        render_batter_ui(selected_hub_player)
