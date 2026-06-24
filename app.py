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
tab_batting, tab_bowling, tab_matchup, tab_fielding, tab_venue = st.tabs([
    "🏏 Batter Profiles", 
    "🍒 Bowler Topography", 
    "📊 Matchup Matrix",
    "📋 Tactical Field Planner",
    "🏟️ Venue Scouting Hub"
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

with tab_matchup:
    st.markdown("## Squad Strategy Matchup Matrix")
    all_batters = sorted(df['Batter'].dropna().unique())
    all_bowlers = sorted(df['Bowler'].dropna().unique())
    default_batters, default_bowlers = all_batters[:8] if len(all_batters) >= 8 else all_batters, all_bowlers[:8] if len(all_bowlers) >= 8 else all_bowlers
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Matchup Configuration")
    selected_matrix_batters = st.sidebar.multiselect("Select Batters for Matrix", all_batters, default=default_batters)
    selected_matrix_bowlers = st.sidebar.multiselect("Select Bowlers for Matrix", all_bowlers, default=default_bowlers)
    if selected_matrix_batters and selected_matrix_bowlers:
        matchup_raw = df[df['Batter'].isin(selected_matrix_batters) & df['Bowler'].isin(selected_matrix_bowlers)].copy()
        matchup_agg = matchup_raw.groupby(['Bowler', 'Batter']).agg(runs=('Runs', 'sum'), balls=('Is_Legal', 'count'), outs=('Is_Bowler_Wicket', 'sum')).reset_index()
        matchup_agg['SR'] = (matchup_agg['runs'] / np.maximum(matchup_agg['balls'], 1)) * 100
        def cell_color(row):
            if row['balls'] < 10: return '#1d242e'
            elif row['SR'] >= 145: return '#6e1a24'
            elif row['SR'] >= 115: return '#4c1d6e'
            else: return '#1d446e'
        matchup_agg['Cell_Color'] = matchup_agg.apply(cell_color, axis=1)
        cell_lookup = {(r['Bowler'], r['Batter']): r for _, r in matchup_agg.iterrows()}
        if 'matrix_focus_batter' not in st.session_state: st.session_state.matrix_focus_batter = selected_matrix_batters[0]
        if 'matrix_focus_bowler' not in st.session_state: st.session_state.matrix_focus_bowler = selected_matrix_bowlers[0]
        matrix_left, details_right = st.columns([2, 1])
        with matrix_left:
            header_cols = st.columns([1.6] + [1] * len(selected_matrix_batters))
            for col_idx, batter in enumerate(selected_matrix_batters):
                is_selected = (batter == st.session_state.matrix_focus_batter)
                header_cols[col_idx + 1].markdown(f"<p style='color:{'#ffa500' if is_selected else '#8b949e'};font-size:10px;font-weight:bold;text-align:center;writing-mode:vertical-rl;transform:rotate(180deg);height:70px;'>{batter}</p>", unsafe_allow_html=True)
            for bowler in selected_matrix_bowlers:
                is_bowler_selected = (bowler == st.session_state.matrix_focus_bowler)
                row_cols = st.columns([1.6] + [1] * len(selected_matrix_batters))
                row_cols[0].markdown(f"<p style='color:{'#ffa500' if is_bowler_selected else '#c9d1d9'};font-size:11px;font-weight:bold;text-align:right;padding-right:6px;line-height:46px;'>{bowler}</p>", unsafe_allow_html=True)
                for col_idx, batter in enumerate(selected_matrix_batters):
                    stats = cell_lookup.get((bowler, batter))
                    bg = stats['Cell_Color'] if stats is not None else '#11161d'
                    text = f"{int(stats['runs'])}({int(stats['balls'])})\n{int(stats['outs'])}W" if stats is not None else "—"
                    border = '2px solid #ffa500' if (batter == st.session_state.matrix_focus_batter and bowler == st.session_state.matrix_focus_bowler) else '1px solid #30363d'
                    with row_cols[col_idx + 1]:
                        st.markdown(f"""<style>div[data-testid="stButton"] > button[key="cell_{bowler}_{batter}"] {{ background-color: {bg}; border: {border}; color: white; font-size: 10px; height: 46px; width: 100%; }}</style>""", unsafe_allow_html=True)
                        if st.button(text, key=f"cell_{bowler}_{batter}", use_container_width=True):
                            st.session_state.matrix_focus_batter, st.session_state.matrix_focus_bowler = batter, bowler
                            st.rerun()
        with details_right:
            focus_b, focus_w = st.session_state.matrix_focus_batter, st.session_state.matrix_focus_bowler
            tgt_df = matchup_raw[(matchup_raw['Batter'] == focus_b) & (matchup_raw['Bowler'] == focus_w)]
            st.markdown(f"<div class='sub-card'><h3>⚔️ {focus_b}</h3><p>vs {focus_w}</p><hr/>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Runs", int(tgt_df['Runs'].sum()))
            m2.metric("Balls", int(tgt_df[tgt_df['Is_Legal'] == True].shape[0]))
            m3.metric("Outs", int(tgt_df['Is_Bowler_Wicket'].sum()))
            st.markdown("</div>", unsafe_allow_html=True)

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
