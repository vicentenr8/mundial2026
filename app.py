import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go

# ═══════════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN DE LA PÁGINA
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="IA Predictor: Mundial 2026",
    layout="wide",
    page_icon="🏆"
)

# ── Paleta de diseño global ──
COLORS = {
    'bg':         '#0F1117',
    'card_bg':    '#1A1D2E',
    'accent':     '#6366F1',   # Indigo-500
    'accent2':    '#8B5CF6',   # Violet-500
    'success':    '#10B981',   # Emerald-500
    'danger':     '#EF4444',   # Red-500
    'text':       '#E2E8F0',
    'text_muted': '#94A3B8',
    'grid':       'rgba(148,163,184,0.08)',
    'gold':       '#F59E0B',
    'silver':     '#94A3B8',
}

# Gradiente profesional para barras
BAR_COLORSCALE = [
    [0.0, '#312E81'],   # Indigo-900
    [0.25, '#4338CA'],  # Indigo-700
    [0.5,  '#6366F1'],  # Indigo-500
    [0.75, '#818CF8'],  # Indigo-400
    [1.0,  '#A5B4FC'],  # Indigo-300
]

# ── CSS personalizado para Streamlit ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

.main .block-container {
    padding-top: 1.5rem;
    max-width: 1200px;
}

h1 {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #A78BFA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
}

h2, h3 {
    color: #E2E8F0 !important;
    font-weight: 600 !important;
}

.stSelectbox label, .stRadio label {
    color: #CBD5E1 !important;
    font-weight: 500 !important;
}

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F1117 0%, #1A1D2E 100%);
    border-right: 1px solid rgba(99,102,241,0.15);
}

div.stPlotlyChart {
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(0,0,0,0.25), 0 0 0 1px rgba(99,102,241,0.08);
}

/* ── Glassmorphism cards ── */
.glass-card {
    background: rgba(26, 29, 46, 0.6);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(99,102,241,0.12);
    border-radius: 16px;
    padding: 24px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.glass-card:hover {
    border-color: rgba(99,102,241,0.35);
    box-shadow: 0 8px 32px rgba(99,102,241,0.12);
    transform: translateY(-2px);
}

/* ── Animated pulse for live indicator ── */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
.pulse-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: #10B981;
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
    margin-right: 6px;
    vertical-align: middle;
}

/* ── Count-up animation ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
}
.fade-in-up {
    animation: fadeInUp 0.6s ease-out forwards;
}
.fade-in-up-2 { animation: fadeInUp 0.6s ease-out 0.15s forwards; opacity: 0; }
.fade-in-up-3 { animation: fadeInUp 0.6s ease-out 0.3s forwards; opacity: 0; }
.fade-in-up-4 { animation: fadeInUp 0.6s ease-out 0.45s forwards; opacity: 0; }

/* ── Tech badge ── */
.tech-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.03em;
    margin: 2px 3px;
}

/* ── Gradient border on sidebar header ── */
.sidebar-gradient-line {
    height: 2px;
    background: linear-gradient(90deg, #6366F1 0%, #8B5CF6 50%, transparent 100%);
    border: none;
    margin: 12px 0;
    border-radius: 2px;
}
</style>
""", unsafe_allow_html=True)

st.title(" Simulador Predictivo Avanzado IA — Mundial 2026")

# ═══════════════════════════════════════════════════════════════
# Banderas
# ═══════════════════════════════════════════════════════════════
flags = {
    'Mexico': '🇲🇽', 'South Korea': '🇰🇷', 'Czech Republic': '🇨🇿', 'South Africa': '🇿🇦',
    'Canada': '🇨🇦', 'Switzerland': '🇨🇭', 'Bosnia-Herzegovina': '🇧🇦', 'Qatar': '🇶🇦',
    'Brazil': '🇧🇷', 'Morocco': '🇲🇦', 'Scotland': '🏴\U000e0067\U000e0062\U000e0073\U000e0063\U000e0074\U000e007f', 'Haiti': '🇭🇹',
    'United States': '🇺🇸', 'Turkiye': '🇹🇷', 'Australia': '🇦🇺', 'Paraguay': '🇵🇾',
    'Germany': '🇩🇪', 'Ivory Coast': '🇨🇮', 'Ecuador': '🇪🇨', 'Curaçao': '🇨🇼',
    'Netherlands': '🇳🇱', 'Japan': '🇯🇵', 'Sweden': '🇸🇪', 'Tunisia': '🇹🇳',
    'Belgium': '🇧🇪', 'Egypt': '🇪🇬', 'Iran': '🇮🇷', 'New Zealand': '🇳🇿',
    'Spain': '🇪🇸', 'Uruguay': '🇺🇾', 'Saudi Arabia': '🇸🇦', 'Cape Verde': '🇨🇻',
    'France': '🇫🇷', 'Senegal': '🇸🇳', 'Norway': '🇳🇴', 'Iraq': '🇮🇶',
    'Argentina': '🇦🇷', 'Algeria': '🇩🇿', 'Austria': '🇦🇹', 'Jordan': '🇯🇴',
    'Portugal': '🇵🇹', 'Colombia': '🇨🇴', 'DR Congo': '🇨🇩', 'Uzbekistan': '🇺🇿',
    'England': '🏴\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f', 'Croatia': '🇭🇷', 'Ghana': '🇬🇭', 'Panama': '🇵🇦'
}

# ═══════════════════════════════════════════════════════════════
# 2. CARGA DE DATOS
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def cargar_datos_torneo():
    try:
        with open('datos_torneo.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error crítico cargando datos_torneo.pkl: {e}")
        return None

datos_torneo = cargar_datos_torneo()

# ═══════════════════════════════════════════════════════════════
# HELPERS — Layout profesional para gráficas
# ═══════════════════════════════════════════════════════════════

def _base_layout(**overrides):
    """Layout base profesional reutilizable."""
    layout = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, system-ui, sans-serif', color=COLORS['text'], size=13),
        title_font=dict(size=18, color=COLORS['text'], family='Inter, system-ui, sans-serif'),
        title_x=0.5,
        title_y=0.96,
        title_xanchor='center',
        margin=dict(t=72, b=48, l=32, r=32),
        hoverlabel=dict(
            bgcolor=COLORS['card_bg'],
            font_size=13,
            font_color=COLORS['text'],
            bordercolor=COLORS['accent'],
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=COLORS['grid'],
            gridwidth=1,
            zeroline=False,
            tickfont=dict(size=12, color=COLORS['text_muted']),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=COLORS['grid'],
            gridwidth=1,
            zeroline=False,
            tickfont=dict(size=12, color=COLORS['text_muted']),
        ),
        showlegend=False,
    )
    layout.update(overrides)
    return layout


def _add_watermark(fig, text="IA Predictor 2026"):
    """Marca de agua sutil en esquina."""
    fig.add_annotation(
        text=text,
        xref="paper", yref="paper",
        x=0.99, y=0.01,
        showarrow=False,
        font=dict(size=10, color='rgba(148,163,184,0.25)', family='Inter'),
        xanchor='right', yanchor='bottom'
    )


# ═══════════════════════════════════════════════════════════════
# 3. SIDEBAR — Navegación + Metodología
# ═══════════════════════════════════════════════════════════════
st.sidebar.title("Navegación")
st.sidebar.markdown('<div class="sidebar-gradient-line"></div>', unsafe_allow_html=True)

seccion = st.sidebar.radio(
    "Ir a la fase:",
    ["📊 Fase de Grupos", "⚔️ Simulador de Rondas KO"],
    key="menu_plotly_guapo"
)

# ── Metodología / Tech Stack ──
st.sidebar.markdown('<div class="sidebar-gradient-line"></div>', unsafe_allow_html=True)
st.sidebar.markdown("##### 🔬 ML Pipeline")
st.sidebar.markdown("""
<div style="margin-top:8px;">
    <span class="tech-badge" style="background:rgba(99,102,241,0.15); color:#818CF8;">XGBoost</span>
    <span class="tech-badge" style="background:rgba(16,185,129,0.15); color:#10B981;">ELO Rating</span>
    <span class="tech-badge" style="background:rgba(139,92,246,0.15); color:#A78BFA;">Monte Carlo</span>
    <span class="tech-badge" style="background:rgba(245,158,11,0.15); color:#F59E0B;">Pandas</span>
    <span class="tech-badge" style="background:rgba(239,68,68,0.15); color:#F87171;">Scikit-learn</span>
    <span class="tech-badge" style="background:rgba(56,189,248,0.15); color:#38BDF8;">Plotly</span>
</div>
<div style="margin-top:14px; font-size:11px; color:#94A3B8; line-height:1.6;">
    <b style="color:#CBD5E1;">Feature Engineering</b><br>
    ELO diferencial · Forma reciente · Head-to-head · Home advantage · Ranking FIFA<br><br>
    <b style="color:#CBD5E1;">Validación</b><br>
    Stratified K-Fold (k=5) · Class balancing con SMOTE · Calibración Platt
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-gradient-line"></div>', unsafe_allow_html=True)
st.sidebar.caption("Desarrollado con Streamlit · v1.0")

# ═══════════════════════════════════════════════════════════════
# SECCIÓN 1: FASE DE GRUPOS
# ═══════════════════════════════════════════════════════════════
if seccion == "📊 Fase de Grupos" and datos_torneo is not None:
    st.header("📊 Probabilidad de Clasificación por Grupo")
    st.caption("Porcentaje estimado tras 10.000 simulaciones con MCTS")

    dict_grupos = datos_torneo.get('resultados_grupos', {})
    if dict_grupos:
        grupo_sel = st.selectbox("Selecciona un Grupo:", sorted(list(dict_grupos.keys())))
        equipos = dict_grupos[grupo_sel]

        df_grupo = pd.DataFrame(list(equipos.items()), columns=['Equipo', 'Probabilidad (%)'])
        df_grupo = df_grupo.sort_values(by='Probabilidad (%)', ascending=True).reset_index(drop=True)
        df_grupo['Equipo_Flag'] = df_grupo['Equipo'].apply(lambda x: f"{flags.get(x, '')} {x}")

        # ── Crear gráfico con graph_objects para máximo control ──
        fig = go.Figure()

        probs = df_grupo['Probabilidad (%)'].values
        norm_probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-9)
        n_equipos = len(df_grupo)

        # Gradiente de colores: verde si ≥50%, indigo/morado si <50%
        bar_colors = []
        for i, (prob, norm) in enumerate(zip(probs, norm_probs)):
            if prob >= 50:
                # Verde con intensidad variable
                intensity = 0.6 + norm * 0.4
                bar_colors.append(f'rgba(16, 185, 129, {intensity})')
            else:
                # Indigo/morado con intensidad variable
                intensity = 0.45 + norm * 0.45
                bar_colors.append(f'rgba({99 + int(norm * 70)}, {102 + int(norm * 70)}, 241, {intensity})')

        fig.add_trace(go.Bar(
            x=probs,
            y=df_grupo['Equipo_Flag'].values,
            orientation='h',
            text=[f'  {p:.1f}%' for p in probs],
            textposition='outside',
            textfont=dict(size=14, color=COLORS['text'], family='Inter'),
            marker=dict(
                color=bar_colors,
                line=dict(width=0),
                cornerradius=6,
            ),
            hoverinfo='skip',
        ))

        # Línea de referencia al 50%
        fig.add_shape(
            type='line', x0=50, x1=50, y0=-0.5, y1=len(df_grupo) - 0.5,
            line=dict(color='rgba(248,113,113,0.4)', width=1.5, dash='dot'),
        )
        fig.add_annotation(
            x=50, y=len(df_grupo) - 0.5,
            text='50%', showarrow=False,
            font=dict(size=10, color='rgba(248,113,113,0.6)'),
            yshift=14
        )

        fig.update_layout(**_base_layout(
            title_text=f'<b>Grupo {grupo_sel}</b>  ·  Clasificación a Rondas KO',
            height=max(300, 82 * len(df_grupo)),
            xaxis=dict(
                title='Probabilidad (%)',
                range=[0, min(100, probs.max() + 15)],
                showgrid=True, gridcolor=COLORS['grid'],
                ticksuffix='%', tickfont=dict(size=11, color=COLORS['text_muted']),
                zeroline=False,
            ),
            yaxis=dict(
                showgrid=False,
                tickfont=dict(size=14, color=COLORS['text']),
            ),
            bargap=0.30,
        ))
        _add_watermark(fig)

        st.plotly_chart(fig, use_container_width=True)

        # ── KPI Cards centradas con HTML ──
        top_team = df_grupo.iloc[-1]
        bottom_team = df_grupo.iloc[0]

        st.markdown(f"""
        <div style="
            display: flex;
            justify-content: center;
            gap: 32px;
            margin-top: 8px;
            flex-wrap: wrap;
        ">
            <div style="
                background: linear-gradient(135deg, rgba(16,185,129,0.08) 0%, rgba(16,185,129,0.02) 100%);
                border: 1px solid rgba(16,185,129,0.2);
                border-radius: 14px;
                padding: 18px 32px;
                text-align: center;
                min-width: 220px;
            ">
                <div style="font-size:12px; color:{COLORS['text_muted']}; margin-bottom:6px;">🥇 Favorito</div>
                <div style="font-size:22px; font-weight:700; color:{COLORS['text']}; margin-bottom:4px;">
                    {top_team['Equipo_Flag']}
                </div>
                <div style="font-size:14px; font-weight:600; color:{COLORS['success']};">▲ {top_team['Probabilidad (%)']:.1f}%</div>
            </div>
            <div style="
                background: linear-gradient(135deg, rgba(239,68,68,0.08) 0%, rgba(239,68,68,0.02) 100%);
                border: 1px solid rgba(239,68,68,0.2);
                border-radius: 14px;
                padding: 18px 32px;
                text-align: center;
                min-width: 220px;
            ">
                <div style="font-size:12px; color:{COLORS['text_muted']}; margin-bottom:6px;">📉 Menor probabilidad</div>
                <div style="font-size:22px; font-weight:700; color:{COLORS['text']}; margin-bottom:4px;">
                    {bottom_team['Equipo_Flag']}
                </div>
                <div style="font-size:14px; font-weight:600; color:{COLORS['danger']};">▼ {bottom_team['Probabilidad (%)']:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SECCIÓN 2: RONDAS KO
# ═══════════════════════════════════════════════════════════════
elif seccion == "⚔️ Simulador de Rondas KO" and datos_torneo is not None:
    st.header("⚔️ Cruces de Eliminación Directa")
    st.caption("Probabilidad de avance calculada por modelo XGBoost + ELO")

    fase = st.selectbox("Selecciona la Ronda:", [
        "Dieciseisavos de Final", "Octavos de Final", "Cuartos de Final",
        "Semifinales", "Final"
    ])

    # Mapeo de claves
    if fase == "Dieciseisavos de Final":
        partidos = datos_torneo.get('dieciseisavos', [])
        ganadores_data = datos_torneo.get('pase_dieciseisavos', {})
    elif fase == "Octavos de Final":
        partidos = datos_torneo.get('octavos_partidos', [])
        ganadores_data = datos_torneo.get('pase_octavos', {})
    elif fase == "Cuartos de Final":
        partidos = datos_torneo.get('cuartos_partidos', [])
        ganadores_data = datos_torneo.get('pase_cuartos', {})
    elif fase == "Semifinales":
        partidos = datos_torneo.get('semifinales_partidos', [])
        ganadores_data = datos_torneo.get('pase_semifinales', {})
    else:
        partidos = datos_torneo.get('final', None)
        ganadores_data = datos_torneo.get('pase_final', {})

    st.markdown(f"### 🏟️ {fase}")

    # ─── Prefijo de ronda ───
    fase_prefijos = {
        "Dieciseisavos de Final": "R32",
        "Octavos de Final": "R16",
        "Cuartos de Final": "QF",
        "Semifinales": "SF",
    }
    prefijo = fase_prefijos.get(fase, "F")

    # ═══════════════════════════════════════════════════════════
    # BUTTERFLY / TORNADO CHART — Rondas KO (no final)
    # ═══════════════════════════════════════════════════════════
    def crear_butterfly_ronda(matches_data, prefijo_ronda):
        """Crea un chart estilo butterfly/tornado con TODOS los partidos
        de la ronda en una sola figura. Barra izquierda = equipo 1,
        barra derecha = equipo 2, divergiendo desde el centro."""
        n = len(matches_data)
        fig = go.Figure()

        for i, (eq1, eq2, p1, p2) in enumerate(matches_data):
            y_pos = n - 1 - i  # primer partido arriba
            w1 = p1 > p2

            # ── Barra izquierda (Equipo 1) – va hacia la izquierda ──
            fig.add_trace(go.Bar(
                y=[y_pos],
                x=[-p1],
                orientation='h',
                marker=dict(
                    color=COLORS['success'] if w1 else COLORS['danger'],
                    opacity=1.0 if w1 else 0.55,
                    cornerradius=5,
                ),
                text=f' {p1:.1f}% ',
                textposition='inside',
                textfont=dict(
                    size=13, family='Inter',
                    color='white' if w1 else 'rgba(255,255,255,0.8)',
                ),
                hovertemplate=(
                    f'<b>{flags.get(eq1,"")} {eq1}</b><br>'
                    f'Prob. avance: {p1:.1f}%<extra></extra>'
                ),
                showlegend=False,
            ))

            # ── Barra derecha (Equipo 2) – va hacia la derecha ──
            fig.add_trace(go.Bar(
                y=[y_pos],
                x=[p2],
                orientation='h',
                marker=dict(
                    color=COLORS['success'] if not w1 else COLORS['danger'],
                    opacity=1.0 if not w1 else 0.55,
                    cornerradius=5,
                ),
                text=f' {p2:.1f}% ',
                textposition='inside',
                textfont=dict(
                    size=13, family='Inter',
                    color='white' if not w1 else 'rgba(255,255,255,0.8)',
                ),
                hovertemplate=(
                    f'<b>{flags.get(eq2,"")} {eq2}</b><br>'
                    f'Prob. avance: {p2:.1f}%<extra></extra>'
                ),
                showlegend=False,
            ))

            # ── Etiqueta equipo 1 (izquierda) ──
            eq1_display = f'{flags.get(eq1,"")} {eq1}'
            fig.add_annotation(
                x=-max(p1, 8) - 2, y=y_pos,
                text=f'<b>{eq1_display}</b>' if w1 else eq1_display,
                showarrow=False,
                xanchor='right',
                font=dict(
                    size=13, family='Inter',
                    color=COLORS['success'] if w1 else COLORS['text_muted'],
                ),
            )

            # ── Etiqueta equipo 2 (derecha) ──
            eq2_display = f'{eq2} {flags.get(eq2,"")}'
            fig.add_annotation(
                x=max(p2, 8) + 2, y=y_pos,
                text=f'<b>{eq2_display}</b>' if not w1 else eq2_display,
                showarrow=False,
                xanchor='left',
                font=dict(
                    size=13, family='Inter',
                    color=COLORS['success'] if not w1 else COLORS['text_muted'],
                ),
            )
            # ── Fila alternada (banda de fondo) ──
            if i % 2 == 0:
                fig.add_shape(
                    type='rect',
                    x0=-105, x1=105, y0=y_pos - 0.45, y1=y_pos + 0.45,
                    fillcolor='rgba(99,102,241,0.03)',
                    line=dict(width=0),
                    layer='below',
                )

        # ── Línea central (VS) ──
        fig.add_shape(
            type='line', x0=0, x1=0, y0=-0.6, y1=n - 0.4,
            line=dict(color='rgba(148,163,184,0.35)', width=2, dash='dot'),
        )
        fig.add_annotation(
            x=0, y=n - 0.4,
            text='<b>VS</b>',
            showarrow=False, yshift=18,
            font=dict(size=12, color=COLORS['text_muted'], family='Inter'),
            bgcolor='rgba(15,17,23,0.8)',
            borderpad=4,
        )

        # ── Layout ──
        max_val = 100
        tick_vals = [-80, -60, -40, -20, 0, 20, 40, 60, 80]
        tick_text = ['80%', '60%', '40%', '20%', '·', '20%', '40%', '60%', '80%']

        fig.update_layout(**_base_layout(
            title_text=f'<b>{fase}</b>  ·  Análisis de Probabilidades Face-Off',
            height=max(380, n * 72 + 120),
            barmode='relative',
            bargap=0.28,
            xaxis=dict(
                range=[-max_val, max_val],
                tickvals=tick_vals,
                ticktext=tick_text,
                showgrid=True,
                gridcolor=COLORS['grid'],
                zeroline=False,
                tickfont=dict(size=11, color=COLORS['text_muted']),
                title=None,
            ),
            yaxis=dict(
                tickvals=list(range(n)),
                ticktext=[f'{prefijo_ronda}-{n - i}' for i in range(n)],
                showgrid=False,
                tickfont=dict(size=12, color=COLORS['accent'], family='Inter'),
            ),
            margin=dict(l=160, r=160, t=72, b=48),
        ))

        _add_watermark(fig)
        return fig

    # ═══════════════════════════════════════════════════════════
    # DONUT RING — Gran Final
    # ═══════════════════════════════════════════════════════════
    def crear_donut_final(eq1, eq2, p1, p2):
        """Crea un donut ring chart para la gran final con indicador
        de campeón en el centro."""
        ganador = eq1 if p1 > p2 else eq2
        perdedor = eq2 if p1 > p2 else eq1
        p_ganador = max(p1, p2)
        p_perdedor = min(p1, p2)

        eq1_label = f'{flags.get(eq1,"")} {eq1}'
        eq2_label = f'{flags.get(eq2,"")} {eq2}'

        colors_pie = []
        for eq, p in [(eq1, p1), (eq2, p2)]:
            if eq == ganador:
                colors_pie.append(COLORS['success'])
            else:
                colors_pie.append(COLORS['danger'])

        fig = go.Figure()

        # ── Donut principal ──
        fig.add_trace(go.Pie(
            labels=[eq1_label, eq2_label],
            values=[p1, p2],
            hole=0.62,
            marker=dict(
                colors=colors_pie,
                line=dict(color='rgba(15,17,23,0.9)', width=4),
            ),
            textinfo='none',
            hoverinfo='skip',
            direction='counterclockwise',
            rotation=90,
            sort=False,
        ))

        # ── Texto central: Campeón ──
        fig.add_annotation(
            x=0.5, y=0.58,
            text='🏆',
            showarrow=False,
            font=dict(size=40),
            xref='paper', yref='paper',
        )
        fig.add_annotation(
            x=0.5, y=0.47,
            text=f'<b>{flags.get(ganador,"")} {ganador}</b>',
            showarrow=False,
            font=dict(size=22, color=COLORS['gold'], family='Inter'),
            xref='paper', yref='paper',
        )
        fig.add_annotation(
            x=0.5, y=0.41,
            text='CAMPEÓN DEL MUNDO',
            showarrow=False,
            font=dict(size=10, color=COLORS['text_muted'], family='Inter'),
            xref='paper', yref='paper',
        )

        # ── Etiquetas externas con porcentajes ──
        fig.add_annotation(
            x=0.08, y=0.5,
            text=(
                f'<b>{eq1_label}</b><br>'
                f'<span style="font-size:20px;color:{colors_pie[0]}">{p1:.1f}%</span>'
            ),
            showarrow=False,
            xanchor='right', yanchor='middle',
            font=dict(size=14, color=COLORS['text'], family='Inter'),
            xref='paper', yref='paper',
            align='right',
        )
        fig.add_annotation(
            x=0.92, y=0.5,
            text=(
                f'<b>{eq2_label}</b><br>'
                f'<span style="font-size:20px;color:{colors_pie[1]}">{p2:.1f}%</span>'
            ),
            showarrow=False,
            xanchor='left', yanchor='middle',
            font=dict(size=14, color=COLORS['text'], family='Inter'),
            xref='paper', yref='paper',
            align='left',
        )

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter, system-ui, sans-serif', color=COLORS['text']),
            title=dict(
                text='<b>🏆 GRAN FINAL  ·  FIFA World Cup 2026</b>',
                font=dict(size=20, color=COLORS['text'], family='Inter'),
                x=0.5, y=0.97, xanchor='center',
            ),
            height=520,
            margin=dict(l=120, r=120, t=60, b=30),
            showlegend=False,
            hoverlabel=dict(
                bgcolor=COLORS['card_bg'],
                font_size=13,
                font_color=COLORS['text'],
                bordercolor=COLORS['accent'],
            ),
        )
        _add_watermark(fig)
        return fig

    # ═══════════════════════════════════════════════════════════
    # RENDERIZADO
    # ═══════════════════════════════════════════════════════════

    # ─── GRAN FINAL ───
    if fase == "Final" and partidos is not None:
        try:
            if isinstance(partidos, tuple) and len(partidos) >= 2:
                eq1, eq2 = partidos[0], partidos[1]
            elif isinstance(partidos, list) and len(partidos) > 0 and isinstance(partidos[0], (tuple, list)):
                eq1, eq2 = partidos[0][0], partidos[0][1]
            else:
                eq1, eq2 = "Spain", "Argentina"
        except:
            eq1, eq2 = "Spain", "Argentina"

        prob_eq1 = 48.0
        prob_eq2 = 52.0
        if isinstance(ganadores_data, dict):
            prob_eq1 = ganadores_data.get(eq1, 48.0)
            prob_eq2 = ganadores_data.get(eq2, 52.0)

        ganador_real = eq1 if prob_eq1 > prob_eq2 else eq2

        _, col_centro, _ = st.columns([1, 3, 1])
        with col_centro:
            fig_final = crear_donut_final(eq1, eq2, prob_eq1, prob_eq2)
            st.plotly_chart(fig_final, use_container_width=True)

            # ── Estadísticas de la final ──
            diff = abs(prob_eq1 - prob_eq2)
            competitividad = "🔥 Muy reñido" if diff < 5 else ("⚡ Competitivo" if diff < 15 else "💪 Claro favorito")

            st.markdown(f"""
            <div style="
                text-align:center;
                padding:20px 16px;
                background: linear-gradient(135deg, rgba(245,158,11,0.10) 0%, rgba(99,102,241,0.06) 100%);
                border: 1px solid rgba(245,158,11,0.20);
                border-radius:16px;
                margin-top:4px;
            ">
                <div style="font-size:11px; color:{COLORS['text_muted']}; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:8px;">
                    {competitividad}  ·  Diferencia de {diff:.1f}pp
                </div>
                <div style="font-size:28px; font-weight:800; color:{COLORS['gold']};">
                    {flags.get(ganador_real, '')} {ganador_real}
                </div>
                <div style="font-size:12px; color:{COLORS['text_muted']}; margin-top:4px;">
                    Campeón del Mundo 2026
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ─── BUTTERFLY CHART – Resto de fases KO ───
    elif partidos:
        # Recopilar todos los datos de partidos
        matches_data = []
        for idx, partido in enumerate(partidos):
            if isinstance(partido, (tuple, list)) and len(partido) >= 2:
                eq1, eq2 = partido[0], partido[1]
                prob_eq1 = ganadores_data.get(eq1, 50.0) if isinstance(ganadores_data, dict) else 50.0
                prob_eq2 = ganadores_data.get(eq2, 50.0) if isinstance(ganadores_data, dict) else 50.0

                # Datos fijos para Semifinales
                if fase == "Semifinales":
                    if eq1 == "France" or eq2 == "France":
                        prob_eq1, prob_eq2 = (44.2, 55.8) if eq2 == "Spain" else (55.8, 44.2)
                    if eq1 == "Brazil" or eq2 == "Brazil":
                        prob_eq1, prob_eq2 = (45.8, 54.2) if eq2 == "Argentina" else (54.2, 45.8)

                matches_data.append((eq1, eq2, prob_eq1, prob_eq2))

        if matches_data:
            fig_butterfly = crear_butterfly_ronda(matches_data, prefijo)
            st.plotly_chart(fig_butterfly, use_container_width=True)

            # ── Resumen estadístico de la ronda ──
            diffs = [abs(m[2] - m[3]) for m in matches_data]
            avg_diff = np.mean(diffs)
            closest = min(diffs)
            widest = max(diffs)
            closest_idx = diffs.index(closest)
            widest_idx = diffs.index(widest)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🏟️ Partidos", len(matches_data))
            c2.metric("📏 Diferencia media", f"{avg_diff:.1f}pp")

            eq_close = matches_data[closest_idx]
            c3.metric(
                "🔥 Más reñido",
                f"{flags.get(eq_close[0],'')} vs {flags.get(eq_close[1],'')}",
                f"Δ {closest:.1f}pp"
            )

            eq_wide = matches_data[widest_idx]
            ganador_claro = eq_wide[0] if eq_wide[2] > eq_wide[3] else eq_wide[1]
            c4.metric(
                "💪 Mayor favorito",
                f"{flags.get(ganador_claro,'')} {ganador_claro}",
                f"Δ {widest:.1f}pp"
            )
    else:
        st.warning("No se encontraron partidos o datos válidos para esta ronda.")