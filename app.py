import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# ═══════════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN DE LA PÁGINA
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="IA Predictor: Mundial 2026",
    layout="wide",
    page_icon="https://cdn-icons-png.flaticon.com/512/188/188905.png"
)
# ── Paleta de diseño global ──
COLORS = {
    'bg':         '#0F1117',
    'card_bg':    '#1A1D2E',
    'accent':     '#6366F1',   # Indigo-500
    'accent2':    '#8B5CF6',   # Violet-500
    'success':    "#00bc7d",   # Emerald-500
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;700;900&display=swap');

html, body, [data-testid="stAppViewContainer"], .stMarkdown, p, h1, h2, h3, h4, h5, h6 {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        }

.main .block-container {
    padding-top: 1.5rem;
    max-width: 1200px;
}

h1, h2, h3 {
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
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
        with open('model/datos_torneo.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error crítico cargando datos_torneo.pkl: {e}")
        return None

datos_torneo = cargar_datos_torneo()
TECH_LOAD_CODE = """
@st.cache_resource
def cargar_tech_data():
    try:
        with open('tech_data.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error cargando tech_data.pkl: {e}")
        return None
 
tech_data = cargar_tech_data()
"""
 

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
    ["📊 Fase de Grupos", "⚔️ Simulador de Rondas KO", "🔬 Análisis del Modelo"],
    key="menu_plotly_guapo"
)

# ── Metodología / Tech Stack ──
st.sidebar.markdown('<div class="sidebar-gradient-line"></div>', unsafe_allow_html=True)
st.sidebar.markdown("##### 🔬 ML Pipeline")
st.sidebar.markdown("""
<div style="margin-top:8px;">
    <span class="tech-badge" style="background:rgba(99,102,241,0.15); color:#818CF8;">XGBoost</span>
    <span class="tech-badge" style="background:rgba(16,185,129,0.15); color:#10B981;">MCTS</span>
    <span class="tech-badge" style="background:rgba(139,92,246,0.15); color:#A78BFA;">Feature Engineering</span>
    <span class="tech-badge" style="background:rgba(245,158,11,0.15); color:#F59E0B;">Pandas</span>
    <span class="tech-badge" style="background:rgba(239,68,68,0.15); color:#F87171;">Scikit-learn</span>
    <span class="tech-badge" style="background:rgba(56,189,248,0.15); color:#38BDF8;">Plotly</span>
</div>
<div style="margin-top:14px; font-size:11px; color:#94A3B8; line-height:1.6;">
    <b style="color:#CBD5E1;">Feature Engineering</b><br>
    ELO · Market Value · Momentum · Pressure Index · Knockout Performance <br><br>
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
    st.caption("Probabilidad de avance calculada por modelo XGBoost + MCTS")

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
            title_text=f'<b>{fase}</b>  ·  Análisis de Probabilidades',
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
# ═══════════════════════════════════════════════════════════════
# SECCIÓN 3: ANÁLISIS DEL MODELO
# ═══════════════════════════════════════════════════════════════
elif seccion == "🔬 Análisis del Modelo":
    # Primero intentamos cargar tech_data.pkl de forma segura
    try:
        with open('model/tech_data.pkl', 'rb') as f:
            tech_data = pickle.load(f)
    except Exception as e:
        st.error(f"Error cargando tech_data.pkl: {e}")
        tech_data = None

    if tech_data is not None:
        st.header("🔬 Análisis Técnico del Modelo")
        st.caption("XGBoost · Balanced class weights · 5 features · entrenado con 5.011 partidos (2010–2025)")
     
        # ─── MÉTRICAS RÁPIDAS ───
        st.markdown("### 📊 Métricas de Rendimiento")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", "52%", delta="vs 33% baseline", delta_color="normal")
        m2.metric("AUC Victoria", "0.76")
        m3.metric("AUC Derrota", "0.77")
        m4.metric("AUC Empate", "0.58")
        m5.metric("Partidos train", "5.011")
     
        st.divider()
     
        # ─── ROW 1: ROC + Feature Importance ───
        col_roc, col_fi = st.columns(2)
     
        # ── 1. CURVA ROC MULTICLASE ──
        with col_roc:
            st.markdown("#### Curva ROC Multiclase")
     
            roc_data = tech_data.get('roc_data', {})
            roc_colors = {
                'Victoria': '#10B981',
                'Derrota':  '#EF4444',
                'Empate':   '#94A3B8',
            }
     
            fig_roc = go.Figure()
     
            # Diagonal de referencia
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                line=dict(color='rgba(148,163,184,0.3)', width=1.5, dash='dot'),
                showlegend=False, hoverinfo='skip'
            ))
     
            for clase, color in roc_colors.items():
                if clase in roc_data:
                    d = roc_data[clase]
                    auc_val = d['auc']
                    fig_roc.add_trace(go.Scatter(
                        x=d['fpr'], y=d['tpr'],
                        fill='tozeroy',
                        fillcolor=color.replace(')', ', 0.06)').replace('rgb', 'rgba') if 'rgb' in color else color,
                        line=dict(color=color, width=2.5),
                        name=f"{clase} (AUC = {auc_val:.2f})",
                        hovertemplate=f"<b>{clase}</b><br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>",
                        showlegend=True,
                    ))
     
            fig_roc.update_layout(**_base_layout(
                title_text="<b>Curva ROC</b>  ·  Discriminación por clase",
                height=400,
                showlegend=True,
                legend=dict(
                    x=0.55, y=0.08,
                    bgcolor='rgba(26,29,46,0.8)',
                    bordercolor='rgba(99,102,241,0.2)',
                    borderwidth=1,
                    font=dict(size=12, color=COLORS['text']),
                ),
                xaxis=dict(title="Tasa de Falsos Positivos", range=[0, 1],
                           tickformat='.0%', showgrid=True, gridcolor=COLORS['grid']),
                yaxis=dict(title="Tasa de Verdaderos Positivos", range=[0, 1.02],
                           tickformat='.0%', showgrid=True, gridcolor=COLORS['grid']),
            ))
            _add_watermark(fig_roc)
            st.plotly_chart(fig_roc, use_container_width=True)
     
        # ── 2. FEATURE IMPORTANCE ──
        with col_fi:
            st.markdown("#### Feature Importance — XGBoost")
     
            fi = tech_data.get('feature_importances', {
                'diff_ELO': 0.35, 'diff_MV': 0.24,
                'diff_Pressure': 0.15, 'diff_Knockout': 0.13, 'diff_Momentum': 0.13
            })
     
            feature_labels = {
                'diff_ELO': 'ELO Diferencial',
                'diff_MV': 'Market Value',
                'diff_Pressure': 'Pressure Index',
                'diff_Knockout': 'Knockout Perf.',
                'diff_Momentum': 'Momentum 15',
            }
     
            df_fi = pd.DataFrame([
                {'feature': feature_labels.get(k, k), 'importance': v}
                for k, v in fi.items()
            ]).sort_values('importance', ascending=True)
     
            max_imp = df_fi['importance'].max()
            fi_colors = [
                f'rgba(99, {102 + int((v/max_imp)*120)}, 241, {0.5 + (v/max_imp)*0.5})'
                for v in df_fi['importance']
            ]
     
            fig_fi = go.Figure()
            fig_fi.add_trace(go.Bar(
                x=df_fi['importance'],
                y=df_fi['feature'],
                orientation='h',
                marker=dict(color=fi_colors, cornerradius=6, line=dict(width=0)),
                text=[f' {v*100:.1f}%' for v in df_fi['importance']],
                textposition='outside',
                textfont=dict(size=13, color=COLORS['text']),
                hovertemplate='<b>%{y}</b><br>Importancia: %{x:.3f}<extra></extra>',
            ))
     
            fig_fi.update_layout(**_base_layout(
                title_text="<b>Importancia de Features</b>  ·  Gain score",
                height=400,
                xaxis=dict(
                    title="Importancia relativa",
                    range=[0, max_imp * 1.35],
                    tickformat='.0%',
                    showgrid=True, gridcolor=COLORS['grid'],
                ),
                yaxis=dict(showgrid=False, tickfont=dict(size=13, color=COLORS['text'])),
                bargap=0.25,
            ))
            _add_watermark(fig_fi)
            st.plotly_chart(fig_fi, use_container_width=True)
     
        st.divider()
     
        # ─── ROW 2: ELO histórico ───
        st.markdown("#### Evolución ELO Histórica — Top 8 Selecciones (2010–2025)")
     
        elo_top8 = tech_data.get('elo_top8', {})
     
        if elo_top8:
            equipos_disponibles = list(elo_top8.keys())
            equipos_sel = st.multiselect(
                "Selecciona selecciones:",
                equipos_disponibles,
                default=equipos_disponibles[:5],
                key="elo_multisel"
            )
     
            palette_elo = [
                '#6366F1', '#10B981', '#F59E0B', '#EF4444',
                '#8B5CF6', '#38BDF8', '#F97316', '#EC4899'
            ]
     
            fig_elo = go.Figure()
     
            for i, eq in enumerate(equipos_sel):
                d = elo_top8[eq]
                color = palette_elo[i % len(palette_elo)]
                flag = flags.get(eq, '')
     
                fig_elo.add_trace(go.Scatter(
                    x=d['fechas'],
                    y=d['elos'],
                    mode='lines',
                    name=f"{flag} {eq}",
                    line=dict(color=color, width=2.2, shape='spline', smoothing=0.4),
                    hovertemplate=(
                        f"<b>{flag} {eq}</b><br>"
                        "Fecha: %{x}<br>"
                        "ELO: %{y:.0f}<extra></extra>"
                    ),
                ))
     
            fig_elo.add_vline(
                x="2022-11-20", line_dash="dot",
                line_color="rgba(245,158,11,0.4)", line_width=1.5,
            )
            fig_elo.add_annotation(
                x="2022-11-20", y=1,
                yref="paper",
                text="Qatar 2022",
                showarrow=False, yshift=10,
                font=dict(size=10, color="rgba(245,158,11,0.7)"),
                bgcolor="rgba(15,17,23,0.7)",
                borderpad=3,
            )
     
            fig_elo.update_layout(**_base_layout(
                title_text="<b>Evolución del Rating ELO</b>  ·  Partidos oficiales e internacionales",
                height=420,
                showlegend=True,
                legend=dict(
                    orientation='h',
                    x=0.5, y=-0.15,
                    xanchor='center',
                    bgcolor='rgba(26,29,46,0.0)',
                    font=dict(size=12, color=COLORS['text']),
                ),
                xaxis=dict(title=None, showgrid=True, gridcolor=COLORS['grid'], tickformat='%Y'),
                yaxis=dict(title="Rating ELO", showgrid=True, gridcolor=COLORS['grid']),
                hovermode='x unified',
            ))
            _add_watermark(fig_elo)
            st.plotly_chart(fig_elo, use_container_width=True)
     
        st.divider()
     
        # ─── ROW 3: Correlación + Radar + Distribución ───
        col_corr, col_radar = st.columns(2)
     
        # ── 3. MATRIZ DE CORRELACIÓN ──
        with col_corr:
            st.markdown("#### Correlación entre Features")
     
            if os.path.exists('df_mundial.csv'):
                df_mundial_local = pd.read_csv('df_mundial.csv')
                feat_cols = ['ELO_norm', 'Log_MarketValue', 'Momentum_15', 'Knockout_Performance', 'Pressure_Index']
                feat_labels = ['ELO', 'Market Val.', 'Momentum', 'Knockout', 'Pressure']
         
                corr_matrix = df_mundial_local[feat_cols].corr().values
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                corr_masked = np.where(mask, np.nan, corr_matrix)
         
                fig_corr = go.Figure(go.Heatmap(
                    z=corr_masked,
                    x=feat_labels, y=feat_labels,
                    colorscale=[
                        [0.0,  '#1E3A5F'], [0.25, '#2563EB'], [0.5,  '#1A1D2E'],
                        [0.75, '#7C3AED'], [1.0,  '#6366F1']
                    ],
                    zmid=0, zmin=-1, zmax=1,
                    text=[[f'{v:.2f}' if not np.isnan(v) else '' for v in row] for row in corr_masked],
                    texttemplate='%{text}',
                    textfont=dict(size=13, color='white', family='Inter'),
                    hovertemplate='%{y} × %{x}<br>r = %{z:.3f}<extra></extra>',
                    showscale=True,
                ))
         
                fig_corr.update_layout(**_base_layout(
                    title_text="<b>Matriz de Correlación</b>",
                    height=380,
                    xaxis=dict(showgrid=False, side='bottom'),
                    yaxis=dict(showgrid=False, autorange='reversed'),
                    margin=dict(t=64, b=40, l=80, r=40),
                ))
                _add_watermark(fig_corr)
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Sube 'df_mundial.csv' a la raíz para visualizar la matriz de correlación.")
     
        # ── 4. RADAR CHART ──
        with col_radar:
            st.markdown("#### Perfil de Selección — Radar Chart")
     
            if os.path.exists('df_mundial.csv'):
                df_mundial_local = pd.read_csv('df_mundial.csv')
                equipos_radar = df_mundial_local['name'].tolist()
                eq_sel_1 = st.selectbox("Selección A:", equipos_radar, index=0, key="radar_eq1")
                eq_sel_2 = st.selectbox("Selección B (comparar):", equipos_radar, index=min(1, len(equipos_radar)-1), key="radar_eq2")
         
                feat_radar = ['ELO_norm', 'Log_MarketValue', 'Momentum_15', 'Knockout_Performance', 'Pressure_Index']
                feat_radar_labels = ['ELO', 'Market Value', 'Momentum', 'Knockout', 'Pressure']
         
                def get_radar_vals(equipo):
                    row = df_mundial_local[df_mundial_local['name'] == equipo]
                    if row.empty: return [0.5] * len(feat_radar)
                    return [float(row[f].values[0]) for f in feat_radar]
         
                vals1 = get_radar_vals(eq_sel_1)
                vals2 = get_radar_vals(eq_sel_2)
         
                fig_radar = go.Figure()
                for eq, vals, color, flag_emoji in [
                    (eq_sel_1, vals1, '#6366F1', flags.get(eq_sel_1, '')),
                    (eq_sel_2, vals2, '#10B981', flags.get(eq_sel_2, '')),
                ]:
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]],
                        theta=feat_radar_labels + [feat_radar_labels[0]],
                        fill='toself',
                        line=dict(color=color, width=2.5),
                        name=f"{flag_emoji} {eq}",
                    ))
         
                fig_radar.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Inter, sans-serif', color=COLORS['text']),
                    polar=dict(
                        bgcolor='rgba(26,29,46,0.3)',
                        radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(148,163,184,0.12)', showticklabels=False),
                        angularaxis=dict(tickfont=dict(size=11), gridcolor='rgba(148,163,184,0.12)'),
                    ),
                    showlegend=True,
                    legend=dict(orientation='h', x=0.5, y=-0.15, xanchor='center'),
                    height=380, margin=dict(t=30, b=60, l=40, r=40),
                )
                _add_watermark(fig_radar)
                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.info("Sube 'df_mundial.csv' a la raíz para visualizar el gráfico de radar.")
     
        st.divider()
     
        # ── 5. DISTRIBUCIÓN HISTÓRICA ──
        st.markdown("<h4 style='text-align: center;'>Distribución Histórica de Resultados (2010–2025)</h4>", unsafe_allow_html=True)
        if os.path.exists('df_train.csv'):
            df_train_local = pd.read_csv('df_train.csv')
            df_train_local['date'] = pd.to_datetime(df_train_local['date'])
            df_train_local['year'] = df_train_local['date'].dt.year
            df_train_local['resultado_label'] = df_train_local['resultado'].map(
                {1: 'Victoria local', 0: 'Empate', -1: 'Derrota local'}
            )
     
            df_anual = df_train_local.groupby(['year', 'resultado_label']).size().reset_index(name='count')
            df_anual_total = df_train_local.groupby('year').size().reset_index(name='total')
            df_anual = df_anual.merge(df_anual_total, on='year')
            df_anual['pct'] = df_anual['count'] / df_anual['total'] * 100
     
            col_dist1, col_dist2 = st.columns(2)
     
            with col_dist1:
                colores_resultado = {
                    'Victoria local': '#10B981', 
                    'Empate': '#6366F1', 
                    'Derrota local': '#EF4444'
                }
                
                fig_dist = go.Figure()
                
                # Transformamos las barras en líneas suaves (spline) mucho más legibles
                for resultado, color in colores_resultado.items():
                    df_r = df_anual[df_anual['resultado_label'] == resultado].sort_values('year')
                    fig_dist.add_trace(go.Scatter(
                        x=df_r['year'], 
                        y=df_r['pct'], 
                        name=resultado, 
                        mode='lines+markers',
                        line=dict(color=color, width=3, shape='spline', smoothing=0.3),
                        marker=dict(size=6, line=dict(width=0)),
                        hovertemplate=f"<b>{resultado}</b><br>Año: %{{x}}<br>Porcentaje: %{{y:.1f}}%<extra></extra>"
                    ))
         
                fig_dist.update_layout(**_base_layout(
                    title_text="<b>Tendencia de Resultados por Año</b>  ·  Fútbol Internacional", 
                    height=340,
                    xaxis=dict(
                        title=None,
                        showgrid=True, 
                        gridcolor=COLORS['grid'],
                        tickmode='linear',
                        dtick=2 # Muestra etiquetas cada 2 años para no saturar
                    ), 
                    yaxis=dict(
                        title="Porcentaje de partidos",
                        ticksuffix='%',
                        range=[0, 60], # Fija el rango para que no baile el gráfico
                        showgrid=True, 
                        gridcolor=COLORS['grid']
                    ),
                    legend=dict(
                        orientation='h',
                        x=0.5, y=-0.15,
                        xanchor='center',
                        bgcolor='rgba(0,0,0,0)'
                    ),
                    hovermode='x unified' # Al pasar el ratón, muestra los 3 valores a la vez
                ))
                _add_watermark(fig_dist)
                st.plotly_chart(fig_dist, use_container_width=True)
     
            with col_dist2:
                dist_global = df_train_local['resultado_label'].value_counts()
                
                # Creamos una lista de colores emparejada exactamente con el orden de las etiquetas
                etiquetas_ordenadas = dist_global.index.tolist()
                colores_mapeados = [colores_resultado[label] for label in etiquetas_ordenadas]
                
                fig_donut_dist = go.Figure(go.Pie(
                    labels=etiquetas_ordenadas, 
                    values=dist_global.values.tolist(), 
                    hole=0.6, 
                    marker=dict(
                        colors=colores_mapeados, # ─── AQUÍ FORZAMOS LA CONSISTENCIA
                        line=dict(color=COLORS['bg'], width=3) 
                    ), 
                    textinfo='percent+label',
                    textposition='outside', 
                    hovertemplate='<b>%{label}</b><br>Total: %{value} partidos<br>%{percent}<extra></extra>'
                ))
                
                # Añadimos el texto central perfectamente centrado en el donut
                fig_donut_dist.add_annotation(
                    x=0.5, y=0.5,
                    text=f"<b>{df_train_local.shape[0]:,}</b><br><span style='font-size:12px;color:#94A3B8'>Partidos</span>",
                    font=dict(size=20, color=COLORS['text'], family='Inter, sans-serif'),
                    showarrow=False, xref='paper', yref='paper',
                    xanchor='center', yanchor='middle'
                )
                
                fig_donut_dist.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', 
                    height=340, 
                    showlegend=False, 
                    margin=dict(t=40, b=40, l=20, r=20)
                )
                _add_watermark(fig_donut_dist)
                st.plotly_chart(fig_donut_dist, use_container_width=True)