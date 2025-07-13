import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import io
import base64

# =============================================================================
# Inisialisasi Aplikasi Dash
# =============================================================================
app = dash.Dash(__name__)
server = app.server # Baris ini penting untuk deployment di platform seperti Heroku/Render

# =============================================================================
# Data Contoh dan Fungsi Backend (Tidak Berubah)
# =============================================================================
R_GAS_CONSTANT = 8.314
sample_csv_data = """Time,Fraction
0,0
9.424369748,0.043697479
18.84873949,0.155462185
28.27310924,0.329411765
37.69747899,0.542857143
47.12184874,0.746218487
56.54621849,0.884033613
65.97058823,0.957983193
75.39495798,0.988235294
84.81932773,0.998319328
"""

# Fungsi kalkulasi model (bisa di-cache di Dash, tapi kita sederhanakan dulu)
def calculate_avrami(n, k, t_points=500):
    if k <= 1e-9 or n <= 1e-9: return np.array([0]), np.array([0])
    t_max = ((-np.log(1 - 0.99)) / k)**(1 / n)
    t = np.linspace(0, t_max * 1.1, t_points)
    y = 1 - np.exp(-k * (t**n))
    return t, y

def calculate_weibull(T_val, b_val, t_points=500):
    if T_val <= 1e-9 or b_val <= 1e-9: return np.array([0]), np.array([0])
    t_max = T_val * (-np.log(1 - 0.99))**(1 / b_val)
    t = np.linspace(0, t_max * 1.1, t_points)
    y = 1 - np.exp(-(t / T_val)**b_val)
    return t, y

def calculate_sbm(A, Ea, T, m, n):
    k = A * np.exp(-Ea / (R_GAS_CONSTANT * T))
    def sbm_ode(t, alpha):
        alpha_clipped = np.clip(alpha, 1e-9, 1 - 1e-9)
        return k * (alpha_clipped**m) * ((1 - alpha_clipped)**n)
    reach_99 = lambda t, y: y[0] - 0.99
    reach_99.terminal = True
    sol = solve_ivp(sbm_ode, [0, 5000], [1e-9], dense_output=True, events=reach_99)
    return sol.t, sol.y[0]

def parse_contents(contents, filename):
    """Fungsi untuk memproses file upload."""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        return df.iloc[:, 0].values, df.iloc[:, 1].values
    except Exception as e:
        print(e)
        return None, None

# =============================================================================
# Layout Aplikasi Dash
# =============================================================================
app.layout = html.Div(style={'fontFamily': 'sans-serif'}, children=[
    # Kolom Kontrol (Sidebar)
    html.Div(style={'width': '24%', 'float': 'left', 'padding': '1%'}, children=[
        html.H2("⚙️ Kontrol Model"),
        html.Hr(),
        html.P("Pilih Model Kinetik:"),
        dcc.Dropdown(
            id='model-selector',
            options=[
                {'label': 'Avrami', 'value': 'avrami'},
                {'label': 'Weibull', 'value': 'weibull'},
                {'label': 'SBM', 'value': 'sbm'}
            ],
            value='avrami',
            clearable=False
        ),

        # Panel Parameter (ditampilkan/disembunyikan secara dinamis)
        html.Div(id='panel-avrami', children=[
            html.H5("Parameter Avrami", style={'marginTop': '20px'}),
            html.P("Exponent (n)"),
            dcc.Slider(id='n-avrami', min=0.5, max=5.0, value=2.0, step=0.01, marks={i: str(i) for i in range(1, 6)}),
            html.P("Rate Constant (k)"),
            dcc.Slider(id='k-avrami', min=0.01, max=5.0, value=0.5, step=0.01, marks={i: str(i) for i in range(1, 6)}),
        ]),
        html.Div(id='panel-weibull', style={'display': 'none'}, children=[
            html.H5("Parameter Weibull", style={'marginTop': '20px'}),
            html.P("Scale / Time (T)"),
            dcc.Slider(id='T-weibull', min=0.1, max=100.0, value=10.0, step=0.1),
            html.P("Shape (b)"),
            dcc.Slider(id='b-weibull', min=0.1, max=10.0, value=1.5, step=0.01),
        ]),
        html.Div(id='panel-sbm', style={'display': 'none'}, children=[
            html.H5("Parameter SBM", style={'marginTop': '20px'}),
            html.P("Pre-exponential (A)"),
            dcc.Input(id='A-sbm', type='number', value=1e10, style={'width':'100%'}),
            html.P("Activation Energy (Ea) J/mol", style={'marginTop': '10px'}),
            dcc.Input(id='Ea-sbm', type='number', value=80000, step=1000, style={'width':'100%'}),
            html.P("Temperature (T) K", style={'marginTop': '10px'}),
            dcc.Slider(id='T-sbm', min=273, max=1500, value=500, step=1),
            html.P("Exponent (m)", style={'marginTop': '10px'}),
            dcc.Slider(id='m-sbm', min=0, max=5, value=1.0, step=0.01),
            html.P("Exponent (n)", style={'marginTop': '10px'}),
            dcc.Slider(id='n-sbm', min=0, max=5, value=1.0, step=0.01),
        ]),
        
        html.Hr(style={'marginTop': '20px'}),
        html.H4("📊 Data Eksperimental"),
        dcc.Checklist(
            id='sample-data-checklist',
            options=[{'label': ' Gunakan Data Contoh', 'value': 'use_sample'}],
            value=['use_sample']
        ),
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag and Drop atau ', html.A('Pilih File CSV')]),
            style={
                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '1px', 'borderStyle': 'dashed',
                'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px 0'
            },
        )
    ]),

    # Kolom Grafik Utama
    html.Div(style={'width': '74%', 'float': 'right', 'padding': '1%'}, children=[
        html.H1("🔬 Universal Kinetic Modeler"),
        html.P("Aplikasi web interaktif untuk memodelkan data kinetik transformasi. Dibuat oleh JeryThom."),
        dcc.Graph(id='kinetic-graph')
    ])
])

# =============================================================================
# Callbacks untuk Interaktivitas
# =============================================================================

# Callback 1: Mengatur panel kontrol mana yang terlihat
@app.callback(
    Output('panel-avrami', 'style'),
    Output('panel-weibull', 'style'),
    Output('panel-sbm', 'style'),
    Input('model-selector', 'value')
)
def toggle_parameter_panels(model):
    if model == 'avrami':
        return {'display': 'block'}, {'display': 'none'}, {'display': 'none'}
    elif model == 'weibull':
        return {'display': 'none'}, {'display': 'block'}, {'display': 'none'}
    elif model == 'sbm':
        return {'display': 'none'}, {'display': 'none'}, {'display': 'block'}

# Callback 2: Callback utama untuk memperbarui grafik
@app.callback(
    Output('kinetic-graph', 'figure'),
    # Input dari semua kontrol
    Input('model-selector', 'value'),
    Input('n-avrami', 'value'), Input('k-avrami', 'value'),
    Input('T-weibull', 'value'), Input('b-weibull', 'value'),
    Input('A-sbm', 'value'), Input('Ea-sbm', 'value'), Input('T-sbm', 'value'), 
    Input('m-sbm', 'value'), Input('n-sbm', 'value'),
    Input('sample-data-checklist', 'value'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_graph(model, n_av, k_av, T_we, b_we, A_sbm, Ea_sbm, T_sbm, m_sbm, n_sbm,
                 checklist_value, upload_contents, upload_filename):
    
    fig = go.Figure()
    
    # 1. Tentukan dan plot data eksperimental
    time_exp, frac_exp = None, None
    if 'use_sample' in checklist_value:
        df = pd.read_csv(io.StringIO(sample_csv_data))
        time_exp, frac_exp = df.iloc[:, 0].values, df.iloc[:, 1].values
    elif upload_contents is not None:
        time_exp, frac_exp = parse_contents(upload_contents, upload_filename)

    if time_exp is not None and frac_exp is not None:
        fig.add_trace(go.Scatter(
            x=time_exp, y=frac_exp, mode='markers', name='Data Eksperimental',
            marker=dict(color='red', size=8, symbol='circle-open')
        ))

    # 2. Hitung dan plot kurva model
    t_model, y_model = [], []
    if model == 'avrami':
        t_model, y_model = calculate_avrami(n_av, k_av)
    elif model == 'weibull':
        t_model, y_model = calculate_weibull(T_we, b_we)
    elif model == 'sbm':
        # Pastikan input tidak None sebelum kalkulasi
        if all(v is not None for v in [A_sbm, Ea_sbm, T_sbm, m_sbm, n_sbm]):
            t_model, y_model = calculate_sbm(A_sbm, Ea_sbm, T_sbm, m_sbm, n_sbm)

    fig.add_trace(go.Scatter(
        x=t_model, y=y_model, mode='lines', name=f'Model ({model.upper()})',
        line=dict(color='royalblue', width=3)
    ))

    # 3. Atur layout, judul, dan watermark
    fig.update_layout(
        title_text='Perbandingan Model Kinetik dengan Data Eksperimental',
        xaxis_title='Waktu (t)',
        yaxis_title='Fraksi Transformasi (α)',
        yaxis_range=[-0.05, 1.1],
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
        template='plotly_white',
        annotations=[
            go.layout.Annotation(
                text="JeryThom program",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=50, color="rgba(128, 128, 128, 0.2)"),
                textangle=-30
            )
        ]
    )
    
    return fig

# =============================================================================
# Menjalankan Aplikasi
# =============================================================================
if __name__ == '__main__':
    app.run(debug=True)

