import streamlit as st
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import io

# =============================================================================
# Konfigurasi Halaman & Judul
# =============================================================================
st.set_page_config(layout="wide", page_title="Universal Kinetic Modeler")
st.title("🔬 Universal Kinetic Modeler")
st.write("Aplikasi web interaktif untuk memodelkan data kinetik transformasi. Dibuat oleh JeryThom.")

# =============================================================================
# Data Contoh (dari file 1 Normalized.csv)
# =============================================================================
# Data dari file CSV Anda disematkan di sini sebagai data contoh.
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

# =============================================================================
# Logika Fungsi Model (Backend) - Sekarang dengan Caching untuk Performa
# =============================================================================
R_GAS_CONSTANT = 8.314

# @st.cache_data akan menyimpan hasil fungsi. Jika input sama, tidak perlu hitung ulang.
@st.cache_data
def calculate_avrami(n, k, t_points=500):
    """Menghitung model Avrami."""
    if k <= 1e-9 or n <= 1e-9:
        return np.array([0]), np.array([0])
    t_max = ((-np.log(1 - 0.99)) / k)**(1 / n)
    t = np.linspace(0, t_max * 1.1, t_points)
    y = 1 - np.exp(-k * (t**n))
    return t, y

@st.cache_data
def calculate_weibull(T_val, b_val, t_points=500):
    """Menghitung model Weibull."""
    if T_val <= 1e-9 or b_val <= 1e-9:
        return np.array([0]), np.array([0])
    t_max = T_val * (-np.log(1 - 0.99))**(1 / b_val)
    t = np.linspace(0, t_max * 1.1, t_points)
    y = 1 - np.exp(-(t / T_val)**b_val)
    return t, y

@st.cache_data
def calculate_sbm(A, Ea, T, m, n):
    """Menghitung model SBM (Šesták-Berggren model)."""
    k = A * np.exp(-Ea / (R_GAS_CONSTANT * T))
    def sbm_ode(t, alpha):
        alpha_clipped = np.clip(alpha, 1e-9, 1 - 1e-9)
        return k * (alpha_clipped**m) * ((1 - alpha_clipped)**n)
    
    def reach_99(t, y):
        return y[0] - 0.99
    reach_99.terminal = True

    sol = solve_ivp(sbm_ode, [0, 5000], [1e-9], dense_output=True, events=reach_99)
    return sol.t, sol.y[0]

# Fungsi untuk memuat data juga di-cache
@st.cache_data
def load_data(data_source):
    """Memuat data dari file yang diupload atau dari data contoh."""
    df = pd.read_csv(data_source)
    return df.iloc[:, 0].values, df.iloc[:, 1].values

# =============================================================================
# UI (User Interface) - Dibuat dengan Streamlit di Sidebar
# =============================================================================
st.sidebar.header("⚙️ Kontrol Model")

model_selector = st.sidebar.selectbox(
    "Pilih Model Kinetik",
    ("Avrami", "Weibull", "SBM")
)

# Kontrol Parameter dinamis berdasarkan model yang dipilih
params = {}
if model_selector == "Avrami":
    st.sidebar.subheader("Parameter Avrami")
    params['n'] = st.sidebar.slider("Exponent (n)", 0.5, 5.0, 2.0, 0.01)
    params['k'] = st.sidebar.slider("Rate Constant (k)", 0.01, 5.0, 0.5, 0.01)
elif model_selector == "Weibull":
    st.sidebar.subheader("Parameter Weibull")
    params['T_val'] = st.sidebar.slider("Scale / Time (T)", 0.1, 100.0, 10.0, 0.1)
    params['b_val'] = st.sidebar.slider("Shape (b)", 0.1, 10.0, 1.5, 0.01)
else: # SBM
    st.sidebar.subheader("Parameter SBM")
    params['A'] = st.sidebar.number_input("Pre-exponential (A)", min_value=1e1, max_value=1e25, value=1e10, format="%e")
    params['Ea'] = st.sidebar.number_input("Activation Energy (Ea) J/mol", min_value=1000, max_value=500000, value=80000, step=1000)
    params['T'] = st.sidebar.slider("Temperature (T) K", 273.0, 1500.0, 500.0, 1.0)
    params['m'] = st.sidebar.slider("Exponent (m)", 0.0, 5.0, 1.0, 0.01)
    params['n'] = st.sidebar.slider("Exponent (n)", 0.0, 5.0, 1.0, 0.01)

# Opsi untuk menggunakan data contoh atau upload file sendiri
st.sidebar.header("📊 Data Eksperimental")
use_sample_data = st.sidebar.checkbox("Gunakan Data Contoh", value=True)

uploaded_file = None
if not use_sample_data:
    uploaded_file = st.sidebar.file_uploader("Upload File CSV Anda", type=["csv"])

# =============================================================================
# Logika Plotting (Backend + Frontend)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Hitung kurva model berdasarkan input dari sidebar
try:
    if model_selector == "Avrami":
        t_model, y_model = calculate_avrami(params['n'], params['k'])
    elif model_selector == "Weibull":
        t_model, y_model = calculate_weibull(params['T_val'], params['b_val'])
    else: # SBM
        t_model, y_model = calculate_sbm(params['A'], params['Ea'], params['T'], params['m'], params['n'])
    
    if len(t_model) > 1:
        ax.plot(t_model, y_model, lw=2.5, color="royalblue", label=f"Model ({model_selector})")
        ax.set_xlim(0, t_model[-1] * 1.05 if t_model[-1] > 0 else 1)
except Exception as e:
    st.error(f"Terjadi error saat menghitung model: {e}")

# Tentukan sumber data (contoh atau upload)
data_source = None
if use_sample_data:
    data_source = io.StringIO(sample_csv_data)
elif uploaded_file is not None:
    data_source = uploaded_file

# Plot data eksperimental jika sumber data tersedia
if data_source:
    try:
        time_exp, fraction_exp = load_data(data_source)
        ax.scatter(time_exp, fraction_exp, s=40, color='red', alpha=0.7, label="Data Eksperimental", zorder=5, edgecolors='black')
    except Exception as e:
        st.error(f"Gagal memproses file data: {e}")

# Pengaturan Tampilan Plot
ax.set_title("Perbandingan Model Kinetik dengan Data Eksperimental", fontsize=16)
ax.set_xlabel("Waktu (t)", fontsize=12)
ax.set_ylabel("Fraksi Transformasi (α)", fontsize=12)
ax.set_ylim(-0.05, 1.1)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()

# --- Menambahkan Watermark ---
fig.text(0.5, 0.5, 'JeryThom program',
         fontsize=50,
         color='gray',
         ha='center', # horizontal alignment
         va='center', # vertical alignment
         alpha=0.15,  # transparansi
         rotation=30) # rotasi watermark

# Tampilkan plot di aplikasi Streamlit
st.pyplot(fig)

