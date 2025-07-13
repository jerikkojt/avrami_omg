import streamlit as st
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# =============================================================================
# Konfigurasi Halaman & Judul
# =============================================================================
st.set_page_config(layout="wide")
st.title("🔬 Universal Kinetic Modeler")
st.write("Aplikasi web interaktif untuk memodelkan data kinetik transformasi.")

# =is_initial_run =============================================================================
# Logika Fungsi Model (Backend) - Hampir tidak berubah
# =============================================================================
R_GAS_CONSTANT = 8.314

def calculate_avrami(n, k, t_points=500):
    """Menghitung model Avrami."""
    # Hindari pembagian dengan nol atau log dari angka non-positif
    if k <= 1e-9 or n <= 1e-9:
        return np.array([0]), np.array([0])
    # Tentukan waktu maksimum untuk mencapai 99% konversi
    t_max = ((-np.log(1 - 0.99)) / k)**(1 / n)
    t = np.linspace(0, t_max * 1.1, t_points)
    y = 1 - np.exp(-k * (t**n))
    return t, y

def calculate_weibull(T_val, b_val, t_points=500):
    """Menghitung model Weibull."""
    if T_val <= 1e-9 or b_val <= 1e-9:
        return np.array([0]), np.array([0])
    t_max = T_val * (-np.log(1 - 0.99))**(1 / b_val)
    t = np.linspace(0, t_max * 1.1, t_points)
    y = 1 - np.exp(-(t / T_val)**b_val)
    return t, y

def calculate_sbm(A, Ea, T, m, n):
    """Menghitung model SBM (Šesták-Berggren model)."""
    k = A * np.exp(-Ea / (R_GAS_CONSTANT * T))
    def sbm_ode(t, alpha):
        # Clip alpha untuk menghindari error numerik di batas 0 dan 1
        alpha_clipped = np.clip(alpha, 1e-9, 1 - 1e-9)
        return k * (alpha_clipped**m) * ((1 - alpha_clipped)**n)
    
    # Event untuk berhenti saat konversi mencapai 99%
    def reach_99(t, y):
        return y[0] - 0.99
    reach_99.terminal = True # Hentikan integrasi saat event terjadi

    sol = solve_ivp(
        sbm_ode, 
        [0, 5000], # Batas waktu integrasi yang wajar
        [1e-9],    # Kondisi awal alpha
        dense_output=True,
        events=reach_99
    )
    return sol.t, sol.y[0]


# =============================================================================
# UI (User Interface) - Dibuat dengan Streamlit di Sidebar
# =============================================================================
st.sidebar.header("⚙️ Kontrol Model")

# 1. Pilihan Model (Menggantikan QComboBox)
model_selector = st.sidebar.selectbox(
    "Pilih Model Kinetik",
    ("Avrami", "Weibull", "SBM")
)

# 2. Kontrol Parameter (Menggantikan QSlider dan QDoubleSpinBox)
if model_selector == "Avrami":
    st.sidebar.subheader("Parameter Avrami")
    n_avrami = st.sidebar.slider("Exponent (n)", 0.5, 5.0, 2.0, 0.01)
    k_avrami = st.sidebar.slider("Rate Constant (k)", 0.01, 5.0, 0.5, 0.01)
elif model_selector == "Weibull":
    st.sidebar.subheader("Parameter Weibull")
    T_weibull = st.sidebar.slider("Scale / Time (T)", 0.1, 100.0, 10.0, 0.1)
    b_weibull = st.sidebar.slider("Shape (b)", 0.1, 10.0, 1.5, 0.01)
else: # SBM
    st.sidebar.subheader("Parameter SBM")
    A_sbm = st.sidebar.number_input("Pre-exponential (A)", min_value=1e1, max_value=1e25, value=1e10, format="%e")
    Ea_sbm = st.sidebar.number_input("Activation Energy (Ea) J/mol", min_value=1000, max_value=500000, value=80000, step=1000)
    T_sbm = st.sidebar.slider("Temperature (T) K", 273.0, 1500.0, 500.0, 1.0)
    m_sbm = st.sidebar.slider("Exponent (m)", 0.0, 5.0, 1.0, 0.01)
    n_sbm = st.sidebar.slider("Exponent (n)", 0.0, 5.0, 1.0, 0.01)

# 3. Upload Data (Menggantikan QFileDialog)
st.sidebar.header("📊 Data Eksperimental")
uploaded_file = st.sidebar.file_uploader("Upload File CSV Anda", type=["csv"])

# =============================================================================
# Logika Plotting (Backend + Frontend)
# =============================================================================
# Siapkan figure untuk plot
fig, ax = plt.subplots(figsize=(10, 6))

# Hitung kurva model berdasarkan input dari sidebar
try:
    if model_selector == "Avrami":
        t_model, y_model = calculate_avrami(n_avrami, k_avrami)
    elif model_selector == "Weibull":
        t_model, y_model = calculate_weibull(T_weibull, b_weibull)
    else: # SBM
        t_model, y_model = calculate_sbm(A_sbm, Ea_sbm, T_sbm, m_sbm, n_sbm)
    
    if len(t_model) > 1:
        ax.plot(t_model, y_model, lw=2.5, color="royalblue", label=f"Model ({model_selector})")
        ax.set_xlim(0, t_model[-1] * 1.05 if t_model[-1] > 0 else 1)

except Exception as e:
    st.error(f"Terjadi error saat menghitung model: {e}")


# Plot data eksperimental jika file diupload
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        time_exp = df.iloc[:, 0]
        fraction_exp = df.iloc[:, 1]
        ax.scatter(time_exp, fraction_exp, s=30, color='red', alpha=0.7, label="Data Eksperimental", zorder=5)
    except Exception as e:
        st.error(f"Gagal membaca file CSV: {e}")


# Pengaturan Tampilan Plot
ax.set_title("Perbandingan Model Kinetik dengan Data Eksperimental", fontsize=16)
ax.set_xlabel("Waktu (t)", fontsize=12)
ax.set_ylabel("Fraksi Transformasi (α)", fontsize=12)
ax.set_ylim(-0.05, 1.1)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()

# Tampilkan plot di aplikasi Streamlit
st.pyplot(fig)