import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ltgm_model import DynamicLTGM

# App configuration
st.set_page_config(
    page_title="World Bank LTGM Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🌍 World Bank Long-Term Growth Model Simulator")
st.markdown("""
    **Simulate global economic scenarios** using the World Bank's Long-Term Growth Model.
    Adjust parameters in the sidebar to explore different policy options.
""")

# Sidebar for parameters
with st.sidebar:
    st.header("⚙️ Model Parameters")
    years = st.slider("Projection Horizon (years)", 10, 100, 50)
    
    # Core parameters
    alpha = st.slider("Capital Share (α)", 0.2, 0.5, 0.35, 0.01)
    delta = st.slider("Depreciation Rate (δ)", 0.01, 0.10, 0.04, 0.01)
    
    # Savings rate controls
    st.subheader("Savings Rate")
    s_init = st.number_input("Initial Savings Rate", 0.15, 0.40, 0.26)
    s_final = st.number_input("Final Savings Rate", 0.10, 0.35, 0.22)
    
    # Labor growth controls
    st.subheader("Labor Force Growth")
    n_init = st.number_input("Initial Growth (%)", 0.1, 5.0, 1.0) / 100
    n_final = st.number_input("Final Growth (%)", -1.0, 3.0, 0.3) / 100
    
    # Education parameters
    st.subheader("Human Capital")
    phi = st.slider("Return to Education (φ)", 0.01, 0.10, 0.065, 0.005)
    S_init = st.number_input("Initial Schooling Years", 0.0, 20.0, 8.2)
    S_final = st.number_input("Final Schooling Years", 0.0, 20.0, 10.5)
    
    # TFP parameters
    st.subheader("Productivity Growth")
    g = st.number_input("TFP Growth Rate (%)", 0.0, 5.0, 1.2) / 100
    
    # Initial conditions
    st.subheader("Initial Conditions (2023)")
    K0 = st.number_input("Capital Stock (Trillion USD)", 100.0, 500.0, 327.0)
    L0 = st.number_input("Labor Force (Billions)", 1.0, 10.0, 3.4)
    
    # Generate time-series parameters
    s_series = np.linspace(s_init, s_final, years)
    n_series = np.linspace(n_init, n_final, years)
    S_series = np.linspace(S_init, S_final, years)

# Run simulation button
if st.button("▶️ Run Simulation", use_container_width=True):
    with st.spinner("Simulating economic scenarios..."):
        # Prepare parameters
        params = {
            'alpha': alpha,
            'delta': delta,
            's': s_series,
            'n': n_series,
            'g': g,
            'phi': phi,
            'S': S_series,
            'T': years,
            'K0': K0 * 1e12,  # Convert to USD
            'L0': L0 * 1e9,   # Convert to workers
            'A0': 1.0
        }
        
        # Run model
        model = DynamicLTGM(params)
        model.run_simulation()
        results = model.get_dataframe()
        
        # Store in session state
        st.session_state.results = results
        st.session_state.params = params

# Display results if available
if 'results' in st.session_state:
    results = st.session_state.results
    params = st.session_state.params
    
    # Convert to trillions for display
    display_df = results.copy()
    display_df['GDP'] = results['GDP'] / 1e12
    display_df['Capital_Stock'] = results['Capital_Stock'] / 1e12
    display_df['Labor_Force'] = results['Labor_Force'] / 1e9
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Final GDP", f"${display_df['GDP'].iloc[-1]:,.1f}T")
    col2.metric("GDP per Worker", f"${results['Output_per_Worker'].iloc[-1]:,.0f}")
    col3.metric("Capital Stock", f"${display_df['Capital_Stock'].iloc[-1]:,.1f}T")
    
    # Main chart
    st.subheader("Economic Projection")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(display_df['Year'], display_df['GDP'], 'b-', label='GDP')
    ax.plot(display_df['Year'], display_df['Capital_Stock'], 'g--', label='Capital Stock')
    ax.set_title(f"{years}-Year Economic Projection")
    ax.set_xlabel("Year")
    ax.set_ylabel("Trillion USD")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["GDP Analysis", "Productivity", "Download Data"])
    
    with tab1:
        st.subheader("GDP Components")
        col1, col2 = st.columns(2)
        
        with col1:
            st.line_chart(display_df, x='Year', y=['GDP', 'Capital_Stock'])
        
        with col2:
            st.line_chart(display_df, x='Year', y='Output_per_Worker')
    
    with tab2:
        st.subheader("Productivity Drivers")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(results['Year'], results['TFP'], 'm-', label='TFP')
        ax.plot(results['Year'], results['Human_Capital'], 'c--', label='Human Capital')
        ax.set_title("Productivity Factors")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        
        st.progress(int(params['phi'] * 100), text="Education Impact Strength")
    
    with tab3:
        st.subheader("Download Results")
        csv = results.to_csv(index=False)
        st.download_button(
            label="📥 Export CSV",
            data=csv,
            file_name='ltgm_simulation.csv',
            mime='text/csv'
        )
        
        st.dataframe(display_df.style.format({
            'GDP': '${:,.1f}T',
            'Capital_Stock': '${:,.1f}T',
            'Output_per_Worker': '${:,.0f}',
            'Labor_Force': '{:,.1f}B'
        }))
else:
    st.info("Configure parameters in the sidebar and click 'Run Simulation'")

# Footer
st.divider()
st.caption("World Bank Long-Term Growth Model Simulator | v1.0 | Developed for Policy Analysis")
