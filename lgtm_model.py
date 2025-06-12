import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DynamicLTGM:
    def __init__(self, params):
        # Unpack parameters with validation
        self.alpha = params['alpha']       # Capital share
        self.delta = params['delta']       # Depreciation rate
        self.s = params['s']               # Savings rate (scalar or time-series)
        self.n = params['n']               # Labor growth rate (scalar or time-series)
        self.g = params['g']               # TFP growth rate (scalar or time-series)
        self.phi = params['phi']           # Return to education
        self.S = params['S']               # Schooling years (scalar or time-series)
        self.T = params['T']               # Time horizon
        self.K0 = params['K0']             # Initial capital
        self.L0 = params['L0']             # Initial labor
        self.A0 = params['A0']             # Initial TFP
        
        # Initialize arrays
        self.time = np.arange(self.T)
        self.Y = np.zeros(self.T)          # Output
        self.K = np.zeros(self.T)          # Capital
        self.L = np.zeros(self.T)          # Labor
        self.A = np.zeros(self.T)          # TFP
        self.h = np.zeros(self.T)          # Human capital
        self.y = np.zeros(self.T)          # Output per worker
        self.k = np.zeros(self.T)          # Capital per effective worker
        
        # Handle scalar vs time-series parameters
        self.s_array = self.s if isinstance(self.s, np.ndarray) else np.full(self.T, self.s)
        self.n_array = self.n if isinstance(self.n, np.ndarray) else np.full(self.T, self.n)
        self.g_array = self.g if isinstance(self.g, np.ndarray) else np.full(self.T, self.g)
        self.S_array = self.S if isinstance(self.S, np.ndarray) else np.full(self.T, self.S)
        
        # Set initial conditions (fully utilize starting values)
        self.K[0] = self.K0
        self.L[0] = self.L0
        self.A[0] = self.A0
        self.h[0] = np.exp(self.phi * self.S_array[0])

    def production_function(self, K, A, L, h):
        """Cobb-Douglas production function with human capital"""
        return A * (K ** self.alpha) * ((L * h) ** (1 - self.alpha))

    def run_simulation(self):
        """Run dynamic simulation with time-varying parameters"""
        for t in range(self.T):
            # Compute current output (use all parameters)
            self.Y[t] = self.production_function(self.K[t], self.A[t], self.L[t], self.h[t])
            self.y[t] = self.Y[t] / self.L[t]  # Output per worker
            self.k[t] = self.K[t] / (self.L[t] * self.A[t])  # Capital per effective worker
            
            # Update next period if not final
            if t < self.T - 1:
                # Capital accumulation (using current savings rate)
                self.K[t+1] = (1 - self.delta) * self.K[t] + self.s_array[t] * self.Y[t]
                
                # Labor growth (using current growth rate)
                self.L[t+1] = self.L[t] * np.exp(self.n_array[t])
                
                # TFP growth (using current growth rate)
                self.A[t+1] = self.A[t] * np.exp(self.g_array[t])
                
                # Human capital update (using current schooling years)
                self.h[t+1] = np.exp(self.phi * self.S_array[t+1])

    def plot_results(self, log_scale=False):
        """Plot comprehensive results with optional log scale"""
        fig, ax = plt.subplots(3, 2, figsize=(15, 12))
        
        # Output and productivity
        ax[0, 0].plot(self.Y, 'b-', linewidth=2)
        ax[0, 0].set_title('GDP (Y)')
        ax[0, 0].grid(True)
        
        ax[0, 1].plot(self.y, 'r-', linewidth=2)
        ax[0, 1].set_title('Output per Worker (y)')
        ax[0, 1].grid(True)
        
        # Factors of production
        ax[1, 0].plot(self.K, 'g-', linewidth=2)
        ax[1, 0].set_title('Capital Stock (K)')
        ax[1, 0].grid(True)
        
        ax[1, 1].plot(self.A, 'm-', linewidth=2)
        ax[1, 1].set_title('Total Factor Productivity (A)')
        ax[1, 1].grid(True)
        
        # Human capital and intensity
        ax[2, 0].plot(self.h, 'c-', linewidth=2)
        ax[2, 0].set_title('Human Capital per Worker (h)')
        ax[2, 0].grid(True)
        
        ax[2, 1].plot(self.k, 'k-', linewidth=2)
        ax[2, 1].set_title('Capital per Effective Worker (k)')
        ax[2, 1].grid(True)
        
        if log_scale:
            for row in ax:
                for col in row:
                    col.set_yscale('log')
        
        plt.tight_layout()
        plt.show()
        
        # Growth rates visualization
        growth_rates = pd.DataFrame({
            'Labor Growth': np.diff(np.log(self.L)) * 100,
            'TFP Growth': np.diff(np.log(self.A)) * 100,
            'Output Growth': np.diff(np.log(self.Y)) * 100
        }).rolling(5).mean()
        
        growth_rates.plot(title='5-Year Moving Average Growth Rates (%)', 
                         figsize=(12, 6), grid=True)
        plt.ylabel('Annual Growth Rate (%)')
        plt.show()

    def steady_state_k(self):
        """Calculate steady-state capital per effective worker"""
        effective_depreciation = self.delta + np.mean(self.n_array) + np.mean(self.g_array)
        return (np.mean(self.s_array) / effective_depreciation) ** (1 / (1 - self.alpha))
    
    def convergence_analysis(self):
        """Compare actual path to steady state"""
        ss_k = self.steady_state_k()
        plt.figure(figsize=(10, 6))
        plt.plot(self.k, 'b-', label='Actual Path')
        plt.axhline(y=ss_k, color='r', linestyle='--', label='Steady State')
        plt.title('Convergence to Steady State')
        plt.xlabel('Time')
        plt.ylabel('Capital per Effective Worker')
        plt.legend()
        plt.grid(True)
        plt.show()

# Example Usage with Time-Varying Parameters
if __name__ == "__main__":
    T = 50  # 50-year simulation
    
    # Create time-varying parameters (example: schooling increases, savings declines)
    params = {
        'alpha': 0.35,
        'delta': 0.04,
        's': np.linspace(0.25, 0.18, T),  # Declining savings rate
        'n': np.full(T, 0.01),             # Constant labor growth
        'g': np.linspace(0.02, 0.015, T),  # Gradually slowing TFP growth
        'phi': 0.065,
        'S': np.linspace(8, 12, T),        # Increasing schooling years
        'T': T,
        'K0': 5000,
        'L0': 1000,
        'A0': 1.0
    }
    
    # Run simulation
    model = DynamicLTGM(params)
    model.run_simulation()
    
    # Analyze results
    print(f"Final Output: {model.Y[-1]:,.2f}")
    print(f"Final Output per Worker: {model.y[-1]:,.2f}")
    print(f"Steady-State k: {model.steady_state_k():.4f}")
    
    # Visualizations
    model.plot_results(log_scale=True)
    model.convergence_analysis()
    
    # Export full results
    results = pd.DataFrame({
        'Year': model.time,
        'GDP': model.Y,
        'Capital': model.K,
        'Labor': model.L,
        'TFP': model.A,
        'Human_Capital': model.h,
        'Output_per_Worker': model.y,
        'Savings_Rate': model.s_array,
        'Schooling_Years': model.S_array
    })
    results.to_csv('dynamic_ltgm_simulation.csv', index=False)
