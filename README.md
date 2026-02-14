# ON NIFTY100 - VaR Forecasting Framework

This project implements a comprehensive framework for Value-at-Risk (VaR) forecasting on NIFTY100 constituents. It compares traditional statistical methods with advanced deep learning approaches (LSTM-MDN) to evaluate market risk during different market conditions (e.g., "Calm" vs. "Covid" periods).

## 📂 Project Structure

- **`r/`**: Core Python codebase (despite the name, these are Python scripts).
  - `initialisation.py`: Configuration file for setting simulation parameters (period, stocks, models).
  - `execution.py`: Main orchestration script for data fetching, model execution, and backtesting.
  - `traditional_models.py` & `GARCH_models.py`: Implementations of Historical, CMM, and GARCH models.
  - `backtesting.py`: VaR backtesting logic (Kupiec, Christoffersen tests).
  - `Visualisations.py`: Plotting modules.
  - `api_yahoo_finance.py`: Data fetching utility using `yfinance`.
  
- **`python/`**: Deep Learning modules.
  - `lstm_mdn_built.py`: Standalone TensorFlow script to train LSTM-Mixture Density Networks (MDN) and generate VaR forecasts.

- **`data/`**: Static data assets.
  - `nifty100_constituents.csv`: List of NIFTY100 stock symbols.
  - `nifty100.py`: Utility script.

- **`NNet_data/`**: Directory where LSTM-MDN model predictions are saved.
- **`r/data_files/`**: Intermediate storage for processed stock data.
- **`r/results/`**: Final summary tables of backtesting results.
- **`r/plots/`**: Generated visualizations.

## 🚀 Setup & Installation

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install numpy pandas scipy matplotlib seaborn yfinance tensorflow
```

## ⚙️ Configuration

Open `r/initialisation.py` to adjust simulation parameters:
- `period`: Set to `"calm"` (2017-2018) or `"covid"` (2021-2022).
- `stocks`: Select specific stocks from the NIFTY100 list.
- `models`: Enable/disable specific models.
- `source_NNet_results`: Set to `True` to include LSTM-MDN results in the final analysis.

## 🏃 Usage Workflow

The project follows a 3-step execution pipeline:

### Step 1: Data Initialization & Baseline Models
Run the execution script to download data from Yahoo Finance and run traditional/GARCH models.
```bash
python r/execution.py
```
*This will create `r/data_files/` with processed CSVs.*

### Step 2: Train Neural Networks (LSTM-MDN)
Run the deep learning script to train the LSTM-MDN models (Vanilla, Regularized, and 3-Component).
```bash
python python/lstm_mdn_built.py
```
*This reads data from `r/data_files/` and saves predictions to `NNet_data/`.*

### Step 3: Final Integration & Backtesting
Run the execution script again to integrate the neural network results and generate final reports.
```bash
python r/execution.py
```
*Ensure `source_NNet_results = True` is set in `r/initialisation.py`.*

## 📊 Outputs

After completing the pipeline, you will find:
- **Visualizations**: `r/plots/` (VaR comparison plots, returns distributions, heatmaps).
- **Backtesting Results**: `r/results/final_tables/` (CSV summaries of model performance).
- **Test Sets**: `r/test_sets/` (Detailed daily VaR forecasts and violations).

## 🧠 Models Implemented

1.  **Historical Simulation**: Non-parametric baseline.
2.  **CMM**: Conditional Moment Model.
3.  **GARCH**: Generalized Autoregressive Conditional Heteroskedasticity (Gaussian/GED).
4.  **LSTM-MDN**: Long Short-Term Memory Network combined with Mixture Density Networks.
    - *Vanilla*: 2-component mixture.
    - *Regularized*: 2-component mixture with regularization.
    - *3-Component*: 3-component mixture (C3).
