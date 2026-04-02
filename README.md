# ML-EV-Energy-Consumption
This project analyzes the application of machine learning models in the electric vehivle sector, specifically analyzing real-world energy consumption patterns.

## Project Overview
Despite widespread adoption of EVs, there still exists uncertainty surrounding the real-world energy consumption of EVs. Energy consumption depends on many interacting factors such as vehicle specifications, charging patterns, environmental conditions, and driving behaviour.
Current energy consumption prediction models are based on ideal scenarios which are not representative of the day-to-day conditions drivers face. This presents a unique challenge for policy makers and consumers, as energy needs of EVs can be ambiguous and over/understated.

This project aims to extract 3 key insights from the 6 data sets analyzed:
1. Predicting EV Range: Estimating how far an EV can travel based on factors including battery level, drivign conditions, and usage patterns.
2. Analyzing Performance Bottlenecks: Identifying factors and conditions that can limit vehicle efficiency with the goal of supporting future improvements.
3. Optimizing Driving Efficiency: Providing insights and reccomnendations to help drivers maximize their energy efficiency and extend battery life.

## Technologies Used
* Python
* Jupyter Notebook
* pandas
* NumPy
* matplotlib
* seaborn
* scikit-learn

To install the main dependencies, run the following:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Reproducability instructions
Clone the repository:
```bash
git clone https://github.com/y3jian/ML-EV-Energy-Consumption.git
cd ML-EV-Energy-Consumption
```
Create & activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate   # linux
.venv\Scripts\activate      # windows
```
Install dependencies

## Running the Code
### 1. Prepare datasets
Open and run the jupyter notebooks:
```bash
jupyter notebook
```
Suggested order:
```text
1-vehicle_emission_data_prep.ipynb
2-Electric_car_norm_data_prep.ipynb
3-EV_population_data_prep.ipynb
4-ev_charging_patterns_data_prep.ipynb
5-EV_energy_consumption_data_prep.ipynb
6-world_gdp_population_co2_emissions_data_prep.ipynb
```
### 2. Run model experiments
#### Predict EV Range
Run:
```bash
python model_implementation/predict_ev_range/ev_range_random_forest.py
python model_implementation/predict_ev_range/ev_range_gradient_boost.py
python model_implementation/predict_ev_range/ev_range_LSTM.py
```
Outputs:
```text
ev_range_rf_results_comparison.csv
ev_range_gb_results_comparison.csv
ev_range_lstm_results_comparison.csv
```
#### Optimize Driving Efficiency
Run:
```bash
python model_implementation/optimize_driving_efficiency/optimize_efficiency_ridge.py
python model_implementation/optimize_driving_efficiency/optimize_efficiency_random_forest.py
python model_implementation/optimize_driving_efficiency/optimize_efficiency_gradient_boosting.py
```
Outputs:
```text
optimize_efficiency_rf_results_comparison.csv
optimize_efficiency_gb_results_comparison.csv
optimize_efficiency_ridge_results_comparison.csv
```
#### Performance Bottlenecks
Run:
```bash
python model_implementation/performance_bottlenecks/performance_bottleneck_knn.py
python model_implementation/performance_bottlenecks/performance_bottlenecks_lasso.py
python model_implementation/performance_bottlenecks/performance_bottlenecks_gradient_boosting.py
python model_implementation/performance_bottlenecks/performance_bottlenecks_random_forest.py
```
Outputs:
```text
summarized in model_implementation/performance_bottlenecks/results.md
```

## Repository Structure
This project organizes several EV-related datasets into a common workflow:
1. Prepare and clean raw datasets,
2. Export processed train/validation/test splits,
3. Train multiple machine learning models,
4. Compare model performance across datasets and tasks.

```text
ML-EV-Energy-Consumption/
│
├── data/
│   ├── ...
│
├── model_implementation/
│   ├── predict_ev_range/
│   │   ├── ev_range_random_forest.py
│   │   ├── ev_range_gradient_boost.py
│   │   ├── ev_range_LSTM.py
│   │   ├── ev_range_knn.py
│   │   ├── ev_range_lasso.py
│   │   └── ev_range_ridge.py
│   │
│   ├── optimize_driving_efficiency/
│   │   ├── optimize_efficiency_data.py
│   │   ├── optimize_efficiency_random_forest.py
│   │   ├── optimize_efficiency_gradient_boosting.py
│   │   └── optimize_efficiency_ridge.py
│   │
│   ├── performance_bottlenecks/
│   │   ├── performance_bottleneck_knn.py
│   │   ├── performance_bottlenecks_lasso.py
│   │   ├── performance_bottlenecks_gradient_boosting.py
│   │   ├── performance_bottlenecks_random_forest.py
│   │   └── results.md
│   │
│   └── compare_behaviour/
│       └── README.md
│
├── notebooks/
│   ├── 1-vehicle_emission_data_prep.ipynb
│   ├── 2-Electric_car_norm_data_prep.ipynb
│   ├── 3-EV_population_data_prep.ipynb
│   ├── 4-ev_charging_patterns_data_prep.ipynb
│   ├── 5-EV_energy_consumption_data_prep.ipynb
│   └── 6-world_gdp_population_co2_emissions_data_prep.ipynb
│
├── results / outputs
│   ├── ev_range_*_results_comparison.csv
│   ├── optimize_efficiency_*_results_comparison.csv
│   ├── rmse_comparison.png
│   └── r2_comparison.png
│
├── ev_range_comparison_plots.py 
├── README.md
└── (other config / helper files)
```