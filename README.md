# ML-EV-Energy-Consumption

## Large files (dataset 3)

GitHub blocks files **> 100 MB**. The full cleaned table `data/processed/3-EV_population_data_cleaned.csv` is **gitignored** when it exceeds that limit. The **train/validation/test splits** (`3-EV_population_data_X_{train,val,test}.csv`, `3-EV_population_data_y_{train,val,test}.csv`) are **committed** so clones get modeling-ready data. To rebuild the cleaned file or splits, run **`3-EV_population_data_prep.ipynb`** (Export section) from `data/raw/`.