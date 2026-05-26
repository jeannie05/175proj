import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# read datasets
logfold_raw= pd.read_csv('/content/primary-screen-replicate-collapsed-logfold-change.csv', index_col=0)
drug_metadata= pd.read_csv('/content/biomarkers.csv')

# drop cell lines with missing data NaN
logfold_v2 = logfold_raw.dropna(axis=0).dropna(axis=1)
print(f"Original shape: {logfold_raw.shape}")
print(f"Shape after dropping drug and cell lines with NaNs: {logfold_v2.shape}")

# cleaning drug dataset - clean missing identifiers and strip whitespace from matching columns
drug_metadata = drug_metadata.dropna(subset=['column_name'])
drug_metadata['column_name'] = drug_metadata['column_name'].str.strip()
logfold_v2.columns = logfold_v2.columns.str.strip()

# Extract the alignment keys that exist simultaneously in both datasets
valid_drug_keys = logfold_v2.columns.intersection(drug_metadata['column_name'])

# Filter target matrix to only retain matched compounds
logfold_v3 = logfold_v2[valid_drug_keys]
Y_matrix = logfold_v3.T # to make drug rows (to match biomarker data)

# Set the index of your drug metadata to match the filtered layout
drug_metadata_filtered = drug_metadata[drug_metadata['column_name'].isin(valid_drug_keys)].set_index('column_name')
print(f"Shape of filtered metadata: {drug_metadata_filtered.shape}")

# Fill missing targets with a placeholder string before encoding
drug_metadata_filtered['target'] = drug_metadata_filtered['target'].fillna('Unknown_Target')

# One-hot encode the target proteins
X_features = pd.get_dummies(drug_metadata_filtered['target'], prefix='target')

# Group by the index (column_name) just in case a drug maps to multiple targets
X_features = X_features.groupby(X_features.index).sum()

# Reindex both matrices to ensure identical row order
common_indices = X_features.index.intersection(Y_matrix.index)
X_final = X_features.loc[common_indices]
Y_final = Y_matrix.loc[common_indices]

# Create masks using the drug_category column from your filtered metadata
is_cancer = drug_metadata_filtered.loc[common_indices, 'drug_category'] == 'targeted cancer'
is_noncancer = drug_metadata_filtered.loc[common_indices, 'drug_category'] == 'noncancer'

# Training Set (Targeted Cancer Drugs)
X_train = X_final[is_cancer].values
Y_train = Y_final[is_cancer].values

# Testing Set (Noncancer Drugs)
X_test = X_final[is_noncancer].values
Y_test = Y_final[is_noncancer].values

print(f"Training set size (Cancer drugs): {X_train.shape[0]}")
print(f"Testing set size (Noncancer drugs): {X_test.shape[0]}")

