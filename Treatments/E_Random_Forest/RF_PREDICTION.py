"""
Random Forest Regression - ALSFRS-R Interval Prediction Evaluation
===================================================================
This script trains and evaluates a Random Forest regressor on every prediction
file produced by PROACT_INTERVALS_CUT.py, covering both Fixed and Sliding
file types for the ALSFRS-R total score target.

For each CSV file the pipeline:
    1. Loads the file and auto-detects the target column.
    2. Drops rows with a missing target and filters uninformative columns
       (> 80 % missing values or a single value dominating > 80 % of rows).
    3. Encodes categorical columns using known value maps, then falls back to
       LabelEncoder for any remaining object columns.
    4. Runs 10-fold cross-validation with a RandomForestRegressor.
    5. Records MAE, RMSE and R² for both mean-aggregated and median-aggregated
       predictions (median is more robust to outlier trees).
    6. Logs the top-20 feature importances averaged across folds.

Prediction file types
---------------------
Fixed_{T}M.csv
    One file per horizon (3M, 6M, ...).  The target column is the ALSFRS-R
    Central statistic of the interval that immediately follows the last
    observed interval, identified from the horizon encoded in the filename.

Sliding_{T}M.csv
    Sliding-window files where all column names have been made relative
    (T1, T2, ..., Ti+1).  The target column is always
    'ALS_ALSFRS_R_Total_Ti+1_Central'.

Output
------
- Console + log file: cross-validation scores and top-20 feature importances
  for every processed file.
- Feature importance PNG charts per file (currently commented out).

Author: Bouclier Lucas
Data:   PROACT dataset (2022-07-29 release)
"""



import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import os
import matplotlib.pyplot as plt
import sys
from contextlib import redirect_stdout
import re





# ------------------------------------------------------------------
# Dual-output logger
# ------------------------------------------------------------------

class Tee:
    """
    Redirect writes simultaneously to multiple file-like objects.

    Used to mirror terminal output to a log file without duplicating
    every print call.  Both sys.stdout and the log file handle are
    flushed after each write to prevent buffering artefacts.

    Parameters
    ----------
    *fileobjs : file-like
        Any number of objects supporting .write() and .flush().
    """

    def __init__(self, *fileobjs):
        self.fileobjs = fileobjs

    def write(self, msg):
        for fileobj in self.fileobjs:
            fileobj.write(msg)
            fileobj.flush()

    def flush(self):
        for fileobj in self.fileobjs:
            fileobj.flush()





# ------------------------------------------------------------------
# Column quality filters
# ------------------------------------------------------------------

def filter_columns_missing_data(df, threshold):
    """
    Remove columns whose fraction of missing values exceeds a threshold.

    A column is kept only when the proportion of NaN values is at or
    below `threshold`.  For example, threshold=0.8 keeps all columns
    that are at least 20 % filled.

    Parameters
    ----------
    df        : pd.DataFrame   Input DataFrame.
    threshold : float          Maximum allowed missing-value ratio [0, 1].

    Returns
    -------
    pd.DataFrame
        DataFrame with uninformative sparse columns removed.
    """
    missing_ratio = df.isnull().mean()
    keep_missing  = missing_ratio[missing_ratio <= threshold].index

    return df[keep_missing].copy()



def filter_columns_dominant_data(df, threshold):
    """
    Remove columns where a single value accounts for more than a given
    fraction of all non-missing entries.

    Columns dominated by one value carry almost no discriminative signal
    for regression.  For example, threshold=0.8 removes any column where
    one value appears in more than 80 % of rows.

    Parameters
    ----------
    df        : pd.DataFrame   Input DataFrame.
    threshold : float          Maximum allowed dominant-value ratio [0, 1].

    Returns
    -------
    pd.DataFrame
        DataFrame with near-constant columns removed.
    """
    dominant_ratio = df.apply(
        lambda col: col.value_counts(normalize=True).iloc[0]
        if col.notna().sum() > 0 else 1
    )
    keep_dominant = dominant_ratio[dominant_ratio <= threshold].index

    return df[keep_dominant].copy()



def filter_columns_missing_and_dominant_data(df, missing_threshold, dominant_threshold):
    """
    Apply both the missing-value and dominant-value filters in sequence.

    A column is retained only when it passes both quality criteria:
        - missing values <= missing_threshold
        - dominant value frequency <= dominant_threshold

    Parameters
    ----------
    df                 : pd.DataFrame   Input DataFrame.
    missing_threshold  : float          Max missing-value ratio [0, 1].
    dominant_threshold : float          Max dominant-value ratio [0, 1].

    Returns
    -------
    pd.DataFrame
        DataFrame with low-quality columns removed.
    """
    keep_missing  = filter_columns_missing_data(df, missing_threshold).columns
    keep_dominant = filter_columns_dominant_data(df, dominant_threshold).columns

    keep_columns = keep_missing.intersection(keep_dominant)
    return df[keep_columns].copy()





# ------------------------------------------------------------------
# Interval file sort key
# ------------------------------------------------------------------

def sort_key_intervals(filepath):
    """
    Return a sort key that orders interval prediction files by type then
    by prediction horizon in months.

    Fixed files precede Sliding files; within each type, files are sorted
    by ascending horizon (3M < 6M < 9M < ...).  This ensures that
    evaluation results are printed in a logical progression.

    Parameters
    ----------
    filepath : str   Path to a Fixed_*.csv or Sliding_*.csv file.

    Returns
    -------
    tuple[int, int]
        (type_order, months) where type_order=0 for Fixed, 1 for Sliding.
    """
    filename = os.path.basename(filepath)

    # Fixed files are listed before Sliding files
    if filename.startswith("Fixed"):
        type_order = 0
    elif filename.startswith("Sliding"):
        type_order = 1
    else:
        type_order = 2  # safety fallback for unexpected file names

    # Extract the numeric horizon (e.g. 12 from 'Fixed_12M.csv')
    match  = re.search(r'_(\d+)M', filename)
    months = int(match.group(1)) if match else float('inf')

    return (type_order, months)





# ------------------------------------------------------------------
# Categorical encoding
# ------------------------------------------------------------------

def encode_simple_categorical(X):
    """
    Map known categorical values to integers using hard-coded dictionaries.

    Columns whose complete value set matches a known mapping (boolean,
    direction, sex, or a column-specific map) are converted to numeric.
    Columns not covered by any mapping are returned unchanged for the
    auto-encoding pass.

    Parameters
    ----------
    X : pd.DataFrame   Feature matrix with potential object-dtype columns.

    Returns
    -------
    pd.DataFrame
        Feature matrix with known categoricals replaced by integers.
    """
    X_encoded = X.copy()

    # Standard boolean surface: True/False and Yes/No indicators
    bool_map = {
        "True": 1, "False": 0,
        "Yes":  1, "No":    0,
    }

    # Hand dominance comparison (dominant vs non-dominant)
    direction_map = {
        "Left": -1, "Equal": 0, "Right": 1
    }

    # Patient sex
    gender_map = {
        "Male": 1, "Female": 0
    }

    # Column-specific mappings for fields with a natural ordinal scale
    simple_maps = {
        "TRE_Study_Arm": {"Active": 1, "Placebo": 0},
        "ELE_el_escorial": {
            "Definite":                    3,
            "Probable Laboratory Supported": 2,
            "Probable":                    1,
            "Possible":                    0
        },
        "DEM_Ethnicity": {
            "Hispanic or Latino":     1,
            "Non-Hispanic or Latino": 0,
            "Unknown":                np.nan
        }
    }

    for col in X.columns:
        if X[col].dtype in ('object', 'string'):
            unique_vals = X[col].dropna().unique()

            if all(str(v).strip() in bool_map for v in unique_vals):
                X_encoded[col] = X[col].map(
                    lambda x: bool_map.get(str(x).strip(), np.nan)
                )
            elif all(str(v).strip() in direction_map for v in unique_vals):
                X_encoded[col] = X[col].map(
                    lambda x: direction_map.get(str(x).strip(), np.nan)
                )
            elif all(str(v).strip() in gender_map for v in unique_vals):
                X_encoded[col] = X[col].map(
                    lambda x: gender_map.get(str(x).strip(), np.nan)
                )
            elif col in simple_maps:
                mapping = simple_maps[col]
                X_encoded[col] = X[col].map(
                    lambda x: mapping.get(str(x).strip(), np.nan)
                )
            # Columns with no matching map are left for pass 2

    return X_encoded



def encode_auto_categorical(X):
    """
    Apply LabelEncoder to any remaining object-dtype columns.

    Columns with > 300 unique values are dropped rather than encoded,
    as they are likely to be free-text fields (drug names, lab units)
    that carry no usable signal after integer conversion.

    Parameters
    ----------
    X : pd.DataFrame   Feature matrix after pass-1 encoding.

    Returns
    -------
    pd.DataFrame
        Fully numeric feature matrix ready for scikit-learn.
    """
    X_encoded = X.copy()

    for col in X.columns:
        if X[col].dtype in ('object', 'string'):
            if X[col].nunique() > 300:
                # Drop free-text / high-cardinality columns
                print(f"Warning: '{col}' has {X[col].nunique()} unique values "
                    f"and will be dropped (likely free text).")
                X_encoded = X_encoded.drop(columns=[col])
            else:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(
                    X[col].astype(str).fillna("MISSING")
                )

    return X_encoded










def run(INTERVALS_CUT_PATH, RESULT_PATH, FEATURE_IMPORTANCE_PATH, PAPER_RESULTS_ONLY):

    print("\n" * 3)
    print("=" * 60)
    print("INTERVALS CUT PIPELINE")
    print("=" * 60)



    # ------------------------------------------------------------------
    # File collection
    # ------------------------------------------------------------------
    # Gather every CSV file from each prediction directory in the correct
    # evaluation order, repeated for two temporal alignment strategies:
    # TARGET ALSFRS STUDY (aligned to first clinical visit) and
    # TARGET ALSFRS FIRST SYMPTOMS STUDY (aligned to first symptom onset).
    
    # Within each study, directories are visited in the following order:
    #   1. Single temporal tables (ALSFRS, FVC, HANDGRIPSTRENGTH, ...)
    #   2. Single temporal tables + non-temporal features (*_NODELTA)
    #   3. All-table merges, temporal features only (ALL_DELTA_*)
    #   4. All-table merges + non-temporal features (ALL_DATA_*)
    #   5. Pairwise merges: ALSFRS + one other table + non-temporal features
    #   6. Three-table merges: ALSFRS + two other + non-temporal
    #   7. Four-table merges: ALSFRS + three other + non-temporal
    
    # Within each directory, files are sorted Fixed < Sliding, then by
    # ascending horizon.

    files       = []
    temp_files  = []



    # ///////////////////////////////////////////////////////////////////////
    # ///////////////////////////////////////////////////////////////////////
    # ========================= TARGET ALSFRS STUDY =========================
    # ///////////////////////////////////////////////////////////////////////
    # ///////////////////////////////////////////////////////////////////////


    # ////////////////////////////////////////////////////
    # Single temporal tables
    # ////////////////////////////////////////////////////

    if PAPER_RESULTS_ONLY==False:
        for table in ('ALSFRS_Central_Only', 'ALSFRS', 'FVC', 'HANDGRIPSTRENGTH',
                    'MUSCLESTRENGTH', 'LABS', 'SVC', 'VITALSIGNS'):
            for root, dirs, filenames in os.walk(f"{INTERVALS_CUT_PATH}/{table}/"):
                for filename in filenames:
                    if filename.endswith(".csv"):
                        temp_files.append(os.path.join(root, filename))
            temp_files.sort(key=sort_key_intervals)
            files.extend(temp_files)
            temp_files.clear()
    else:
        for table in ('ALSFRS_Central_Only', 'ALSFRS'):
            for root, dirs, filenames in os.walk(f"{INTERVALS_CUT_PATH}/{table}/"):
                for filename in filenames:
                    if filename.endswith(".csv"):
                        temp_files.append(os.path.join(root, filename))
            temp_files.sort(key=sort_key_intervals)
            files.extend(temp_files)
            temp_files.clear()


    if PAPER_RESULTS_ONLY==False:
        # ////////////////////////////////////////////////////
        # Single temporal tables + non-temporal features
        # ////////////////////////////////////////////////////

        for table in ('ALSFRS_NODELTA', 'FVC_NODELTA', 'HANDGRIPSTRENGTH_NODELTA',
                    'MUSCLESTRENGTH_NODELTA', 'LABS_NODELTA', 'SVC_NODELTA',
                    'VITALSIGNS_NODELTA'):
            for root, dirs, filenames in os.walk(f"{INTERVALS_CUT_PATH}/{table}/"):
                for filename in filenames:
                    if filename.endswith(".csv"):
                        temp_files.append(os.path.join(root, filename))
            temp_files.sort(key=sort_key_intervals)
            files.extend(temp_files)
            temp_files.clear()


    if PAPER_RESULTS_ONLY==False:
        # ////////////////////////////////////////////////////
        # All-table merges (temporal features only)
        # ////////////////////////////////////////////////////

        for merge_dir in ('ALL_DELTA_INTERSECTION', 'ALL_DELTA_LEFT', 'ALL_DELTA_UNION'):
            for root, dirs, filenames in os.walk(f"{INTERVALS_CUT_PATH}/{merge_dir}/"):
                for filename in filenames:
                    if filename.endswith(".csv"):
                        temp_files.append(os.path.join(root, filename))
            temp_files.sort(key=sort_key_intervals)
            files.extend(temp_files)
            temp_files.clear()


    # ////////////////////////////////////////////////////
    # All-table merges + non-temporal features
    # ////////////////////////////////////////////////////

    if PAPER_RESULTS_ONLY==False:
        for merge_dir in ('ALL_DATA_INTERSECTION', 'ALL_DATA_LEFT', 'ALL_DATA_UNION'):
            for root, dirs, filenames in os.walk(f"{INTERVALS_CUT_PATH}/{merge_dir}/"):
                for filename in filenames:
                    if filename.endswith(".csv"):
                        temp_files.append(os.path.join(root, filename))
            temp_files.sort(key=sort_key_intervals)
            files.extend(temp_files)
            temp_files.clear()
    else:
        for merge_dir in ('ALL_DATA_UNION'):
            for root, dirs, filenames in os.walk(f"{INTERVALS_CUT_PATH}/{merge_dir}/"):
                for filename in filenames:
                    if filename.endswith(".csv"):
                        temp_files.append(os.path.join(root, filename))
            temp_files.sort(key=sort_key_intervals)
            files.extend(temp_files)
            temp_files.clear()


    # ////////////////////////////////////////////////////
    # Pairwise merges: ALSFRS + one other table + non-temporal features
    # ////////////////////////////////////////////////////

    if PAPER_RESULTS_ONLY==False:
        for merge_dir in (
            'MERGE_ALSFRS_NODELTA_FVC_UNION',               'MERGE_ALSFRS_NODELTA_LABS_UNION', 
            'MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_UNION',  'MERGE_ALSFRS_NODELTA_MUSCLESTRENGTH_UNION',    
            'MERGE_ALSFRS_NODELTA_SVC_UNION',               'MERGE_ALSFRS_NODELTA_VITALSIGNS_UNION',

            'MERGE_ALSFRS_NODELTA_FVC_INTERSECTION',                'MERGE_ALSFRS_NODELTA_LABS_INTERSECTION',               
            'MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_INTERSECTION',   'MERGE_ALSFRS_NODELTA_MUSCLESTRENGTH_INTERSECTION',     
            'MERGE_ALSFRS_NODELTA_SVC_INTERSECTION',                'MERGE_ALSFRS_NODELTA_VITALSIGNS_INTERSECTION',
            
            'MERGE_ALSFRS_NODELTA_FVC_LEFT',                'MERGE_ALSFRS_NODELTA_LABS_LEFT',
            'MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_LEFT',   'MERGE_ALSFRS_NODELTA_MUSCLESTRENGTH_LEFT',
            'MERGE_ALSFRS_NODELTA_SVC_LEFT',                'MERGE_ALSFRS_NODELTA_VITALSIGNS_LEFT',
        ):
            for root, dirs, filenames in os.walk(f"{INTERVALS_CUT_PATH}/{merge_dir}/"):
                for filename in filenames:
                    if filename.endswith(".csv"):
                        temp_files.append(os.path.join(root, filename))
            temp_files.sort(key=sort_key_intervals)
            files.extend(temp_files)
            temp_files.clear()
    else:
        for merge_dir in ('MERGE_ALSFRS_NODELTA_SVC_INTERSECTION'):
            for root, dirs, filenames in os.walk(f"{INTERVALS_CUT_PATH}/{merge_dir}/"):
                for filename in filenames:
                    if filename.endswith(".csv"):
                        temp_files.append(os.path.join(root, filename))
            temp_files.sort(key=sort_key_intervals)
            files.extend(temp_files)
            temp_files.clear()


    # ////////////////////////////////////////////////////
    # Three-table merges: ALSFRS + HANDGRIPSTRENGTH 
    # + one other table + non-temporal features 
    # UNION & LEFT only / FIXED windows only
    # ////////////////////////////////////////////////////

    if PAPER_RESULTS_ONLY==False:
        for merge_dir in (
            'MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_FVC_UNION',              'MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_LABS_UNION',         
            'MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_MUSCLESTRENGTH_UNION',   'MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_SVC_UNION',          
            'MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_VITALSIGNS_UNION',
            
            'MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_FVC_LEFT',               'MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_LABS_LEFT',
            'MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_MUSCLESTRENGTH_LEFT',    'MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_SVC_LEFT',
            'MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_VITALSIGNS_LEFT',
        ):
            for root, dirs, filenames in os.walk(f"{INTERVALS_CUT_PATH}/{merge_dir}/"):
                for filename in filenames:
                    if filename.endswith(".csv"):
                        temp_files.append(os.path.join(root, filename))
            temp_files.sort(key=sort_key_intervals)
            files.extend(temp_files)
            temp_files.clear()
    else:
        for merge_dir in ('MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_FVC_UNION', 'MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_FVC_LEFT'):
            for root, dirs, filenames in os.walk(f"{INTERVALS_CUT_PATH}/{merge_dir}/"):
                for filename in filenames:
                    if filename.endswith(".csv"):
                        temp_files.append(os.path.join(root, filename))
            temp_files.sort(key=sort_key_intervals)
            files.extend(temp_files)
            temp_files.clear()


    # ////////////////////////////////////////////////
    # Three-table merges: ALSFRS + SVC 
    # + one other table + non-temporal features
    # INTERSECTION only
    # ////////////////////////////////////////////////

    if PAPER_RESULTS_ONLY==False:
        for merge_dir in (
            'MERGE_ALSFRS_NODELTA_SVC_FVC_INTERSECTION',                'MERGE_ALSFRS_NODELTA_SVC_LABS_INTERSECTION',
            'MERGE_ALSFRS_NODELTA_SVC_HANDGRIPSTRENGTH_INTERSECTION',   'MERGE_ALSFRS_NODELTA_SVC_MUSCLESTRENGTH_INTERSECTION',
            'MERGE_ALSFRS_NODELTA_SVC_VITALSIGNS_INTERSECTION',
        ):
            for root, dirs, filenames in os.walk(f"{INTERVALS_CUT_PATH}/{merge_dir}/"):
                for filename in filenames:
                    if filename.endswith(".csv"):
                        temp_files.append(os.path.join(root, filename))
            temp_files.sort(key=sort_key_intervals)
            files.extend(temp_files)
            temp_files.clear()
    else:
        for merge_dir in ('MERGE_ALSFRS_NODELTA_SVC_LABS_INTERSECTION'):
            for root, dirs, filenames in os.walk(f"{INTERVALS_CUT_PATH}/{merge_dir}/"):
                for filename in filenames:
                    if filename.endswith(".csv"):
                        temp_files.append(os.path.join(root, filename))
            temp_files.sort(key=sort_key_intervals)
            files.extend(temp_files)
            temp_files.clear()


    # ////////////////////////////////////////////////////
    # Three-table merges: ALSFRS + FVC 
    # + one other table + non-temporal features
    # UNION & LEFT only / SLIDING windows only
    # ////////////////////////////////////////////////////

    if PAPER_RESULTS_ONLY==False:
        for merge_dir in (
            'MERGE_ALSFRS_NODELTA_FVC_LABS_UNION',              'MERGE_ALSFRS_NODELTA_FVC_HANDGRIPSTRENGTH_UNION', 
            'MERGE_ALSFRS_NODELTA_FVC_MUSCLESTRENGTH_UNION',    'MERGE_ALSFRS_NODELTA_FVC_SVC_UNION',           
            'MERGE_ALSFRS_NODELTA_FVC_VITALSIGNS_UNION',

            'MERGE_ALSFRS_NODELTA_FVC_LABS_LEFT',               'MERGE_ALSFRS_NODELTA_FVC_HANDGRIPSTRENGTH_LEFT',
            'MERGE_ALSFRS_NODELTA_FVC_MUSCLESTRENGTH_LEFT',     'MERGE_ALSFRS_NODELTA_FVC_SVC_LEFT',
            'MERGE_ALSFRS_NODELTA_FVC_VITALSIGNS_LEFT',
        ):
            for root, dirs, filenames in os.walk(f"{INTERVALS_CUT_PATH}/{merge_dir}/"):
                for filename in filenames:
                    if filename.endswith(".csv"):
                        temp_files.append(os.path.join(root, filename))
            temp_files.sort(key=sort_key_intervals)
            files.extend(temp_files)
            temp_files.clear()
    else:
        for merge_dir in ('MERGE_ALSFRS_NODELTA_FVC_SVC_UNION', 'MERGE_ALSFRS_NODELTA_FVC_SVC_LEFT'):
            for root, dirs, filenames in os.walk(f"{INTERVALS_CUT_PATH}/{merge_dir}/"):
                for filename in filenames:
                    if filename.endswith(".csv"):
                        temp_files.append(os.path.join(root, filename))
            temp_files.sort(key=sort_key_intervals)
            files.extend(temp_files)
            temp_files.clear()


    if PAPER_RESULTS_ONLY==False:
        # ////////////////////////////////////////////////////
        # Four-table merges: ALSFRS + HANDGRIPSTRENGTH + FVC 
        # + one other + non-temporal features
        # UNION & LEFT only / FIXED windows only
        # ////////////////////////////////////////////////////

        for merge_dir in (
            'MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_FVC_LABS_UNION', 'MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_FVC_MUSCLESTRENGTH_UNION', 
            'MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_FVC_SVC_UNION',  'MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_FVC_VITALSIGNS_UNION',
            
            'MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_FVC_LABS_LEFT',  'MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_FVC_MUSCLESTRENGTH_LEFT',
            'MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_FVC_SVC_LEFT',   'MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_FVC_VITALSIGNS_LEFT',
        ):
            for root, dirs, filenames in os.walk(f"{INTERVALS_CUT_PATH}/{merge_dir}/"):
                for filename in filenames:
                    if filename.endswith(".csv"):
                        temp_files.append(os.path.join(root, filename))
            temp_files.sort(key=sort_key_intervals)
            files.extend(temp_files)
            temp_files.clear()


    if PAPER_RESULTS_ONLY==False:
        # ////////////////////////////////////////////////////
        # Four-table merges: ALSFRS + SVC + LABS 
        # + one other + non-temporal features
        # INTERSECTION only / SLIDING windows only
        # ////////////////////////////////////////////////////

        for merge_dir in (
            'MERGE_ALSFRS_NODELTA_SVC_LABS_FVC_INTERSECTION',               'MERGE_ALSFRS_NODELTA_SVC_LABS_HANDGRIPSTRENGTH_INTERSECTION',
            'MERGE_ALSFRS_NODELTA_SVC_LABS_MUSCLESTRENGTH_INTERSECTION',    'MERGE_ALSFRS_NODELTA_SVC_LABS_VITALSIGNS_INTERSECTION',
        ):
            for root, dirs, filenames in os.walk(f"{INTERVALS_CUT_PATH}/{merge_dir}/"):
                for filename in filenames:
                    if filename.endswith(".csv"):
                        temp_files.append(os.path.join(root, filename))
            temp_files.sort(key=sort_key_intervals)
            files.extend(temp_files)
            temp_files.clear()


    # ////////////////////////////////////////////////////
    # Four-table merges: ALSFRS + FVC + SVC 
    # + one other + non-temporal features
    # UNION & LEFT only / SLIDING windows only
    # ////////////////////////////////////////////////////

    if PAPER_RESULTS_ONLY==False:
        for merge_dir in (
            'MERGE_ALSFRS_NODELTA_FVC_SVC_LABS_UNION',              'MERGE_ALSFRS_NODELTA_FVC_SVC_HANDGRIPSTRENGTH_UNION', 
            'MERGE_ALSFRS_NODELTA_FVC_SVC_MUSCLESTRENGTH_UNION',    'MERGE_ALSFRS_NODELTA_FVC_SVC_VITALSIGNS_UNION',

            'MERGE_ALSFRS_NODELTA_FVC_SVC_LABS_LEFT',           'MERGE_ALSFRS_NODELTA_FVC_SVC_HANDGRIPSTRENGTH_LEFT',
            'MERGE_ALSFRS_NODELTA_FVC_SVC_MUSCLESTRENGTH_LEFT', 'MERGE_ALSFRS_NODELTA_FVC_SVC_VITALSIGNS_LEFT',
        ):
            for root, dirs, filenames in os.walk(f"{INTERVALS_CUT_PATH}/{merge_dir}/"):
                for filename in filenames:
                    if filename.endswith(".csv"):
                        temp_files.append(os.path.join(root, filename))
            temp_files.sort(key=sort_key_intervals)
            files.extend(temp_files)
            temp_files.clear()
    else:
        for merge_dir in ('MERGE_ALSFRS_NODELTA_FVC_SVC_HANDGRIPSTRENGTH_LEFT'):
            for root, dirs, filenames in os.walk(f"{INTERVALS_CUT_PATH}/{merge_dir}/"):
                for filename in filenames:
                    if filename.endswith(".csv"):
                        temp_files.append(os.path.join(root, filename))
            temp_files.sort(key=sort_key_intervals)
            files.extend(temp_files)
            temp_files.clear()





    # ////////////////////////////////////////////////////////////////////////////////
    # ////////////////////////////////////////////////////////////////////////////////
    # ========================= TARGET ALSFRS FIRST SYMPTOMS =========================
    # ////////////////////////////////////////////////////////////////////////////////
    # ////////////////////////////////////////////////////////////////////////////////


    # ////////////////////////////////////////////////////
    # Single temporal tables
    # ////////////////////////////////////////////////////

    if PAPER_RESULTS_ONLY==False:
        for table in ('ALSFRS_Central_Only_First_Symptoms', 'ALSFRS_First_Symptoms', 'FVC_First_Symptoms', 
                    'HANDGRIPSTRENGTH_First_Symptoms', 'MUSCLESTRENGTH_First_Symptoms', 'LABS_First_Symptoms', 
                    'SVC_First_Symptoms', 'VITALSIGNS_First_Symptoms'):
            for root, dirs, filenames in os.walk(f"{INTERVALS_CUT_PATH}/{table}/"):
                for filename in filenames:
                    if filename.endswith(".csv"):
                        temp_files.append(os.path.join(root, filename))
            temp_files.sort(key=sort_key_intervals)
            files.extend(temp_files)
            temp_files.clear()
    else:
        for table in ('ALSFRS_Central_Only_First_Symptoms', 'ALSFRS_First_Symptoms'):
            for root, dirs, filenames in os.walk(f"{INTERVALS_CUT_PATH}/{table}/"):
                for filename in filenames:
                    if filename.endswith(".csv"):
                        temp_files.append(os.path.join(root, filename))
            temp_files.sort(key=sort_key_intervals)
            files.extend(temp_files)
            temp_files.clear()


    # ////////////////////////////////////////////////////
    # Single temporal tables + non-temporal features
    # ////////////////////////////////////////////////////

    if PAPER_RESULTS_ONLY==False:
        for table in ('ALSFRS_NODELTA_First_Symptoms', 'FVC_NODELTA_First_Symptoms', 'HANDGRIPSTRENGTH_NODELTA_First_Symptoms',
                    'MUSCLESTRENGTH_NODELTA_First_Symptoms', 'LABS_NODELTA_First_Symptoms', 'SVC_NODELTA_First_Symptoms',
                    'VITALSIGNS_NODELTA_First_Symptoms'):
            for root, dirs, filenames in os.walk(f"{INTERVALS_CUT_PATH}/{table}/"):
                for filename in filenames:
                    if filename.endswith(".csv"):
                        temp_files.append(os.path.join(root, filename))
            temp_files.sort(key=sort_key_intervals)
            files.extend(temp_files)
            temp_files.clear()
    else:
        for table in ('ALSFRS_NODELTA_First_Symptoms'):
            for root, dirs, filenames in os.walk(f"{INTERVALS_CUT_PATH}/{table}/"):
                for filename in filenames:
                    if filename.endswith(".csv"):
                        temp_files.append(os.path.join(root, filename))
            temp_files.sort(key=sort_key_intervals)
            files.extend(temp_files)
            temp_files.clear()


    if PAPER_RESULTS_ONLY==False:
        # ////////////////////////////////////////////////////
        # All-table merges (temporal features only)
        # ////////////////////////////////////////////////////

        for merge_dir in ('ALL_DELTA_First_Symptoms_INTERSECTION', 'ALL_DELTA_First_Symptoms_LEFT', 'ALL_DELTA_First_Symptoms_UNION'):
            for root, dirs, filenames in os.walk(f"{INTERVALS_CUT_PATH}/{merge_dir}/"):
                for filename in filenames:
                    if filename.endswith(".csv"):
                        temp_files.append(os.path.join(root, filename))
            temp_files.sort(key=sort_key_intervals)
            files.extend(temp_files)
            temp_files.clear()


    # ////////////////////////////////////////////////////
    # All-table merges + non-temporal features
    # ////////////////////////////////////////////////////

    if PAPER_RESULTS_ONLY==False:
        for merge_dir in ('ALL_DATA_First_Symptoms_INTERSECTION', 'ALL_DATA_First_Symptoms_LEFT', 'ALL_DATA_First_Symptoms_UNION'):
            for root, dirs, filenames in os.walk(f"{INTERVALS_CUT_PATH}/{merge_dir}/"):
                for filename in filenames:
                    if filename.endswith(".csv"):
                        temp_files.append(os.path.join(root, filename))
            temp_files.sort(key=sort_key_intervals)
            files.extend(temp_files)
            temp_files.clear()
    else:
        for merge_dir in ('ALL_DATA_First_Symptoms_UNION'):
            for root, dirs, filenames in os.walk(f"{INTERVALS_CUT_PATH}/{merge_dir}/"):
                for filename in filenames:
                    if filename.endswith(".csv"):
                        temp_files.append(os.path.join(root, filename))
            temp_files.sort(key=sort_key_intervals)
            files.extend(temp_files)
            temp_files.clear()
        
        
    # ////////////////////////////////////////////////////
    # Pairwise merges: ALSFRS + one other table + non-temporal features
    # ////////////////////////////////////////////////////

    if PAPER_RESULTS_ONLY==False:
        for merge_dir in (
            'MERGE_ALSFRS_NODELTA_FVC_UNION',               'MERGE_ALSFRS_NODELTA_LABS_UNION', 
            'MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_UNION',  'MERGE_ALSFRS_NODELTA_MUSCLESTRENGTH_UNION',    
            'MERGE_ALSFRS_NODELTA_SVC_UNION',               'MERGE_ALSFRS_NODELTA_VITALSIGNS_UNION',

            'MERGE_ALSFRS_NODELTA_FVC_INTERSECTION',                'MERGE_ALSFRS_NODELTA_LABS_INTERSECTION',               
            'MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_INTERSECTION',   'MERGE_ALSFRS_NODELTA_MUSCLESTRENGTH_INTERSECTION',     
            'MERGE_ALSFRS_NODELTA_SVC_INTERSECTION',                'MERGE_ALSFRS_NODELTA_VITALSIGNS_INTERSECTION',
            
            'MERGE_ALSFRS_NODELTA_FVC_LEFT',                'MERGE_ALSFRS_NODELTA_LABS_LEFT',
            'MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_LEFT',   'MERGE_ALSFRS_NODELTA_MUSCLESTRENGTH_LEFT',
            'MERGE_ALSFRS_NODELTA_SVC_LEFT',                'MERGE_ALSFRS_NODELTA_VITALSIGNS_LEFT',
        ):
            for root, dirs, filenames in os.walk(f"{INTERVALS_CUT_PATH}/{merge_dir}/"):
                for filename in filenames:
                    if filename.endswith(".csv"):
                        temp_files.append(os.path.join(root, filename))
            temp_files.sort(key=sort_key_intervals)
            files.extend(temp_files)
            temp_files.clear()
    else:
        for merge_dir in ('MERGE_ALSFRS_NODELTA_SVC_INTERSECTION'):
            for root, dirs, filenames in os.walk(f"{INTERVALS_CUT_PATH}/{merge_dir}/"):
                for filename in filenames:
                    if filename.endswith(".csv"):
                        temp_files.append(os.path.join(root, filename))
            temp_files.sort(key=sort_key_intervals)
            files.extend(temp_files)
            temp_files.clear()


    if PAPER_RESULTS_ONLY==False:
        # ////////////////////////////////////////////////
        # Three-table merges: ALSFRS + SVC 
        # + one other table + non-temporal features
        # INTERSECTION only / FIXED windows only
        # ////////////////////////////////////////////////

        for merge_dir in (
            'MERGE_ALSFRS_NODELTA_SVC_FVC_First_Symptoms_INTERSECTION',                 'MERGE_ALSFRS_NODELTA_SVC_LABS_First_Symptoms_INTERSECTION',
            'MERGE_ALSFRS_NODELTA_SVC_HANDGRIPSTRENGTH_First_Symptoms_INTERSECTION',    'MERGE_ALSFRS_NODELTA_SVC_MUSCLESTRENGTH_First_Symptoms_INTERSECTION',
            'MERGE_ALSFRS_NODELTA_SVC_VITALSIGNS_First_Symptoms_INTERSECTION',
        ):
            for root, dirs, filenames in os.walk(f"{INTERVALS_CUT_PATH}/{merge_dir}/"):
                for filename in filenames:
                    if filename.endswith(".csv"):
                        temp_files.append(os.path.join(root, filename))
            temp_files.sort(key=sort_key_intervals)
            files.extend(temp_files)
            temp_files.clear()










    # ------------------------------------------------------------------
    # Output directories and log file
    # ------------------------------------------------------------------

    # Single log file that captures everything printed to the terminal
    log_file = open(f"{RESULT_PATH}/PROACT - RF Results.txt", "w")





    # ==============================================================================
    # Main evaluation loop
    # ==============================================================================

    for file in files:

        # ------------------------------------------------------------------
        # 1. Load CSV
        # ------------------------------------------------------------------
        # Skip zero-byte files produced by generate_interval_prediction_files
        # when no valid patients were found for a given horizon.

        if os.path.getsize(file) > 0:
            df = pd.read_csv(file, low_memory=False)
        else:
            with redirect_stdout(Tee(sys.stdout, log_file)):
                print(f"\n\n\nProcessing file: {file}")
                print("\t=> Empty file, skipped.")
            continue

        # Sub-directory for feature importance charts (one folder per source directory)
        feature_importance_dir = os.path.join(
            FEATURE_IMPORTANCE_PATH, file.split('/')[-2]
        )

        # ------------------------------------------------------------------
        # 2. Identify the target column
        # ------------------------------------------------------------------
        # Fixed files encode the horizon in their filename; the target column
        # is the Central statistic of the interval immediately following the
        # last training interval.
        # Sliding files always use the generic 'Ti+1' target name.

        target_col = None

        fixed_match = re.search(r'Fixed_(\d+)M', file)

        if fixed_match:
            # Derive target interval bounds from the horizon in months
            months    = int(fixed_match.group(1))
            start_day = months * 30
            end_day   = start_day + 90  # 90-day interval width

            candidate_target = f"ALS_ALSFRS_R_Total_{start_day}_{end_day}_Central"

            if candidate_target in df.columns:
                target_col             = candidate_target
                feature_importance_dir = os.path.join(feature_importance_dir, 'Fixed')
            else:
                target_col = None

        elif 'Sliding' in file:
            # Sliding files use relative column names; target is always Ti+1
            if 'ALS_ALSFRS_R_Total_Ti+1_Central' in df.columns:
                target_col             = 'ALS_ALSFRS_R_Total_Ti+1_Central'
                feature_importance_dir = os.path.join(feature_importance_dir, 'Sliding')
            else:
                target_col = None

        if not os.path.exists(feature_importance_dir):
            os.makedirs(feature_importance_dir)

        with redirect_stdout(Tee(sys.stdout, log_file)):
            print(f"\n\n\nProcessing file: {file}")

        # Skip files where the expected target column is absent or all-NaN
        if target_col not in df.columns or target_col is None or df[target_col].isnull().all():
            with redirect_stdout(Tee(sys.stdout, log_file)):
                print("\t=> Target column not found or all missing, file skipped.")
            continue

        # ------------------------------------------------------------------
        # 3. Row-level cleaning
        # ------------------------------------------------------------------

        # Remove rows with a missing target value
        df = df.dropna(subset=[target_col])

        # Remove rows that are completely empty across all feature columns
        df = df.dropna(
            how='all',
            subset=[col for col in df.columns
                    if col != target_col and 'subject_id' not in col]
        )

        # ------------------------------------------------------------------
        # 4. Column-level filtering
        # ------------------------------------------------------------------
        # subject_id and the target are excluded before quality filtering
        # so that their missingness does not influence column retention.

        df_train_columns = df.drop(
            columns=[target_col] + [col for col in df.columns if 'subject_id' in col]
        )

        # Remove columns with > 80 % missing values or a dominant value > 80 %
        # (e.g. a binary flag that is 'No' for 95 % of patients adds no signal)
        df_train_columns = filter_columns_missing_and_dominant_data(
            df_train_columns,
            missing_threshold=0.8,
            dominant_threshold=0.8
        )

        # Align the main DataFrame to the retained columns
        df = df[[col for col in df.columns
                if col in df_train_columns.columns
                or col == target_col
                or 'subject_id' in col]]

        with redirect_stdout(Tee(sys.stdout, log_file)):
            print(f"\t=> {df.shape[0]} rows, {df.shape[1]} columns, "
                f"target: {target_col}")

        # ------------------------------------------------------------------
        # 5. Feature and target separation
        # ------------------------------------------------------------------

        X = df.drop(
            columns=[target_col] + [col for col in df.columns if 'subject_id' in col]
        )
        y = df[target_col]

        # ------------------------------------------------------------------
        # 6. Categorical encoding - pass 1: known value maps
        # ------------------------------------------------------------------
        # Apply deterministic mappings for columns whose value sets are fully
        # known (boolean flags, laterality, sex, specific coded fields).
        # Using fixed mappings rather than LabelEncoder preserves the ordinal
        # meaning of the values and ensures reproducibility across files.

        X = encode_simple_categorical(X)

        # ------------------------------------------------------------------
        # 7. Categorical encoding - pass 2: automatic LabelEncoder fallback
        # ------------------------------------------------------------------
        # Any object column not handled in pass 1 receives a LabelEncoder.
        # Columns with more than 300 unique values are dropped instead:
        # they almost certainly contain free-text units or identifiers that
        # would produce meaningless integer codes.

        X = encode_auto_categorical(X)

        # ------------------------------------------------------------------
        # 8. Minimum sample check
        # ------------------------------------------------------------------
        # 10-fold cross-validation requires at least 20 rows; files with
        # fewer patients (very restrictive intersection merges at long horizons)
        # are skipped.

        if X.shape[0] < 20:
            with redirect_stdout(Tee(sys.stdout, log_file)):
                print(f"\t=> Only {X.shape[0]} rows - insufficient for 10-fold "
                    f"cross-validation, file skipped.")
            continue

        # ------------------------------------------------------------------
        # 9. Model definition
        # ------------------------------------------------------------------
        # max_features = floor(p / 4): using a quarter of features as the
        # candidate split pool.  The standard sqrt(p) heuristic is designed
        # for classification; p/3 or p/4 tends to work better for regression
        # by reducing tree correlation while retaining enough candidate splits.

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_features=int(X.shape[1] / 4)
        )

        # ------------------------------------------------------------------
        # 10. 10-fold cross-validation
        # ------------------------------------------------------------------

        scoring = {
            'MAE':  make_scorer(mean_absolute_error),
            'RMSE': make_scorer(root_mean_squared_error),
            'R²':   make_scorer(r2_score)
        }

        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        fold_scores_mean = {metric: [] for metric in scoring}
        fold_scores_med = {metric: [] for metric in scoring}
        fold_importances = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_train, y_train)

            # Median-aggregated prediction: collect each tree's output and
            # take the element-wise median.  More robust to outlier trees
            # than the default mean (model.predict) when the error distribution
            # is skewed or a few trees overfit noisy folds.
            all_predictions = np.array([
                tree.predict(X_val.values) for tree in model.estimators_
            ])
            y_pred_med = np.median(all_predictions, axis=0)

            # Mean-aggregated prediction (standard sklearn behaviour)
            y_pred_mean = model.predict(X_val)

            # Clip predictions to the valid ALSFRS-R range [0, 48]
            y_pred_med  = np.clip(y_pred_med, 0, 48)
            y_pred_mean = np.clip(y_pred_mean, 0, 48)

            # Median-aggregate scores
            fold_scores_med['MAE'].append(mean_absolute_error(y_val, y_pred_med))
            fold_scores_med['RMSE'].append(root_mean_squared_error(y_val, y_pred_med))
            fold_scores_med['R²'].append(r2_score(y_val, y_pred_med))

            # Mean-aggregate scores
            fold_scores_mean['MAE'].append(mean_absolute_error(y_val, y_pred_mean))
            fold_scores_mean['RMSE'].append(root_mean_squared_error(y_val, y_pred_mean))
            fold_scores_mean['R²'].append(r2_score(y_val, y_pred_mean))

            fold_importances.append(model.feature_importances_)

        # ------------------------------------------------------------------
        # 11. Score summary
        # ------------------------------------------------------------------

        with redirect_stdout(Tee(sys.stdout, log_file)):
            print("\nCross-validation results (10-fold):")

        for metric in fold_scores_mean:
            mean_score = np.mean(fold_scores_mean[metric])
            std_score  = np.std(fold_scores_mean[metric])
            with redirect_stdout(Tee(sys.stdout, log_file)):
                print(f"\t{metric} (mean) : {mean_score:.2f} ± {std_score:.2f} "
                    f"({np.round(fold_scores_mean[metric], 2)})")

        for metric in fold_scores_med:
            mean_score = np.mean(fold_scores_med[metric])
            std_score  = np.std(fold_scores_med[metric])
            with redirect_stdout(Tee(sys.stdout, log_file)):
                print(f"\t{metric} (median): {mean_score:.2f} ± {std_score:.2f} "
                    f"({np.round(fold_scores_med[metric], 2)})")

        # ------------------------------------------------------------------
        # 12. Feature importance
        # ------------------------------------------------------------------
        # Average importances across folds to reduce the variance introduced
        # by the random train/validation splits.

        importances_array = np.array(fold_importances)
        mean_importances  = np.mean(importances_array, axis=0)
        std_importances   = np.std(importances_array, axis=0)

        # Top-20 features by average importance
        indices_top_20      = np.argsort(mean_importances)[::-1][:20]
        top_features        = X.columns[indices_top_20]
        top_mean_importance = mean_importances[indices_top_20]
        top_std_importance  = std_importances[indices_top_20]

        # Feature importance bar chart (commented out to speed up batch runs)
        plt.figure(figsize=(10, 6))
        plt.title("Top 20 feature importances (mean ± std across 10 folds)")
        plt.barh(top_features, top_mean_importance, xerr=top_std_importance, align="center",
                color='skyblue', ecolor='gray')
        plt.xlabel("Mean importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plot_filename = os.path.join(
            feature_importance_dir,
            f"RF_model_{os.path.basename(file).replace('.csv', '_feature_importance_avg.png')}"
        )
        plt.savefig(plot_filename)

        # DataFrame with one column per fold for detailed post-hoc analysis
        feature_importance_df = pd.DataFrame({
            'Feature':          X.columns,
            'Mean_Importance':  mean_importances,
            'Std_Importance':   std_importances,
            **{f'Fold_{i+1}': fold_importances[i] for i in range(10)}
        }).sort_values(by='Mean_Importance', ascending=False)

        with redirect_stdout(Tee(sys.stdout, log_file)):
            print("\nTop 20 features (mean importance across folds):")
            print(feature_importance_df.head(20).to_string(index=False))
            print("\n\n\n" + "-" * 80)