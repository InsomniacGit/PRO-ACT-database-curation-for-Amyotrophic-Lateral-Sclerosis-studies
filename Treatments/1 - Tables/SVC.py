"""
PROACT Slow Vital Capacity (SVC) Processing Pipeline
=====================================================
This script processes the PROACT (PRO-ACT ALS) slow vital capacity dataset
through a sequential cleaning, imputation, and reshaping pipeline. It produces
seven intermediate CSV files, culminating in a wide-format one-row-per-patient
matrix with visits stored as sequentially prefixed columns.

SVC is measured across up to three trials per visit. For each trial, three
quantities are recorded and linked by the identity:

    pct_of_Normal = (Subject_Liters / subject_normal) * 100

When two of the three quantities are present, the third can be derived
algebraically. This imputation strategy mirrors the one applied in the
FVC pipeline (PROACT_FVC_processing.py).

Pipeline stages:
    v2  - Drop the redundant units column; fix inconsistent raw column names
    v3  - Impute missing trial values using the three-quantity identity
    v4  - Add per-patient observation count
    v5  - Add best-trial columns (maximum Subject_Liters across the three trials
          and its corresponding pct_of_Normal)
    v6  - Reshape to wide format (one row per patient, visits as prefixed columns)
    v7  - Prefix all feature columns with 'SVC_' for downstream merging

Author: Bouclier Lucas
Data:   PROACT dataset (2022-07-29 release)
"""



import pandas as pd
import os
from pathlib import Path



# ------------------------------------------------------------------
# Path configuration
# ------------------------------------------------------------------

# Root directory for all processed outputs
data_path = str(Path.home() / "Desktop" / "DATA_PROACT_V2" / "BDDfiltre2")

# Root directory containing raw PROACT CSV exports
proact_path = str(Path.home() / "Desktop" / "DATA_PROACT_V2" / "2022_07_29_PROACT_ALL_FORMS")

# Create the output subdirectory if it does not already exist
if not os.path.exists(data_path):
    os.makedirs(data_path)




# ///////////////////////////////////////////////////////
# ------------------------- SVC -------------------------
# ///////////////////////////////////////////////////////




# ------------------------------------------------------------------
# Stage v2 - Drop redundant column and fix raw column names
# ------------------------------------------------------------------

def modify_proact_svc(file_path):
    """
    Remove the constant units column and standardise inconsistent column names
    inherited from the raw PROACT export.

    Dropped column:
        Slow_Vital_Capacity_Units  - measurement unit is constant (Liters)

    Column renames applied:
        Subject_Liters__Trial_2_   -> Subject_Liters_Trial_2  (spurious underscores)
        Subject_Liters__Trial_3_   -> Subject_Liters_Trial_3  (spurious underscores)
        Subject_Normal             -> subject_normal           (case consistency with FVC)
        Slow_vital_Capacity_Delta  -> Slow_Vital_Capacity_Delta (capitalisation fix)

    These renames align the SVC column naming convention with that of the FVC
    pipeline so that downstream code can treat both datasets uniformly.

    Parameters
    ----------
    file_path : str
        Path to the raw PROACT_SVC.csv file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with standardised column names
        (-> saved as PROACT_SVC_v2.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)

    df.drop(columns=['Slow_Vital_Capacity_Units'], inplace=True)

    df.rename(columns={
        'Subject_Liters__Trial_2_':  'Subject_Liters_Trial_2',
        'Subject_Liters__Trial_3_':  'Subject_Liters_Trial_3',
        'Subject_Normal':            'subject_normal',
        'Slow_vital_Capacity_Delta': 'Slow_Vital_Capacity_Delta',
    }, inplace=True)

    return df


df = modify_proact_svc(proact_path + '/PROACT_SVC.csv')
df.to_csv(data_path + '/PROACT_SVC_v2.csv', index=False)




# ------------------------------------------------------------------
# Stage v3 - Impute missing trial values from the SVC identity
# ------------------------------------------------------------------

def compute_missing_data_svc(file_path):
    """
    Impute missing SVC trial values using the three-quantity algebraic identity:

        pct_of_Normal = (Subject_Liters / subject_normal) * 100

    which can be rearranged to:

        Subject_Liters = subject_normal * (pct_of_Normal / 100)
        subject_normal = Subject_Liters / (pct_of_Normal / 100)

    Imputation is applied in three sequential passes (identical logic to the
    FVC pipeline in PROACT_FVC_processing.py):

    1. subject_normal: computed from each available (Subject_Liters, pct_of_Normal)
       trial pair and averaged across trials. Only imputed when the raw value
       is missing; existing values are preserved.

    2. pct_of_Normal_Trial_X: imputed from Subject_Liters_Trial_X and
       subject_normal when the percentage is missing for that trial.

    3. Subject_Liters_Trial_X: imputed from pct_of_Normal_Trial_X and
       subject_normal when the volume is missing for that trial.

    subject_normal is computed first because it is shared across all three
    trials and is needed as a denominator in both subsequent passes.

    Parameters
    ----------
    file_path : str
        Path to PROACT_SVC_v2.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with trial columns imputed where algebraically possible
        (-> saved as PROACT_SVC_v3.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)

    # Pass 1: impute subject_normal as the average across available trial pairs
    def calculate_subject_normal(row):
        """
        Derive subject_normal from each trial where both Subject_Liters and
        pct_of_Normal are present, then return the rounded mean.
        Division by zero is guarded by checking pct_of_Normal != 0.
        """
        results = []
        for trial in ['1', '2', '3']:
            liters = row[f'Subject_Liters_Trial_{trial}']
            pct    = row[f'pct_of_Normal_Trial_{trial}']
            if pd.notna(liters) and pd.notna(pct) and pct != 0:
                results.append(round(liters / (pct / 100), 2))
        return round(sum(results) / len(results), 2) if results else None

    df['subject_normal'] = df.apply(
        lambda row: calculate_subject_normal(row) if pd.isna(row['subject_normal']) else row['subject_normal'],
        axis=1
    )

    # Pass 2: impute missing pct_of_Normal for each trial
    def calculate_pct_of_normal(row):
        """
        For the first trial whose pct_of_Normal is missing, compute it from
        the corresponding Subject_Liters and subject_normal.
        Only one trial is imputed per call (the first missing one found).
        """
        for trial in ['1', '2', '3']:
            if pd.isna(row[f'pct_of_Normal_Trial_{trial}']):
                liters = row[f'Subject_Liters_Trial_{trial}']
                normal = row['subject_normal']
                if pd.notna(liters) and pd.notna(normal):
                    return round((liters / normal) * 100, 0)
        return None

    for trial in ['1', '2', '3']:
        col = f'pct_of_Normal_Trial_{trial}'
        df[col] = df.apply(
            lambda row: calculate_pct_of_normal(row) if pd.isna(row[col]) else row[col],
            axis=1
        )

    # Pass 3: impute missing Subject_Liters for each trial
    def calculate_subject_liters(row):
        """
        For the first trial whose Subject_Liters is missing, compute it from
        pct_of_Normal and subject_normal.
        Only one trial is imputed per call (the first missing one found).
        """
        for trial in ['1', '2', '3']:
            if pd.isna(row[f'Subject_Liters_Trial_{trial}']):
                pct    = row[f'pct_of_Normal_Trial_{trial}']
                normal = row['subject_normal']
                if pd.notna(pct) and pd.notna(normal):
                    return round(normal * (pct / 100), 2)
        return None

    for trial in ['1', '2', '3']:
        col = f'Subject_Liters_Trial_{trial}'
        df[col] = df.apply(
            lambda row: calculate_subject_liters(row) if pd.isna(row[col]) else row[col],
            axis=1
        )

    return df


df_fvc = compute_missing_data_svc(data_path + '/PROACT_SVC_v2.csv')
df_fvc.to_csv(data_path + '/PROACT_SVC_v3.csv', index=False)




# ------------------------------------------------------------------
# Stage v4 - Add per-patient observation count
# ------------------------------------------------------------------

def observation_counter_svc(file_path):
    """
    Add an `observation_count` column recording how many SVC visit rows exist
    per patient.

    Each row represents one spirometry assessment visit. This count reflects
    the number of valid visits retained after cleaning and imputation.

    Parameters
    ----------
    file_path : str
        Path to PROACT_SVC_v3.csv.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with `observation_count` inserted as the second column
        (immediately after `subject_id`).
    """
    df = pd.read_csv(file_path, low_memory=False)

    # Map each patient to the number of rows they have in the DataFrame
    df['observation_count'] = df['subject_id'].map(df['subject_id'].value_counts())

    # Move the new column to position 1 (right after subject_id)
    cols = df.columns.tolist()
    cols.insert(1, cols.pop(cols.index('observation_count')))
    df = df[cols]

    return df


df_fvc = observation_counter_svc(data_path + '/PROACT_SVC_v3.csv')
df_fvc.to_csv(data_path + '/PROACT_SVC_v4.csv', index=False)




# ------------------------------------------------------------------
# Stage v5 - Add best-trial summary columns
# ------------------------------------------------------------------

def add_max_trials_columns(file_path):
    """
    Derive two summary columns representing the best performance across the
    three spirometry trials within each visit:

        Subject_Liters_Trials_Max  - the highest volume recorded across the
                                     three trials (standard clinical practice
                                     is to report the best effort)
        pct_of_Normal_Trials_Max   - the pct_of_Normal corresponding to the
                                     trial that produced Subject_Liters_Trials_Max

    When multiple trials share the maximum volume, Trial 1 takes precedence,
    followed by Trial 2 then Trial 3 (first-match logic in get_pct_of_normal_max).

    Parameters
    ----------
    file_path : str
        Path to PROACT_SVC_v4.csv.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with two additional summary columns
        (-> saved as PROACT_SVC_v5.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)

    df['Subject_Liters_Trials_Max'] = df[[
        'Subject_Liters_Trial_1',
        'Subject_Liters_Trial_2',
        'Subject_Liters_Trial_3',
    ]].max(axis=1)

    def get_pct_of_normal_max(row):
        """Return the pct_of_Normal from the trial that achieved the max volume."""
        if row['Subject_Liters_Trials_Max'] == row['Subject_Liters_Trial_1']:
            return row['pct_of_Normal_Trial_1']
        elif row['Subject_Liters_Trials_Max'] == row['Subject_Liters_Trial_2']:
            return row['pct_of_Normal_Trial_2']
        elif row['Subject_Liters_Trials_Max'] == row['Subject_Liters_Trial_3']:
            return row['pct_of_Normal_Trial_3']
        return None

    df['pct_of_Normal_Trials_Max'] = df.apply(get_pct_of_normal_max, axis=1)

    return df


df_fvc = add_max_trials_columns(data_path + '/PROACT_SVC_v4.csv')
df_fvc.to_csv(data_path + '/PROACT_SVC_v5.csv', index=False)




# ------------------------------------------------------------------
# Stage v6 - Reshape to wide format
# ------------------------------------------------------------------

def reshape_to_wide_format(csv_file):
    """
    Reshape the long-format SVC data (multiple rows per patient) into a
    wide-format DataFrame (one row per patient) where each visit's values
    are stored in sequentially prefixed columns.

    Visits are sorted by Slow_Vital_Capacity_Delta (time since study baseline)
    before pivoting, so that column prefix 1_ always corresponds to the
    earliest recorded visit.

    Only the ten clinically relevant columns are retained in the wide format;
    administrative columns are excluded.

    Column naming convention:
        {visit_index}_{original_column_name}
        e.g. "1_Slow_Vital_Capacity_Delta", "2_Subject_Liters_Trials_Max"

    Parameters
    ----------
    csv_file : str
        Path to PROACT_SVC_v5.csv.

    Returns
    -------
    pd.DataFrame
        Wide-format patient-level DataFrame
        (-> saved as PROACT_SVC_v6.csv).
    """
    df = pd.read_csv(csv_file, low_memory=False)

    # Columns to carry into the wide format (one set per visit)
    colonnes = [
        'Slow_Vital_Capacity_Delta',
        'Subject_Liters_Trial_1',
        'pct_of_Normal_Trial_1',
        'Subject_Liters_Trial_2',
        'pct_of_Normal_Trial_2',
        'Subject_Liters_Trial_3',
        'pct_of_Normal_Trial_3',
        'subject_normal',
        'Subject_Liters_Trials_Max',
        'pct_of_Normal_Trials_Max',
    ]

    # Sort visits chronologically within each patient
    df = df.sort_values(by=['subject_id', 'Slow_Vital_Capacity_Delta'])

    rows = []

    for subject_id, group in df.groupby('subject_id'):
        group = group.reset_index(drop=True)
        row_data = {
            'subject_id':        subject_id,
            'observation_count': group.shape[0],
        }

        for i, row in group.iterrows():
            for col in colonnes:
                row_data[f'{i + 1}_{col}'] = row[col]

        rows.append(row_data)

    df_final = pd.DataFrame(rows)

    return df_final


df_fvc = reshape_to_wide_format(data_path + '/PROACT_SVC_v5.csv')
df_fvc.to_csv(data_path + '/PROACT_SVC_v6.csv', index=False)




# ------------------------------------------------------------------
# Stage v7 - Add 'SVC_' prefix to all feature columns
# ------------------------------------------------------------------

def rename_all_columns(file_path):
    """
    Prefix every feature column with 'SVC_' to namespace the slow vital
    capacity variables when merging with other PROACT sub-datasets.

    `subject_id` is the join key and is left unchanged.

    Parameters
    ----------
    file_path : str
        Path to PROACT_SVC_v6.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns
        (-> saved as PROACT_SVC_v7.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)
    df = df.rename(columns={col: f'SVC_{col}' for col in df.columns if col != 'subject_id'})
    return df


df_renamed = rename_all_columns(data_path + '/PROACT_SVC_v6.csv')
df_renamed.to_csv(data_path + '/PROACT_SVC_v7.csv', index=False)