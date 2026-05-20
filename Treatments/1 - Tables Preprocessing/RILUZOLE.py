"""
PROACT Riluzole Processing Pipeline
=====================================
This script processes the PROACT (PRO-ACT ALS) riluzole treatment dataset
through a minimal cleaning pipeline. It produces one output CSV file,
namespacing the columns for downstream merging after a diagnostic uniqueness
check.

Riluzole is the reference neuroprotective treatment for ALS. The dataset
records whether each patient received riluzole and is expected to contain
exactly one record per patient.

Pipeline stages:
    Diagnostic  - Verify that no patient appears more than once
    v2          - Prefix all feature columns with 'RIL_' for downstream merging

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




# ////////////////////////////////////////////////////////////
# ------------------------- RILUZOLE -------------------------
# ////////////////////////////////////////////////////////////




# ------------------------------------------------------------------
# Diagnostic - Check for unexpected duplicate patients
# ------------------------------------------------------------------

def check_unique_rows(file_path):
    """
    Report the number of patients with more than one row in the dataset.

    Riluzole records are expected to contain exactly one entry per patient.
    This check is run on the raw file before any transformation to confirm
    that assumption holds before namespacing and merging.

    Parameters
    ----------
    file_path : str
        Path to the raw PROACT_RILUZOLE.csv file.
    """
    df = pd.read_csv(file_path, low_memory=False)
    duplicated_subjects = df['subject_id'][df['subject_id'].duplicated(keep=False)]
    num_errors = len(duplicated_subjects.unique())
    print(f"Patients with more than one row: {num_errors}")


check_unique_rows(proact_path + '/PROACT_RILUZOLE.csv')




# ------------------------------------------------------------------
# Stage v2 - Add 'RIL_' prefix to all feature columns
# ------------------------------------------------------------------

def rename_all_columns(file_path):
    """
    Prefix every feature column with 'RIL_' to namespace the riluzole
    variables when merging with other PROACT sub-datasets.

    `subject_id` is the join key and is left unchanged.

    Parameters
    ----------
    file_path : str
        Path to the raw PROACT_RILUZOLE.csv file.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns
        (-> saved as PROACT_RILUZOLE_v2.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)
    df = df.rename(columns={col: f'RIL_{col}' for col in df.columns if col != 'subject_id'})
    return df


df_renamed = rename_all_columns(proact_path + '/PROACT_RILUZOLE.csv')
df_renamed.to_csv(data_path + '/PROACT_RILUZOLE_v2.csv', index=False)