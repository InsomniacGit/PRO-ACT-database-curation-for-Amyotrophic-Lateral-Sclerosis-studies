"""
PROACT Death Data Processing Pipeline
======================================
This script processes the PROACT (PRO-ACT ALS) death data dataset through
a minimal cleaning pipeline. It produces two intermediate CSV files,
removing ambiguous duplicate records and namespacing columns for downstream
merging.

Pipeline stages:
    v2  - Remove patients with duplicate records
    v3  - Prefix all feature columns with 'DEA_' for downstream merging

Author: Bouclier Lucas
Data:   PROACT dataset (2022-07-29 release)
"""



import pandas as pd





# /////////////////////////////////////////////////////////////
# ------------------------- DEATHDATA -------------------------
# /////////////////////////////////////////////////////////////



# ------------------------------------------------------------------
# Stage v2 - Remove patients with duplicate records
# ------------------------------------------------------------------

def remove_duplicates(file_path):
    """
    Drop all rows belonging to patients who appear more than once.

    Unlike other PROACT sub-datasets where duplicates can be resolved by
    merging or taking the first non-null value, death records are expected
    to be unique per patient. A patient appearing multiple times indicates
    a data integrity issue that cannot be safely resolved automatically;
    all their records are therefore discarded entirely.

    Parameters
    ----------
    file_path : str
        Path to the raw PROACT_DEATHDATA.csv file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only patients with a single unambiguous record
        (-> saved as PROACT_DEATHDATA_v2.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)

    # Flag all rows whose subject_id appears more than once
    duplicated_subjects = df['subject_id'].duplicated(keep=False)

    # Remove every row flagged as a duplicate (keep=False marks both occurrences)
    df_cleaned = df[~duplicated_subjects]

    return df_cleaned





# ------------------------------------------------------------------
# Stage v3 - Add 'DEA_' prefix to all feature columns
# ------------------------------------------------------------------

def rename_all_columns(file_path):
    """
    Prefix every feature column with 'DEA_' to namespace the death data
    variables when merging with other PROACT sub-datasets.

    `subject_id` is the join key and is left unchanged.

    Parameters
    ----------
    file_path : str
        Path to PROACT_DEATHDATA_v2.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns
        (-> saved as PROACT_DEATHDATA_v3.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)
    df = df.rename(columns={col: f'DEA_{col}' for col in df.columns if col != 'subject_id'})

    return df










# ==================================================================
# ------------------------- PIPELINE EXECUTION ---------------------
# ==================================================================

def run(DATA_PATH, PROACT_PATH):

    print("\n" * 3)
    print("=" * 60)
    print("DEATHDATA PIPELINE")
    print("=" * 60)

    

    # Stage v2 - Remove patients with duplicate records
    df_cleaned = remove_duplicates(PROACT_PATH + '/PROACT_DEATHDATA.csv')
    df_cleaned.to_csv(DATA_PATH + '/PROACT_DEATHDATA_v2.csv', index=False)



    # Stage v3 - Add 'DEA_' prefix to all feature columns
    df_renamed = rename_all_columns(DATA_PATH + '/PROACT_DEATHDATA_v2.csv')
    df_renamed.to_csv(DATA_PATH + '/PROACT_DEATHDATA_v3.csv', index=False)