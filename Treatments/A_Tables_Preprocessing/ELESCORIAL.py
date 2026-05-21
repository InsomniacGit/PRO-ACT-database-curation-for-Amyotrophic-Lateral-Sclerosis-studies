"""
PROACT El Escorial Processing Pipeline
========================================
This script processes the PROACT (PRO-ACT ALS) El Escorial diagnostic criteria
dataset through a minimal cleaning pipeline. It produces three intermediate CSV
files, removing an unused column and namespacing the remaining columns for
downstream merging.

The El Escorial criteria provide a standardised classification of ALS diagnostic
certainty (Definite, Probable, Possible, Suspected) and are expected to contain
at most one record per patient.

Pipeline stages:
    v2  - Drop the uninformative delta_days column
    v3  - Prefix all feature columns with 'ELE_' for downstream merging

Author: Bouclier Lucas
Data:   PROACT dataset (2022-07-29 release)
"""



import pandas as pd





# //////////////////////////////////////////////////////////////
# ------------------------- ELESCORIAL -------------------------
# //////////////////////////////////////////////////////////////



# ------------------------------------------------------------------
# Stage v2 - Drop uninformative column
# ------------------------------------------------------------------

def modify_proact_elescorial(file_path):
    """
    Remove the delta_days column from the El Escorial dataset.

    delta_days is an administrative timing column that does not contribute
    clinical information for this study and is therefore discarded.

    Parameters
    ----------
    file_path : str
        Path to the raw PROACT_ELESCORIAL.csv file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame (-> saved as PROACT_ELESCORIAL_v2.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)
    df = df.drop(columns=['delta_days'], errors='ignore')

    return df





# ------------------------------------------------------------------
# Diagnostic - Check for unexpected duplicate patients
# ------------------------------------------------------------------

def check_unique_lines(file_path):
    """
    Report the number of patients with more than one row in the dataset.

    El Escorial records are expected to contain exactly one classification
    per patient. This check confirms that assumption holds before the
    column-renaming stage.

    Parameters
    ----------
    file_path : str
        Path to PROACT_ELESCORIAL_v2.csv.
    """
    df = pd.read_csv(file_path, low_memory=False)
    duplicated_subjects = df['subject_id'][df['subject_id'].duplicated(keep=False)]
    num_errors = len(duplicated_subjects.unique())
    print(f"Patients with more than one row: {num_errors}")





# ------------------------------------------------------------------
# Stage v3 - Add 'ELE_' prefix to all feature columns
# ------------------------------------------------------------------

def rename_all_columns(file_path):
    """
    Prefix every feature column with 'ELE_' to namespace the El Escorial
    variables when merging with other PROACT sub-datasets.

    `subject_id` is the join key and is left unchanged.

    Parameters
    ----------
    file_path : str
        Path to PROACT_ELESCORIAL_v2.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns
        (-> saved as PROACT_ELESCORIAL_v3.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)
    df = df.rename(columns={col: f'ELE_{col}' for col in df.columns if col != 'subject_id'})

    return df










# ==================================================================
# ------------------------- PIPELINE EXECUTION ---------------------
# ==================================================================

def run(DATA_PATH, PROACT_PATH):

    print("\n" * 3)
    print("=" * 60)
    print("ELESCORIAL PIPELINE")
    print("=" * 60)

    

    # Stage v2 - Drop uninformative column
    df = modify_proact_elescorial(PROACT_PATH + '/PROACT_ELESCORIAL.csv')
    df.to_csv(DATA_PATH + '/PROACT_ELESCORIAL_v2.csv', index=False)



    # Diagnostic - Check for unexpected duplicate patients
    check_unique_lines(DATA_PATH + '/PROACT_ELESCORIAL_v2.csv')



    # Stage v3 - Add 'ELE_' prefix to all feature columns
    df_renamed = rename_all_columns(DATA_PATH + '/PROACT_ELESCORIAL_v2.csv')
    df_renamed.to_csv(DATA_PATH + '/PROACT_ELESCORIAL_v3.csv', index=False)