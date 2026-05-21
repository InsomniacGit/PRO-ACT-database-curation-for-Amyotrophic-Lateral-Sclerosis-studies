"""
PROACT Treatment Processing Pipeline
=====================================
This script processes the PROACT (PRO-ACT ALS) treatment dataset through a
minimal pipeline. It produces one output CSV file, namespacing the columns
for downstream merging without any prior cleaning step.

The treatment dataset records the clinical trial arm assignment for each
patient (e.g. active treatment vs. placebo) and is expected to be clean
and one-row-per-patient as delivered by the PROACT consortium.

Pipeline stages:
    v2  - Prefix all feature columns with 'TRE_' for downstream merging

Author: Bouclier Lucas
Data:   PROACT dataset (2022-07-29 release)
"""



import pandas as pd





# /////////////////////////////////////////////////////////////
# ------------------------- TREATMENT -------------------------
# /////////////////////////////////////////////////////////////



# ------------------------------------------------------------------
# Stage v2 - Add 'TRE_' prefix to all feature columns
# ------------------------------------------------------------------

def rename_all_columns(file_path):
    """
    Prefix every feature column with 'TRE_' to namespace the treatment
    variables when merging with other PROACT sub-datasets.

    `subject_id` is the join key and is left unchanged.

    Parameters
    ----------
    file_path : str
        Path to the raw PROACT_TREATMENT.csv file.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns
        (-> saved as PROACT_TREATMENT_v2.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)
    df = df.rename(columns={col: f'TRE_{col}' for col in df.columns if col != 'subject_id'})

    return df










# ==================================================================
# ------------------------- PIPELINE EXECUTION ---------------------
# ==================================================================

def run(DATA_PATH, PROACT_PATH):

    print("\n" * 3)
    print("=" * 60)
    print("TREATMENT PIPELINE")
    print("=" * 60)

    

    # Stage v2 - Add 'TRE_' prefix to all feature columns
    df_renamed = rename_all_columns(PROACT_PATH + '/PROACT_TREATMENT.csv')
    df_renamed.to_csv(DATA_PATH + '/PROACT_TREATMENT_v2.csv', index=False)