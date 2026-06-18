"""
PROACT Non-Temporal Tables Merging and Supervised Dataset Generation
====================================================================
This script merges all nine preprocessed non-temporal PROACT tables into a
single wide-format patient-level dataset (union join), then generates one
supervised learning dataset per prediction horizon by appending the
corresponding ALSFRS-R target variable.

The resulting files are the non-temporal feature matrices (¬δ tables) used as
auxiliary inputs alongside the ALSFRS temporal table in the regression
experiments described in the paper. They correspond to the ¬δ component of
the best-performing merge strategy (ALSFRS ∩ ¬δ ∩ SVC and variants).

Two dataset versions are managed:
    V2  - Full non-temporal merge including ADVERSE EVENTS and CONMEDS
    V1  - Lighter merge without ADVERSE EVENTS and CONMEDS (archived, commented out)

For each prediction horizon present in the target variable file, a separate
CSV is produced containing the merged non-temporal features joined with the
target column, with rows missing the target value dropped.

Prediction horizons:
    Fixed_0_3M   - target at trimester 1 (days 0-90)
    Fixed_0_6M   - target at trimester 2 (days 0-180)
    Fixed_0_9M   - target at trimester 3 (days 0-270)
    Fixed_0_12M  - target at trimester 4 (days 0-360)

Author: Bouclier Lucas
Data:   PROACT dataset (2022-07-29 release)
"""



import pandas as pd





# ==================================================================
# ------------------------- PIPELINE EXECUTION ---------------------
# ==================================================================

def run(DATA_PATH, MERGE_NODELTA_PATH):

    print("\n" * 3)
    print("=" * 60)
    print("MERGE NODELTA PIPELINE")
    print("=" * 60)



    # ------------------------------------------------------------------
    # Load all preprocessed non-temporal tables
    # ------------------------------------------------------------------

    # Each file is the final versioned output of its respective pipeline script.
    # All feature columns carry a three-letter source prefix (e.g. ADV_, HIS_)
    # to preserve traceability after merging.

    df_adverseevents = pd.read_csv(DATA_PATH + '/PROACT_ADVERSEEVENTS_v7.csv')
    df_alshistory    = pd.read_csv(DATA_PATH + '/PROACT_ALSHISTORY_v6.csv')
    df_conmeds       = pd.read_csv(DATA_PATH + '/PROACT_CONMEDS_v7.csv')
    df_deathdata     = pd.read_csv(DATA_PATH + '/PROACT_DEATHDATA_v3.csv')
    df_demographics  = pd.read_csv(DATA_PATH + '/PROACT_DEMOGRAPHICS_v6.csv')
    df_elescorial    = pd.read_csv(DATA_PATH + '/PROACT_ELESCORIAL_v3.csv')
    df_familyhistory = pd.read_csv(DATA_PATH + '/PROACT_FAMILYHISTORY_v9.csv')
    df_riluzole      = pd.read_csv(DATA_PATH + '/PROACT_RILUZOLE_v2.csv')
    df_treatment     = pd.read_csv(DATA_PATH + '/PROACT_TREATMENT_v2.csv')



    # ------------------------------------------------------------------
    # Merge non-temporal tables (union join)
    # ------------------------------------------------------------------

    # V1 - Lighter merge excluding ADVERSE EVENTS and CONMEDS.
    # Archived for comparison; commented out as V2 is the active version.
    #
    # df_merge_v1 = df_alshistory.merge(df_deathdata,     on='subject_id', how='outer')
    # df_merge_v1 = df_merge_v1.merge(df_demographics,   on='subject_id', how='outer')
    # df_merge_v1 = df_merge_v1.merge(df_elescorial,     on='subject_id', how='outer')
    # df_merge_v1 = df_merge_v1.merge(df_familyhistory,  on='subject_id', how='outer')
    # df_merge_v1 = df_merge_v1.merge(df_riluzole,       on='subject_id', how='outer')
    # df_merge_v1 = df_merge_v1.merge(df_treatment,      on='subject_id', how='outer')
    # df_merge_v1 = df_merge_v1.sort_values(by=['subject_id']).reset_index(drop=True)
    # df_merge_v1.to_csv(MERGE_NODELTA_PATH + '/PROACT_MERGE_NODELTA_V1.csv', index=False)

    # V2 - Full merge including ADVERSE EVENTS and CONMEDS.
    # An outer (union) join is used at each step so that no patient is discarded
    # if they are absent from one of the tables. This maximises cohort size at
    # the cost of introducing sparsity for patients missing in some tables,
    # consistent with the union merging strategy described in the paper.
    df_merge_v2 = df_adverseevents.merge(df_alshistory,    on='subject_id', how='outer')
    df_merge_v2 = df_merge_v2.merge(df_conmeds,            on='subject_id', how='outer')
    df_merge_v2 = df_merge_v2.merge(df_deathdata,          on='subject_id', how='outer')
    df_merge_v2 = df_merge_v2.merge(df_demographics,       on='subject_id', how='outer')
    df_merge_v2 = df_merge_v2.merge(df_elescorial,         on='subject_id', how='outer')
    df_merge_v2 = df_merge_v2.merge(df_familyhistory,      on='subject_id', how='outer')
    df_merge_v2 = df_merge_v2.merge(df_riluzole,           on='subject_id', how='outer')
    df_merge_v2 = df_merge_v2.merge(df_treatment,          on='subject_id', how='outer')
    df_merge_v2 = df_merge_v2.sort_values(by=['subject_id']).reset_index(drop=True)
    df_merge_v2.to_csv(MERGE_NODELTA_PATH + '/PROACT_MERGE_NODELTA_V2.csv', index=False)
    print(f'df_merge_v2 shape: {df_merge_v2.shape}')

    # V2 with intra-interval statistics (archived; requires pre-computed stats file)
    # df_merge_v2_stats = pd.read_csv(MERGE_NODELTA_PATH + '/PROACT_MERGE_NODELTA_V2_IntraStats.csv')