"""
PROACT First Symptoms Temporal Alignment
=========================================
This script re-indexes all temporal PROACT tables from study-inclusion time
(δ relative to clinical trial entry) to disease-onset time (δ relative to
first reported symptom onset), as described in Section IV.D of the paper.

In the raw PROACT dataset, all longitudinal measurements are indexed by the
number of days since a patient's inclusion in a clinical trial. However,
different patients are enrolled at different stages of disease progression,
making direct comparison across patients difficult. Re-aligning trajectories
to symptom onset reduces inter-patient variability and produces more coherent
disease progression curves.

The alignment is performed by subtracting each patient's HIS_Onset_Delta
(the day of first symptom onset relative to trial inclusion, a negative value
in most cases) from every Delta column in each temporal table. Patients for
whom HIS_Onset_Delta is not recorded are excluded from the aligned dataset.

Tables processed (all Delta columns shifted):
    ALSFRS, FVC, HANDGRIPSTRENGTH, LABS, MUSCLESTRENGTH, SVC, VITALSIGNS,
    MERGE_NODELTA (the non-temporal union table, which carries the target
    variable Delta columns added at the supervised dataset generation step)

Tables skipped:
    ADVERSE EVENTS   - Delta columns were dropped during preprocessing
    CONMEDS          - Delta columns were dropped during preprocessing
    ALSHISTORY       - non-temporal (no Delta column)
    DEATHDATA        - non-temporal (no Delta column)
    DEMOGRAPHICS     - non-temporal (no Delta column)
    EL ESCORIAL      - non-temporal (no Delta column)
    FAMILY HISTORY   - non-temporal (no Delta column)
    RILUZOLE         - non-temporal (no Delta column)
    TREATMENT        - non-temporal (no Delta column)

Author: Bouclier Lucas
Data:   PROACT dataset (2022-07-29 release)
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




    
# ------------------------------------------------------------------
# Load the symptom onset reference values
# ------------------------------------------------------------------

def get_first_symptoms_delta(data_path, first_symptoms_path):
    """
    Load HIS_Onset_Delta from the preprocessed ALS HISTORY table and return
    a patient-to-onset dictionary used for temporal re-alignment.

    HIS_Onset_Delta is the day of first reported symptom onset relative to
    the patient's inclusion in the clinical trial. It is typically a negative
    value (symptoms precede enrolment). Subtracting it from any Delta column
    shifts the time axis so that day 0 corresponds to symptom onset rather
    than trial inclusion.

    Patients for whom HIS_Onset_Delta is missing are excluded from the
    dictionary and will be dropped from all aligned tables downstream.

    Two audit artefacts are produced:
        HIS_Onset_Delta_Distribution.png  - histogram of onset delta values
        HIS_Onset_Delta_Distribution.csv  - frequency table of onset delta values

    Parameters
    ----------
    data_path           : str   Path to the Root directory for all processed outputs
    first_symptoms_path : str   Path to the Root directory for first-symptoms-aligned output datasets

    Returns
    -------
    dict
        Mapping from subject_id (int) to HIS_Onset_Delta (float).
    """
    df_alshistory = pd.read_csv(
        data_path + "/PROACT_ALSHISTORY_v6.csv",
        usecols=["subject_id", "HIS_Onset_Delta"]
    )

    # Drop patients with no onset delta; they cannot be aligned
    df_alshistory = df_alshistory.dropna(subset=["HIS_Onset_Delta"])

    # Build a fast lookup dictionary for use in the alignment function
    delta_dict = pd.Series(
        df_alshistory.HIS_Onset_Delta.values,
        index=df_alshistory.subject_id
    ).to_dict()

    # Audit: histogram of onset delta distribution
    plt.hist(df_alshistory["HIS_Onset_Delta"], bins=50, color='blue', alpha=0.7)
    plt.xlabel("HIS_Onset_Delta")
    plt.ylabel("Frequency")
    plt.title("Distribution of HIS_Onset_Delta values")
    plt.savefig(first_symptoms_path + "/HIS_Onset_Delta_Distribution.png")
    plt.close()

    # Audit: frequency table of onset delta values sorted chronologically
    delta_distribution = df_alshistory["HIS_Onset_Delta"].value_counts().reset_index()
    delta_distribution.columns = ["HIS_Onset_Delta", "Frequency"]
    delta_distribution = delta_distribution.sort_values(by="HIS_Onset_Delta")
    delta_distribution.to_csv(
        first_symptoms_path + "/HIS_Onset_Delta_Distribution.csv", index=False
    )

    return delta_dict





# ------------------------------------------------------------------
# Temporal re-alignment function
# ------------------------------------------------------------------

def align_first_symptoms_delta(file_path, output_csv, delta_dict):
    """
    Shift all Delta columns in a table from study-inclusion time to
    first-symptom-onset time by subtracting each patient's HIS_Onset_Delta.

    The transformation applied to each Delta column d for patient p is:

        d_aligned = d_original - HIS_Onset_Delta(p)

    After alignment, day 0 in the output corresponds to the date of first
    reported symptom, making trajectories comparable across patients who
    were enrolled at different disease stages.

    Patients absent from delta_dict (i.e. those without a recorded onset
    date) are dropped before processing, as they cannot be aligned.

    All columns whose name contains the substring 'Delta' are shifted.
    Other columns (scores, boolean indicators, counts, etc.) are unchanged.

    Parameters
    ----------
    file_path  : str   Path to the input preprocessed CSV file.
    output_csv : str   Path where the aligned CSV will be saved.
    delta_dict : dict  Mapping from subject_id to HIS_Onset_Delta.
    """
    print(f"Processing: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)

    # Drop patients for whom no onset delta is available
    df = df[df["subject_id"].map(delta_dict).notna()]

    # Subtract HIS_Onset_Delta from every Delta column
    for col in df.columns:
        if "Delta" in col:
            print(f"  - Aligning column: {col}")
            df[col] = df.apply(
                lambda row: row[col] - delta_dict.get(row["subject_id"], np.nan),
                axis=1
            )

    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")










# ==================================================================
# ------------------------- PIPELINE EXECUTION ---------------------
# ==================================================================

def run(DATA_PATH, MERGE_NODELTA_PATH, FIRST_SYMPTOMS_PATH):

    print("\n" * 3)
    print("=" * 60)
    print("FIRST SYMPTOMS ALIGNMENT PIPELINE")
    print("=" * 60)



    # Load the symptom onset reference values
    delta_dict = get_first_symptoms_delta(data_path=DATA_PATH, first_symptoms_path=FIRST_SYMPTOMS_PATH)
    print(f"Patients with HIS_Onset_Delta: {len(delta_dict)}")



    # ALSFRS
    align_first_symptoms_delta(
        file_path  = DATA_PATH + "/PROACT_ALSFRS_v8.csv",
        output_csv = FIRST_SYMPTOMS_PATH + "/PROACT_ALSFRS_FIRST_SYMPTOMS.csv",
        delta_dict = delta_dict
    )



    # FVC
    align_first_symptoms_delta(
        file_path  = DATA_PATH + "/PROACT_FVC_v7.csv",
        output_csv = FIRST_SYMPTOMS_PATH + "/PROACT_FVC_FIRST_SYMPTOMS.csv",
        delta_dict = delta_dict
    )



    # HANDGRIPSTRENGTH
    align_first_symptoms_delta(
        file_path  = DATA_PATH + "/PROACT_HANDGRIPSTRENGTH_v8.csv",
        output_csv = FIRST_SYMPTOMS_PATH + "/PROACT_HANDGRIPSTRENGTH_FIRST_SYMPTOMS.csv",
        delta_dict = delta_dict
    )



    # LABS
    align_first_symptoms_delta(
        file_path  = DATA_PATH + "/PROACT_LABS_v10.csv",
        output_csv = FIRST_SYMPTOMS_PATH + "/PROACT_LABS_FIRST_SYMPTOMS.csv",
        delta_dict = delta_dict
    )



    # MUSCLESTRENGTH
    align_first_symptoms_delta(
        file_path  = DATA_PATH + "/PROACT_MUSCLESTRENGTH_v8.csv",
        output_csv = FIRST_SYMPTOMS_PATH + "/PROACT_MUSCLESTRENGTH_FIRST_SYMPTOMS.csv",
        delta_dict = delta_dict
    )



    # SVC
    align_first_symptoms_delta(
        file_path  = DATA_PATH + "/PROACT_SVC_v7.csv",
        output_csv = FIRST_SYMPTOMS_PATH + "/PROACT_SVC_FIRST_SYMPTOMS.csv",
        delta_dict = delta_dict
    )



    # VITALSIGNS
    align_first_symptoms_delta(
        file_path  = DATA_PATH + "/PROACT_VITALSIGNS_v7.csv",
        output_csv = FIRST_SYMPTOMS_PATH + "/PROACT_VITALSIGNS_FIRST_SYMPTOMS.csv",
        delta_dict = delta_dict
    )



    # ------------------------------------------------------------------
    # MERGE_NODELTA
    # ------------------------------------------------------------------

    # The non-temporal union table contains no Delta columns to shift.
    # This call only drops patients without a recorded symptom onset date,
    # ensuring cohort consistency with the aligned temporal tables.
    align_first_symptoms_delta(
        file_path  = MERGE_NODELTA_PATH + "/PROACT_MERGE_NODELTA_V2.csv",
        output_csv = FIRST_SYMPTOMS_PATH + "/PROACT_MERGE_NODELTA_FIRST_SYMPTOMS.csv",
        delta_dict = delta_dict
    )



    # ------------------------------------------------------------------
    # Tables skipped - no Delta columns to shift
    # ------------------------------------------------------------------

    # ADVERSE EVENTS   - Delta columns were dropped during preprocessing (v1->v2)
    # CONMEDS          - Delta columns were dropped during preprocessing (v3->v4)
    # ALSHISTORY       - non-temporal dataset, no Delta column after preprocessing
    # DEATHDATA        - non-temporal dataset, no Delta column after preprocessing
    # DEMOGRAPHICS     - non-temporal dataset, no Delta column after preprocessing
    # EL ESCORIAL      - non-temporal dataset, no Delta column after preprocessing
    # FAMILY HISTORY   - non-temporal dataset, no Delta column after preprocessing
    # RILUZOLE         - non-temporal dataset, no Delta column after preprocessing
    # TREATMENT        - non-temporal dataset, no Delta column after preprocessing