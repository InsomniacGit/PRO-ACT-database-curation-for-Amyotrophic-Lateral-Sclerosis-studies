"""
PROACT Concomitant Medications (CONMEDS) Processing Pipeline
=============================================================
This script processes the PROACT (PRO-ACT ALS) concomitant medications dataset
through a sequential cleaning and feature engineering pipeline. It produces
seven intermediate CSV files, culminating in a wide-format boolean feature
matrix with one row per patient and one column per medication.

Pipeline stages:
    v2  - Split semicolon-separated Medication_Coded entries into individual rows
    v3  - Filter out medications reported by fewer than a prevalence threshold
    v4  - Drop dosing/timing columns and deduplicate per-patient medication rows
    v5  - Add per-patient observation count
    v6  - Encode medications as binary (True/False) indicator columns
    v7  - Prefix all feature columns with 'CON_' for downstream merging

Author: Bouclier Lucas
Data:   PROACT dataset (2022-07-29 release)
"""



import pandas as pd





# ///////////////////////////////////////////////////////////
# ------------------------- CONMEDS -------------------------
# ///////////////////////////////////////////////////////////



# ------------------------------------------------------------------
# Stage v2 - Split multi-valued Medication_Coded entries
# ------------------------------------------------------------------

def split_medication_coded(file_path):
    """
    Expand rows where Medication_Coded contains multiple medications joined
    by ';' into one row per individual medication.

    Some records in the raw file list several medications in a single
    Medication_Coded cell separated by semicolons. This function splits those
    cells so that each row refers to exactly one medication, which is required
    for accurate per-medication counting and filtering in later stages.

    Rows where Medication_Coded is missing are dropped beforehand, as they
    carry no usable medication information.

    Parameters
    ----------
    file_path : str
        Path to the raw PROACT_CONMEDS.csv file.

    Returns
    -------
    pd.DataFrame
        Expanded one-medication-per-row DataFrame
        (-> saved as PROACT_CONMEDS_v2.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)

    # Drop rows with no medication name before splitting
    df = df.dropna(subset=['Medication_Coded'])

    # Split semicolon-separated values and explode into individual rows
    df = df.assign(Medication_Coded=df['Medication_Coded'].str.split(';')).explode('Medication_Coded')
    df['Medication_Coded'] = df['Medication_Coded'].str.strip()
    df = df.reset_index(drop=True)

    return df





# ------------------------------------------------------------------
# Prevalence reference - Count patients per medication
# ------------------------------------------------------------------

def summarize_conmeds_prevalence(file_path):
    """
    Compute the number and percentage of patients who received each medication.

    Text is normalised (stripped, title-cased, quotes removed) before
    aggregation to avoid spurious duplicates from formatting inconsistencies.
    The resulting table is used by filter_conmeds_by_percentage() to decide
    which medications to retain.

    Parameters
    ----------
    file_path : str
        Path to PROACT_CONMEDS_v2.csv.

    Returns
    -------
    pd.DataFrame
        Summary DataFrame with columns [Medication_Coded, Occurrences, Percentage],
        sorted by Occurrences descending
        (-> saved as PROACT_CONMEDS_list.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)

    # Normalise medication names before grouping
    df['Medication_Coded'] = (
        df['Medication_Coded']
        .str.strip()
        .str.title()
        .str.replace('"', '', regex=False)
        .str.replace("'", '', regex=False)
    )

    list_conmeds = df['Medication_Coded'].dropna().unique().tolist()
    list_conmeds.sort()
    print(f'Unique medications: {len(list_conmeds)}')

    # Count distinct patients per medication
    total_subjects = df['subject_id'].nunique()
    df_conmeds = (
        df[df['Medication_Coded'].isin(list_conmeds)]
        .groupby('Medication_Coded')['subject_id']
        .nunique()
        .reset_index(name='Occurrences')
    )
    df_conmeds['Percentage'] = (df_conmeds['Occurrences'] / total_subjects * 100).round(2)
    df_conmeds = df_conmeds.sort_values(by='Occurrences', ascending=False).reset_index(drop=True)

    return df_conmeds





# ------------------------------------------------------------------
# Stage v3 - Filter rare medications by prevalence threshold
# ------------------------------------------------------------------

def filter_conmeds_by_percentage(file_path, threshold, data_path):
    """
    Retain only rows corresponding to medications reported by at least
    `threshold` percent of the total patient cohort.

    The prevalence reference file produced by list_conmeds() is used to
    build the allow-list of medications to keep. Text normalisation is
    re-applied before filtering to guarantee consistency with the reference.

    Parameters
    ----------
    file_path : str   Path to PROACT_CONMEDS_v2.csv.
    threshold : float Minimum patient prevalence (%) required to keep a medication.
    data_path : str   Path to the Root directory for all processed outputs

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only sufficiently prevalent medications
        (-> saved as PROACT_CONMEDS_v3.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)
    df_list = pd.read_csv(data_path + '/PROACT_CONMEDS_list.csv', low_memory=False)

    # Build allow-list from the prevalence reference table
    conmeds_to_keep = df_list[df_list['Percentage'] >= threshold]['Medication_Coded'].tolist()

    # Re-apply normalisation to match the reference table exactly
    df['Medication_Coded'] = (
        df['Medication_Coded']
        .str.strip()
        .str.title()
        .str.replace('"', '', regex=False)
        .str.replace("'", '', regex=False)
    )
    df_filtered = df[df['Medication_Coded'].isin(conmeds_to_keep)]

    return df_filtered





# ------------------------------------------------------------------
# Stage v4 - Drop dosing columns and deduplicate
# ------------------------------------------------------------------

def clean_conmeds_data(file_path):
    """
    Remove dosing and timing columns that are not used in this study, and
    deduplicate so that each (patient, medication) pair appears only once.

    Dropped columns:
        Start_Delta, Stop_Delta  - temporal information not retained here
        Dose, Unit, Frequency,
        Route                    - dosing details not used in this study

    After dropping these columns, a patient may have multiple identical rows
    for the same medication (e.g. recorded at different visits). These
    duplicates are removed so that stage v6 correctly encodes presence/absence.

    Parameters
    ----------
    file_path : str
        Path to PROACT_CONMEDS_v3.csv.

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame with one row per (subject_id, Medication_Coded)
        pair (-> saved as PROACT_CONMEDS_v4.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)

    cols_to_drop = ['Start_Delta', 'Stop_Delta', 'Dose', 'Unit', 'Frequency', 'Route']
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Drop rows with no medication name (safety check after column removal)
    df = df.dropna(subset=['Medication_Coded'])

    # One row per patient-medication pair is sufficient for boolean encoding
    df = df.drop_duplicates(subset=['subject_id', 'Medication_Coded'])

    return df





# ------------------------------------------------------------------
# Stage v5 - Add per-patient observation count
# ------------------------------------------------------------------

def observation_counter_conmeds(file_path):
    """
    Add an `observation_count` column recording how many distinct medications
    each patient has recorded after filtering and deduplication.

    Parameters
    ----------
    file_path : str
        Path to PROACT_CONMEDS_v4.csv.

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





# ------------------------------------------------------------------
# Stage v6 - Encode medications as binary indicator columns
# ------------------------------------------------------------------

def pivot_conmeds_to_boolean_matrix(file_path):
    """
    Pivot the long-format medication data into a wide-format boolean matrix.

    For every distinct medication present in the filtered dataset, a boolean
    column is created in a patient-level DataFrame, indicating whether that
    patient received the medication (True) or not (False).

    All medication names are normalised (stripped, title-cased, spaces replaced
    by underscores) before being used as column names, so that the resulting
    identifiers are valid Python/pandas column names.

    A progress message is printed every 500 patients to monitor execution,
    as this function iterates over all patients individually.

    Parameters
    ----------
    file_path : str
        Path to PROACT_CONMEDS_v5.csv.

    Returns
    -------
    pd.DataFrame
        Wide-format patient-level boolean feature matrix with one row per
        patient (-> saved as PROACT_CONMEDS_v6.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)

    # Normalise medication names and replace spaces with underscores for
    # use as valid column identifiers
    df['Medication_Coded'] = (
        df['Medication_Coded']
        .str.strip()
        .str.title()
        .str.replace('"', '', regex=False)
        .str.replace("'", '', regex=False)
        .str.replace(' ', '_', regex=False)
    )

    # Sorted list of all medications present after filtering
    unique_conmeds = sorted(df['Medication_Coded'].unique().tolist())

    list_results = []

    for subject_id, group in df.groupby('subject_id'):
        row_data = {
            'subject_id': subject_id,
            'observation_count': group.shape[0],
        }

        # Initialise all medication columns to False for this patient
        for conmed in unique_conmeds:
            row_data[conmed] = False

        # Set to True for each medication this patient actually received
        for conmed in group['Medication_Coded'].unique():
            row_data[conmed] = True

        list_results.append(row_data)

        # Progress indicator: print every 500 patients processed
        if len(list_results) % 500 == 0:
            print(f'Processed {len(list_results)} patients out of {df["subject_id"].nunique()}')

    df_final = pd.DataFrame(list_results)

    return df_final





# ------------------------------------------------------------------
# Stage v7 - Add 'CON_' prefix to all feature columns
# ------------------------------------------------------------------

def rename_all_columns(file_path):
    """
    Prefix every feature column with 'CON_' to namespace the concomitant
    medication variables when merging with other PROACT sub-datasets.

    `subject_id` is the join key and is left unchanged.

    Parameters
    ----------
    file_path : str
        Path to PROACT_CONMEDS_v6.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns
        (-> saved as PROACT_CONMEDS_v7.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)
    df = df.rename(columns={col: f'CON_{col}' for col in df.columns if col != 'subject_id'})
    return df










# ==================================================================
# ------------------------- PIPELINE EXECUTION ---------------------
# ==================================================================

def run(DATA_PATH, PROACT_PATH):

    print("\n" * 3)
    print("=" * 60)
    print("CONMEDS PIPELINE")
    print("=" * 60)

    

    # Stage v2 - Split multi-valued Medication_Coded entries
    df_duplicated = split_medication_coded(PROACT_PATH + '/PROACT_CONMEDS.csv')
    df_duplicated.to_csv(DATA_PATH + '/PROACT_CONMEDS_v2.csv', index=False)



    # Prevalence reference - Count patients per medication
    df_boolean_conmeds = summarize_conmeds_prevalence(DATA_PATH + '/PROACT_CONMEDS_v2.csv')
    df_boolean_conmeds.to_csv(DATA_PATH + '/PROACT_CONMEDS_list.csv', index=False)



    # Stage v3 - Filter rare medications by prevalence threshold
    df_filtered_conmeds = filter_conmeds_by_percentage(
        DATA_PATH + '/PROACT_CONMEDS_v2.csv', threshold=5, data_path=DATA_PATH
    )
    df_filtered_conmeds.to_csv(DATA_PATH + '/PROACT_CONMEDS_v3.csv', index=False)



    # Stage v4 - Drop dosing columns and deduplicate
    df_cleaned_conmeds = clean_conmeds_data(DATA_PATH + '/PROACT_CONMEDS_v3.csv')
    df_cleaned_conmeds.to_csv(DATA_PATH + '/PROACT_CONMEDS_v4.csv', index=False)



    # Stage v5 - Add per-patient observation count
    df_counter = observation_counter_conmeds(DATA_PATH + '/PROACT_CONMEDS_v4.csv')
    df_counter.to_csv(DATA_PATH + '/PROACT_CONMEDS_v5.csv', index=False)



    # Stage v6 - Encode medications as binary indicator columns
    df_transformed = pivot_conmeds_to_boolean_matrix(DATA_PATH + '/PROACT_CONMEDS_v5.csv')
    df_transformed.to_csv(DATA_PATH + '/PROACT_CONMEDS_v6.csv', index=False)



    # Stage v7 - Add 'CON_' prefix to all feature columns
    df_renamed = rename_all_columns(DATA_PATH + '/PROACT_CONMEDS_v6.csv')
    df_renamed.to_csv(DATA_PATH + '/PROACT_CONMEDS_v7.csv', index=False)