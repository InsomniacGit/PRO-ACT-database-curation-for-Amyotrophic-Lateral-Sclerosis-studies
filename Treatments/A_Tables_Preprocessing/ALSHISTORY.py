"""
PROACT ALSHISTORY Processing Pipeline
======================================
This script processes the PROACT (PRO-ACT ALS) disease history dataset through
a sequential cleaning and feature engineering pipeline. It produces six
intermediate CSV files, culminating in a wide-format one-row-per-patient
binary feature matrix suitable for machine learning or statistical analysis.

Pipeline stages:
    v2  - Consolidate duplicate rows into one row per patient;
          concatenate multi-valued fields (Symptom, Location) with ';'
    v3  - Clean and impute Site_of_Onset from binary indicator columns;
          drop redundant and always-zero columns
    v4  - Correct inverted Onset_Delta / Diagnosis_Delta pairs
    v5  - Encode Site_of_Onset and Symptom as binary indicator columns;
          drop sparse free-text fields (Symptom_Other_Specify, Location)
    v6  - Prefix all feature columns with 'HIS_' for downstream merging

Author: Bouclier Lucas
Data:   PROACT dataset (2022-07-29 release)
"""



import pandas as pd
import numpy as np





# //////////////////////////////////////////////////////////////
# ------------------------- ALSHISTORY -------------------------
# //////////////////////////////////////////////////////////////



# ------------------------------------------------------------------
# Stage v2 - Consolidate duplicate rows into one row per patient
# ------------------------------------------------------------------

# NOTE: Exploratory analysis (commented out below) showed that only three
# columns carry genuinely different values across rows for the same patient:
# 'Symptom', 'Symptom_Other_Specify', and 'Location'. All other columns are
# either constant within a patient or differ only by NaN vs. a real value.
# These three columns are therefore concatenated with ';' as a separator,
# while all other columns take the first non-null value available.

# def display_duplicated_subjects(csv_file):
#     df = pd.read_csv(csv_file, low_memory=False)
#     duplicated_subjects = df['subject_id'].duplicated(keep=False)
#     df_duplicated = df[duplicated_subjects]
#     print(df_duplicated)
#     print(f'Number of subject_id values with multiple rows: {df_duplicated["subject_id"].nunique()}')
#     columns_to_check = [col for col in df.columns if col != 'subject_id']
#     for col in columns_to_check:
#         non_null_values = df_duplicated.groupby('subject_id')[col].apply(lambda x: x.dropna().nunique())
#         subject_ids_multiple_values = non_null_values[non_null_values > 1].index.tolist()
#         if subject_ids_multiple_values:
#             print(f'Column "{col}" has multiple non-null values for the same subject_id: {subject_ids_multiple_values}')



def group_by_subject_id(csv_file):
    """
    Collapse multiple rows per patient into a single row.

    For multi-valued columns (Symptom, Symptom_Other_Specify, Location),
    all distinct non-null values found across rows for the same patient are
    joined into a single string separated by '; '. For all other columns,
    the first non-null value encountered is kept.

    Parameters
    ----------
    csv_file : str
        Path to the raw PROACT_ALSHISTORY.csv file.

    Returns
    -------
    pd.DataFrame
        One-row-per-patient DataFrame (-> saved as PROACT_ALSHISTORY_v2.csv).
    """
    df = pd.read_csv(csv_file, low_memory=False)
    columns = [col for col in df.columns if col not in ['subject_id']]
    result_list = []

    for subject_id, group in df.groupby('subject_id'):

        group = group.reset_index(drop=True)
        row_data = {'subject_id': subject_id}

        for col in columns:
            if col in ['Symptom', 'Symptom_Other_Specify', 'Location']:
                # Concatenate all distinct non-null values with '; ' separator
                unique_values = group[col].dropna().unique()
                row_data[col] = '; '.join(map(str, unique_values)) if len(unique_values) > 0 else np.nan
            else:
                # For stable columns, take the first non-null value
                first_value = group[col].dropna().iloc[0] if not group[col].dropna().empty else np.nan
                row_data[col] = first_value

        result_list.append(row_data)

    df_final = pd.DataFrame(result_list)

    return df_final





# ------------------------------------------------------------------
# Stage v3 - Clean and impute Site_of_Onset
# ------------------------------------------------------------------

def impute_site_of_onset(csv_file):
    """
    Clean the Site_of_Onset column and impute its value from binary indicator
    columns when it is missing.

    The raw column contains a redundant 'Onset: ' prefix that is stripped.
    When Site_of_Onset is NaN, its value is reconstructed from three binary
    columns (Site_of_Onset___Bulbar, Site_of_Onset___Limb) following this
    priority order:
        1. Both Bulbar and Limb are 1  -> 'Limb and Bulbar'
        2. Only Bulbar is 1            -> 'Bulbar'
        3. Only Limb is 1              -> 'Limb'

    Columns dropped after imputation:
        Site_of_Onset___Bulbar, ___Limb, ___Limb_and_Bulbar,
        ___Other, ___Other_Specify, ___Spine  - now encoded in Site_of_Onset
        Subject_ALS_History_Delta              - always 0, carries no information

    Fully empty columns are also removed, and key identifier columns are
    moved to the front of the DataFrame for readability.

    Parameters
    ----------
    csv_file : str
        Path to PROACT_ALSHISTORY_v2.csv.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with imputed Site_of_Onset
        (-> saved as PROACT_ALSHISTORY_v3.csv).
    """
    df = pd.read_csv(csv_file, low_memory=False)

    # Strip the redundant 'Onset: ' prefix present in some raw values
    df['Site_of_Onset'] = df['Site_of_Onset'].str.replace('Onset: ', '', regex=False)

    # Impute Site_of_Onset from binary indicator columns when it is missing
    df['Site_of_Onset'] = df.apply(
        lambda row: (
            'Limb and Bulbar' if pd.isna(row['Site_of_Onset'])
                and row.get('Site_of_Onset___Bulbar', 0) == 1
                and row.get('Site_of_Onset___Limb', 0) == 1
            else 'Bulbar' if pd.isna(row['Site_of_Onset'])
                and row.get('Site_of_Onset___Bulbar', 0) == 1
            else 'Limb' if pd.isna(row['Site_of_Onset'])
                and row.get('Site_of_Onset___Limb', 0) == 1
            else row['Site_of_Onset']
        ),
        axis=1
    )

    # Drop binary indicator columns now that Site_of_Onset has been imputed
    df = df.drop(columns=[
        'Site_of_Onset___Bulbar', 'Site_of_Onset___Limb',
        'Site_of_Onset___Limb_and_Bulbar', 'Site_of_Onset___Other',
        'Site_of_Onset___Other_Specify', 'Site_of_Onset___Spine'
    ], errors='ignore')

    # Drop column that is always 0 and carries no information
    df = df.drop(columns=['Subject_ALS_History_Delta'], errors='ignore')

    # Remove any columns that are entirely empty
    df = df.dropna(axis=1, how='all')

    # Bring the four key columns to the front for readability
    cols = (
        ['subject_id', 'Onset_Delta', 'Diagnosis_Delta', 'Site_of_Onset']
        + [col for col in df.columns if col not in ['subject_id', 'Onset_Delta', 'Diagnosis_Delta', 'Site_of_Onset']]
    )
    df = df[cols]

    return df





# ------------------------------------------------------------------
# Stage v4 - Correct inverted Onset_Delta / Diagnosis_Delta pairs
# ------------------------------------------------------------------

def correct_onset_diagnosis(csv_file):
    """
    Ensure that Onset_Delta is always earlier (smaller) than Diagnosis_Delta.

    By definition, symptom onset must precede clinical diagnosis. Rows where
    Onset_Delta > Diagnosis_Delta are assumed to be data entry errors and are
    corrected by swapping the two values.

    Parameters
    ----------
    csv_file : str
        Path to PROACT_ALSHISTORY_v3.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with corrected temporal ordering
        (-> saved as PROACT_ALSHISTORY_v4.csv).
    """
    df = pd.read_csv(csv_file, low_memory=False)

    def validate_and_correct_row(row):
        """Swap Onset_Delta and Diagnosis_Delta if their order is inverted."""
        onset_delta     = row['Onset_Delta']
        diagnosis_delta = row['Diagnosis_Delta']
        if pd.notna(onset_delta) and pd.notna(diagnosis_delta):
            if onset_delta > diagnosis_delta:
                row['Onset_Delta'], row['Diagnosis_Delta'] = diagnosis_delta, onset_delta
        return row

    df = df.apply(validate_and_correct_row, axis=1)

    return df





# ------------------------------------------------------------------
# Stage v5 - Encode categorical columns as binary indicators
# ------------------------------------------------------------------

def binary_site_of_onset(csv_file):
    """
    Replace the free-text Site_of_Onset column with four binary indicator columns.

    The four anatomical onset regions are encoded as boolean columns:
        Site_of_Onset_Limb, Site_of_Onset_Bulbar,
        Site_of_Onset_Other, Site_of_Onset_Spine

    'Limb and Bulbar' patients receive True in both Site_of_Onset_Limb and
    Site_of_Onset_Bulbar, as the str.contains() check handles partial matches.

    Parameters
    ----------
    csv_file : str
        Path to PROACT_ALSHISTORY_v4.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with Site_of_Onset replaced by four boolean columns.
    """
    df = pd.read_csv(csv_file, low_memory=False)

    df['Site_of_Onset_Limb']   = df['Site_of_Onset'].str.contains("Limb",   na=False).astype(bool)
    df['Site_of_Onset_Bulbar'] = df['Site_of_Onset'].str.contains("Bulbar", na=False).astype(bool)
    df['Site_of_Onset_Other']  = df['Site_of_Onset'].str.contains("Other",  na=False).astype(bool)
    df['Site_of_Onset_Spine']  = df['Site_of_Onset'].str.contains("Spine",  na=False).astype(bool)

    # Drop the original free-text column now that it is encoded
    df = df.drop(columns=['Site_of_Onset'])

    return df



def create_binary_columns(df, column, prefix):
    """
    Create one boolean indicator column per unique value found in a
    semicolon-separated multi-valued column.

    Values are normalised (stripped and title-cased) before deduplication to
    avoid spurious duplicates from inconsistent capitalisation. The resulting
    column names follow the pattern: {prefix}_{value_with_underscores}.

    NOTE: Symptom_Other_Specify and Location were assessed but found too
    sparse to be informative at the chosen prevalence threshold; they are
    dropped rather than encoded (see below).

    Parameters
    ----------
    df     : pd.DataFrame  Input DataFrame.
    column : str           Name of the semicolon-separated column to encode.
    prefix : str           Prefix to prepend to each generated column name.

    Returns
    -------
    binary_df    : pd.DataFrame  Boolean indicator columns, one per unique value.
    unique_values: list          Sorted list of unique normalised term strings.
    """
    # Collect all distinct terms across all rows
    unique_values = (
        df[column]
        .dropna()
        .str.split('; ')
        .explode()
        .str.strip()
        .str.title()
        .unique()
        .tolist()
    )
    unique_values.sort()

    new_columns = []

    for val in unique_values:
        col_name = f"{prefix}_{val.replace(' ', '_')}"

        # A patient is True if the exact term appears in their semicolon list
        new_col = df[column].apply(
            lambda x: True if (
                isinstance(x, str)
                and val in [s.strip().title() for s in x.split(';')]
            ) else False
        )
        new_col.name = col_name
        new_columns.append(new_col)

    # Concatenate all indicator columns at once (more efficient than iterative assignment)
    binary_df = pd.concat(new_columns, axis=1)

    return binary_df, unique_values





# ------------------------------------------------------------------
# Stage v6 - Add 'HIS_' prefix to all feature columns
# ------------------------------------------------------------------

def rename_all_columns(csv_file):
    """
    Prefix every feature column with 'HIS_' to namespace the disease history
    variables when merging with other PROACT sub-datasets.

    `subject_id` is the join key and is left unchanged.

    Parameters
    ----------
    csv_file : str
        Path to PROACT_ALSHISTORY_v5.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns
        (-> saved as PROACT_ALSHISTORY_v6.csv).
    """
    df = pd.read_csv(csv_file, low_memory=False)
    df = df.rename(columns={col: f'HIS_{col}' for col in df.columns if col != 'subject_id'})
    return df










# ==================================================================
# ------------------------- PIPELINE EXECUTION ---------------------
# ==================================================================

def run(DATA_PATH, PROACT_PATH):

    print("\n" * 3)
    print("=" * 60)
    print("ALSHISTORY PIPELINE")
    print("=" * 60)

    

    # Exploratory analysis commented before v2 - shows that only three columns have multiple non-null values across rows for the same patient_id
    # display_duplicated_subjects(PROACT_PATH + '/PROACT_ALSHISTORY.csv')



    # Stage v2 - Consolidate duplicate rows into one row per patient
    df_grouped = group_by_subject_id(PROACT_PATH + '/PROACT_ALSHISTORY.csv')
    df_grouped.to_csv(DATA_PATH + '/PROACT_ALSHISTORY_v2.csv', index=False)



    # Stage v3 - Clean and impute Site_of_Onset
    df_alshistory_cleaned = impute_site_of_onset(DATA_PATH + '/PROACT_ALSHISTORY_v2.csv')
    df_alshistory_cleaned.to_csv(DATA_PATH + '/PROACT_ALSHISTORY_v3.csv', index=False)



    # Stage v4 - Correct inverted Onset_Delta / Diagnosis_Delta pairs
    df_corrected = correct_onset_diagnosis(DATA_PATH + '/PROACT_ALSHISTORY_v3.csv')
    df_corrected.to_csv(DATA_PATH + '/PROACT_ALSHISTORY_v4.csv', index=False)



    # Stage v5 - Encode categorical columns as binary indicators
    # Apply binary encoding to Site_of_Onset
    df = binary_site_of_onset(DATA_PATH + '/PROACT_ALSHISTORY_v4.csv')

    # Encode Symptom as binary indicators
    binary_symptom_df, all_symptoms = create_binary_columns(df, "Symptom", "Symptom")
    df = pd.concat([df, binary_symptom_df], axis=1)
    df = df.drop(columns=["Symptom"])

    # Symptom_Other_Specify: values are too sparse to be informative - dropped
    # binary_sym_other_df, all_other_symptoms = create_binary_columns(df, "Symptom_Other_Specify", "Symptom_Other_Specify")
    # df = pd.concat([df, binary_sym_other_df], axis=1)
    df = df.drop(columns=["Symptom_Other_Specify"])

    # Location: values are too sparse to be informative - dropped
    # binary_location_df, all_locations = create_binary_columns(df, "Location", "Location")
    # df = pd.concat([df, binary_location_df], axis=1)
    df = df.drop(columns=["Location"])

    df.to_csv(DATA_PATH + '/PROACT_ALSHISTORY_v5.csv', index=False)



    # Stage v6 - Add 'HIS_' prefix to all feature columns
    df_renamed = rename_all_columns(DATA_PATH + '/PROACT_ALSHISTORY_v5.csv')
    df_renamed.to_csv(DATA_PATH + '/PROACT_ALSHISTORY_v6.csv', index=False)