"""
PROACT Demographics Processing Pipeline
========================================
This script processes the PROACT (PRO-ACT ALS) demographics dataset through
a sequential cleaning and feature engineering pipeline. It produces six
intermediate CSV files, culminating in a one-row-per-patient feature matrix
suitable for machine learning or statistical analysis.

Pipeline stages:
    v2  - Consolidate binary race indicator columns into a single Race column
    v3  - Impute missing Age values from Date_of_Birth where available
    v4  - Drop administrative columns; reorder key columns
    v5  - Encode Race as binary indicator columns;
          drop sparse free-text field (Race_Other_Specify)
    v6  - Prefix all feature columns with 'DEM_' for downstream merging

Author: Bouclier Lucas
Data:   PROACT dataset (2022-07-29 release)
"""



import pandas as pd
import numpy as np





# ////////////////////////////////////////////////////////////////
# ------------------------- DEMOGRAPHICS -------------------------
# ////////////////////////////////////////////////////////////////



# ------------------------------------------------------------------
# Stage v2 - Consolidate binary race columns into a single Race column
# ------------------------------------------------------------------

def create_race_column(file_path):
    """
    Replace the seven binary race indicator columns with a single categorical
    Race column, then drop the original indicator columns.

    The raw dataset encodes race as a set of mutually exclusive binary flags
    (e.g. Race_Caucasian = 1.0). This function converts them into a single
    string column using np.select(), which evaluates conditions in order and
    assigns the label of the first matching condition. The binary columns are
    then dropped.

    Race_Other_Specify (a free-text field) is moved to the end of the
    DataFrame so that structured columns remain grouped together.

    Parameters
    ----------
    file_path : str
        Path to the raw PROACT_DEMOGRAPHICS.csv file.

    Returns
    -------
    pd.DataFrame
        DataFrame with a single Race column replacing the seven binary flags
        (-> saved as PROACT_DEMOGRAPHICS_v2.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)

    # Conditions evaluated in order; the first match determines the Race value
    conditions = [
        (df['Race_Americ_Indian_Alaska_Native'] == 1.0),
        (df['Race_Asian']                       == 1.0),
        (df['Race_Black_African_American']       == 1.0),
        (df['Race_Hawaiian_Pacific_Islander']    == 1.0),
        (df['Race_Unknown']                      == 1.0),
        (df['Race_Caucasian']                    == 1.0),
        (df['Race_Other']                        == 1.0),
    ]
    choices = [
        'Americ_Indian_Alaska_Native',
        'Asian',
        'Black_African_American',
        'Hawaiian_Pacific_Islander',
        'Unknown',
        'Caucasian',
        'Other',
    ]

    # default=None produces NaN for patients with no race indicator set
    df['Race'] = np.select(conditions, choices, default=None)

    df = df.drop(columns=[
        'Race_Americ_Indian_Alaska_Native',
        'Race_Asian',
        'Race_Black_African_American',
        'Race_Hawaiian_Pacific_Islander',
        'Race_Unknown',
        'Race_Caucasian',
        'Race_Other',
    ], errors='ignore')

    # Move the free-text field to the end to keep structured columns grouped
    cols = [col for col in df.columns if col != 'Race_Other_Specify']
    df = df[cols + ['Race_Other_Specify']]

    return df





# ------------------------------------------------------------------
# Stage v3 - Impute missing Age from Date_of_Birth
# ------------------------------------------------------------------

def age_add(file_path):
    """
    Impute Age (in years) for patients where it is missing but
    Date_of_Birth is available.

    In the PROACT dataset, Date_of_Birth is stored as a negative integer
    representing the patient's age in days relative to an arbitrary reference
    date. Age in years is approximated by taking the absolute value and
    dividing by 365.

    Parameters
    ----------
    file_path : str
        Path to PROACT_DEMOGRAPHICS_v2.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with Age imputed where possible
        (-> saved as PROACT_DEMOGRAPHICS_v3.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)

    # Impute only when Age is missing but Date_of_Birth is present
    df['Age'] = df.apply(
        lambda row: abs(row['Date_of_Birth']) // 365
        if pd.isna(row['Age']) and pd.notna(row['Date_of_Birth'])
        else row['Age'],
        axis=1
    )

    return df





# ------------------------------------------------------------------
# Stage v4 - Drop administrative columns and reorder
# ------------------------------------------------------------------

def remove_unused_columns_demographics(file_path):
    """
    Remove columns not used in this study and move Sex to the front.

    Dropped columns:
        Demographics_Delta  - administrative timing column, always 0
        Date_of_Birth       - superseded by the imputed Age column (stage v3)

    Parameters
    ----------
    file_path : str
        Path to PROACT_DEMOGRAPHICS_v3.csv.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with Sex as the second column after subject_id
        (-> saved as PROACT_DEMOGRAPHICS_v4.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)

    df = df.drop(columns=['Demographics_Delta', 'Date_of_Birth'])

    # Bring Sex forward as the first substantive feature after subject_id
    cols = ['subject_id', 'Sex'] + [col for col in df.columns if col not in ['subject_id', 'Sex']]
    df = df[cols]

    return df





# ------------------------------------------------------------------
# Diagnostic - Check for unexpected duplicate patients
# ------------------------------------------------------------------

def check_unique_lines(file_path):
    """
    Report the number of patients with more than one row in the dataset.

    Demographics data is expected to contain exactly one row per patient.
    This check is run after the cleaning steps to confirm that the assumption
    holds before proceeding to binary encoding.

    Parameters
    ----------
    file_path : str
        Path to PROACT_DEMOGRAPHICS_v4.csv.
    """
    df = pd.read_csv(file_path, low_memory=False)
    duplicated_subjects = df['subject_id'][df['subject_id'].duplicated(keep=False)]
    num_errors = len(duplicated_subjects.unique())
    print(f"Patients with more than one row: {num_errors}")





# ------------------------------------------------------------------
# Stage v5 - Encode Race as binary indicator columns
# ------------------------------------------------------------------

def create_binary_columns(df, column, prefix):
    """
    Create one boolean indicator column per unique value found in a
    semicolon-separated multi-valued column.

    Values are normalised (stripped and title-cased) before deduplication to
    avoid spurious duplicates from inconsistent capitalisation. The resulting
    column names follow the pattern: {prefix}_{value_with_underscores}.

    NOTE: Race_Other_Specify was assessed but found too sparse to be
    informative; it is dropped rather than encoded (see below).

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
# Stage v6 - Add 'DEM_' prefix to all feature columns
# ------------------------------------------------------------------

def rename_all_columns(file_path):
    """
    Prefix every feature column with 'DEM_' to namespace the demographics
    variables when merging with other PROACT sub-datasets.

    `subject_id` is the join key and is left unchanged.

    Parameters
    ----------
    file_path : str
        Path to PROACT_DEMOGRAPHICS_v5.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns
        (-> saved as PROACT_DEMOGRAPHICS_v6.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)
    df = df.rename(columns={col: f'DEM_{col}' for col in df.columns if col != 'subject_id'})

    return df










# ==================================================================
# ------------------------- PIPELINE EXECUTION ---------------------
# ==================================================================

def run(DATA_PATH, PROACT_PATH):

    print("\n" * 3)
    print("=" * 60)
    print("DEMOGRAPHICS PIPELINE")
    print("=" * 60)

    

    # Stage v2 - Consolidate binary race columns into a single Race column
    df_race = create_race_column(PROACT_PATH + '/PROACT_DEMOGRAPHICS.csv')
    df_race.to_csv(DATA_PATH + '/PROACT_DEMOGRAPHICS_v2.csv', index=False)



    # Stage v3 - Impute missing Age from Date_of_Birth
    df_age = age_add(DATA_PATH + '/PROACT_DEMOGRAPHICS_v2.csv')
    df_age.to_csv(DATA_PATH + '/PROACT_DEMOGRAPHICS_v3.csv', index=False)



    # Stage v4 - Drop administrative columns and reorder
    df = remove_unused_columns_demographics(DATA_PATH + '/PROACT_DEMOGRAPHICS_v3.csv')
    df.to_csv(DATA_PATH + '/PROACT_DEMOGRAPHICS_v4.csv', index=False)



    # Diagnostic - Check for unexpected duplicate patients
    check_unique_lines(DATA_PATH + '/PROACT_DEMOGRAPHICS_v4.csv')



    # Stage v5 - Encode Race as binary indicator columns
    df = pd.read_csv(DATA_PATH + '/PROACT_DEMOGRAPHICS_v4.csv', low_memory=False)

    # Encode Race as binary indicators
    binary_race_df, all_races = create_binary_columns(df, 'Race', 'Race')
    df = pd.concat([df, binary_race_df], axis=1)
    df = df.drop(columns=["Race"])

    # Race_Other_Specify: values are too sparse to be informative - dropped
    # binary_race_other_df, all_race_others = create_binary_columns(df, 'Race_Other_Specify', 'Race_Other_Specify')
    # df = pd.concat([df, binary_race_other_df], axis=1)
    df = df.drop(columns=["Race_Other_Specify"])

    df.to_csv(DATA_PATH + '/PROACT_DEMOGRAPHICS_v5.csv', index=False)



    # Stage v6 - Add 'DEM_' prefix to all feature columns
    df_renamed = rename_all_columns(DATA_PATH + '/PROACT_DEMOGRAPHICS_v5.csv')
    df_renamed.to_csv(DATA_PATH + '/PROACT_DEMOGRAPHICS_v6.csv', index=False)
