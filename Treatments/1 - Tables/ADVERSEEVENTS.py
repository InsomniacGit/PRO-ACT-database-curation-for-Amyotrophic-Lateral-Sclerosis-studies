"""
PROACT Adverse Events Processing Pipeline
==========================================
This script processes the PROACT (PRO-ACT ALS) adverse events dataset through
a sequential cleaning and feature engineering pipeline. It produces seven
intermediate CSV files, culminating in a wide-format boolean feature matrix
suitable for machine learning or statistical analysis.

Pipeline stages:
    v2  - Remove irrelevant columns and duplicate rows
    v3  - Filter out rare adverse events below a prevalence threshold
    v4  - Add per-patient observation count
    v5  - Add per-patient unique adverse event counts per classification level
    v6  - Encode adverse events as binary (True/False) indicator columns
    v7  - Prefix all feature columns with 'ADV_' for downstream merging

Author: Bouclier Lucas
Data:   PROACT dataset (2022-07-29 release)
"""



import pandas as pd
import numpy as np
import re
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





# //////////////////////////////////////////////////////////////////
# ------------------------- ADVERSE EVENTS -------------------------
# //////////////////////////////////////////////////////////////////



# ------------------------------------------------------------------
# Prevalence reference - Adverse event patient counts (MedDRA levels)
# ------------------------------------------------------------------

def list_adverse_events(csv_file):
    """
    Compute per-term patient prevalence across all MedDRA classification levels.

    For each of the seven MedDRA fields (Lowest Level Term, Preferred Term,
    High Level Term, High Level Group Term, System Organ Class, SOC Abbreviation,
    SOC Code), this function returns a summary DataFrame containing:
        - the term name
        - the number of distinct patients who reported it
        - the percentage of the total cohort who reported it

    Text columns are normalised (stripped, quotes removed, title-cased) before
    aggregation to avoid spurious duplicates caused by inconsistent formatting.

    Parameters
    ----------
    csv_file : str
        Path to the raw PROACT_ADVERSEEVENTS.csv file.

    Returns
    -------
    tuple of 7 pd.DataFrame
        One summary DataFrame per MedDRA level, sorted by patient count descending.
    """
    df = pd.read_csv(csv_file, low_memory=False)

    # Normalise all free-text classification columns uniformly before any
    # grouping, so that case/quote differences do not create duplicate terms.
    text_cols = [
        'Lowest_Level_Term', 'Preferred_Term', 'High_Level_Term',
        'High_Level_Group_Term', 'System_Organ_Class', 'SOC_Abbreviation'
    ]
    for col in text_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace('"', '', regex=False)
            .str.replace("'", '', regex=False)
            .str.title()
        )

    # Convert SOC_Code from float to int where applicable (avoids "10010.0" display)
    df['SOC_Code'] = df['SOC_Code'].apply(
        lambda x: int(x) if isinstance(x, float) and x.is_integer() else x
    )

    # Total number of unique patients in the dataset (denominator for percentages)
    total_subjects = df['subject_id'].nunique()

    # Build sorted unique-value lists for each classification column
    list_lowest_level_term      = sorted(df['Lowest_Level_Term'].dropna().unique().tolist())
    list_preferred_term         = sorted(df['Preferred_Term'].dropna().unique().tolist())
    list_high_level_term        = sorted(df['High_Level_Term'].dropna().unique().tolist())
    list_high_level_group_term  = sorted(df['High_Level_Group_Term'].dropna().unique().tolist())
    list_system_organ_class     = sorted(df['System_Organ_Class'].dropna().unique().tolist())
    list_soc_abbreviation       = sorted(df['SOC_Abbreviation'].dropna().unique().tolist())
    list_soc_code               = sorted(df['SOC_Code'].dropna().unique().tolist())

    print(f'lowest_level_term : {len(list_lowest_level_term)}')
    print(f'preferred_term : {len(list_preferred_term)}')
    print(f'high_level_term : {len(list_high_level_term)}')
    print(f'high_level_group_term : {len(list_high_level_group_term)}')
    print(f'system_organ_class : {len(list_system_organ_class)}')
    print(f'soc_abbreviation : {len(list_soc_abbreviation)}')
    print(f'soc_code : {len(list_soc_code)}')

    def summarize(df, column, list_values):
        """
        For a given classification column, count distinct patients per term
        and compute their percentage relative to the full cohort.
        """
        temp = df[df[column].isin(list_values)]
        temp = temp.groupby(column)['subject_id'].nunique().reset_index(name='Patients')
        temp['Percentage'] = (temp['Patients'] / total_subjects * 100).round(2)
        temp = temp.sort_values(by='Patients', ascending=False).reset_index(drop=True)
        return temp

    return (
        summarize(df, 'Lowest_Level_Term',     list_lowest_level_term),
        summarize(df, 'Preferred_Term',         list_preferred_term),
        summarize(df, 'High_Level_Term',        list_high_level_term),
        summarize(df, 'High_Level_Group_Term',  list_high_level_group_term),
        summarize(df, 'System_Organ_Class',     list_system_organ_class),
        summarize(df, 'SOC_Abbreviation',       list_soc_abbreviation),
        summarize(df, 'SOC_Code',               list_soc_code),
    )


# Generate prevalence summary tables and save them as reference files
(
    df_lowest_level_term,
    df_preferred_term,
    df_high_level_term,
    df_high_level_group_term,
    df_system_organ_class,
    df_soc_abbreviation,
    df_soc_code,
) = list_adverse_events(proact_path + '/PROACT_ADVERSEEVENTS.csv')

df_lowest_level_term.to_csv(    data_path + '/PROACT_ADVERSEEVENTS_list_lowest_level_term.csv',     index=False)
df_preferred_term.to_csv(       data_path + '/PROACT_ADVERSEEVENTS_list_preferred_term.csv',        index=False)
df_high_level_term.to_csv(      data_path + '/PROACT_ADVERSEEVENTS_list_high_level_term.csv',       index=False)
df_high_level_group_term.to_csv(data_path + '/PROACT_ADVERSEEVENTS_list_high_level_group_term.csv', index=False)
df_system_organ_class.to_csv(   data_path + '/PROACT_ADVERSEEVENTS_list_system_organ_class.csv',    index=False)
df_soc_abbreviation.to_csv(     data_path + '/PROACT_ADVERSEEVENTS_list_soc_abbreviation.csv',      index=False)
df_soc_code.to_csv(             data_path + '/PROACT_ADVERSEEVENTS_list_soc_code.csv',              index=False)





# ------------------------------------------------------------------
# Stage v2 - Clean raw adverse events
# ------------------------------------------------------------------

def clean_adverse_events(csv_file):
    """
    Remove administratively redundant columns and duplicate rows.

    Dropped columns:
        SOC_Abbreviation, SOC_Code  - redundant with System_Organ_Class
        Severity, Outcome           - too sparse / not used in this study
        Start_Date_Delta,
        End_Date_Delta              - temporal information not retained here

    Text normalisation is applied (same logic as list_adverse_events) to
    ensure consistency with the reference prevalence files produced above.

    Parameters
    ----------
    csv_file : str
        Path to the raw PROACT_ADVERSEEVENTS.csv file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame (→ saved as PROACT_ADVERSEEVENTS_v2.csv).
    """
    df = pd.read_csv(csv_file, low_memory=False)

    # Re-apply text normalisation for consistency with the reference lists
    text_cols = [
        'Lowest_Level_Term', 'Preferred_Term', 'High_Level_Term',
        'High_Level_Group_Term', 'System_Organ_Class', 'SOC_Abbreviation'
    ]
    for col in text_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace('"', '', regex=False)
            .str.replace("'", '', regex=False)
            .str.title()
        )

    cols_to_drop = [
        'SOC_Abbreviation', 'SOC_Code', 'Severity', 'Outcome',
        'Start_Date_Delta', 'End_Date_Delta',
    ]
    df = df.drop(columns=cols_to_drop, errors='ignore')

    print(f'Rows before cleaning: {df.shape[0]}')
    df = df.drop_duplicates()
    print(f'Rows after cleaning: {df.shape[0]}')

    return df


df_cleaned_adverse = clean_adverse_events(proact_path + '/PROACT_ADVERSEEVENTS.csv')
df_cleaned_adverse.to_csv(data_path + '/PROACT_ADVERSEEVENTS_v2.csv', index=False)





# ------------------------------------------------------------------
# Stage v3 - Filter rare adverse events by prevalence threshold
# ------------------------------------------------------------------

def filter_adverse_events_by_completeness(csv_file, threshold):
    """
    Set to NaN any term that is reported by fewer than `threshold` percent
    of the total cohort, then drop rows where all five classification columns
    are NaN (i.e. no usable information remains for that row).

    The prevalence reference files produced by list_adverse_events() are used
    to determine which terms to keep. This function operates independently on
    each of the five MedDRA levels: a term may be retained at a coarser level
    (e.g. System_Organ_Class) even if it is removed at a finer level
    (e.g. Lowest_Level_Term).

    Parameters
    ----------
    csv_file  : str   Path to PROACT_ADVERSEEVENTS_v2.csv.
    threshold : float Minimum patient prevalence (%) required to keep a term.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame (→ saved as PROACT_ADVERSEEVENTS_v3.csv).
    """
    df = pd.read_csv(csv_file, low_memory=False)
    print(f'Rows before prevalence filtering: {df.shape[0]}')

    # Load prevalence reference tables generated at the previous step
    df_list_hlt  = pd.read_csv(data_path + '/PROACT_ADVERSEEVENTS_list_high_level_term.csv',       low_memory=False)
    df_list_pt   = pd.read_csv(data_path + '/PROACT_ADVERSEEVENTS_list_preferred_term.csv',        low_memory=False)
    df_list_llt  = pd.read_csv(data_path + '/PROACT_ADVERSEEVENTS_list_lowest_level_term.csv',     low_memory=False)
    df_list_hlgt = pd.read_csv(data_path + '/PROACT_ADVERSEEVENTS_list_high_level_group_term.csv', low_memory=False)
    df_list_soc  = pd.read_csv(data_path + '/PROACT_ADVERSEEVENTS_list_system_organ_class.csv',    low_memory=False)

    # Build allow-lists of terms that exceed the prevalence threshold
    hlt_to_keep  = df_list_hlt [df_list_hlt ['Percentage'] >= threshold]['High_Level_Term'].tolist()
    pt_to_keep   = df_list_pt  [df_list_pt  ['Percentage'] >= threshold]['Preferred_Term'].tolist()
    llt_to_keep  = df_list_llt [df_list_llt ['Percentage'] >= threshold]['Lowest_Level_Term'].tolist()
    hlgt_to_keep = df_list_hlgt[df_list_hlgt['Percentage'] >= threshold]['High_Level_Group_Term'].tolist()
    soc_to_keep  = df_list_soc [df_list_soc ['Percentage'] >= threshold]['System_Organ_Class'].tolist()

    # Replace rare terms with NaN (rather than dropping the whole row)
    df['High_Level_Term']      = df['High_Level_Term'].apply(     lambda x: x if x in hlt_to_keep  else np.nan)
    df['Preferred_Term']       = df['Preferred_Term'].apply(      lambda x: x if x in pt_to_keep   else np.nan)
    df['Lowest_Level_Term']    = df['Lowest_Level_Term'].apply(   lambda x: x if x in llt_to_keep  else np.nan)
    df['High_Level_Group_Term']= df['High_Level_Group_Term'].apply(lambda x: x if x in hlgt_to_keep else np.nan)
    df['System_Organ_Class']   = df['System_Organ_Class'].apply(  lambda x: x if x in soc_to_keep  else np.nan)

    # Drop rows that carry no usable classification information at any level
    df = df.dropna(
        subset=['High_Level_Term', 'Preferred_Term', 'Lowest_Level_Term',
                'High_Level_Group_Term', 'System_Organ_Class'],
        how='all'
    )

    df = df.drop_duplicates()
    print(f'Rows after prevalence filtering: {df.shape[0]}')

    return df


df_filtered_adverse = filter_adverse_events_by_completeness(
    data_path + '/PROACT_ADVERSEEVENTS_v2.csv', threshold=5
)
df_filtered_adverse.to_csv(data_path + '/PROACT_ADVERSEEVENTS_v3.csv', index=False)





# ------------------------------------------------------------------
# Stage v4 - Add per-patient observation count
# ------------------------------------------------------------------

def observation_count_adverse(csv_file):
    """
    Add an `observation_count` column recording how many rows exist per patient.

    Each row in the dataset represents one recorded adverse event for a patient.
    This count therefore reflects the total number of adverse event records
    retained after filtering.

    Parameters
    ----------
    csv_file : str  Path to PROACT_ADVERSEEVENTS_v3.csv.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with `observation_count` inserted as the second column
        (immediately after `subject_id`).
    """
    df = pd.read_csv(csv_file, low_memory=False)

    # Map each patient to the number of rows they have in the DataFrame
    df['observation_count'] = df['subject_id'].map(df['subject_id'].value_counts())

    # Move the new column to position 1 (right after subject_id)
    cols = df.columns.tolist()
    cols.insert(1, cols.pop(cols.index('observation_count')))
    df = df[cols]

    return df


df_adversevenent = observation_count_adverse(data_path + '/PROACT_ADVERSEEVENTS_v3.csv')
df_adversevenent.to_csv(data_path + '/PROACT_ADVERSEEVENTS_v4.csv', index=False)





# ------------------------------------------------------------------
# Stage v5 - Add per-patient unique adverse event counts
# ------------------------------------------------------------------

def adversevent_count(csv_file):
    """
    For each patient, count the number of distinct terms they exhibit at each
    MedDRA classification level and store these counts as new columns.

    Columns added:
        Lowest_Level_Term_Count, Preferred_Term_Count, High_Level_Term_Count,
        High_Level_Group_Term_Count, System_Organ_Class_Count

    These counts complement the binary indicators produced in stage v6 by
    providing a continuous summary of adverse event breadth per patient.

    Parameters
    ----------
    csv_file : str  Path to PROACT_ADVERSEEVENTS_v4.csv.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with five count columns inserted after `observation_count`.
    """
    df = pd.read_csv(csv_file, low_memory=False)

    # Count distinct non-null terms per patient for each MedDRA level
    lowest_level_term_count      = df.groupby('subject_id')['Lowest_Level_Term'].nunique().reset_index(     name='Lowest_Level_Term_Count')
    preferred_term_count         = df.groupby('subject_id')['Preferred_Term'].nunique().reset_index(        name='Preferred_Term_Count')
    high_level_term_count        = df.groupby('subject_id')['High_Level_Term'].nunique().reset_index(       name='High_Level_Term_Count')
    high_level_group_term_count  = df.groupby('subject_id')['High_Level_Group_Term'].nunique().reset_index( name='High_Level_Group_Term_Count')
    system_organ_class_count     = df.groupby('subject_id')['System_Organ_Class'].nunique().reset_index(    name='System_Organ_Class_Count')

    # Left-join count columns onto the main DataFrame (preserves row-level granularity)
    df = df.merge(lowest_level_term_count,     on='subject_id', how='left')
    df = df.merge(preferred_term_count,        on='subject_id', how='left')
    df = df.merge(high_level_term_count,       on='subject_id', how='left')
    df = df.merge(high_level_group_term_count, on='subject_id', how='left')
    df = df.merge(system_organ_class_count,    on='subject_id', how='left')

    # Reorder: insert all five count columns directly after observation_count
    cols = df.columns.tolist()
    obs_index = cols.index('observation_count') + 1
    cols.insert(obs_index,     cols.pop(cols.index('Lowest_Level_Term_Count')))
    cols.insert(obs_index + 1, cols.pop(cols.index('Preferred_Term_Count')))
    cols.insert(obs_index + 2, cols.pop(cols.index('High_Level_Term_Count')))
    cols.insert(obs_index + 3, cols.pop(cols.index('High_Level_Group_Term_Count')))
    cols.insert(obs_index + 4, cols.pop(cols.index('System_Organ_Class_Count')))
    df = df[cols]

    return df


df_adversevenent_count = adversevent_count(data_path + '/PROACT_ADVERSEEVENTS_v4.csv')
df_adversevenent_count.to_csv(data_path + '/PROACT_ADVERSEEVENTS_v5.csv', index=False)





# ------------------------------------------------------------------
# Stage v6 - Encode adverse events as binary indicator columns
# ------------------------------------------------------------------

def create_boolean_columns(csv_file):
    """
    Pivot the long-format adverse event data into a wide-format boolean matrix.

    For every distinct term present in each MedDRA classification column, a
    new boolean column is created in a patient-level DataFrame, indicating
    whether that patient had at least one record with that term (True) or not
    (False).

    Column naming convention:
        {MedDRA_Level}_{term_with_spaces_and_hyphens_replaced_by_underscores}
        e.g. "Preferred_Term_Muscle_Weakness"

    The output DataFrame has one row per patient and contains:
        - subject_id
        - observation_count + five term-count columns (from stage v5)
        - boolean columns for each term, grouped by MedDRA level in alphabetical
          order within each level

    Parameters
    ----------
    csv_file : str  Path to PROACT_ADVERSEEVENTS_v5.csv.

    Returns
    -------
    pd.DataFrame
        Wide-format patient-level boolean feature matrix
        (→ saved as PROACT_ADVERSEEVENTS_v6.csv).
    """
    df = pd.read_csv(csv_file, low_memory=False)

    # Collect the unique terms for each MedDRA level (after filtering)
    adverse_event_cols = {
        'Lowest_Level_Term':     df['Lowest_Level_Term'].dropna().unique().tolist(),
        'Preferred_Term':        df['Preferred_Term'].dropna().unique().tolist(),
        'High_Level_Term':       df['High_Level_Term'].dropna().unique().tolist(),
        'High_Level_Group_Term': df['High_Level_Group_Term'].dropna().unique().tolist(),
        'System_Organ_Class':    df['System_Organ_Class'].dropna().unique().tolist(),
    }

    def clean_name(value: str) -> str:
        """
        Convert a term string into a valid Python/column identifier by replacing
        spaces and hyphens with underscores and removing parentheses and slashes.
        Consecutive underscores are collapsed to one.
        """
        if pd.isna(value):
            return ""
        cleaned = value.replace(" ", "_").replace("-", "_")
        cleaned = re.sub(r"[()\-/]", "", cleaned)
        cleaned = re.sub(r"_+", "_", cleaned)
        return cleaned.strip("_")

    # Apply clean_name and sort alphabetically within each level
    for key in adverse_event_cols:
        adverse_event_cols[key] = sorted(clean_name(v) for v in adverse_event_cols[key])

    # Start with one row per patient
    df_boolean = df[['subject_id']].drop_duplicates().reset_index(drop=True)

    # Columns to carry over from the long-format DataFrame
    count_cols = [
        'observation_count',
        'Lowest_Level_Term_Count',
        'Preferred_Term_Count',
        'High_Level_Term_Count',
        'High_Level_Group_Term_Count',
        'System_Organ_Class_Count',
    ]

    # Extract count columns once per patient (keep first occurrence)
    df_counts = (
        df[['subject_id'] + [c for c in count_cols if c in df.columns]]
        .drop_duplicates(subset='subject_id', keep='first')
        .reset_index(drop=True)
    )
    df_boolean = df_boolean.merge(df_counts, on='subject_id', how='left')

    # Build boolean columns for each MedDRA level
    bool_dfs = []
    for col, values in adverse_event_cols.items():
        sub_df = pd.DataFrame({'subject_id': df_boolean['subject_id']})
        cleaned_col = df[col].fillna("").map(clean_name)
        for value in values:
            bool_col_name = f"{col}_{value}"
            # Patients who have at least one row with this term
            subject_mask = df['subject_id'][cleaned_col == value].unique()
            sub_df[bool_col_name] = df_boolean['subject_id'].isin(subject_mask)
        bool_dfs.append(sub_df.drop(columns=['subject_id']))

    # Concatenate count columns and all boolean column blocks
    df_boolean = pd.concat([df_boolean] + bool_dfs, axis=1)

    # Sort boolean columns alphabetically within each MedDRA level
    ordered_bool_cols = []
    for category in [
        'Lowest_Level_Term', 'Preferred_Term', 'High_Level_Term',
        'High_Level_Group_Term', 'System_Organ_Class',
    ]:
        ordered_bool_cols.extend(sorted([
            c for c in df_boolean.columns
            if c.startswith(f"{category}_") and not c.endswith("_Count")
        ]))

    final_order = (
        ['subject_id']
        + [c for c in count_cols if c in df_boolean.columns]
        + ordered_bool_cols
    )
    df_boolean = df_boolean[final_order]

    return df_boolean


df_boolean_adverse = create_boolean_columns(data_path + '/PROACT_ADVERSEEVENTS_v5.csv')
df_boolean_adverse.to_csv(data_path + '/PROACT_ADVERSEEVENTS_v6.csv', index=False)





# ------------------------------------------------------------------
# Stage v7 - Add 'ADV_' prefix to all feature columns
# ------------------------------------------------------------------

def rename_all_columns(csv_file):
    """
    Prefix every feature column with 'ADV_' to namespace the adverse events
    variables when merging with other PROACT sub-datasets.

    `subject_id` is the join key and is left unchanged.

    Parameters
    ----------
    csv_file : str  Path to PROACT_ADVERSEEVENTS_v6.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns (→ saved as PROACT_ADVERSEEVENTS_v7.csv).
    """
    df = pd.read_csv(csv_file, low_memory=False)
    df = df.rename(columns={col: f'ADV_{col}' for col in df.columns if col != 'subject_id'})
    return df


df_renamed = rename_all_columns(data_path + '/PROACT_ADVERSEEVENTS_v6.csv')
df_renamed.to_csv(data_path + '/PROACT_ADVERSEEVENTS_v7.csv', index=False)