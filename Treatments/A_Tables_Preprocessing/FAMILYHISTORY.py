"""
PROACT Family History Processing Pipeline
==========================================
This script processes the PROACT (PRO-ACT ALS) family history dataset through
a sequential cleaning and feature engineering pipeline. It produces nine
intermediate CSV files, culminating in a one-row-per-patient boolean feature
matrix suitable for machine learning or statistical analysis.

Pipeline stages:
    v2  - Reorder columns into a logical family-relation hierarchy
    v3  - Drop fully empty columns and the uninformative timing column
    v4  - Merge maternal/paternal sub-columns into a single column per relation
    v5  - Resolve duplicate patient records by taking the maximum value per
          binary column and concatenating free-text fields with ';'
    v6  - Count affected family members; propagate ALS diagnosis label;
          drop Family_Hx_of_Neuro_Disease
    v7  - Convert family-member columns from 1.0/NaN to True/False
    v8  - Encode Neurological_Disease as binary indicator columns;
          drop sparse free-text field (Neurological_Disease_Other_Specify)
    v9  - Prefix all feature columns with 'FAM_' for downstream merging

Author: Bouclier Lucas
Data:   PROACT dataset (2022-07-29 release)
"""



import pandas as pd
import numpy as np





# /////////////////////////////////////////////////////////////////
# ------------------------- FAMILYHISTORY -------------------------
# /////////////////////////////////////////////////////////////////



# ------------------------------------------------------------------
# Stage v2 - Reorder columns into a logical family-relation hierarchy
# ------------------------------------------------------------------

def reorder_family_history(file_path):
    """
    Reorder the raw columns into a logical hierarchy grouping family members
    by kinship proximity, followed by medical history fields.

    The raw CSV presents columns in an arbitrary order. Reordering them by
    family-relation group (immediate family, grandparents, uncles/aunts,
    cousins, nephews/nieces, then medical flags) makes the dataset easier to
    inspect and audit. Columns absent from the raw file are silently skipped.

    Parameters
    ----------
    file_path : str
        Path to the raw PROACT_FAMILYHISTORY.csv file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns in hierarchical family order
        (-> saved as PROACT_FAMILYHISTORY_v2.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)

    ordered_columns = [
        # Patient identifier and timing
        'subject_id', 'Family_History_Delta',
        # Immediate family
        'Father', 'Mother', 'Brother', 'Sister', 'Son', 'Daughter',
        # Grandparents
        'Grandfather', 'Grandfather__Maternal_', 'Grandfather__Paternal_',
        'Grandmother', 'Grandmother__Maternal_', 'Grandmother__Paternal_',
        # Uncles and aunts
        'Uncle', 'Uncle__Maternal_', 'Uncle__Paternal_',
        'Aunt', 'Aunt__Maternal_', 'Aunt__Paternal_',
        # Cousins
        'Cousin', 'Cousin__Maternal_', 'Cousin__Paternal_',
        # Nephews and nieces
        'Nephew', 'Nephew__Maternal_', 'Nephew__Paternal_',
        'Niece', 'Niece__Maternal_', 'Niece__Paternal_',
        # Other relation types
        'Sibling', 'Other', 'Volunteer',
        # Medical history fields
        'Family_Hx_of_ALS_Mutation', 'Family_Hx_of_ALS_Mutation_Other',
        'Family_Hx_of_Neuro_Disease', 'Family_Hx_of_Neuro_Disease_Other',
        'Neurological_Disease', 'Neurological_Disease_Other', 'Other_Specify',
    ]

    # Keep only columns that actually exist in this release of the dataset
    ordered_columns_existing = [col for col in ordered_columns if col in df.columns]
    df = df[ordered_columns_existing]

    return df





# ------------------------------------------------------------------
# Stage v3 - Drop empty and uninformative columns
# ------------------------------------------------------------------

def clean_family_history(file_path):
    """
    Remove fully empty columns and the administrative timing column.

    Family_History_Delta is an administrative offset that carries no clinical
    information for this study. Columns that are entirely NaN across all rows
    are also removed as they cannot contribute to any analysis.

    Parameters
    ----------
    file_path : str
        Path to PROACT_FAMILYHISTORY_v2.csv.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame (-> saved as PROACT_FAMILYHISTORY_v3.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)

    # Drop columns that are entirely empty
    df = df.dropna(axis=1, how='all')

    # Drop administrative timing column
    df.drop(columns=['Family_History_Delta'], errors='ignore', inplace=True)

    return df





# ------------------------------------------------------------------
# Stage v4 - Merge maternal/paternal sub-columns into unified columns
# ------------------------------------------------------------------

def merge_family_columns(file_path):
    """
    Consolidate maternal and paternal sub-columns into a single column per
    family relation using the row-wise maximum.

    For four relations (Grandfather, Grandmother, Uncle, Aunt), the raw data
    provides up to three columns: a general indicator and two side-specific
    variants (e.g. Grandfather__Maternal_, Grandfather__Paternal_). Since this
    study does not distinguish maternal from paternal lineage, all three are
    collapsed into one column by taking the maximum value (1.0 > NaN). The
    side-specific columns are then dropped.

    Parameters
    ----------
    file_path : str
        Path to PROACT_FAMILYHISTORY_v3.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with one column per family relation
        (-> saved as PROACT_FAMILYHISTORY_v4.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)

    # Map each unified column name to the sub-columns it absorbs
    family_relations = {
        'Grandfather': ['Grandfather', 'Grandfather__Maternal_', 'Grandfather__Paternal_'],
        'Grandmother': ['Grandmother', 'Grandmother__Maternal_', 'Grandmother__Paternal_'],
        'Uncle':       ['Uncle',       'Uncle__Maternal_',       'Uncle__Paternal_'],
        'Aunt':        ['Aunt',        'Aunt__Maternal_',        'Aunt__Paternal_'],
    }

    for new_col, cols_to_merge in family_relations.items():
        # max() treats 1.0 as higher than NaN, so any positive report is preserved
        df[new_col] = df[cols_to_merge].max(axis=1)
        # Drop the side-specific columns (keep index 0, the general column, now overwritten)
        df.drop(columns=cols_to_merge[1:], inplace=True)

    return df





# ------------------------------------------------------------------
# Diagnostic - Identify patients with multiple records
# ------------------------------------------------------------------

def list_subjects_with_multiple_lines(file_path):
    """
    Return the list of patient IDs that appear more than once in the dataset.

    Multiple records per patient indicate conflicting family history entries
    that must be resolved before the data can be used in a one-row-per-patient
    matrix. The list is inspected manually to inform the merging strategy
    applied in stage v5.

    Parameters
    ----------
    file_path : str
        Path to PROACT_FAMILYHISTORY_v4.csv.

    Returns
    -------
    list
        Sorted list of subject_id values with more than one row.
    """
    df = pd.read_csv(file_path, low_memory=False)
    duplicated_subjects = df['subject_id'][df['subject_id'].duplicated(keep=False)]

    return duplicated_subjects.unique().tolist()



# Patients identified above as having multiple records; exported for audit
# before being resolved by combine_family_history() in stage v5.
def filter_subject_ids(
        file_path, 
        subject_ids = [
            24755, 27702, 33770, 37171, 43427, 60885, 74899, 93510, 98303, 155551,
            194137, 211529, 218348, 224639, 228071, 239612, 269593, 376662, 377636,
            380514, 382892, 398097, 400991, 406412, 413145, 416756, 421158, 423030,
            423619, 424119, 427558, 430805, 434916, 455263, 484508, 485993, 499055,
            509862, 525247, 551989, 556823, 569878, 577625, 605404, 757989, 773939,
            774869, 776467, 827520, 829054, 833579, 844156, 853005, 863694, 874150,
            885248, 896339, 933971, 941016, 955372,
        ]
    ):
    """
    Extract the rows corresponding to a specific list of patient IDs.

    Used here to isolate the duplicate patients for visual inspection before
    the automated merging step.

    Parameters
    ----------
    file_path   : str   Path to PROACT_FAMILYHISTORY_v4.csv.
    subject_ids : list  List of subject_id values to extract.

    Returns
    -------
    pd.DataFrame
        Subset of rows whose subject_id is in subject_ids
        (-> saved as PROACT_FAMILYHISTORY_multipleID.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)
    return df[df['subject_id'].isin(subject_ids)]





# ------------------------------------------------------------------
# Stage v5 - Resolve duplicate records into one row per patient
# ------------------------------------------------------------------

def combine_family_history(file_path):
    """
    Collapse multiple rows per patient into a single row.

    Merging strategy:
        - Binary family-member columns (0/1/NaN): take the row-wise maximum
          so that a positive report in any row is preserved (1.0 > NaN).
        - Free-text disease columns (Family_Hx_of_Neuro_Disease,
          Neurological_Disease, Neurological_Disease_Other): concatenate all
          distinct non-null values with '; ' so no information is lost.
        - All other columns: take the row-wise maximum.

    Parameters
    ----------
    file_path : str
        Path to PROACT_FAMILYHISTORY_v4.csv.

    Returns
    -------
    pd.DataFrame
        One-row-per-patient DataFrame
        (-> saved as PROACT_FAMILYHISTORY_v5.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)
    result_list = []

    for subject_id, group in df.groupby('subject_id'):
        row_data = {'subject_id': subject_id}

        for col in group.columns:
            if col != 'subject_id':
                if col in ['Family_Hx_of_Neuro_Disease', 'Neurological_Disease', 'Neurological_Disease_Other']:
                    # Preserve all distinct disease labels by joining them
                    unique_values = group[col].dropna().unique()
                    unique_values.sort()
                    row_data[col] = '; '.join(map(str, unique_values)) if len(unique_values) > 0 else np.nan
                else:
                    # For binary columns, any positive report takes precedence
                    row_data[col] = group[col].max()

        result_list.append(row_data)

    return pd.DataFrame(result_list)





# ------------------------------------------------------------------
# Stage v6 - Count affected family members and propagate ALS label
# ------------------------------------------------------------------

def count_neurological_conditions(file_path):
    """
    Derive three count features and propagate the ALS diagnosis label.

    Count columns added:
        Nb_Family_Member          - total number of family members with a
                                    reported neurological condition (value 1.0),
                                    plus 1 if Family_Hx_of_Neuro_Disease is 'ALS'
        Nb_Close_Family_Member    - same count restricted to immediate family
                                    (parents, siblings, children, grandparents)
        Nb_Extended_Family_Member - same count restricted to extended family
                                    (uncles, aunts, cousins, nephews, nieces)

    When Family_Hx_of_Neuro_Disease is 'ALS' and Neurological_Disease is
    missing, Neurological_Disease is filled with 'ALS' to ensure the label
    is available for binary encoding in stage v8.
    Family_Hx_of_Neuro_Disease is then dropped as it is now redundant.

    Parameters
    ----------
    file_path : str
        Path to PROACT_FAMILYHISTORY_v5.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with three count columns prepended and ALS label propagated
        (-> saved as PROACT_FAMILYHISTORY_v6.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)

    family_columns = [
        'Father', 'Mother', 'Brother', 'Sister', 'Son', 'Daughter',
        'Grandfather', 'Grandmother', 'Uncle', 'Aunt', 'Cousin', 'Nephew', 'Niece',
    ]
    close_family_columns = [
        'Father', 'Mother', 'Brother', 'Sister', 'Son', 'Daughter',
        'Grandfather', 'Grandmother',
    ]
    extended_family_columns = [
        'Uncle', 'Aunt', 'Cousin', 'Nephew', 'Niece',
    ]

    # Count family members flagged with a neurological condition (value 1.0)
    df['Nb_Family_Member'] = df[family_columns].apply(
        lambda row: sum(1 for val in row if val == 1.0), axis=1
    )
    # Add 1 if an ALS family history is reported outside the per-member columns
    df['Nb_Family_Member'] += df['Family_Hx_of_Neuro_Disease'].apply(
        lambda x: 1 if x == 'ALS' else 0
    )

    df['Nb_Close_Family_Member'] = df[close_family_columns].apply(
        lambda row: sum(1 for val in row if val == 1.0), axis=1
    )
    df['Nb_Close_Family_Member'] += df['Family_Hx_of_Neuro_Disease'].apply(
        lambda x: 1 if x == 'ALS' else 0
    )

    df['Nb_Extended_Family_Member'] = df[extended_family_columns].apply(
        lambda row: sum(1 for val in row if val == 1.0), axis=1
    )

    # Propagate ALS label to Neurological_Disease when it is not already set
    df['Neurological_Disease'] = df.apply(
        lambda row: 'ALS'
        if row['Family_Hx_of_Neuro_Disease'] == 'ALS' and pd.isna(row['Neurological_Disease'])
        else row['Neurological_Disease'],
        axis=1
    )

    # Family_Hx_of_Neuro_Disease is now encoded in the count columns and the
    # Neurological_Disease label; it can be safely dropped
    df.drop(columns=['Family_Hx_of_Neuro_Disease'], inplace=True)

    # Bring count columns to the front for readability
    cols = (
        ['subject_id', 'Nb_Family_Member', 'Nb_Close_Family_Member', 'Nb_Extended_Family_Member']
        + [col for col in df.columns if col not in ['subject_id', 'Nb_Family_Member', 'Nb_Close_Family_Member', 'Nb_Extended_Family_Member']]
    )
    df = df[cols]

    return df





# ------------------------------------------------------------------
# Stage v7 - Convert family-member columns to boolean
# ------------------------------------------------------------------

def convert_family_columns_to_boolean(file_path):
    """
    Convert the binary family-member indicator columns from numeric (1.0/NaN)
    to boolean (True/False).

    This makes the encoding consistent with the boolean indicator columns
    produced in later stages (v8) and avoids mixed numeric/boolean types in
    the final feature matrix.

    Parameters
    ----------
    file_path : str
        Path to PROACT_FAMILYHISTORY_v6.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with family-member columns cast to bool
        (-> saved as PROACT_FAMILYHISTORY_v7.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)

    family_columns = [
        'Father', 'Mother', 'Brother', 'Sister', 'Son', 'Daughter',
        'Grandfather', 'Grandmother', 'Uncle', 'Aunt', 'Cousin', 'Nephew', 'Niece',
    ]

    for col in family_columns:
        if col in df.columns:
            # 1.0 -> True, NaN or 0 -> False
            df[col] = df[col].apply(lambda x: True if x == 1.0 else False)

    return df





# ------------------------------------------------------------------
# Stage v8 - Encode Neurological_Disease as binary indicator columns
# ------------------------------------------------------------------

def create_binary_columns(df, column, prefix):
    """
    Create one boolean indicator column per unique value found in a
    semicolon-separated multi-valued column.

    Values are normalised (stripped and title-cased) before deduplication to
    avoid spurious duplicates from inconsistent capitalisation. The resulting
    column names follow the pattern: {prefix}_{value_with_underscores}.

    NOTE: Neurological_Disease_Other_Specify was assessed but found too sparse
    to be informative; it is dropped rather than encoded (see below).

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
# Stage v9 - Add 'FAM_' prefix to all feature columns
# ------------------------------------------------------------------

def rename_all_columns(file_path):
    """
    Prefix every feature column with 'FAM_' to namespace the family history
    variables when merging with other PROACT sub-datasets.

    `subject_id` is the join key and is left unchanged.

    Parameters
    ----------
    file_path : str
        Path to PROACT_FAMILYHISTORY_v8.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns
        (-> saved as PROACT_FAMILYHISTORY_v9.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)
    df = df.rename(columns={col: f'FAM_{col}' for col in df.columns if col != 'subject_id'})

    return df










# ==================================================================
# ------------------------- PIPELINE EXECUTION ---------------------
# ==================================================================

def run(DATA_PATH, PROACT_PATH):

    print("\n" * 3)
    print("=" * 60)
    print("FAMILYHISTORY PIPELINE")
    print("=" * 60)

    

    # Stage v2 - Reorder columns into a logical family-relation hierarchy
    df_family = reorder_family_history(PROACT_PATH + '/PROACT_FAMILYHISTORY.csv')
    df_family.to_csv(DATA_PATH + '/PROACT_FAMILYHISTORY_v2.csv', index=False)



    # Stage v3 - Drop empty and uninformative columns
    df_family = clean_family_history(DATA_PATH + '/PROACT_FAMILYHISTORY_v2.csv')
    df_family.to_csv(DATA_PATH + '/PROACT_FAMILYHISTORY_v3.csv', index=False)



    # Stage v4 - Merge maternal/paternal sub-columns into unified columns
    df_family = merge_family_columns(DATA_PATH + '/PROACT_FAMILYHISTORY_v3.csv')
    df_family.to_csv(DATA_PATH + '/PROACT_FAMILYHISTORY_v4.csv', index=False)



    # Diagnostic - Identify patients with multiple records
    print("Subject IDs with multiple lines in FAMILYHISTORY:")
    multiple_line_ids = list_subjects_with_multiple_lines(DATA_PATH + '/PROACT_FAMILYHISTORY_v4.csv')
    print(multiple_line_ids)

    df_family = filter_subject_ids(DATA_PATH + '/PROACT_FAMILYHISTORY_v4.csv')
    df_family.to_csv(DATA_PATH + '/PROACT_FAMILYHISTORY_multipleID.csv', index=False)



    # Stage v5 - Resolve duplicate records into one row per patient
    df_family = combine_family_history(DATA_PATH + '/PROACT_FAMILYHISTORY_v4.csv')
    df_family.to_csv(DATA_PATH + '/PROACT_FAMILYHISTORY_v5.csv', index=False)



    # Stage v6 - Count affected family members and propagate ALS label
    df_family = count_neurological_conditions(DATA_PATH + '/PROACT_FAMILYHISTORY_v5.csv')
    df_family.to_csv(DATA_PATH + '/PROACT_FAMILYHISTORY_v6.csv', index=False)



    # Stage v7 - Convert family-member columns to boolean
    df_family = convert_family_columns_to_boolean(DATA_PATH + '/PROACT_FAMILYHISTORY_v6.csv')
    df_family.to_csv(DATA_PATH + '/PROACT_FAMILYHISTORY_v7.csv', index=False)



    # Stage v8 - Encode Neurological_Disease as binary indicator columns
    df = pd.read_csv(DATA_PATH + '/PROACT_FAMILYHISTORY_v7.csv', low_memory=False)

    # Rename for consistency with the naming convention used across other scripts
    df = df.rename(columns={"Neurological_Disease_Other": "Neurological_Disease_Other_Specify"})

    # Encode Neurological_Disease as binary indicators
    binary_neuro_df, all_neuros = create_binary_columns(df, "Neurological_Disease", "Neurological_Disease")
    df = pd.concat([df, binary_neuro_df], axis=1)
    df = df.drop(columns=["Neurological_Disease"])

    # Neurological_Disease_Other_Specify: values are too sparse to be informative - dropped
    # binary_neuro_other_df, all_neuro_others = create_binary_columns(df, "Neurological_Disease_Other_Specify", "Neurological_Disease_Other_Specify")
    # df = pd.concat([df, binary_neuro_other_df], axis=1)
    df = df.drop(columns=["Neurological_Disease_Other_Specify"])

    df.to_csv(DATA_PATH + '/PROACT_FAMILYHISTORY_v8.csv', index=False)



    # Stage v9 - Add 'FAM_' prefix to all feature columns
    df_renamed = rename_all_columns(DATA_PATH + '/PROACT_FAMILYHISTORY_v8.csv')
    df_renamed.to_csv(DATA_PATH + '/PROACT_FAMILYHISTORY_v9.csv', index=False)