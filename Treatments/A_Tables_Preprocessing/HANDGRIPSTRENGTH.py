"""
PROACT Hand Grip Strength Processing Pipeline
=============================================
This script processes the PROACT (PRO-ACT ALS) hand grip strength dataset
through a sequential cleaning, restructuring, and feature engineering pipeline.
It produces eight intermediate CSV files, culminating in a wide-format
one-row-per-patient matrix with visits stored as sequentially prefixed columns.

Grip strength is measured separately for the left and right hand across up to
three trials per visit. The pipeline pivots this long-format data into a
structured wide format and derives several summary features per visit:
the best trial per side, the most affected (weakest) side, and whether the
weaker side corresponds to the patient's dominant hand.

Pipeline stages:
    v2  - Drop administrative metadata columns
    v3  - Impute missing Test_trial numbers within each visit group
    v4  - Pivot left/right laterality and trial rows into one row per visit
    v5  - Add per-patient observation count
    v6  - Add best-trial summary columns and laterality features
    v7  - Reshape to wide format (one row per patient, visits as prefixed columns)
    v8  - Prefix all feature columns with 'HAN_' for downstream merging

Author: Bouclier Lucas
Data:   PROACT dataset (2022-07-29 release)
"""



import pandas as pd





# ////////////////////////////////////////////////////////////////////
# ------------------------- HANDGRIPSTRENGTH -------------------------
# ////////////////////////////////////////////////////////////////////



# ------------------------------------------------------------------
# Stage v2 - Drop administrative metadata columns
# ------------------------------------------------------------------

def modify_handgrip_strength(file_path):
    """
    Remove administrative columns that are constant or redundant across the
    entire hand grip strength dataset.

    Dropped columns:
        Test_Name      - always the same test name, carries no information
        Test_Category  - constant category label, carries no information
        Test_Location  - administrative field not used in this study
        Test_Unit      - measurement unit is constant (Pounds); encoded in
                         the renamed column Test_Result_Pounds in stage v4

    Parameters
    ----------
    file_path : str
        Path to the raw PROACT_HANDGRIPSTRENGTH.csv file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame (-> saved as PROACT_HANDGRIPSTRENGTH_v2.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)
    df = df.drop(columns=['Test_Name', 'Test_Category', 'Test_Location', 'Test_Unit'])

    return df





# ------------------------------------------------------------------
# Stage v3 - Impute missing trial numbers
# ------------------------------------------------------------------

def correct_test_trial(file_path):
    """
    Fill missing Test_trial numbers by assigning sequential integers within
    each (subject_id, MS_Delta, Test_Laterality) group.

    Some rows have NaN in Test_trial even though they represent distinct
    measurement attempts. Within each visit/laterality group the rows are
    sorted and assigned trial numbers 1, 2, 3, ... in order. Existing
    non-null trial numbers are preserved.

    Test_trial is cast to integer after filling to avoid downstream issues
    caused by float representations (e.g. 1.0 instead of 1).

    Parameters
    ----------
    file_path : str
        Path to PROACT_HANDGRIPSTRENGTH_v2.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with Test_trial fully populated as integers
        (-> saved as PROACT_HANDGRIPSTRENGTH_v3.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)

    # Sort so that rows within each group appear in a deterministic order
    df = df.sort_values(by=['subject_id', 'MS_Delta', 'Test_Laterality', 'Test_trial'])

    def fill_test_trial(group):
        """Assign sequential trial numbers starting at 1 to NaN entries."""
        trials = pd.Series(range(1, len(group) + 1), index=group.index)
        group['Test_trial'] = group['Test_trial'].fillna(trials)
        return group

    # Explicit column selection suppresses the pandas groupby/apply warning
    df = df.groupby(
        ['subject_id', 'MS_Delta', 'Test_Laterality'],
        group_keys=False
    )[df.columns].apply(fill_test_trial)

    df['Test_trial'] = df['Test_trial'].astype(int)

    return df





# ------------------------------------------------------------------
# Stage v4 - Pivot laterality and trial rows into one row per visit
# ------------------------------------------------------------------

def regroup_left_right_trials_handgrip(file_path):
    """
    Reshape the long-format data (one row per trial per side) into a
    semi-wide format with one row per (subject_id, MS_Delta) visit.

    Within each visit, left and right trial measurements are spread across
    dedicated columns following the naming convention:

        Test_Result_Pounds_{Left|Right}_Trial_{n}
        Test_Setting_{Left|Right}_Trial_{n}

    where n is the trial number (1, 2, 3, ...). This pivot makes it possible
    to compare left vs. right performance and to select the best trial per
    side in the next stage.

    DominantHand is taken from the first row of each visit group, as it is
    constant within a patient.

    A progress message is printed every 500 patients to monitor execution.

    Parameters
    ----------
    file_path : str
        Path to PROACT_HANDGRIPSTRENGTH_v3.csv.

    Returns
    -------
    pd.DataFrame
        Semi-wide DataFrame with one row per visit
        (-> saved as PROACT_HANDGRIPSTRENGTH_v4.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)

    # Rename for clarity: the unit (Pounds) is now encoded in the column name
    df.rename(columns={'Test_Result': 'Test_Result_Pounds'}, inplace=True)
    df['MS_Delta'] = pd.to_numeric(df['MS_Delta'], errors='coerce')

    # Stable sort before grouping
    df = df.sort_values(by=['subject_id', 'MS_Delta', 'Test_Laterality', 'Test_trial'])

    data_rows = []
    patient_counter = 0
    patient_total = df['subject_id'].nunique()

    for subject_id, group in df.groupby('subject_id'):
        patient_counter += 1
        if patient_counter % 500 == 0:
            print(f"Processed {patient_counter} patients out of {patient_total}...")

        # One output row per (subject_id, MS_Delta) combination
        for (delta,), sub_group in group.groupby(['MS_Delta']):
            row_data = {
                'subject_id': subject_id,
                'MS_Delta': delta,
                # DominantHand is stable within a patient; take the first value
                'DominantHand': sub_group['DominantHand'].iloc[0],
            }

            # Spread each trial into a laterality-specific column pair
            for _, row in sub_group.iterrows():
                lat   = str(row['Test_Laterality']).lower()
                trial = int(row['Test_trial'])
                if lat == 'left':
                    row_data[f'Test_Result_Pounds_Left_Trial_{trial}']  = row['Test_Result_Pounds']
                    row_data[f'Test_Setting_Left_Trial_{trial}']         = row['Test_Setting']
                elif lat == 'right':
                    row_data[f'Test_Result_Pounds_Right_Trial_{trial}'] = row['Test_Result_Pounds']
                    row_data[f'Test_Setting_Right_Trial_{trial}']        = row['Test_Setting']

            data_rows.append(row_data)

    return pd.DataFrame(data_rows)





# ------------------------------------------------------------------
# Stage v5 - Add per-patient observation count
# ------------------------------------------------------------------

def observation_counter_handgrip(file_path):
    """
    Add an `observation_count` column recording how many visit rows exist
    per patient after the laterality pivot.

    Each row now represents one spirometry visit (one MS_Delta timepoint).
    This count reflects the number of valid visits retained after pivoting.

    Parameters
    ----------
    file_path : str
        Path to PROACT_HANDGRIPSTRENGTH_v4.csv.

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
# Stage v6 - Add best-trial summary and laterality feature columns
# ------------------------------------------------------------------

def add_max_trials_columns_handgrip(file_path):
    """
    Derive four summary features per visit:

        Test_Result_Pounds_{Left|Right}_Trial_Max
            - the highest grip force recorded across all trials for that side
        Test_Setting_{Left|Right}_Trial_Max
            - the test setting corresponding to the best trial for that side
        Test_Result_Most_Affected_Side
            - the weaker side (Left / Right / Equal / None);
              'None' when no measurement is available for either side
        Test_Result_Most_Affected_Side_isDominant
            - True if the weaker side matches the patient's dominant hand,
              False otherwise, None when either value is missing

    The maximum search iterates over dynamically detected trial columns so
    that the function handles any number of trials without hardcoding.

    Parameters
    ----------
    file_path : str
        Path to PROACT_HANDGRIPSTRENGTH_v5.csv.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with four additional summary columns per side
        (-> saved as PROACT_HANDGRIPSTRENGTH_v6.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)

    def get_max_trial(side):
        """
        Add *_Trial_Max result and setting columns for the given side.
        Columns are detected dynamically so the function scales to any number
        of trials present in the data.
        """
        result_cols  = [col for col in df.columns if f'Test_Result_Pounds_{side}_Trial_' in col]
        setting_cols = [col for col in df.columns if f'Test_Setting_{side}_Trial_' in col]

        def max_result_and_setting(row):
            """Return the maximum result value and its corresponding setting."""
            max_value   = None
            max_setting = None
            for res_col, set_col in zip(result_cols, setting_cols):
                value = row[res_col]
                if pd.notna(value) and (max_value is None or value > max_value):
                    max_value   = value
                    max_setting = row[set_col]
            return pd.Series([max_value, max_setting])

        df[[
            f'Test_Result_Pounds_{side}_Trial_Max',
            f'Test_Setting_{side}_Trial_Max',
        ]] = df.apply(max_result_and_setting, axis=1)

    get_max_trial('Left')
    get_max_trial('Right')

    def determine_best_side(row):
        """
        Identify the weaker side by comparing left and right maxima.
        Returns 'Left' if left grip < right grip, 'Right' if right < left,
        'Equal' if both are equal, or None if no data is available.
        """
        left_max  = row['Test_Result_Pounds_Left_Trial_Max']
        right_max = row['Test_Result_Pounds_Right_Trial_Max']
        if pd.notna(left_max) and pd.notna(right_max):
            if left_max < right_max:
                return 'Left'
            elif left_max > right_max:
                return 'Right'
            else:
                return 'Equal'
        elif pd.notna(left_max):
            return 'Left'
        elif pd.notna(right_max):
            return 'Right'
        return None

    df['Test_Result_Most_Affected_Side'] = df.apply(determine_best_side, axis=1)

    def is_dominant(row):
        """Return True if the weaker side matches the patient's dominant hand."""
        if pd.isna(row['DominantHand']) or pd.isna(row['Test_Result_Most_Affected_Side']):
            return None
        return row['DominantHand'].lower() == row['Test_Result_Most_Affected_Side'].lower()

    df['Test_Result_Most_Affected_Side_isDominant'] = df.apply(is_dominant, axis=1)

    return df





# ------------------------------------------------------------------
# Stage v7 - Reshape to wide format
# ------------------------------------------------------------------

def reshape_to_wide_format(csv_file):
    """
    Reshape the semi-wide visit-level data into a fully wide patient-level
    DataFrame (one row per patient) where each visit's values are stored in
    sequentially prefixed columns.

    Visits are sorted by MS_Delta (time since study baseline) before pivoting,
    so that column prefix 1_ always corresponds to the earliest recorded visit.

    DominantHand and observation_count are patient-level constants and are
    stored once per row rather than repeated per visit.

    A progress message is printed every 500 patients to monitor execution.

    Column naming convention:
        {visit_index}_{original_column_name}
        e.g. "1_MS_Delta", "2_Test_Result_Pounds_Left_Trial_Max"

    Parameters
    ----------
    csv_file : str
        Path to PROACT_HANDGRIPSTRENGTH_v6.csv.

    Returns
    -------
    pd.DataFrame
        Wide-format patient-level DataFrame
        (-> saved as PROACT_HANDGRIPSTRENGTH_v7.csv).
    """
    df = pd.read_csv(csv_file, low_memory=False)

    total_subjects = df['subject_id'].nunique()

    # Columns to carry into the wide format (one set per visit)
    colonnes = [
        'MS_Delta',
        'Test_Result_Pounds_Left_Trial_1',  'Test_Setting_Left_Trial_1',
        'Test_Result_Pounds_Left_Trial_2',  'Test_Setting_Left_Trial_2',
        'Test_Result_Pounds_Right_Trial_1', 'Test_Setting_Right_Trial_1',
        'Test_Result_Pounds_Right_Trial_2', 'Test_Setting_Right_Trial_2',
        'Test_Result_Pounds_Left_Trial_Max',  'Test_Setting_Left_Trial_Max',
        'Test_Result_Pounds_Right_Trial_Max', 'Test_Setting_Right_Trial_Max',
        'Test_Result_Most_Affected_Side',
        'Test_Result_Most_Affected_Side_isDominant',
    ]

    # Sort visits chronologically within each patient
    df = df.sort_values(by=['subject_id', 'MS_Delta'])

    rows = []
    processed_subjects = set()
    counter = 0

    for subject_id, group in df.groupby('subject_id'):
        if subject_id not in processed_subjects:
            processed_subjects.add(subject_id)
            counter += 1
            if counter % 500 == 0:
                print(f"Processed {counter} patients out of {total_subjects}...")

        group = group.reset_index(drop=True)

        # Patient-level constants stored once per row
        row_data = {
            'subject_id':        subject_id,
            'observation_count': group['observation_count'].iloc[0],
            'DominantHand':      group['DominantHand'].iloc[0],
        }

        for i, row in group.iterrows():
            for col in colonnes:
                row_data[f'{i + 1}_{col}'] = row[col]

        rows.append(row_data)

    return pd.DataFrame(rows)





# ------------------------------------------------------------------
# Stage v8 - Add 'HAN_' prefix to all feature columns
# ------------------------------------------------------------------

def rename_all_columns(file_path):
    """
    Prefix every feature column with 'HAN_' to namespace the hand grip
    strength variables when merging with other PROACT sub-datasets.

    `subject_id` is the join key and is left unchanged.

    Parameters
    ----------
    file_path : str
        Path to PROACT_HANDGRIPSTRENGTH_v7.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns
        (-> saved as PROACT_HANDGRIPSTRENGTH_v8.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)
    df = df.rename(columns={col: f'HAN_{col}' for col in df.columns if col != 'subject_id'})

    return df










# ==================================================================
# ------------------------- PIPELINE EXECUTION ---------------------
# ==================================================================

def run(DATA_PATH, PROACT_PATH):

    print("\n" * 3)
    print("=" * 60)
    print("HANDGRIPSTRENGTH PIPELINE")
    print("=" * 60)

    

    # Stage v2 - Drop administrative metadata columns
    df = modify_handgrip_strength(PROACT_PATH + '/PROACT_HANDGRIPSTRENGTH.csv')
    df.to_csv(DATA_PATH + '/PROACT_HANDGRIPSTRENGTH_v2.csv', index=False)



    # Stage v3 - Impute missing trial numbers
    df_handgrip = correct_test_trial(DATA_PATH + '/PROACT_HANDGRIPSTRENGTH_v2.csv')
    df_handgrip.to_csv(DATA_PATH + '/PROACT_HANDGRIPSTRENGTH_v3.csv', index=False)



    # Stage v4 - Pivot laterality and trial rows into one row per visit
    df_handgrip = regroup_left_right_trials_handgrip(DATA_PATH + '/PROACT_HANDGRIPSTRENGTH_v3.csv')
    df_handgrip.to_csv(DATA_PATH + '/PROACT_HANDGRIPSTRENGTH_v4.csv', index=False)



    # Stage v5 - Add per-patient observation count
    df_handgrip = observation_counter_handgrip(DATA_PATH + '/PROACT_HANDGRIPSTRENGTH_v4.csv')
    df_handgrip.to_csv(DATA_PATH + '/PROACT_HANDGRIPSTRENGTH_v5.csv', index=False)



    # Stage v6 - Add best-trial summary and laterality feature columns
    df_handgrip = add_max_trials_columns_handgrip(DATA_PATH + '/PROACT_HANDGRIPSTRENGTH_v5.csv')
    df_handgrip.to_csv(DATA_PATH + '/PROACT_HANDGRIPSTRENGTH_v6.csv', index=False)



    # Stage v7 - Reshape to wide format
    df_handgrip = reshape_to_wide_format(DATA_PATH + '/PROACT_HANDGRIPSTRENGTH_v6.csv')
    df_handgrip.to_csv(DATA_PATH + '/PROACT_HANDGRIPSTRENGTH_v7.csv', index=False)



    # Stage v8 - Add 'HAN_' prefix to all feature columns
    df_renamed = rename_all_columns(DATA_PATH + '/PROACT_HANDGRIPSTRENGTH_v7.csv')
    df_renamed.to_csv(DATA_PATH + '/PROACT_HANDGRIPSTRENGTH_v8.csv', index=False)