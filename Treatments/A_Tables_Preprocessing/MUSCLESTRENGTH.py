"""
PROACT Muscle Strength Processing Pipeline
==========================================
This script processes the PROACT (PRO-ACT ALS) isometric muscle strength
dataset through a sequential cleaning, restructuring, and feature engineering
pipeline. It produces eight intermediate CSV files, culminating in a
wide-format one-row-per-patient matrix with visits stored as sequentially
prefixed column blocks.

Muscle strength is measured for multiple muscle groups (Test_Name) at multiple
anatomical locations (Test_Location), separately for left and right sides,
across up to three trials per visit. The pipeline pivots this long-format data
into a structured wide format and derives several summary features per
test/location combination: the best trial per side, the most affected
(weakest) side, and whether the weaker side corresponds to the patient's
dominant hand (joined from the HANDGRIPSTRENGTH pipeline output).

Pipeline stages:
    v2  - Drop constant metadata columns; clean Test_Name prefix;
          standardise Test_Location casing; convert Test_Break to boolean
    v3  - Pivot left/right laterality and trial rows into one row per
          (subject_id, MS_Delta, Test_Name, Test_Location)
    v4  - Add best-trial summary columns, laterality features, and
          DominantHand (joined from PROACT_HANDGRIPSTRENGTH_v8.csv)
    v5  - Pivot test/location combinations into one row per
          (subject_id, MS_Delta), with one column block per test/location
    v6  - Add per-patient observation count
    v7  - Reshape to fully wide format (one row per patient, visits as
          sequentially prefixed column blocks with custom suffix ordering)
    v8  - Prefix all feature columns with 'MUS_' for downstream merging

Author: Bouclier Lucas
Data:   PROACT dataset (2022-07-29 release)
"""



import pandas as pd





# //////////////////////////////////////////////////////////////////
# ------------------------- MUSCLESTRENGTH -------------------------
# //////////////////////////////////////////////////////////////////



# ------------------------------------------------------------------
# Stage v2 - Drop constant columns and normalise values
# ------------------------------------------------------------------

def modify_muscle_strength(file_path):
    """
    Remove constant metadata columns, clean the Test_Name prefix, standardise
    Test_Location casing, and convert Test_Break to a boolean column.

    Dropped columns:
        Test_Category - constant across the entire dataset
        Test_Unit     - constant (Pounds); encoded in column names from stage v3

    Test_Name cleaning:
        The raw values carry a redundant 'Isometric Muscle Strength, ' prefix
        (e.g. 'Isometric Muscle Strength, Abduction'). This prefix is stripped
        so that Test_Name contains only the muscle action label.

    Test_Location normalisation:
        Values are title-cased and a known inconsistency ('Wristjoint') is
        corrected to 'Wrist Joint' for consistent column naming downstream.

    Test_Break conversion:
        The raw string values ('yes' / NaN) are converted to booleans
        (True / None) to match the type convention used across the pipeline.

    Parameters
    ----------
    file_path : str
        Path to the raw PROACT_MUSCLESTRENGTH.csv file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame (-> saved as PROACT_MUSCLESTRENGTH_v2.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)

    df = df.drop(columns=['Test_Category', 'Test_Unit'])

    # Remove redundant prefix present on all Test_Name values
    df['Test_Name'] = df['Test_Name'].str.replace(
        r'^Isometric Muscle Strength, ', '', regex=True
    )

    # Normalise Test_Location: title-case and fix known typo
    df['Test_Location'] = df['Test_Location'].str.strip().str.title()
    df['Test_Location'] = df['Test_Location'].replace({'Wristjoint': 'Wrist Joint'})

    def is_test_break(row):
        """Convert 'yes' string to True; NaN stays None."""
        if pd.isna(row['Test_Break']):
            return None
        return row['Test_Break'].lower() == 'yes'

    df['Test_Break'] = df.apply(is_test_break, axis=1)

    return df





# ------------------------------------------------------------------
# Stage v3 - Pivot laterality and trial rows into one row per test/visit
# ------------------------------------------------------------------

def group_left_right_muscle_strength_trials(file_path):
    """
    Reshape the long-format data (one row per trial per side) into a
    semi-wide format with one row per (subject_id, MS_Delta, Test_Name,
    Test_Location) combination.

    Within each group, left and right trial measurements are spread across
    dedicated columns following the naming convention:

        Test_Result_Pounds_{Left|Right}_Trial_{n}
        Test_Break_{Left|Right}_Trial_{n}

    where n is the trial number. This pivot makes it possible to compare
    left vs. right performance and to select the best trial per side in
    the next stage.

    A progress message is printed every 500 patients to monitor execution.

    Parameters
    ----------
    file_path : str
        Path to PROACT_MUSCLESTRENGTH_v2.csv.

    Returns
    -------
    pd.DataFrame
        Semi-wide DataFrame with one row per test/visit combination
        (-> saved as PROACT_MUSCLESTRENGTH_v3.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)

    # Rename for clarity: the unit (Pounds) is now encoded in the column name
    df.rename(columns={'Test_Result': 'Test_Result_Pounds'}, inplace=True)

    # Stable sort before grouping
    df = df.sort_values(
        by=['subject_id', 'MS_Delta', 'Test_Name', 'Test_Location', 'Test_trial', 'Test_Laterality']
    )

    data_rows     = []
    patient_counter = 0
    patient_total   = df['subject_id'].nunique()

    for subject_id, group in df.groupby('subject_id'):
        patient_counter += 1
        if patient_counter % 500 == 0:
            print(f"Processed {patient_counter} patients out of {patient_total}...")

        # One output row per (MS_Delta, Test_Name, Test_Location) combination
        for (delta, test_name, location), sub_group in group.groupby(
            ['MS_Delta', 'Test_Name', 'Test_Location']
        ):
            row_data = {
                'subject_id':    subject_id,
                'MS_Delta':      delta,
                'Test_Name':     test_name,
                'Test_Location': location,
            }

            # Spread each trial into a laterality-specific column pair
            for _, row in sub_group.iterrows():
                lat   = str(row['Test_Laterality']).lower()
                trial = row['Test_trial']
                if lat == 'left':
                    row_data[f'Test_Result_Pounds_Left_Trial_{trial}'] = row['Test_Result_Pounds']
                    row_data[f'Test_Break_Left_Trial_{trial}']         = row['Test_Break']
                elif lat == 'right':
                    row_data[f'Test_Result_Pounds_Right_Trial_{trial}'] = row['Test_Result_Pounds']
                    row_data[f'Test_Break_Right_Trial_{trial}']          = row['Test_Break']

            data_rows.append(row_data)

    return pd.DataFrame(data_rows)





# ------------------------------------------------------------------
# Stage v4 - Add best-trial summary and laterality feature columns
# ------------------------------------------------------------------

def add_max_trials_columns_muscle_strength(file_path, data_path):
    """
    Derive four summary features per test/visit row and join DominantHand
    from the hand grip strength pipeline output.

    Summary columns added:
        Test_Result_Pounds_{Left|Right}_Trial_Max
            - the highest force recorded across all trials for that side
        Test_Break_{Left|Right}_Trial_Max
            - the Test_Break value corresponding to the best trial
        Test_Result_Most_Affected_Side
            - the weaker side (Left / Right / Equal / None)
        Test_Result_Most_Affected_Side_isDominant
            - True if the weaker side matches the patient's dominant hand

    DominantHand is joined from PROACT_HANDGRIPSTRENGTH_v8.csv rather than
    being stored in the muscle strength raw data. The 'HAN_' prefix from the
    hand grip pipeline is removed after the join.

    Trial columns are detected dynamically so the function handles any number
    of trials without hardcoding.

    Parameters
    ----------
    file_path : str
        Path to PROACT_MUSCLESTRENGTH_v3.csv.
    data_path : str   
        Path to the Root directory for all processed outputs

    Returns
    -------
    pd.DataFrame
        Input DataFrame enriched with summary and laterality columns
        (-> saved as PROACT_MUSCLESTRENGTH_v4.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)

    def get_max_trial(side):
        """
        Add *_Trial_Max result and break columns for the given side.
        Columns are detected dynamically to scale to any number of trials.
        """
        result_cols = [col for col in df.columns if f'Test_Result_Pounds_{side}_Trial_' in col]
        break_cols  = [col for col in df.columns if f'Test_Break_{side}_Trial_' in col]

        def max_result_and_break(row):
            """Return the maximum result value and its corresponding break flag."""
            max_value = None
            max_break = None
            for res_col, brk_col in zip(result_cols, break_cols):
                value = row[res_col]
                if pd.notna(value) and (max_value is None or value > max_value):
                    max_value = value
                    max_break = row[brk_col]
            return pd.Series([max_value, max_break])

        df[[
            f'Test_Result_Pounds_{side}_Trial_Max',
            f'Test_Break_{side}_Trial_Max',
        ]] = df.apply(max_result_and_break, axis=1)

    get_max_trial('Left')
    get_max_trial('Right')

    def determine_best_side(row):
        """
        Identify the weaker side by comparing left and right maxima.
        Returns 'Left' if left < right, 'Right' if right < left,
        'Equal' if both are identical, or None if no data is available.
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

    # Join DominantHand from the hand grip strength pipeline output
    # (not present in the raw muscle strength file)
    df_handgrip = pd.read_csv(data_path + '/PROACT_HANDGRIPSTRENGTH_v8.csv', low_memory=False)
    df_handgrip = df_handgrip[['subject_id', 'HAN_DominantHand']].drop_duplicates()

    df = df.merge(df_handgrip, on='subject_id', how='left')

    # Move DominantHand to position 1 and drop the 'HAN_' namespace prefix
    cols = df.columns.tolist()
    cols.insert(1, cols.pop(cols.index('HAN_DominantHand')))
    df = df[cols]
    df.rename(columns={'HAN_DominantHand': 'DominantHand'}, inplace=True)

    # Coerce 'nan' strings introduced by astype(str) back to proper NaN
    df['DominantHand'] = df['DominantHand'].astype(str).replace('nan', pd.NA)

    def is_most_affected_side_dominant(row):
        """Return True if the weaker side matches the patient's dominant hand."""
        if pd.isna(row['Test_Result_Most_Affected_Side']) or pd.isna(row['DominantHand']):
            return None
        return row['Test_Result_Most_Affected_Side'].lower() == row['DominantHand'].lower()

    df['Test_Result_Most_Affected_Side_isDominant'] = df.apply(
        is_most_affected_side_dominant, axis=1
    )

    return df





# ------------------------------------------------------------------
# Stage v5 - Pivot test/location combinations into one row per visit
# ------------------------------------------------------------------

def regrouper_par_subject_id_ms_delta(csv_file):
    """
    Reshape the test/location-level rows into a visit-level wide DataFrame
    with one row per (subject_id, MS_Delta), where each (Test_Name,
    Test_Location) combination is represented by a dedicated column block.

    Column naming convention:
        {Test_Name}_{clean_Test_Location}_{measurement_column}
        e.g. 'Abduction_FirstDorsalInterosseousMuscleOfTheHand_Test_Result_Pounds_Left_Trial_Max'

    Test_Location is cleaned by removing spaces, hyphens, and parentheses
    before being used as a column name component. Test_Name is used as-is.

    Columns are sorted alphabetically by (Test_Name, Test_Location), and
    within each combination the measurement columns follow the order defined
    in the `columns` list.

    DominantHand is stored once per row as a patient-level constant.
    A progress message is printed every 2000 groups to monitor execution.

    Parameters
    ----------
    csv_file : str
        Path to PROACT_MUSCLESTRENGTH_v4.csv.

    Returns
    -------
    pd.DataFrame
        Wide-format visit-level DataFrame with one row per (subject_id, MS_Delta)
        (-> saved as PROACT_MUSCLESTRENGTH_v5.csv).
    """
    df = pd.read_csv(csv_file, low_memory=False)

    # Ordered list of measurement columns to carry per test/location block
    columns = [
        'DominantHand', 'MS_Delta',
        'Test_Result_Pounds_Left_Trial_1',   'Test_Break_Left_Trial_1',
        'Test_Result_Pounds_Left_Trial_2',   'Test_Break_Left_Trial_2',
        'Test_Result_Pounds_Left_Trial_3',   'Test_Break_Left_Trial_3',
        'Test_Result_Pounds_Right_Trial_1',  'Test_Break_Right_Trial_1',
        'Test_Result_Pounds_Right_Trial_2',  'Test_Break_Right_Trial_2',
        'Test_Result_Pounds_Right_Trial_3',  'Test_Break_Right_Trial_3',
        'Test_Result_Pounds_Left_Trial_Max',  'Test_Break_Left_Trial_Max',
        'Test_Result_Pounds_Right_Trial_Max', 'Test_Break_Right_Trial_Max',
        'Test_Result_Most_Affected_Side', 'Test_Result_Most_Affected_Side_isDominant',
    ]

    def clean_name(name):
        """Remove special characters from location names for use in column identifiers."""
        return (
            name.replace(" ", "").replace("-", "")
                .replace("(", "").replace(")", "")
        )

    df = df.sort_values(by=['subject_id', 'MS_Delta', 'Test_Name', 'Test_Location'])

    grouped      = df.groupby(['subject_id', 'MS_Delta'])
    rows         = []
    total_groups = len(grouped)
    counter     = 0

    for (subject_id, ms_delta), group in grouped:
        counter += 1
        if counter % 2000 == 0:
            print(f"Processed {counter} groups out of {total_groups}...")

        # Patient-level constants stored once per visit row
        row_data = {
            'subject_id':  subject_id,
            'DominantHand': group['DominantHand'].iloc[0],
            'MS_Delta':    ms_delta,
        }

        # One column block per (Test_Name, Test_Location) combination present at this visit
        for _, test_row in group.iterrows():
            test_name     = test_row['Test_Name']
            test_location = clean_name(test_row['Test_Location'])
            for col in columns:
                if col in ('DominantHand', 'MS_Delta'):
                    continue
                row_data[f"{test_name}_{test_location}_{col}"] = test_row[col]

        rows.append(row_data)

    df_wide = pd.DataFrame(rows)

    # Sort columns: fixed headers first, then alphabetically by (Test_Name, Test_Location),
    # with measurement columns in the order defined by `columns`
    test_combinations = sorted(
        df[['Test_Name', 'Test_Location']].drop_duplicates().values.tolist(),
        key=lambda x: (x[0], x[1])
    )
    print(test_combinations)

    final_cols = ['subject_id', 'DominantHand', 'MS_Delta']
    for test_name, test_location in test_combinations:
        clean_loc = clean_name(test_location)
        for col in columns:
            if col in ('DominantHand', 'MS_Delta'):
                continue
            final_cols.append(f"{test_name}_{clean_loc}_{col}")

    df_wide = df_wide.reindex(columns=final_cols)

    return df_wide





# ------------------------------------------------------------------
# Stage v6 - Add per-patient observation count
# ------------------------------------------------------------------

def observation_counter_muscle_strength(file_path):
    """
    Add an `observation_count` column recording how many visit rows exist
    per patient in the visit-level wide dataset.

    Each row represents one MS_Delta timepoint for a patient.

    Parameters
    ----------
    file_path : str
        Path to PROACT_MUSCLESTRENGTH_v5.csv.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with `observation_count` inserted as the second column
        (immediately after `subject_id`).
    """
    df = pd.read_csv(file_path, low_memory=False)

    # Map each patient to the number of rows they have in the DataFrame
    df['observation_count'] = df['subject_id'].map(df['subject_id'].value_counts())

    cols = df.columns.tolist()
    cols.insert(1, cols.pop(cols.index('observation_count')))
    df = df[cols]

    return df





# ------------------------------------------------------------------
# Stage v7 - Reshape to fully wide format
# ------------------------------------------------------------------

def reshape_to_fully_wide_format(csv_file):
    """
    Reshape the visit-level data (one row per MS_Delta) into a fully wide
    patient-level DataFrame (one row per patient) where each visit's column
    block is prefixed by the visit index.

    Visits are sorted chronologically by MS_Delta before pivoting, so that
    prefix 1_ always corresponds to the earliest recorded visit.

    Column ordering within each visit block follows a custom suffix order
    (defined in `custom_suffix_order`) that groups trial results before
    break flags, left before right, individual trials before the max summary,
    and keeps MS_Delta as the first column of each block.

    The sort key handles the composite column structure:
        {visit_index}_{Test_Name}_{Test_Location}_{measurement_suffix}
    by separating the numeric prefix, identifying the matched suffix from
    the custom order, and sorting on (visit_index, type_part, suffix_order).

    Patient-level constants (subject_id, observation_count, DominantHand)
    are stored once per row and excluded from the visit-prefixed columns.

    A progress message is printed every 500 patients to monitor execution.

    Parameters
    ----------
    csv_file : str
        Path to PROACT_MUSCLESTRENGTH_v6.csv.

    Returns
    -------
    pd.DataFrame
        Wide-format patient-level DataFrame
        (-> saved as PROACT_MUSCLESTRENGTH_v7.csv).
    """
    df = pd.read_csv(csv_file, low_memory=False)

    df = df.sort_values(by=['subject_id', 'MS_Delta'])

    # Patient-level constants — stored once, not repeated per visit
    fixed_cols = ['subject_id', 'observation_count', 'DominantHand']
    value_cols = [col for col in df.columns if col not in fixed_cols]

    grouped_rows   = []
    total_subjects = df['subject_id'].nunique()
    counter       = 0

    for subject_id, group in df.groupby('subject_id'):
        counter += 1
        if counter % 500 == 0:
            print(f"Processed {counter} patients out of {total_subjects}...")

        group = group.reset_index(drop=True)
        row_data = {
            'subject_id':        subject_id,
            'observation_count': group['observation_count'].iloc[0],
            'DominantHand':      group['DominantHand'].iloc[0],
        }

        for i, (_, obs) in enumerate(group.iterrows(), start=1):
            for col in value_cols:
                # MS_Delta is the visit anchor and receives its own prefixed column
                row_data[f"{i}_{col}"] = obs[col]

        grouped_rows.append(row_data)

    df_grouped = pd.DataFrame(grouped_rows)

    # Custom suffix ordering: groups measurement types in a clinically logical sequence
    custom_suffix_order = [
        'Test_Result_Pounds_Left_Trial_1',   'Test_Break_Left_Trial_1',
        'Test_Result_Pounds_Left_Trial_2',   'Test_Break_Left_Trial_2',
        'Test_Result_Pounds_Left_Trial_3',   'Test_Break_Left_Trial_3',
        'Test_Result_Pounds_Right_Trial_1',  'Test_Break_Right_Trial_1',
        'Test_Result_Pounds_Right_Trial_2',  'Test_Break_Right_Trial_2',
        'Test_Result_Pounds_Right_Trial_3',  'Test_Break_Right_Trial_3',
        'Test_Result_Pounds_Left_Trial_Max',  'Test_Break_Left_Trial_Max',
        'Test_Result_Pounds_Right_Trial_Max', 'Test_Break_Right_Trial_Max',
        'Test_Result_Most_Affected_Side', 'Test_Result_Most_Affected_Side_isDominant',
    ]
    custom_order_map = {suffix: i for i, suffix in enumerate(custom_suffix_order)}

    def sort_key(col):
        """
        Sort prefixed visit columns by:
            1. visit index (numeric prefix)
            2. type_part (Test_Name_Test_Location prefix of the column)
            3. suffix position in custom_suffix_order
        MS_Delta is forced to position -1 within each visit block so it
        always appears first.
        """
        parts = col.split('_', 1)
        if len(parts) < 2:
            return (0, "zzz", 9999, col)

        num    = int(parts[0]) if parts[0].isdigit() else 0
        suffix = parts[1]

        if suffix == 'MS_Delta':
            return (num, "", -1, suffix)

        # Match the longest suffix from the custom order list
        matched_suffix = next(
            (s for s in custom_suffix_order if suffix.endswith(s)), None
        )

        if matched_suffix:
            type_part = suffix[: -len(matched_suffix)].rstrip('_')
            order_idx = custom_order_map.get(matched_suffix, 9999)
        else:
            type_part = suffix
            order_idx = 9999

        return (num, str(type_part), int(order_idx), str(suffix))

    id_cols    = ['subject_id', 'observation_count', 'DominantHand']
    other_cols = sorted(
        [c for c in df_grouped.columns if c not in id_cols], key=sort_key
    )

    df_grouped = df_grouped[id_cols + other_cols]

    return df_grouped





# ------------------------------------------------------------------
# Stage v8 - Add 'MUS_' prefix to all feature columns
# ------------------------------------------------------------------

def rename_all_columns(file_path):
    """
    Prefix every feature column with 'MUS_' to namespace the muscle strength
    variables when merging with other PROACT sub-datasets.

    `subject_id` is the join key and is left unchanged.

    Parameters
    ----------
    file_path : str
        Path to PROACT_MUSCLESTRENGTH_v7.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns
        (-> saved as PROACT_MUSCLESTRENGTH_v8.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)
    df = df.rename(columns={col: f'MUS_{col}' for col in df.columns if col != 'subject_id'})
    return df










# ==================================================================
# ------------------------- PIPELINE EXECUTION ---------------------
# ==================================================================

def run(DATA_PATH, PROACT_PATH):

    print("\n" * 3)
    print("=" * 60)
    print("MUSCLESTRENGTH PIPELINE")
    print("=" * 60)

    

    # Stage v2 - Drop constant columns and normalise values
    df = modify_muscle_strength(PROACT_PATH + '/PROACT_MUSCLESTRENGTH.csv')
    df.to_csv(DATA_PATH + '/PROACT_MUSCLESTRENGTH_v2.csv', index=False)



    # Stage v3 - Pivot laterality and trial rows into one row per test/visit
    df_muscle_strength = group_left_right_muscle_strength_trials(
        DATA_PATH + '/PROACT_MUSCLESTRENGTH_v2.csv'
    )
    df_muscle_strength.to_csv(DATA_PATH + '/PROACT_MUSCLESTRENGTH_v3.csv', index=False)



    # Stage v4 - Add best-trial summary and laterality feature columns
    df_muscle_strength = add_max_trials_columns_muscle_strength(
        DATA_PATH + '/PROACT_MUSCLESTRENGTH_v3.csv',
        data_path=DATA_PATH
    )
    df_muscle_strength.to_csv(DATA_PATH + '/PROACT_MUSCLESTRENGTH_v4.csv', index=False)



    # Stage v5 - Pivot test/location combinations into one row per visit
    df_muscle_strength = regrouper_par_subject_id_ms_delta(
        DATA_PATH + '/PROACT_MUSCLESTRENGTH_v4.csv'
    )
    df_muscle_strength.to_csv(DATA_PATH + '/PROACT_MUSCLESTRENGTH_v5.csv', index=False)



    # Stage v6 - Add per-patient observation count
    df_muscle_strength = observation_counter_muscle_strength(
        DATA_PATH + '/PROACT_MUSCLESTRENGTH_v5.csv'
    )
    df_muscle_strength.to_csv(DATA_PATH + '/PROACT_MUSCLESTRENGTH_v6.csv', index=False)



    # Stage v7 - Reshape to fully wide format
    df_final = reshape_to_fully_wide_format(DATA_PATH + '/PROACT_MUSCLESTRENGTH_v6.csv')
    df_final.to_csv(DATA_PATH + '/PROACT_MUSCLESTRENGTH_v7.csv', index=False)



    # Stage v8 - Add 'MUS_' prefix to all feature columns
    df_renamed = rename_all_columns(DATA_PATH + '/PROACT_MUSCLESTRENGTH_v7.csv')
    df_renamed.to_csv(DATA_PATH + '/PROACT_MUSCLESTRENGTH_v8.csv', index=False)