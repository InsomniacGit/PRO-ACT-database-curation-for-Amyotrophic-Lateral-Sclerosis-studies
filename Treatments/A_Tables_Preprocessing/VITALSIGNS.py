"""
PROACT Vital Signs Processing Pipeline
=======================================
This script processes the PROACT (PRO-ACT ALS) vital signs dataset through
a sequential cleaning, unit conversion, and reshaping pipeline. It produces
seven intermediate CSV files, culminating in a wide-format one-row-per-patient
matrix with visits stored as sequentially prefixed column blocks.

The dataset contains a heterogeneous mix of measurements (blood pressure,
pulse, respiratory rate, temperature, height, weight) recorded in different
units across study sites. All values are converted to a single standardised
unit per measurement type before reshaping.

Pipeline stages:
    v2  - Drop constant unit columns; encode units in measurement column names
    v3  - Convert Height (inches -> cm) and Weight (pounds -> kg); encode
          units in column names; drop unit columns
    v4  - Encode units in the remaining positional blood pressure and pulse
          column names; reorder key columns
    v5  - Add per-patient observation count
    v6  - Reshape to wide format (one row per patient, visits as prefixed
          column blocks with clinically grouped ordering)
    v7  - Prefix all feature columns with 'VIT_' for downstream merging

Author: Bouclier Lucas
Data:   PROACT dataset (2022-07-29 release)
"""



import pandas as pd





# //////////////////////////////////////////////////////////////
# ------------------------- VITALSIGNS -------------------------
# //////////////////////////////////////////////////////////////



# ------------------------------------------------------------------
# Stage v2 - Drop unit columns and encode units in measurement column names
# ------------------------------------------------------------------

def clean_vital_signs(csv_file):
    """
    Remove the five unit columns for the core vital sign measurements and
    encode the unit directly in each measurement column name.

    In the raw dataset, each measurement has a companion *_Units column that
    is constant across the entire dataset. Dropping these columns and moving
    the unit into the column name makes the DataFrame self-documenting and
    reduces its width.

    Renames applied:
        Blood_Pressure_Diastolic  -> Blood_Pressure_Diastolic_mmHg
        Blood_Pressure_Systolic   -> Blood_Pressure_Systolic_mmHg
        Pulse                     -> Pulse_Beats_per_minute
        Respiratory_Rate          -> Respiratory_Rate_breaths_per_minute
        Temperature               -> Temperature_Celsius

    Parameters
    ----------
    csv_file : str
        Path to the raw PROACT_VITALSIGNS.csv file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with units encoded in column names
        (-> saved as PROACT_VITALSIGNS_v2.csv).
    """
    df = pd.read_csv(csv_file, low_memory=False)

    # Unit columns are constant across the dataset and therefore uninformative
    columns_to_delete = [
        'Blood_Pressure_Diastolic_Units',
        'Blood_Pressure_Systolic_Units',
        'Pulse_Units',
        'Respiratory_Rate_Units',
        'Temperature_Units',
    ]
    df.drop(columns=columns_to_delete, inplace=True)

    df.rename(columns={
        'Blood_Pressure_Diastolic': 'Blood_Pressure_Diastolic_mmHg',
        'Blood_Pressure_Systolic':  'Blood_Pressure_Systolic_mmHg',
        'Pulse':                    'Pulse_Beats_per_minute',
        'Respiratory_Rate':         'Respiratory_Rate_breaths_per_minute',
        'Temperature':              'Temperature_Celsius',
    }, inplace=True)

    return df





# ------------------------------------------------------------------
# Stage v3 - Convert Height and Weight to metric units
# ------------------------------------------------------------------

def convert_vital_signs(csv_file):
    """
    Convert Height from inches to centimetres and Weight (including
    Baseline_Weight and Endpoint_Weight) from pounds to kilograms for
    records recorded in imperial units.

    Conversion factors applied:
        Height:  1 inch  = 2.54 cm
        Weight:  1 pound = 0.45359237 kg

    Only rows where the unit column indicates imperial values are converted;
    rows already in metric units are left unchanged. After conversion, unit
    columns are dropped and the measurement columns are renamed to encode
    the now-uniform metric unit.

    Parameters
    ----------
    csv_file : str
        Path to PROACT_VITALSIGNS_v2.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with Height and Weight expressed in metric units
        (-> saved as PROACT_VITALSIGNS_v3.csv).
    """
    df = pd.read_csv(csv_file, low_memory=False)

    # Convert Height: inches -> centimetres
    mask_inches = df['Height_Units'] == 'Inches'
    df.loc[mask_inches, 'Height'] = df.loc[mask_inches, 'Height'].apply(
        lambda x: x * 2.54 if pd.notnull(x) else x
    )

    # Convert Weight, Baseline_Weight, and Endpoint_Weight: pounds -> kilograms
    mask_pounds = df['Weight_Units'] == 'Pounds'
    for col in ['Weight', 'Baseline_Weight', 'Endpoint_Weight']:
        df.loc[mask_pounds, col] = df.loc[mask_pounds, col].apply(
            lambda x: x * 0.45359237 if pd.notnull(x) else x
        )

    df.rename(columns={
        'Height':           'Height_cm',
        'Weight':           'Weight_kg',
        'Baseline_Weight':  'Baseline_Weight_kg',
        'Endpoint_Weight':  'Endpoint_Weight_kg',
    }, inplace=True)

    # Drop unit columns now that all values are in a single metric unit
    df.drop(columns=['Height_Units', 'Weight_Units'], inplace=True)

    return df





# ------------------------------------------------------------------
# Stage v4 - Encode units in positional blood pressure and pulse column names
# ------------------------------------------------------------------

def rename_vital_signs(csv_file):
    """
    Encode the measurement unit in the remaining blood pressure and pulse
    column names that were not renamed in stage v2, and reorder key columns.

    The raw dataset contains additional positional blood pressure and pulse
    measurements (supine, standing, baseline, endpoint) whose names do not
    carry unit suffixes. This function applies the same naming convention
    established in stage v2 (unit encoded in column name) to all remaining
    columns of this type.

    Vital_Signs_Delta is moved to the second position (immediately after
    subject_id) as the chronological anchor for the visit.

    Parameters
    ----------
    csv_file : str
        Path to PROACT_VITALSIGNS_v3.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with fully unit-annotated column names and reordered columns
        (-> saved as PROACT_VITALSIGNS_v4.csv).
    """
    df = pd.read_csv(csv_file, low_memory=False)

    df.rename(columns={
        # Blood pressure - diastolic (positional and temporal variants)
        'Baseline_Standing_BP_Diastolic':  'Baseline_Standing_BP_Diastolic_mmHg',
        'Baseline_Supine_BP_Diastolic':    'Baseline_Supine_BP_Diastolic_mmHg',
        'Supine_BP_Diastolic':             'Supine_BP_Diastolic_mmHg',
        'Standing_BP_Diastolic':           'Standing_BP_Diastolic_mmHg',
        'Endpoint_Supine_BP_Diastolic':    'Endpoint_Supine_BP_Diastolic_mmHg',
        'Endpoint_Standing_BP_Diastolic':  'Endpoint_Standing_BP_Diastolic_mmHg',
        # Blood pressure - systolic (positional and temporal variants)
        'Baseline_Standing_BP_Systolic':   'Baseline_Standing_BP_Systolic_mmHg',
        'Baseline_Supine_BP_Systolic':     'Baseline_Supine_BP_Systolic_mmHg',
        'Supine_BP_Systolic':              'Supine_BP_Systolic_mmHg',
        'Standing_BP_Systolic':            'Standing_BP_Systolic_mmHg',
        'Endpoint_Supine_BP_Systolic':     'Endpoint_Supine_BP_Systolic_mmHg',
        'Endpoint_Standing_BP_Systolic':   'Endpoint_Standing_BP_Systolic_mmHg',
        # Pulse (positional and temporal variants)
        'Supine_Pulse':                    'Supine_Pulse_Beats_per_minute',
        'Standing_Pulse':                  'Standing_Pulse_Beats_per_minute',
        'Baseline_Supine_Pulse':           'Baseline_Supine_Pulse_Beats_per_minute',
        'Baseline_Standing_Pulse':         'Baseline_Standing_Pulse_Beats_per_minute',
        'Endpoint_Supine_Pulse':           'Endpoint_Supine_Pulse_Beats_per_minute',
        'Endpoint_Standing_Pulse':         'Endpoint_Standing_Pulse_Beats_per_minute',
    }, inplace=True)

    # Bring the visit time anchor to the front for readability
    column = (
        ['subject_id', 'Vital_Signs_Delta']
        + [col for col in df.columns if col not in ['subject_id', 'Vital_Signs_Delta']]
    )
    df = df[column]

    return df





# ------------------------------------------------------------------
# Stage v5 - Add per-patient observation count
# ------------------------------------------------------------------

def observation_counter_vital_signs(file_path):
    """
    Add an `observation_count` column recording how many visit rows exist
    per patient.

    Each row represents one vital signs assessment visit. This count reflects
    the number of valid visits retained after all cleaning and renaming steps.

    Parameters
    ----------
    file_path : str
        Path to PROACT_VITALSIGNS_v4.csv.

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
# Stage v6 - Reshape to wide format
# ------------------------------------------------------------------

def reshape_to_wide_format(csv_file):
    """
    Reshape the long-format vital signs data (multiple rows per patient) into
    a wide-format DataFrame (one row per patient) where each visit's values
    are stored in sequentially prefixed column blocks.

    Visits are sorted by Vital_Signs_Delta (time since study baseline) before
    pivoting, so that prefix 1_ always corresponds to the earliest recorded
    visit.

    Within each visit block, columns are grouped by measurement category in
    the following clinically logical order:
        1. Visit time anchor (Vital_Signs_Delta)
        2. Respiratory rate and temperature
        3. Anthropometry (height, weight, baseline weight, endpoint weight)
        4. Blood pressure - diastolic (general, supine, standing variants)
        5. Blood pressure - systolic (general, supine, standing variants)
        6. Pulse (general, supine, standing variants)

    Values for columns that may not be present in all rows are retrieved with
    .get() defaulting to None, so the output matrix is always rectangular.

    A progress message is printed every 500 patients to monitor execution.

    Parameters
    ----------
    csv_file : str
        Path to PROACT_VITALSIGNS_v5.csv.

    Returns
    -------
    pd.DataFrame
        Wide-format patient-level DataFrame
        (-> saved as PROACT_VITALSIGNS_v6.csv).
    """
    df = pd.read_csv(csv_file, low_memory=False)

    # Sort visits chronologically within each patient
    df = df.sort_values(by=['subject_id', 'Vital_Signs_Delta'])

    result = []

    for subject_id, group in df.groupby('subject_id'):
        row_data = {
            'subject_id':        subject_id,
            'observation_count': group.shape[0],
        }

        for idx in range(len(group)):
            p = idx + 1  # 1-based visit prefix

            # Visit time anchor
            row_data[f'{p}_Vital_Signs_Delta'] = group.iloc[idx]['Vital_Signs_Delta']

            # Other core vital signs
            row_data[f'{p}_Respiratory_Rate_breaths_per_minute'] = group.iloc[idx]['Respiratory_Rate_breaths_per_minute']
            row_data[f'{p}_Temperature_Celsius']                  = group.iloc[idx]['Temperature_Celsius']

            # Anthropometry
            row_data[f'{p}_Height_cm']           = group.iloc[idx]['Height_cm']
            row_data[f'{p}_Weight_kg']            = group.iloc[idx]['Weight_kg']
            row_data[f'{p}_Baseline_Weight_kg']   = group.iloc[idx].get('Baseline_Weight_kg',  None)
            row_data[f'{p}_Endpoint_Weight_kg']   = group.iloc[idx].get('Endpoint_Weight_kg',  None)

            # Blood pressure - diastolic
            row_data[f'{p}_Blood_Pressure_Diastolic_mmHg']          = group.iloc[idx]['Blood_Pressure_Diastolic_mmHg']
            row_data[f'{p}_Baseline_Supine_BP_Diastolic_mmHg']      = group.iloc[idx].get('Baseline_Supine_BP_Diastolic_mmHg',     None)
            row_data[f'{p}_Baseline_Standing_BP_Diastolic_mmHg']    = group.iloc[idx].get('Baseline_Standing_BP_Diastolic_mmHg',   None)
            row_data[f'{p}_Supine_BP_Diastolic_mmHg']               = group.iloc[idx].get('Supine_BP_Diastolic_mmHg',              None)
            row_data[f'{p}_Standing_BP_Diastolic_mmHg']             = group.iloc[idx].get('Standing_BP_Diastolic_mmHg',            None)
            row_data[f'{p}_Endpoint_Supine_BP_Diastolic_mmHg']      = group.iloc[idx].get('Endpoint_Supine_BP_Diastolic_mmHg',     None)
            row_data[f'{p}_Endpoint_Standing_BP_Diastolic_mmHg']    = group.iloc[idx].get('Endpoint_Standing_BP_Diastolic_mmHg',   None)

            # Blood pressure - systolic
            row_data[f'{p}_Blood_Pressure_Systolic_mmHg']           = group.iloc[idx]['Blood_Pressure_Systolic_mmHg']
            row_data[f'{p}_Baseline_Supine_BP_Systolic_mmHg']       = group.iloc[idx].get('Baseline_Supine_BP_Systolic_mmHg',      None)
            row_data[f'{p}_Baseline_Standing_BP_Systolic_mmHg']     = group.iloc[idx].get('Baseline_Standing_BP_Systolic_mmHg',    None)
            row_data[f'{p}_Supine_BP_Systolic_mmHg']                = group.iloc[idx].get('Supine_BP_Systolic_mmHg',               None)
            row_data[f'{p}_Standing_BP_Systolic_mmHg']              = group.iloc[idx].get('Standing_BP_Systolic_mmHg',             None)
            row_data[f'{p}_Endpoint_Supine_BP_Systolic_mmHg']       = group.iloc[idx].get('Endpoint_Supine_BP_Systolic_mmHg',      None)
            row_data[f'{p}_Endpoint_Standing_BP_Systolic_mmHg']     = group.iloc[idx].get('Endpoint_Standing_BP_Systolic_mmHg',    None)

            # Pulse
            row_data[f'{p}_Pulse_Beats_per_minute']                      = group.iloc[idx]['Pulse_Beats_per_minute']
            row_data[f'{p}_Baseline_Supine_Pulse_Beats_per_minute']      = group.iloc[idx].get('Baseline_Supine_Pulse_Beats_per_minute',   None)
            row_data[f'{p}_Baseline_Standing_Pulse_Beats_per_minute']    = group.iloc[idx].get('Baseline_Standing_Pulse_Beats_per_minute', None)
            row_data[f'{p}_Supine_Pulse_Beats_per_minute']               = group.iloc[idx].get('Supine_Pulse_Beats_per_minute',            None)
            row_data[f'{p}_Standing_Pulse_Beats_per_minute']             = group.iloc[idx].get('Standing_Pulse_Beats_per_minute',          None)
            row_data[f'{p}_Endpoint_Supine_Pulse_Beats_per_minute']      = group.iloc[idx].get('Endpoint_Supine_Pulse_Beats_per_minute',   None)
            row_data[f'{p}_Endpoint_Standing_Pulse_Beats_per_minute']    = group.iloc[idx].get('Endpoint_Standing_Pulse_Beats_per_minute', None)

        result.append(row_data)

        if len(result) % 500 == 0:
            print(f'Processed {len(result)} subjects out of {df["subject_id"].nunique()}')

    return pd.DataFrame(result)





# ------------------------------------------------------------------
# Stage v7 - Add 'VIT_' prefix to all feature columns
# ------------------------------------------------------------------

def rename_all_columns(file_path):
    """
    Prefix every feature column with 'VIT_' to namespace the vital signs
    variables when merging with other PROACT sub-datasets.

    `subject_id` is the join key and is left unchanged.

    Parameters
    ----------
    file_path : str
        Path to PROACT_VITALSIGNS_v6.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns
        (-> saved as PROACT_VITALSIGNS_v7.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)
    df = df.rename(columns={col: f'VIT_{col}' for col in df.columns if col != 'subject_id'})

    return df










# ==================================================================
# ------------------------- PIPELINE EXECUTION ---------------------
# ==================================================================

def run(DATA_PATH, PROACT_PATH):

    print("\n" * 3)
    print("=" * 60)
    print("VITALSIGNS PIPELINE")
    print("=" * 60)

    

    # Stage v2 - Drop unit columns and encode units in measurement column names
    df_vital_signs = clean_vital_signs(PROACT_PATH + '/PROACT_VITALSIGNS.csv')
    df_vital_signs.to_csv(DATA_PATH + '/PROACT_VITALSIGNS_v2.csv', index=False)



    # Stage v3 - Convert Height and Weight to metric units
    df_vital_signs = convert_vital_signs(DATA_PATH + '/PROACT_VITALSIGNS_v2.csv')
    df_vital_signs.to_csv(DATA_PATH + '/PROACT_VITALSIGNS_v3.csv', index=False)



    # Stage v4 - Encode units in positional blood pressure and pulse column names
    df_vital_signs = rename_vital_signs(DATA_PATH + '/PROACT_VITALSIGNS_v3.csv')
    df_vital_signs.to_csv(DATA_PATH + '/PROACT_VITALSIGNS_v4.csv', index=False)



    # Stage v5 - Add per-patient observation count
    df_vital_signs = observation_counter_vital_signs(DATA_PATH + '/PROACT_VITALSIGNS_v4.csv')
    df_vital_signs.to_csv(DATA_PATH + '/PROACT_VITALSIGNS_v5.csv', index=False)



    # Stage v6 - Reshape to wide format
    df_vital_signs_wide = reshape_to_wide_format(DATA_PATH + '/PROACT_VITALSIGNS_v5.csv')
    df_vital_signs_wide.to_csv(DATA_PATH + '/PROACT_VITALSIGNS_v6.csv', index=False)



    # Stage v7 - Add 'VIT_' prefix to all feature columns
    df_renamed = rename_all_columns(DATA_PATH + '/PROACT_VITALSIGNS_v6.csv')
    df_renamed.to_csv(DATA_PATH + '/PROACT_VITALSIGNS_v7.csv', index=False)