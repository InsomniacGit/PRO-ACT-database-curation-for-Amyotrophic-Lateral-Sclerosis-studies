"""
PROACT ALSFRS Processing Pipeline
==================================
This script processes the PROACT (PRO-ACT ALS) ALSFRS and ALSFRS-R dataset
through a sequential cleaning, validation, and feature engineering pipeline.
It produces several intermediate CSV files and two final wide-format datasets
(one with imputed total scores, one without).

Pipeline stages:
    v2          - Remove rows with missing core items; merge Q5a/Q5b cutting items
    v3          - Drop rows with structurally inconsistent total scores
    v4          - Correct arithmetic errors in ALSFRS_Total and ALSFRS_R_Total
    v5          - Compute domain subscores (bulbar, upper limb, lower limb, respiratory)
    v6          - Add per-patient observation count
    v6_filled   - Impute missing ALSFRS_Total or ALSFRS_R_Total from the other scale
    v7 / v7_filled  - Reshape to wide format (one row per patient, visits as prefixed columns)
    v8 / v8_filled  - Prefix all feature columns with 'ALS_' for downstream merging

Author: Bouclier Lucas
Data:   PROACT dataset (2022-07-29 release)
"""



import pandas as pd
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
# ------------------------- ALSFRS ---------------------------------
# //////////////////////////////////////////////////////////////////



# ------------------------------------------------------------------
# Stage v2 - Clean raw ALSFRS data
# ------------------------------------------------------------------

def clean_data(csv_file):
    """
    Remove rows with missing core ALSFRS items and consolidate the two
    cutting subscale columns (Q5a/Q5b) into a single Q5_Cutting column.

    The ALSFRS form includes two mutually exclusive cutting items:
        Q5a_Cutting_without_Gastrostomy  - used when no gastrostomy tube is present
        Q5b_Cutting_with_Gastrostomy     - used when a gastrostomy tube is present
    Rows where both are missing are dropped. When only Q5a is missing, its
    value is filled from Q5b, then Q5b is dropped and Q5a is renamed Q5_Cutting.

    Columns dropped:
        Mode_of_Administration  - administrative metadata, not used in analysis
        ALSFRS_Responded_By     - respondent identity, not used in analysis
        Q5b_Cutting_with_Gastrostomy  - merged into Q5_Cutting

    Parameters
    ----------
    csv_file : str
        Path to the raw PROACT_ALSFRS.csv file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame (-> saved as PROACT_ALSFRS_v2.csv).
    """
    df = pd.read_csv(csv_file, low_memory=False)

    # Core motor/functional items that must all be present for a row to be usable
    columns_to_check = [
        'Q1_Speech', 'Q2_Salivation', 'Q3_Swallowing', 'Q4_Handwriting',
        'Q6_Dressing_and_Hygiene', 'Q7_Turning_in_Bed', 'Q8_Walking', 'Q9_Climbing_Stairs'
    ]
    df = df.dropna(subset=columns_to_check)

    # Drop rows where neither cutting item is available
    df = df.dropna(subset=['Q5a_Cutting_without_Gastrostomy', 'Q5b_Cutting_with_Gastrostomy'], how='all')

    # For patients with a gastrostomy, Q5a is NaN: fill it from Q5b
    df['Q5a_Cutting_without_Gastrostomy'] = df.apply(
        lambda row: row['Q5b_Cutting_with_Gastrostomy']
        if pd.isna(row['Q5a_Cutting_without_Gastrostomy'])
        else row['Q5a_Cutting_without_Gastrostomy'],
        axis=1
    )
    df = df.drop(columns=['Q5b_Cutting_with_Gastrostomy'])
    df = df.rename(columns={'Q5a_Cutting_without_Gastrostomy': 'Q5_Cutting'})

    df = df.drop(columns=['Mode_of_Administration', 'ALSFRS_Responded_By'])

    return df


cleaned_file = clean_data(proact_path + '/PROACT_ALSFRS.csv')
cleaned_file.to_csv(data_path + '/PROACT_ALSFRS_v2.csv', index=False)





# ------------------------------------------------------------------
# Stage v3 - Remove structurally inconsistent rows
# ------------------------------------------------------------------

def value_checker(csv_file):
    """
    Drop rows where the reported total score is present but its required
    component items are missing, and rows where both total scores are absent.

    Consistency rules applied:
        - If ALSFRS_Total is recorded, Q10_Respiratory must also be present.
        - If ALSFRS_R_Total is recorded, all three R-subscale items
          (R_1_Dyspnea, R_2_Orthopnea, R_3_Respiratory_Insufficiency)
          must be present.
        - Rows where both ALSFRS_Total and ALSFRS_R_Total are missing are removed
          as they carry no usable score information.

    Parameters
    ----------
    csv_file : str
        Path to PROACT_ALSFRS_v2.csv.

    Returns
    -------
    pd.DataFrame
        Structurally consistent DataFrame (-> saved as PROACT_ALSFRS_v3.csv).
    """
    df = pd.read_csv(csv_file, low_memory=False)

    # ALSFRS_Total requires Q10_Respiratory to be computable
    df = df[~(df['ALSFRS_Total'].notna() & df['Q10_Respiratory'].isna())]

    # ALSFRS_R_Total requires all three respiratory R-items
    df = df[~(
        df['ALSFRS_R_Total'].notna() &
        (df['R_1_Dyspnea'].isna() | df['R_2_Orthopnea'].isna() | df['R_3_Respiratory_Insufficiency'].isna())
    )]

    # Rows with no total score at all are not usable
    df = df[~(df['ALSFRS_Total'].isna() & df['ALSFRS_R_Total'].isna())]

    return df


validated_file = value_checker(data_path + '/PROACT_ALSFRS_v2.csv')
validated_file.to_csv(data_path + '/PROACT_ALSFRS_v3.csv', index=False)





# ------------------------------------------------------------------
# Stage v4 - Correct arithmetic errors in total scores
# ------------------------------------------------------------------

def total_score_checker(csv_file):
    """
    Detect and correct rows where ALSFRS_Total or ALSFRS_R_Total does not
    match the sum of its component items.

    Discrepant rows are exported to a separate audit file before correction
    so that the nature and frequency of data entry errors can be reported.

    Correction logic:
        ALSFRS_Total   = Q1 + Q2 + Q3 + Q4 + Q5 + Q6 + Q7 + Q8 + Q9 + Q10
        ALSFRS_R_Total = Q1 + Q2 + Q3 + Q4 + Q5 + Q6 + Q7 + Q8 + Q9
                         + R_1 + R_2 + R_3

    Parameters
    ----------
    csv_file : str
        Path to PROACT_ALSFRS_v3.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with corrected total scores
        (-> saved as PROACT_ALSFRS_v4.csv).
        Discrepant rows are also saved to PROACT_ALSFRS_value_error.csv.
    """
    df = pd.read_csv(csv_file, low_memory=False)

    # Recompute both totals from their constituent items
    ALSFRS_Total_calculated = df[[
        'Q1_Speech', 'Q2_Salivation', 'Q3_Swallowing', 'Q4_Handwriting',
        'Q5_Cutting', 'Q6_Dressing_and_Hygiene', 'Q7_Turning_in_Bed',
        'Q8_Walking', 'Q9_Climbing_Stairs', 'Q10_Respiratory'
    ]].sum(axis=1)

    ALSFRS_R_Total_calculated = df[[
        'Q1_Speech', 'Q2_Salivation', 'Q3_Swallowing', 'Q4_Handwriting',
        'Q5_Cutting', 'Q6_Dressing_and_Hygiene', 'Q7_Turning_in_Bed',
        'Q8_Walking', 'Q9_Climbing_Stairs',
        'R_1_Dyspnea', 'R_2_Orthopnea', 'R_3_Respiratory_Insufficiency'
    ]].sum(axis=1)

    # Identify rows where the recorded total differs from the computed total
    incorrect_ALSFRS_Total   = df[(df['ALSFRS_Total'].notna())   & (df['ALSFRS_Total']   != ALSFRS_Total_calculated)]
    incorrect_ALSFRS_R_Total = df[(df['ALSFRS_R_Total'].notna()) & (df['ALSFRS_R_Total'] != ALSFRS_R_Total_calculated)]

    print(f'Rows with incorrect ALSFRS_Total: {incorrect_ALSFRS_Total.shape[0]}')
    print(f'Rows with incorrect ALSFRS_R_Total: {incorrect_ALSFRS_R_Total.shape[0]}')

    # Export discrepant rows for audit / reporting
    df_incorrect_scores = pd.concat([incorrect_ALSFRS_Total, incorrect_ALSFRS_R_Total]).drop_duplicates()
    df_incorrect_scores.to_csv(data_path + '/PROACT_ALSFRS_value_error.csv', index=False)

    # Overwrite erroneous totals with recomputed values
    df.loc[incorrect_ALSFRS_Total.index,   'ALSFRS_Total']   = ALSFRS_Total_calculated[incorrect_ALSFRS_Total.index]
    df.loc[incorrect_ALSFRS_R_Total.index, 'ALSFRS_R_Total'] = ALSFRS_R_Total_calculated[incorrect_ALSFRS_R_Total.index]

    return df


df_corrected_scores = total_score_checker(data_path + '/PROACT_ALSFRS_v3.csv')
df_corrected_scores.to_csv(data_path + '/PROACT_ALSFRS_v4.csv', index=False)





# ------------------------------------------------------------------
# Stage v5 - Compute domain subscores
# ------------------------------------------------------------------

def compute_domain_scores(csv_file):
    """
    Derive four functional domain subscores from individual ALSFRS items
    and append them as new columns.

    Subscores computed:
        Bulbar_Score       = Q1 + Q2 + Q3                     (range 0-12)
        Upper_Limb_Score   = Q4 + Q5 + Q6                     (range 0-12)
        Lower_Limb_Score   = Q7 + Q8 + Q9                     (range 0-12)
        Respiratory_Score:
            If ALSFRS_R_Total is available -> R_1 + R_2 + R_3 (range 0-12)
            If only ALSFRS_Total is available -> Q10 * 3       (range 0-12)
            The Q10 * 3 approximation maps the single 0-4 respiratory item
            onto the same 0-12 range as the three R-subscale items.

    Parameters
    ----------
    csv_file : str
        Path to PROACT_ALSFRS_v4.csv.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with four subscore columns appended
        (-> saved as PROACT_ALSFRS_v5.csv).
    """
    df = pd.read_csv(csv_file, low_memory=False)

    df['Bulbar_Score']     = df[['Q1_Speech', 'Q2_Salivation', 'Q3_Swallowing']].sum(axis=1)
    df['Upper_Limb_Score'] = df[['Q4_Handwriting', 'Q5_Cutting', 'Q6_Dressing_and_Hygiene']].sum(axis=1)
    df['Lower_Limb_Score'] = df[['Q7_Turning_in_Bed', 'Q8_Walking', 'Q9_Climbing_Stairs']].sum(axis=1)

    # Default: use the three R-items (ALSFRS-R respiratory subscore)
    df['Respiratory_Score'] = df[['R_1_Dyspnea', 'R_2_Orthopnea', 'R_3_Respiratory_Insufficiency']].sum(axis=1)

    # Override with Q10 * 3 for rows that only have ALSFRS (not ALSFRS-R)
    df.loc[df['ALSFRS_Total'].notna(), 'Respiratory_Score'] = (
        df.loc[df['ALSFRS_Total'].notna(), 'Q10_Respiratory'] * 3
    )

    return df


domain_scores_file = compute_domain_scores(data_path + '/PROACT_ALSFRS_v4.csv')
domain_scores_file.to_csv(data_path + '/PROACT_ALSFRS_v5.csv', index=False)





# ------------------------------------------------------------------
# Diagnostic - Evaluate cross-scale respiratory imputation strategies
# ------------------------------------------------------------------

def evaluate_imputation_strategies(csv_file):
    """
    Benchmark candidate formulas for converting between ALSFRS Q10_Respiratory
    and the ALSFRS-R three-item respiratory subscore (R_1 + R_2 + R_3),
    using patients who have both scores recorded as a ground-truth reference.

    For each candidate formula, the mean absolute difference (MAD) and standard
    deviation are printed to allow selection of the best approximation for use
    in the imputation step (stage v6_filled).

    Two imputation directions are evaluated:
        ALSFRS -> ALSFRS-R: predicting (R_1 + R_2 + R_3) from Q10
            Candidates: Q10*3, Q10*2.5, Q10*3.5
        ALSFRS-R -> ALSFRS: predicting Q10 from (R_1 + R_2 + R_3)
            Candidates: mean of R-items (integer and float),
                        individual R-items alone, and pairwise means

    The subset of patients with both scores is also saved as an audit file.

    Parameters
    ----------
    csv_file : str
        Path to PROACT_ALSFRS_v5.csv.

    Returns
    -------
    pd.DataFrame
        Subset of rows where both ALSFRS_Total and ALSFRS_R_Total are present
        (-> saved as PROACT_ALSFRS_list_both_score.csv).
    """
    df = pd.read_csv(csv_file, low_memory=False)

    # Restrict to patients with both scale totals for a fair ground-truth comparison
    df = df[df['ALSFRS_Total'].notna() & df['ALSFRS_R_Total'].notna()]

    print("--- Imputing ALSFRS-R from ALSFRS (predicting R_1+R_2+R_3 from Q10) ---")

    # Candidate multipliers applied to Q10 to approximate the 0-12 R-subscore
    Q10x3_diff   = abs(df['Q10_Respiratory'] * 3   - (df['R_1_Dyspnea'] + df['R_2_Orthopnea'] + df['R_3_Respiratory_Insufficiency']))
    Q10x2_5_diff = abs(df['Q10_Respiratory'] * 2.5 - (df['R_1_Dyspnea'] + df['R_2_Orthopnea'] + df['R_3_Respiratory_Insufficiency']))
    Q10x3_5_diff = abs(df['Q10_Respiratory'] * 3.5 - (df['R_1_Dyspnea'] + df['R_2_Orthopnea'] + df['R_3_Respiratory_Insufficiency']))

    print(f'Q10*3   vs R_sum (/12): MAD={Q10x3_diff.mean():.4f}   STD={Q10x3_diff.std():.4f}')
    print(f'Q10*2.5 vs R_sum (/12): MAD={Q10x2_5_diff.mean():.4f} STD={Q10x2_5_diff.std():.4f}')
    print(f'Q10*3.5 vs R_sum (/12): MAD={Q10x3_5_diff.mean():.4f} STD={Q10x3_5_diff.std():.4f}')

    print("\n--- Imputing ALSFRS from ALSFRS-R (predicting Q10 from R-items) ---")

    R_sum  = df['R_1_Dyspnea'] + df['R_2_Orthopnea'] + df['R_3_Respiratory_Insufficiency']

    # Mean of all three R-items (integer-truncated and float)
    R_sumint_diff = abs((R_sum / 3).astype(int) - df['Q10_Respiratory'])
    R_sum_diff    = abs((R_sum / 3)              - df['Q10_Respiratory'])
    print(f'floor(R_sum/3) vs Q10 (/4): MAD={R_sumint_diff.mean():.4f} STD={R_sumint_diff.std():.4f}')
    print(f'R_sum/3        vs Q10 (/4): MAD={R_sum_diff.mean():.4f}    STD={R_sum_diff.std():.4f}')

    # Individual R-items used alone as a proxy for Q10
    R1_Q10_diff = abs(df['R_1_Dyspnea']                  - df['Q10_Respiratory'])
    R2_Q10_diff = abs(df['R_2_Orthopnea']                 - df['Q10_Respiratory'])
    R3_Q10_diff = abs(df['R_3_Respiratory_Insufficiency'] - df['Q10_Respiratory'])
    print(f'R1 vs Q10 (/4): MAD={R1_Q10_diff.mean():.4f} STD={R1_Q10_diff.std():.4f}')
    print(f'R2 vs Q10 (/4): MAD={R2_Q10_diff.mean():.4f} STD={R2_Q10_diff.std():.4f}')
    print(f'R3 vs Q10 (/4): MAD={R3_Q10_diff.mean():.4f} STD={R3_Q10_diff.std():.4f}')

    # Pairwise means of R-items (integer-truncated and float)
    R12int_diff = abs(((df['R_1_Dyspnea'] + df['R_2_Orthopnea'])                  / 2).astype(int) - df['Q10_Respiratory'])
    R12_diff    = abs(((df['R_1_Dyspnea'] + df['R_2_Orthopnea'])                  / 2)             - df['Q10_Respiratory'])
    R13int_diff = abs(((df['R_1_Dyspnea'] + df['R_3_Respiratory_Insufficiency'])  / 2).astype(int) - df['Q10_Respiratory'])
    R13_diff    = abs(((df['R_1_Dyspnea'] + df['R_3_Respiratory_Insufficiency'])  / 2)             - df['Q10_Respiratory'])
    R23int_diff = abs(((df['R_2_Orthopnea'] + df['R_3_Respiratory_Insufficiency'])/ 2).astype(int) - df['Q10_Respiratory'])
    R23_diff    = abs(((df['R_2_Orthopnea'] + df['R_3_Respiratory_Insufficiency'])/ 2)             - df['Q10_Respiratory'])

    print(f'floor((R1+R2)/2) vs Q10 (/4): MAD={R12int_diff.mean():.4f} STD={R12int_diff.std():.4f}')
    print(f'(R1+R2)/2        vs Q10 (/4): MAD={R12_diff.mean():.4f}    STD={R12_diff.std():.4f}')
    print(f'floor((R1+R3)/2) vs Q10 (/4): MAD={R13int_diff.mean():.4f} STD={R13int_diff.std():.4f}')
    print(f'(R1+R3)/2        vs Q10 (/4): MAD={R13_diff.mean():.4f}    STD={R13_diff.std():.4f}')
    print(f'floor((R2+R3)/2) vs Q10 (/4): MAD={R23int_diff.mean():.4f} STD={R23int_diff.std():.4f}')
    print(f'(R2+R3)/2        vs Q10 (/4): MAD={R23_diff.mean():.4f}    STD={R23_diff.std():.4f}')

    return df


benchmark_results = evaluate_imputation_strategies(data_path + '/PROACT_ALSFRS_v5.csv')
benchmark_results.to_csv(data_path + '/PROACT_ALSFRS_list_both_score.csv', index=False)





# ------------------------------------------------------------------
# Stage v6 - Add per-patient observation count
# ------------------------------------------------------------------

def observation_counter_alsfrs(csv_file):
    """
    Add an `observation_count` column recording how many rows exist per patient.

    Each row represents one ALSFRS assessment visit. This count therefore
    reflects the number of valid visits retained after all cleaning steps.

    Parameters
    ----------
    csv_file : str
        Path to PROACT_ALSFRS_v5.csv.

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


df_counter = observation_counter_alsfrs(data_path + '/PROACT_ALSFRS_v5.csv')
df_counter.to_csv(data_path + '/PROACT_ALSFRS_v6.csv', index=False)





# ------------------------------------------------------------------
# Stage v6_filled - Impute missing total scores across scales
# ------------------------------------------------------------------

def impute_total_scores(csv_file):
    """
    Impute whichever of ALSFRS_Total or ALSFRS_R_Total is missing, using the
    conversion formulas selected on the basis of the diagnostic step above.

    Imputation formulas:
        Missing ALSFRS_R_Total (have ALSFRS_Total):
            ALSFRS_R_Total = ALSFRS_Total - Q10 + (Q10 * 3)
            i.e. replace the single Q10 item with its three-item equivalent.

        Missing ALSFRS_Total (have ALSFRS_R_Total):
            ALSFRS_Total = ALSFRS_R_Total - R_1 - R_2 - R_3
                           + floor((R_1 + R_2 + R_3) / 3)
            i.e. replace the three R-items with the integer mean mapped onto
            the 0-4 Q10 scale.

    Parameters
    ----------
    csv_file : str
        Path to PROACT_ALSFRS_v6.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with both total score columns fully populated where at
        least one was available (-> saved as PROACT_ALSFRS_v6_filled.csv).
    """
    df = pd.read_csv(csv_file, low_memory=False)

    print(f'Missing ALSFRS_Total before imputation:   {df["ALSFRS_Total"].isna().sum()}')
    print(f'Missing ALSFRS_R_Total before imputation: {df["ALSFRS_R_Total"].isna().sum()}')

    # Mask: rows that need ALSFRS_R_Total filled in
    mask_r = df['ALSFRS_R_Total'].isna() & df['ALSFRS_Total'].notna()
    df.loc[mask_r, 'ALSFRS_R_Total'] = (
        df.loc[mask_r, 'ALSFRS_Total']
        - df.loc[mask_r, 'Q10_Respiratory']
        + df.loc[mask_r, 'Q10_Respiratory'] * 3
    )

    # Mask: rows that need ALSFRS_Total filled in
    mask_t = df['ALSFRS_Total'].isna() & df['ALSFRS_R_Total'].notna()
    R_sum = (
        df.loc[mask_t, 'R_1_Dyspnea']
        + df.loc[mask_t, 'R_2_Orthopnea']
        + df.loc[mask_t, 'R_3_Respiratory_Insufficiency']
    )
    df.loc[mask_t, 'ALSFRS_Total'] = (
        df.loc[mask_t, 'ALSFRS_R_Total']
        - df.loc[mask_t, 'R_1_Dyspnea']
        - df.loc[mask_t, 'R_2_Orthopnea']
        - df.loc[mask_t, 'R_3_Respiratory_Insufficiency']
        + (R_sum / 3).astype(int)
    )

    print(f'Missing ALSFRS_Total after imputation:   {df["ALSFRS_Total"].isna().sum()}')
    print(f'Missing ALSFRS_R_Total after imputation: {df["ALSFRS_R_Total"].isna().sum()}')

    return df


imputed_file = impute_total_scores(data_path + '/PROACT_ALSFRS_v6.csv')
imputed_file.to_csv(data_path + '/PROACT_ALSFRS_v6_filled.csv', index=False)





# ------------------------------------------------------------------
# Stage v7 / v7_filled - Reshape to wide format
# ------------------------------------------------------------------

def reshape_to_wide_format(csv_file):
    """
    Reshape the long-format ALSFRS data (multiple rows per patient) into a
    wide-format DataFrame (one row per patient) where each visit's values are
    stored in sequentially prefixed columns.

    Visits are sorted by ALSFRS_Delta (time since symptom onset or study
    baseline) before pivoting, so that column prefix 1_ always corresponds
    to the earliest recorded visit.

    Column naming convention:
        {visit_index}_{original_column_name}
        e.g. "1_ALSFRS_Delta", "2_ALSFRS_Total", "3_Bulbar_Score"

    Parameters
    ----------
    csv_file : str
        Path to PROACT_ALSFRS_v6.csv or PROACT_ALSFRS_v6_filled.csv.

    Returns
    -------
    pd.DataFrame
        Wide-format patient-level DataFrame
        (-> saved as PROACT_ALSFRS_v7.csv or PROACT_ALSFRS_v7_filled.csv).
    """
    df = pd.read_csv(csv_file, low_memory=False)
    result_list = []

    for subject_id, group in df.groupby('subject_id'):

        # Sort visits chronologically within each patient
        group = group.sort_values(by='ALSFRS_Delta').reset_index(drop=True)

        patient_data = {
            'subject_id': subject_id,
            'observation_count': group.shape[0],
        }

        for i, row in group.iterrows():
            prefix = f"{i + 1}_"
            # ALSFRS_Delta is placed first within each visit block
            patient_data[f"{prefix}ALSFRS_Delta"] = row['ALSFRS_Delta']
            for col in df.columns:
                if col not in ['subject_id', 'ALSFRS_Delta', 'observation_count']:
                    patient_data[f"{prefix}{col}"] = row[col]

        result_list.append(patient_data)

    df_merged = pd.DataFrame(result_list)

    return df_merged


reshaped_file = reshape_to_wide_format(data_path + '/PROACT_ALSFRS_v6.csv')
reshaped_file.to_csv(data_path + '/PROACT_ALSFRS_v7.csv', index=False)

reshaped_imputed_file = reshape_to_wide_format(data_path + '/PROACT_ALSFRS_v6_filled.csv')
reshaped_imputed_file.to_csv(data_path + '/PROACT_ALSFRS_v7_filled.csv', index=False)





# ------------------------------------------------------------------
# Stage v8 / v8_filled - Add 'ALS_' prefix to all feature columns
# ------------------------------------------------------------------

def rename_all_columns(csv_file):
    """
    Prefix every feature column with 'ALS_' to namespace the ALSFRS variables
    when merging with other PROACT sub-datasets.

    `subject_id` is the join key and is left unchanged.

    Parameters
    ----------
    csv_file : str
        Path to PROACT_ALSFRS_v7.csv or PROACT_ALSFRS_v7_filled.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns
        (-> saved as PROACT_ALSFRS_v8.csv or PROACT_ALSFRS_v8_filled.csv).
    """
    df = pd.read_csv(csv_file, low_memory=False)
    df = df.rename(columns={col: f'ALS_{col}' for col in df.columns if col != 'subject_id'})
    return df


df_renamed = rename_all_columns(data_path + '/PROACT_ALSFRS_v7.csv')
df_renamed.to_csv(data_path + '/PROACT_ALSFRS_v8.csv', index=False)

df_renamed_filled = rename_all_columns(data_path + '/PROACT_ALSFRS_v7_filled.csv')
df_renamed_filled.to_csv(data_path + '/PROACT_ALSFRS_v8_filled.csv', index=False)