"""
PROACT Interval Statistical Feature Extraction
=====================================================
This script transforms the wide-format temporal PROACT tables into
interval-aggregated feature matrices by computing a set of descriptive
statistics for every variable within each 90-day time interval, under
both temporal reference frames (study-inclusion and first-symptom alignment).

This corresponds to the statistical aggregation step described in
Section IV.B of the paper. For each (patient, variable, interval) triplet,
up to 18 statistics are computed for numerical variables and modality counts
plus a majority label for categorical variables.

Output column naming convention:
    {prefix}_{variable}_{interval_start}_{interval_end}_{statistic}
    e.g. 'ALS_ALSFRS_R_Total_0_90_Mean', 'HAN_Test_Result_Most_Affected_Side_0_90_Majority'

Statistics computed for numerical variables (Section IV.B.1):
    Central tendency:  Mean, Median
    Dispersion:        Std, Var, MAD, Amplitude
    Position:          Min, Max
    Temporal dynamics: Central (value closest to interval midpoint),
                       First, Last, Slope, MeanDiff, MaxDiff, MinDiff, MedianDiff
    Count:             number of observations in the interval

    Excluded (insufficient reliability at typical sample sizes):
        Mode, TrimMean  - require n >= 5
        CV              - unstable when mean is near 0
        Skewness, Kurtosis, Q25, Q75, IQR, Entropy - require n >= 5 to 10

Statistics computed for categorical variables (Section IV.B.2):
    Left/Right/Equal:  Count_Left, Count_Right, Count_Equal, Majority
    True/False:        Count_True, Count_False, Majority
    Test_Unit (LABS):  Majority (first non-null value; units are constant per test)
    Count:             total number of observations in the interval

Tables processed:
    ALSFRS, FVC, HANDGRIPSTRENGTH, LABS, MUSCLESTRENGTH, SVC, VITALSIGNS
    (each in both study-inclusion-aligned and first-symptoms-aligned versions)

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
import re
import time
from tqdm import tqdm





# ------------------------------------------------------------------
# Custom median helper
# ------------------------------------------------------------------

def median_reelle(arr, mode='normal'):
    """
    Compute the median of an array with configurable tie-breaking behaviour
    for even-length arrays.

    For odd-length arrays the result is always the unique central value.
    For even-length arrays, three modes are available:

        'lower'  - return the lower of the two central values
        'upper'  - return the upper of the two central values
        'normal' - return the arithmetic mean of the two central values
                   (equivalent to numpy.median, the default)

    This wrapper exists to allow explicit control over tie-breaking in
    contexts where a strictly observed data point is preferred over an
    interpolated value (e.g. clinical scores that can only take integer values).

    Parameters
    ----------
    arr  : array-like  Input values (will be sorted internally).
    mode : str         Tie-breaking strategy (default: 'normal').

    Returns
    -------
    float
        The median value, or NaN if arr is empty.
    """
    arr = np.sort(np.asarray(arr, dtype=float))
    n   = len(arr)
    if n == 0:
        return np.nan

    mid = n // 2
    if n % 2 == 1:
        return arr[mid]
    if mode == 'lower':
        return arr[mid - 1]
    elif mode == 'upper':
        return arr[mid]
    elif mode == 'normal':
        return np.median(arr)
    else:
        raise ValueError("Unknown mode: choose 'lower', 'upper', or 'normal'")





# ------------------------------------------------------------------
# Generic intra-interval statistical aggregation
# ------------------------------------------------------------------

def generic_intervals(
    file_path,
    prefix,
    observation_count_col,
    delta_suffix,
    output_csv,
    interval_length=90,
    negative_intervals=None,
    positive_intervals=None,
    stats_intra_interval=True
):
    """
    Aggregate all observations within each 90-day interval into a set of
    descriptive statistics, producing a wide-format patient-level feature matrix.

    For each patient, every recorded visit is assigned to the 90-day interval
    it falls into:

        interval_index = floor(Delta / interval_length)

    Observations whose interval index falls outside [-negative_intervals,
    positive_intervals] are discarded. Pre-study visits (negative Delta values)
    are excluded by default (negative_intervals=0).

    Variable detection:
        Column names matching the pattern {prefix}_{visit_index}_{variable}
        are parsed to extract the unique variable names. Columns that do not
        match this pattern (e.g. observation_count) are treated as 
        complementary columns and carried over unchanged to the output.

    Value type handling:
        - Numeric (int, float): full or reduced statistical summary (see below)
        - Boolean: converted to the strings "True"/"False" and treated as
          categorical, because pandas may read booleans as numeric types
        - String: categorical summary (modality counts + majority label)

    Statistical summary modes (controlled by stats_intra_interval):
        True  (full):    Mean, Median, Std, Var, MAD, Amplitude, Min, Max,
                         Central, First, Last, Slope, MeanDiff, MaxDiff,
                         MinDiff, MedianDiff, Count
        False (reduced): Mean, Median, Min, Max, Central only

    Excluded statistics (commented out in the code) and their minimum reliable
    sample size:
        Mode, TrimMean  (n >= 5), CV (unstable near mean=0),
        Skewness, Kurtosis (n >= 5-8), Q25, Q75, IQR (n >= 5),
        Entropy (n >= 10)

    Column sort order in the output:
        1. subject_id
        2. complementary columns (in their original CSV order)
        3. statistical columns, sorted by (interval_start, variable_order, stat_order)

    Parameters
    ----------
    file_path             : str   Path to the input wide-format CSV file.
    prefix                : str   Three-letter table prefix (e.g. 'ALS', 'FVC').
    observation_count_col : str   Name of the per-patient observation count column.
    delta_suffix          : str   Suffix identifying the Delta column within each
                                  visit block (e.g. 'ALSFRS_Delta').
    output_csv            : str   Path for the output interval-statistics CSV.
    interval_length       : int   Width of each interval in days (default: 90).
    negative_intervals    : int   Number of pre-baseline intervals to include.
                                  None = include all; 0 = exclude pre-study visits.
    positive_intervals    : int   Maximum post-baseline interval index to include.
                                  None = include all (commented out in calls below
                                  to retain all available follow-up data).
    stats_intra_interval  : bool  If True, compute the full statistical summary;
                                  if False, compute only Mean, Median, Min, Max,
                                  Central (faster, smaller output).

    Returns
    -------
    pd.DataFrame
        Wide-format patient-level feature matrix with one row per patient and
        one column per (variable, interval, statistic) triplet.
        Also saved to output_csv.
    """
    start_time = time.time()
    print(f"Processing: {file_path}")
    df         = pd.read_csv(file_path, low_memory=False)
    total_rows = len(df)

    # ------------------------------------------------------------------
    # Detect variable names and complementary columns from the CSV schema
    # ------------------------------------------------------------------
    all_columns = df.columns.tolist()

    # Extract variable names from prefixed visit columns (e.g. ALS_1_ALSFRS_Delta)
    # preserving the order they first appear in the CSV
    variable_names_raw = [
        c for c in all_columns
        if re.match(rf"{prefix}_(\d+)_(.+)", c) and not c.endswith(delta_suffix)
    ]
    variable_names = [re.match(rf"{prefix}_(\d+)_(.+)", c).group(2) for c in variable_names_raw]
    # Deduplicate while preserving first-occurrence order
    variable_names = sorted(set(variable_names), key=lambda x: variable_names.index(x))
    print(f"Variables detected: {variable_names}")

    # Columns not part of the visit-indexed block (e.g. DominantHand) are
    # carried over unchanged to the output
    complementary_cols = [
        c for c in all_columns
        if not re.match(rf"{prefix}_(\d+)_(.+)", c)
        and c not in variable_names
        and not c.endswith(delta_suffix)
        and c != "subject_id"
    ]
    print(f"Complementary columns: {complementary_cols}")

    # Interval index bounds derived from the parameters
    min_interval = -negative_intervals if negative_intervals is not None else None
    max_interval =  positive_intervals if positive_intervals is not None else None

    # Reference order for statistics in the final column sort
    stat_order = [
        "Mean", "Median", "Mode", "TrimMean",
        "Std", "Var", "MAD", "Amplitude", "CV",
        "Skewness", "Kurtosis",
        "Min", "Max", "Q25", "Q75", "IQR",
        "Central", "First", "Last", "Slope",
        "MeanDiff", "MaxDiff", "MinDiff", "MedianDiff",
        "Entropy",
        "Count_Left", "Count_Right", "Count_Equal",
        "Count_False", "Count_True", "Majority",
        "Count",
    ]

    records = []

    print(f"Processing {total_rows} patients...")
    for _, row in tqdm(df.iterrows(), total=total_rows):
        record            = {"subject_id": row["subject_id"]}
        observation_count = int(row[observation_count_col]) if pd.notna(row[observation_count_col]) else 0

        # Accumulate values and their associated Delta timestamps per variable
        # per interval, keeping numeric and categorical observations separate
        interval_numeric_values = {var: {} for var in variable_names}
        interval_numeric_deltas = {var: {} for var in variable_names}
        interval_text_values    = {var: {} for var in variable_names}
        interval_text_deltas    = {var: {} for var in variable_names}

        for i in range(1, observation_count + 1):
            delta_col = f"{prefix}_{i}_{delta_suffix}"
            if pd.notna(row.get(delta_col, np.nan)):
                delta_val      = float(row[delta_col])
                interval_index = int(np.floor(delta_val / interval_length))

                # Apply interval bounds filtering
                if min_interval is not None and interval_index < min_interval:
                    continue
                if max_interval is not None and interval_index > max_interval:
                    continue

                for var in variable_names:
                    value_col = f"{prefix}_{i}_{var}"
                    value     = row.get(value_col, np.nan)
                    if pd.notna(value):
                        if isinstance(value, bool):
                            # Booleans are converted to strings to prevent
                            # pandas from treating True/False as 1/0 numerically
                            interval_text_values[var].setdefault(interval_index, []).append(str(value))
                            interval_text_deltas[var].setdefault(interval_index, []).append(delta_val)
                        elif isinstance(value, (int, float)):
                            interval_numeric_values[var].setdefault(interval_index, []).append(float(value))
                            interval_numeric_deltas[var].setdefault(interval_index, []).append(delta_val)
                        elif isinstance(value, str):
                            interval_text_values[var].setdefault(interval_index, []).append(value.strip())
                            interval_text_deltas[var].setdefault(interval_index, []).append(delta_val)

        # ==============================================================
        # Numerical variable statistics per interval
        # ==============================================================
        for var, intervals in interval_numeric_values.items():
            for interval_index, values in intervals.items():
                start    = int(interval_index * interval_length)
                end      = int((interval_index + 1) * interval_length)
                col_base = f"{prefix}_{var}_{start}_{end}"

                arr = np.array(values, dtype=float)
                n   = len(arr)

                if n == 0:
                    record[f"{col_base}_Count"] = 0
                    continue

                # ============================================================
                # 1. CENTRAL TENDENCY
                # ============================================================
                mean_val   = float(np.mean(arr))                          # Mean   - arithmetic centre; computable from n=1, meaningful from n>=3
                median_val = float(median_reelle(arr, mode='normal'))     # Median - central value;     computable from n=1, meaningful from n>=3

                # Excluded (minimum reliable sample size too high for typical interval densities):
                # mode_val         = float(stats.mode(arr, keepdims=True)[0][0])       # Mode        - most frequent value;   n>=5
                # trimmed_mean_val = float(stats.trim_mean(arr, 0.1)) if n>=3 else mean_val  # TrimMean - trimmed mean (10%); n>=5

                # ============================================================
                # 2. DISPERSION / VARIABILITY
                # ============================================================
                std_val       = float(np.std(arr, ddof=0))                          if n >= 2 else 0  # Std       - spread around the mean;         n>=2, meaningful n>=3
                var_val       = float(np.var(arr, ddof=0))                          if n >= 2 else 0  # Var       - variance;                       n>=2, meaningful n>=3
                mad_val       = float(np.median(np.abs(arr - np.median(arr))))      if n >= 2 else 0  # MAD       - median absolute deviation;      n>=2
                amplitude_val = float(np.max(arr) - np.min(arr))                    if n >= 2 else 0  # Amplitude - range (max - min);             n>=2

                # Excluded:
                # cv_val = float(std_val / mean_val) if mean_val != 0 else np.nan  # CV - coefficient of variation; unstable when mean ≈ 0, n>=5

                # ============================================================
                # 3. DISTRIBUTION SHAPE
                # ============================================================
                # Excluded (require n>=5-8 for reliable estimates):
                # skew_val     = float(stats.skew(arr, bias=False))       # Skewness - asymmetry;     n>=5-8
                # kurtosis_val = float(stats.kurtosis(arr, bias=False))   # Kurtosis - peakedness;    n>=5-8

                # ============================================================
                # 4. POSITION / RANK
                # ============================================================
                min_val = float(np.min(arr))  # Min - minimum value; computable from n=1, always meaningful
                max_val = float(np.max(arr))  # Max - maximum value; computable from n=1, always meaningful

                # Excluded (require n>=5 for reliable quantile estimates):
                # q25_val = float(np.percentile(arr, 25)) if n >= 3 else np.nan  # Q25 - 25th percentile; n>=5
                # q75_val = float(np.percentile(arr, 75)) if n >= 3 else np.nan  # Q75 - 75th percentile; n>=5
                # iqr_val = float(q75_val - q25_val)      if n >= 3 else np.nan  # IQR - interquartile range; n>=5

                # ============================================================
                # 5. TEMPORAL DYNAMICS
                # ============================================================
                deltas       = np.array(interval_numeric_deltas[var][interval_index], dtype=float)
                center_point = (start + end) / 2.0
                central_val  = arr[np.argmin(np.abs(deltas - center_point))]   # Central  - value closest to the interval midpoint; n>=1, meaningful n>=2
                first_val    = float(arr[0])                                   # First    - first observed value;                   n>=1, always meaningful
                last_val     = float(arr[-1])                                  # Last     - last observed value;                    n>=1, always meaningful

                slope_val = 0.0
                if n >= 2:
                    delta_diff = deltas[-1] - deltas[0]
                    slope_val  = (arr[-1] - arr[0]) / delta_diff if delta_diff != 0 else 0.0  # Slope - rate of change between first and last observation; n>=2, meaningful n>=3

                if n >= 2:
                    diff_arr    = np.diff(arr)
                    mean_diff   = float(np.mean(diff_arr))    # MeanDiff   - mean of successive differences;   n>=2, meaningful n>=3
                    max_diff    = float(np.max(diff_arr))     # MaxDiff    - max of successive differences;    n>=2, meaningful n>=3
                    min_diff    = float(np.min(diff_arr))     # MinDiff    - min of successive differences;    n>=2, meaningful n>=3
                    median_diff = float(np.median(diff_arr))  # MedianDiff - median of successive differences; n>=2, meaningful n>=3
                else:
                    mean_diff = max_diff = min_diff = median_diff = 0.0

                # Excluded:
                # hist_counts, _ = np.histogram(arr, bins='auto')
                # entropy_val = float(stats.entropy(hist_counts + 1e-6))  # Entropy - measure of disorder; n>=10

                # ============================================================
                # 6. WRITE STATISTICS TO THE OUTPUT RECORD
                # ============================================================
                if stats_intra_interval:
                    # Full statistical summary
                    record[f"{col_base}_Mean"]       = mean_val
                    record[f"{col_base}_Median"]     = median_val
                    # record[f"{col_base}_Mode"]     = mode_val
                    # record[f"{col_base}_TrimMean"] = trimmed_mean_val

                    record[f"{col_base}_Std"]        = std_val
                    record[f"{col_base}_Var"]        = var_val
                    record[f"{col_base}_MAD"]        = mad_val
                    record[f"{col_base}_Amplitude"]  = amplitude_val
                    # record[f"{col_base}_CV"]       = cv_val

                    # record[f"{col_base}_Skewness"] = skew_val
                    # record[f"{col_base}_Kurtosis"] = kurtosis_val

                    record[f"{col_base}_Min"]        = min_val
                    record[f"{col_base}_Max"]        = max_val
                    # record[f"{col_base}_Q25"]      = q25_val
                    # record[f"{col_base}_Q75"]      = q75_val
                    # record[f"{col_base}_IQR"]      = iqr_val

                    record[f"{col_base}_Central"]    = central_val
                    record[f"{col_base}_First"]      = first_val
                    record[f"{col_base}_Last"]       = last_val
                    record[f"{col_base}_Slope"]      = slope_val
                    record[f"{col_base}_MeanDiff"]   = mean_diff
                    record[f"{col_base}_MaxDiff"]    = max_diff
                    record[f"{col_base}_MinDiff"]    = min_diff
                    record[f"{col_base}_MedianDiff"] = median_diff
                    # record[f"{col_base}_Entropy"]  = entropy_val

                    record[f"{col_base}_Count"]      = n  # number of observations in this interval

                else:
                    # Reduced summary (faster, smaller output)
                    record[f"{col_base}_Mean"]    = mean_val
                    record[f"{col_base}_Median"]  = median_val
                    record[f"{col_base}_Min"]     = min_val
                    record[f"{col_base}_Max"]     = max_val
                    record[f"{col_base}_Central"] = central_val

        # ==============================================================
        # Categorical variable statistics per interval
        # ==============================================================
        for var, intervals in interval_text_values.items():
            for interval_index, values in intervals.items():
                start    = int(interval_index * interval_length)
                end      = int((interval_index + 1) * interval_length)
                col_base = f"{prefix}_{var}_{start}_{end}"

                arr = np.array(values, dtype=object)
                n   = len(arr)

                if n == 0:
                    record[f"{col_base}_Count"] = 0
                    continue

                # Laterality variable (Left / Right / Equal)
                if set(arr).issubset({"Left", "Right", "Equal"}):
                    count_left  = int(np.sum(arr == "Left"))
                    count_right = int(np.sum(arr == "Right"))
                    count_equal = int(np.sum(arr == "Equal"))
                    record[f"{col_base}_Count_Left"]  = count_left
                    record[f"{col_base}_Count_Right"] = count_right
                    record[f"{col_base}_Count_Equal"] = count_equal
                    # Majority ignores Equal ties; Equal is used only as a fallback
                    if count_left > count_right:
                        majority_val = "Left"
                    elif count_right > count_left:
                        majority_val = "Right"
                    else:
                        majority_val = "Equal"

                # Boolean variable (True / False)
                elif set(arr).issubset({"True", "False"}) or set(arr).issubset({True, False}):
                    count_false = int(np.sum(arr == "False") + np.sum(arr == False))
                    count_true  = int(np.sum(arr == "True")  + np.sum(arr == True))
                    record[f"{col_base}_Count_False"] = count_false
                    record[f"{col_base}_Count_True"]  = count_true
                    # In case of a tie, False is given priority (conservative assumption)
                    majority_val = "True" if count_true > count_false else "False"

                # Lab test unit: constant within a test, so the first non-null value
                # is taken as the majority rather than computing a frequency count
                elif re.search(r'_Test_Unit$', var):
                    non_null   = [v for v in arr if v not in (None, '', 'NaN', np.nan)]
                    majority_val = non_null[0] if non_null else np.nan

                else:
                    majority_val = np.nan

                record[f"{col_base}_Majority"] = majority_val
                record[f"{col_base}_Count"]    = n

        # Carry over patient-level complementary columns unchanged
        for col in complementary_cols:
            record[col] = row.get(col, np.nan)

        records.append(record)

    # ------------------------------------------------------------------
    # Build output DataFrame and sort columns
    # ------------------------------------------------------------------
    result = pd.DataFrame(records)

    def sort_key(col):
        """
        Sort columns in the output DataFrame:
            1. subject_id (always first)
            2. complementary columns in their original CSV order
            3. statistical columns sorted by (interval_start, variable_order,
               stat_order), grouping all statistics for the same variable and
               interval together
            4. any unrecognised columns last
        """
        if col == "subject_id":
            return (0, 0, 0, 0, 0)
        if col in complementary_cols:
            return (1, complementary_cols.index(col), 0, 0, 0)
        m = re.match(rf"{prefix}_(.+)_(\d+)_(\d+)_(.+)", col)
        if m:
            var, start, end, stat = m.groups()
            start, end  = int(start), int(end)
            var_index   = variable_names.index(var) if var in variable_names else 999
            stat_index  = stat_order.index(stat) if stat in stat_order else 999
            return (2, start, end, var_index, stat_index)
        return (3, 0, 0, 0, 0)

    result = result[sorted(result.columns, key=sort_key)]
    result.to_csv(output_csv, index=False)

    elapsed = time.time() - start_time
    print(f"Saved: {output_csv}")
    print(f"Elapsed: {int(elapsed // 60)} min {int(elapsed % 60)} s")

    return result










def run(DATA_PATH, FIRST_SYMPTOMS_PATH, INTERVALS_FULL_PATH):

    print("\n" * 3)
    print("=" * 60)
    print("FULL INTERVALS CREATION PIPELINE")
    print("=" * 60)



    # //////////////////////////////////////////////////////////
    # ------------------------- ALSFRS -------------------------
    # //////////////////////////////////////////////////////////

    # Study-inclusion-aligned intervals
    df_alsfrs = generic_intervals(
        file_path             = DATA_PATH + "/PROACT_ALSFRS_v8.csv",
        prefix                = "ALS",
        observation_count_col = "ALS_observation_count",
        delta_suffix          = "ALSFRS_Delta",
        output_csv            = INTERVALS_FULL_PATH + "/PROACT_ALSFRS_INTERVALS.csv",
        interval_length       = 90,
        negative_intervals    = 0,   # exclude pre-study visits
        # positive_intervals  = 4    # uncomment to cap at 12 months
    )

    # First-symptom-aligned intervals
    df_alsfrs_first_symptoms = generic_intervals(
        file_path             = FIRST_SYMPTOMS_PATH + "/PROACT_ALSFRS_FIRST_SYMPTOMS.csv",
        prefix                = "ALS",
        observation_count_col = "ALS_observation_count",
        delta_suffix          = "ALSFRS_Delta",
        output_csv            = INTERVALS_FULL_PATH + "/PROACT_ALSFRS_FIRST_SYMPTOMS_INTERVALS.csv",
        interval_length       = 90,
        negative_intervals    = 0,
        # positive_intervals  = 16   # uncomment to cap at 4 years post-onset
    )





    # ///////////////////////////////////////////////////////
    # ------------------------- FVC -------------------------
    # ///////////////////////////////////////////////////////

    df_fvc = generic_intervals(
        file_path             = DATA_PATH + "/PROACT_FVC_v7.csv",
        prefix                = "FVC",
        observation_count_col = "FVC_observation_count",
        delta_suffix          = "Forced_Vital_Capacity_Delta",
        output_csv            = INTERVALS_FULL_PATH + "/PROACT_FVC_INTERVALS.csv",
        interval_length       = 90,
        negative_intervals    = 0,
    )

    df_fvc_first_symptoms = generic_intervals(
        file_path             = FIRST_SYMPTOMS_PATH + "/PROACT_FVC_FIRST_SYMPTOMS.csv",
        prefix                = "FVC",
        observation_count_col = "FVC_observation_count",
        delta_suffix          = "Forced_Vital_Capacity_Delta",
        output_csv            = INTERVALS_FULL_PATH + "/PROACT_FVC_FIRST_SYMPTOMS_INTERVALS.csv",
        interval_length       = 90,
        negative_intervals    = 0,
    )





    # ////////////////////////////////////////////////////////////////////
    # ------------------------- HANDGRIPSTRENGTH -------------------------
    # ////////////////////////////////////////////////////////////////////

    df_handgripstrength = generic_intervals(
        file_path             = DATA_PATH + "/PROACT_HANDGRIPSTRENGTH_v8.csv",
        prefix                = "HAN",
        observation_count_col = "HAN_observation_count",
        delta_suffix          = "MS_Delta",
        output_csv            = INTERVALS_FULL_PATH + "/PROACT_HANDGRIPSTRENGTH_INTERVALS.csv",
        interval_length       = 90,
        negative_intervals    = 0,
    )

    df_handgripstrength_first_symptoms = generic_intervals(
        file_path             = FIRST_SYMPTOMS_PATH + "/PROACT_HANDGRIPSTRENGTH_FIRST_SYMPTOMS.csv",
        prefix                = "HAN",
        observation_count_col = "HAN_observation_count",
        delta_suffix          = "MS_Delta",
        output_csv            = INTERVALS_FULL_PATH + "/PROACT_HANDGRIPSTRENGTH_FIRST_SYMPTOMS_INTERVALS.csv",
        interval_length       = 90,
        negative_intervals    = 0,
    )





    # ////////////////////////////////////////////////////////
    # ------------------------- LABS -------------------------
    # ////////////////////////////////////////////////////////

    df_labs = generic_intervals(
        file_path             = DATA_PATH + "/PROACT_LABS_v10.csv",
        prefix                = "LAB",
        observation_count_col = "LAB_observation_count",
        delta_suffix          = "Laboratory_Delta",
        output_csv            = INTERVALS_FULL_PATH + "/PROACT_LABS_INTERVALS.csv",
        interval_length       = 90,
        negative_intervals    = 0,
    )

    df_labs_first_symptoms = generic_intervals(
        file_path             = FIRST_SYMPTOMS_PATH + "/PROACT_LABS_FIRST_SYMPTOMS.csv",
        prefix                = "LAB",
        observation_count_col = "LAB_observation_count",
        delta_suffix          = "Laboratory_Delta",
        output_csv            = INTERVALS_FULL_PATH + "/PROACT_LABS_FIRST_SYMPTOMS_INTERVALS.csv",
        interval_length       = 90,
        negative_intervals    = 0,
    )





    # //////////////////////////////////////////////////////////////////
    # ------------------------- MUSCLESTRENGTH -------------------------
    # //////////////////////////////////////////////////////////////////

    df_musclestrength = generic_intervals(
        file_path             = DATA_PATH + "/PROACT_MUSCLESTRENGTH_v8.csv",
        prefix                = "MUS",
        observation_count_col = "MUS_observation_count",
        delta_suffix          = "MS_Delta",
        output_csv            = INTERVALS_FULL_PATH + "/PROACT_MUSCLESTRENGTH_INTERVALS.csv",
        interval_length       = 90,
        negative_intervals    = 0,
    )

    df_musclestrength_first_symptoms = generic_intervals(
        file_path             = FIRST_SYMPTOMS_PATH + "/PROACT_MUSCLESTRENGTH_FIRST_SYMPTOMS.csv",
        prefix                = "MUS",
        observation_count_col = "MUS_observation_count",
        delta_suffix          = "MS_Delta",
        output_csv            = INTERVALS_FULL_PATH + "/PROACT_MUSCLESTRENGTH_FIRST_SYMPTOMS_INTERVALS.csv",
        interval_length       = 90,
        negative_intervals    = 0,
    )





    # ///////////////////////////////////////////////////////
    # ------------------------- SVC -------------------------
    # ///////////////////////////////////////////////////////

    df_svc = generic_intervals(
        file_path             = DATA_PATH + "/PROACT_SVC_v7.csv",
        prefix                = "SVC",
        observation_count_col = "SVC_observation_count",
        delta_suffix          = "Slow_Vital_Capacity_Delta",
        output_csv            = INTERVALS_FULL_PATH + "/PROACT_SVC_INTERVALS.csv",
        interval_length       = 90,
        negative_intervals    = 0,
    )

    df_svc_first_symptoms = generic_intervals(
        file_path             = FIRST_SYMPTOMS_PATH + "/PROACT_SVC_FIRST_SYMPTOMS.csv",
        prefix                = "SVC",
        observation_count_col = "SVC_observation_count",
        delta_suffix          = "Slow_Vital_Capacity_Delta",
        output_csv            = INTERVALS_FULL_PATH + "/PROACT_SVC_FIRST_SYMPTOMS_INTERVALS.csv",
        interval_length       = 90,
        negative_intervals    = 0,
    )





    # //////////////////////////////////////////////////////////////
    # ------------------------- VITALSIGNS -------------------------
    # //////////////////////////////////////////////////////////////

    df_vitalsigns_intervals = generic_intervals(
        file_path             = DATA_PATH + "/PROACT_VITALSIGNS_v7.csv",
        prefix                = "VIT",
        observation_count_col = "VIT_observation_count",
        delta_suffix          = "Vital_Signs_Delta",
        output_csv            = INTERVALS_FULL_PATH + "/PROACT_VITALSIGNS_INTERVALS.csv",
        interval_length       = 90,
        negative_intervals    = 0,
    )

    df_vitalsigns_first_symptoms = generic_intervals(
        file_path             = FIRST_SYMPTOMS_PATH + "/PROACT_VITALSIGNS_FIRST_SYMPTOMS.csv",
        prefix                = "VIT",
        observation_count_col = "VIT_observation_count",
        delta_suffix          = "Vital_Signs_Delta",
        output_csv            = INTERVALS_FULL_PATH + "/PROACT_VITALSIGNS_FIRST_SYMPTOMS_INTERVALS.csv",
        interval_length       = 90,
        negative_intervals    = 0,
    )





    # ------------------------------------------------------------------
    # Tables skipped - no interval analysis applicable
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