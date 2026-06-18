"""
PROACT Interval-Based Prediction File Generation
=================================================
This script transforms interval-aggregated feature matrices (produced by
INTERVALS_ALL.py) into ready-to-use supervised learning datasets for 
prediction tasks:

    ALSFRS-R progression - predict the ALSFRS-R total score at the next
                           90-day interval given observations up to time t

Two temporal reference frames are supported:
    - Study-inclusion alignment  : Delta = 0 at trial enrolment
    - First-symptom alignment    : Delta = 0 at self-reported symptom onset

Two output file types are produced per prediction horizon:

    Fixed_{T}M.csv
        Static design matrix where features come from intervals [0, t] and
        the target is the value at interval t+1.  One file per horizon
        (3M, 6M, 9M, 12M, ...).  Mirrors the setup used by prior work and
        allows direct comparisons.

    Sliding_{T}M.csv
        Sliding-window design matrix that pools all consecutive sequences of
        length (horizon + 1).  Feature columns are renamed to relative labels
        T1, T2, ..., Th and the target becomes Ti+1, making the file
        horizon-aware but time-position-agnostic.  Enables training a single
        generic sequential model.

Three merge strategies control which patients are retained when features from
multiple tables are combined (merge_interval_options):
    'union'        - patient present in at least one source table
    'intersection' - patient present in all source tables
    'left'         - patient present in the first (primary) source table

Non-temporal features (no Delta column) can optionally be appended via
add_no_delta_to_prediction_files after the interval files are generated.

Pipeline overview
-----------------
Stage 1  Extract target variable columns from the interval CSV
         (extract_target_als_variables)
Stage 2  Generate Fixed and/or Sliding prediction files
         (generate_interval_prediction_files)
Stage 3  Optionally merge interval tables across multiple clinical sources
         (merge_interval_dataframes)
Stage 4  Optionally enrich with non-temporal features
         (add_no_delta_to_prediction_files)

Output directory structure
--------------------------
Intervals/CSV/   - source interval feature matrices (one per table)
Intervals/Cut/   - prediction-ready files (Fixed_*.csv, Sliding_*.csv)
                   organised in one subdirectory per table/merge combination

Tables with no Delta column (ALSHISTORY, DEATHDATA, DEMOGRAPHICS,
EL ESCORIAL, FAMILY HISTORY, RILUZOLE, TREATMENT) and tables whose
Delta columns were dropped during preprocessing (ADVERSE EVENTS, CONMEDS)
cannot produce interval features and are therefore excluded.

Author: Bouclier Lucas
Data:   PROACT dataset (2022-07-29 release)
"""



import pandas as pd
import os
import re
import pyarrow.csv as pacsv
import pyarrow as pa
from tqdm import tqdm
from collections import defaultdict





# ------------------------------------------------------------------
# Fast I/O helpers
# ------------------------------------------------------------------

def fast_read_csv(path, delimiter=","):
    """
    Read a CSV file as a pandas DataFrame, using PyArrow for speed.

    PyArrow's columnar reader is significantly faster than pandas for large
    wide files (tens of thousands of columns).  If PyArrow fails for any
    reason (encoding issues, malformed rows) the function falls back to a
    chunked pandas read that skips bad lines rather than raising an error.

    Parameters
    ----------
    path      : str   Path to the CSV file.
    delimiter : str   Column delimiter (default: ',').

    Returns
    -------
    pd.DataFrame
        The file contents, or an empty DataFrame if the file is unreadable.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()

    if first_line == "":
        return pd.DataFrame()

    columns = first_line.split(delimiter)

    try:
        table = pacsv.read_csv(
            path,
            parse_options=pacsv.ParseOptions(
                delimiter=delimiter,
                ignore_empty_lines=True
            ),
            convert_options=pacsv.ConvertOptions(
                check_utf8=False
            )
        )
        print(f"Fast read succeeded (PyArrow): {path}")
        return table.to_pandas()

    except Exception as e:
        print(f"PyArrow failed ({e}), falling back to pandas: {path}")
        try:
            return pandas_fallback(path, delimiter)
        except Exception:
            print(f"Read failed: {path}")
            return pd.DataFrame(columns=columns)



def pandas_fallback(path, delimiter=",", chunksize=500_000):
    """
    Safe chunked pandas CSV reader used when PyArrow fails.

    Reads the file in chunks of 500,000 rows, skipping malformed lines,
    and concatenates the result.  This avoids loading the entire file into
    memory at once and tolerates encoding or parsing errors gracefully.

    Parameters
    ----------
    path      : str   Path to the CSV file.
    delimiter : str   Column delimiter (default: ',').
    chunksize : int   Number of rows per chunk (default: 500,000).

    Returns
    -------
    pd.DataFrame
    """
    chunks = []
    for chunk in pd.read_csv(
        path,
        delimiter=delimiter,
        chunksize=chunksize,
        engine="python",
        on_bad_lines="skip"
    ):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)



def fast_write_csv(df, output_path):
    """
    Write a DataFrame to CSV using PyArrow for maximum throughput.

    PyArrow's columnar writer is considerably faster than pandas.to_csv for
    large wide DataFrames.  The index is never written to the file.

    Parameters
    ----------
    df          : pd.DataFrame   DataFrame to write.
    output_path : str            Destination file path.
    """
    table = pa.Table.from_pandas(df, preserve_index=False)
    pacsv.write_csv(table, output_path)





# ------------------------------------------------------------------
# Target variable extraction - ALSFRS
# ------------------------------------------------------------------

def extract_target_als_variables(
        df_or_path,
        variable_prefix="ALS_ALSFRS_R_Total",
        variable_suffix="Central"
    ):
    """
    Extract interval-level ALSFRS-R total score columns to use as prediction
    targets.

    The function selects all columns whose name matches the pattern:

        {variable_prefix}_{interval_start}_{interval_end}_{variable_suffix}

    e.g. 'ALS_ALSFRS_R_Total_0_90_Central', 'ALS_ALSFRS_R_Total_90_180_Central'

    These columns represent the ALSFRS-R total score at the centre of each
    90-day interval (the 'Central' statistic).  Retaining all intervals at
    once allows generate_interval_prediction_files to look up any horizon
    without re-reading the source file.

    Rows where all target columns are missing are dropped (patients with no
    recorded ALSFRS-R score in any interval are not usable as training
    examples).

    Parameters
    ----------
    df_or_path        : str | pd.DataFrame
        Path to an interval feature CSV, or an already-loaded DataFrame.
    variable_prefix   : str
        Column name prefix (default: 'ALS_ALSFRS_R_Total').
    variable_suffix   : str
        Column name suffix (default: 'Central').

    Returns
    -------
    pd.DataFrame
        Subset containing 'subject_id' plus all matching target columns.
        Returns an empty DataFrame if no matching columns are found.
    """
    df = df_or_path if isinstance(df_or_path, pd.DataFrame) else fast_read_csv(df_or_path)

    pattern = re.compile(
        rf"^{re.escape(variable_prefix)}_\d+_\d+_{re.escape(variable_suffix)}$"
    )
    matching_columns = [col for col in df.columns if pattern.match(col)]

    if not matching_columns:
        print(f"No columns found matching '{variable_prefix}_*_*_{variable_suffix}'")
        return pd.DataFrame()

    print(f"{len(matching_columns)} target columns found: {matching_columns}")

    cols_to_keep = (
        ["subject_id"] + matching_columns if "subject_id" in df.columns
        else matching_columns
    )

    # Drop patients with no target value in any interval
    df = df.dropna(subset=matching_columns, how='all')

    return df[cols_to_keep]





# ------------------------------------------------------------------
# Interval fill detection
# ------------------------------------------------------------------

def intervals_filled(
    df,
    interval_labels,
    columns_by_interval,
    merge_tables=None,
    merge_interval_options='union'
):
    """
    Determine, for each patient and each temporal interval, whether at least
    one usable measurement is present.

    This is a prerequisite for generate_interval_prediction_files: only
    intervals that actually contain data are included in training sequences.
    The definition of "filled" depends on the merge strategy:

    No merge (single table):
        An interval is filled if any variable in that interval is non-missing.

    Merge with multiple source tables:
        'union'        - at least one source table has data in the interval
        'intersection' - every source table has data in the interval
        'left'         - only the first (primary) source table must have data

    Implementation notes:
        - The DataFrame is cleaned once (empty strings -> NA) rather than
          inside each loop iteration.
        - A boolean presence matrix (not_na) is built once for all columns and
          reused for every interval, avoiding redundant boolean operations.
        - Column-to-table mappings are pre-indexed per interval to avoid
          repeated 'startswith' scans.

    Parameters
    ----------
    df                    : pd.DataFrame
        Wide-format patient feature matrix (one row per patient).
    interval_labels       : list[str]
        Ordered list of interval labels, e.g. ['0_90', '90_180', ...].
    columns_by_interval   : dict
        Mapping interval_label -> set of column names belonging to that
        interval.
    merge_tables          : list[str] | None
        Three-letter table prefixes (e.g. ['ALS', 'FVC']) when the DataFrame
        is a multi-table merge.  None for single-table inputs.
    merge_interval_options : str
        Fill evaluation strategy for merged inputs: 'union', 'intersection',
        or 'left'.

    Returns
    -------
    pd.DataFrame
        Boolean DataFrame with one column per interval label and one row per
        patient.  subject_id is appended as the last column for downstream
        filtering.
    """

    # 1. Normalise: replace empty strings with NA once for the whole DataFrame
    df_clean = df.replace("", pd.NA)

    # 2. Boolean presence matrix: True if a cell contains a non-missing value
    not_na = df_clean.notna()

    # 3. Pre-index columns by interval and source table to avoid inner-loop scans
    interval_table_columns = {}
    if merge_tables is not None:
        for interval, cols in columns_by_interval.items():
            interval_table_columns[interval] = {
                table: [c for c in cols if c.startswith(table)]
                for table in merge_tables
            }

    # 4. Compute the fill mask for each interval
    filled_dict = {}

    for interval in interval_labels:
        cols = list(columns_by_interval.get(interval, []))

        if not cols:
            # No feature exists for this interval; treat all patients as empty
            filled_dict[interval] = pd.Series(False, index=df.index)
            continue

        if merge_tables is None or merge_interval_options == 'union':
            # Standard case or union: filled if any variable is present
            interval_mask = not_na[cols].any(axis=1)

        else:
            # Merge case: build one boolean mask per source table
            table_masks = {
                table: not_na[table_cols].any(axis=1)
                for table in merge_tables
                for table_cols in [interval_table_columns[interval][table]]
                if table_cols
            }
            table_df = pd.DataFrame(table_masks)

            if merge_interval_options == "intersection":
                interval_mask = table_df.all(axis=1)
            elif merge_interval_options == "left":
                first_table = merge_tables[0]
                if first_table not in table_df.columns:
                    interval_mask = pd.Series(False, index=df.index)
                else:
                    interval_mask = table_df[first_table]
            else:
                raise ValueError(
                    "merge_interval_options must be 'intersection', 'left' or 'union'"
                )

        filled_dict[interval] = interval_mask

    # 5. Assemble result DataFrame
    df_filled = pd.DataFrame(filled_dict, index=df.index)
    df_filled["subject_id"] = df["subject_id"]  # keep for groupby and filtering

    return df_filled





# ------------------------------------------------------------------
# Core prediction file generator
# ------------------------------------------------------------------

def generate_interval_prediction_files(
    df_intervals_path,
    df_target_path,
    output_dir,
    interval_length=90,
    num_intervals=None,
    merge_interval_options='union',
    stats_added=False,
    generate_fixed=True,
    generate_sliding=True,
    central_statistic_only=False
):
    """
    Generate Fixed and Sliding prediction files from interval feature matrices.

    Fixed files (generate_fixed=True)
    -----------------------------------
    One file per prediction horizon (3M, 6M, ...):
        Features  = all observed intervals [0, t] for each patient.
        Target    = the Central statistic at interval t+1.
    File name: Fixed_{T}M.csv

    Sliding files (generate_sliding=True)
    -----------------------------------------
    One file per prediction horizon aggregating ALL consecutive (T1, ..., Th)
    -> Ti+1 windows found across all patients and all starting positions:
        Features  = h consecutive filled intervals, renamed T1 ... Th.
        Target    = the interval immediately following, renamed Ti+1.
    This produces a larger, time-position-agnostic training set suitable for
    generic sequential models.
    File name: Sliding_{T}M.csv

    Temporal leakage prevention
    ---------------------------
    keep_column / keep_column_sliding ensure that no feature column whose
    interval overlaps the target interval is included.  Columns that encode
    cross-interval spans (__to__) are only kept when all spanned intervals
    fall within the feature window.

    Death target handling
    ---------------------
    DEA_Death_Days and DEA_Subject_Died are always removed from the feature
    set to prevent direct target leakage.

    Memory management (stats_added=True)
    --------------------------------------
    When the feature matrix is very wide (AllStats / InterStats / PopStats
    variants with hundreds of statistics per variable), the Sliding file can
    grow very large.  With stats_added=True, rows are flushed to disk in
    batches of 200 and merged at the end, keeping peak memory usage bounded.

    Parameters
    ----------
    df_intervals_path       : str   Path to the interval feature CSV.
    df_target_path          : str   Path to the target variable CSV.
    output_dir              : str   Directory where output files are written.
    interval_length         : int   Interval width in days (default: 90).
    num_intervals           : int | None
        Total number of intervals.  Inferred from column names if None.
    merge_interval_options  : str
        Patient retention strategy for multi-table merges: 'union',
        'intersection', or 'left'.
    stats_added             : bool
        If True, use disk-based batching to limit memory usage (for very wide
        AllStats / InterStats / PopStats feature matrices).
    generate_fixed          : bool  Whether to generate Fixed files.
    generate_sliding        : bool  Whether to generate Sliding files.
    central_statistic_only  : bool
        If True, only include the central statistic for each interval.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Clean up any stale files from a previous run
    delete_count = 0
    for f in os.listdir(output_dir):
        file_path = os.path.join(output_dir, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
            delete_count += 1
    print(f"Output directory cleaned: {delete_count} file(s) removed")

    df_intervals = fast_read_csv(df_intervals_path)
    df_target    = fast_read_csv(df_target_path)

    print(f"Interval features: {df_intervals.shape}")
    print(f"Target variables:  {df_target.shape}")

    # ------------------------------------------------------------------
    # Central statistic filter
    # ------------------------------------------------------------------
    # When central_statistic_only=True, reduce the feature matrix to
    # columns holding only the Central statistic per interval.
    # subject_id and observation_count columns are always kept:
    # the former is the join key, the latter is required by the
    # OBS_COUNT early-exit guard in the Sliding loop.

    if central_statistic_only:
        df_intervals = df_intervals[[
            col for col in df_intervals.columns
            if col == "subject_id"
            or "observation_count" in col
            or "_Central" in col
        ]]
        print(
            f"Central-statistic-only filter applied: "
            f"{df_intervals.shape[1]} columns retained."
        )

    # ------------------------------------------------------------------
    # Auto-detect the number of intervals from column names
    # ------------------------------------------------------------------
    if num_intervals is None:
        all_intervals = set()
        for col in df_intervals.columns:
            for start, end in re.findall(r"(\d+)_(\d+)", col):
                all_intervals.add((int(start), int(end)))

        if all_intervals:
            max_end       = max(end for _, end in all_intervals)
            num_intervals = max_end // interval_length
            print(
                f"Auto-detected intervals: last interval ends at "
                f"{max_end} days -> {num_intervals} intervals"
            )
        else:
            raise ValueError(
                "Cannot detect intervals from column names. "
                "Specify num_intervals manually."
            )

    if num_intervals < 2:
        raise ValueError(f"num_intervals must be >= 2, got {num_intervals}")

    # ------------------------------------------------------------------
    # Detect source table prefixes for MERGE files
    # ------------------------------------------------------------------
    merge_components = []
    merge_tables     = []

    if (
        "MERGE"     in df_intervals_path
        or "ALL_DATA"  in df_intervals_path
        or "ALL_DELTA" in df_intervals_path
    ):
        print("MERGE file detected, identifying source tables...")
        match = re.search(
            r"PROACT_MERGE_([^/]+)_INTERVALS", df_intervals_path
        )
        if match:
            merge_components = match.group(1).split("_")

        print(f"Components: {merge_components}")

        for component in merge_components:
            if "ALL" in component and "AllStats" not in component:
                merge_tables.extend(["ALS", "FVC", "HAN", "MUS", "LAB", "SVC", "VIT"])
            elif "ALSFRS"          in component: merge_tables.append("ALS")
            elif "FVC"             in component: merge_tables.append("FVC")
            elif "HANDGRIPSTRENGTH" in component: merge_tables.append("HAN")
            elif "MUSCLESTRENGTH"  in component: merge_tables.append("MUS")
            elif "LABS"            in component: merge_tables.append("LAB")
            elif "SVC"             in component: merge_tables.append("SVC")
            elif "VITALSIGNS"      in component: merge_tables.append("VIT")

        print(f"Source tables: {merge_tables}")

    # ------------------------------------------------------------------
    # Pre-compute column-interval index structures
    # ------------------------------------------------------------------

    # Map each column to the set of interval labels it belongs to
    COLUMN_INTERVALS = {
        col: re.findall(r"(\d+_\d+)", col)
        for col in df_intervals.columns
    }
    COLUMN_INTERVAL_SET = {
        col: set(intervals)
        for col, intervals in COLUMN_INTERVALS.items()
    }

    # Reverse index: interval label -> set of columns
    COLUMNS_BY_INTERVAL = defaultdict(set)
    for col, intervals in COLUMN_INTERVALS.items():
        for i in intervals:
            COLUMNS_BY_INTERVAL[i].add(col)

    SOURCE_COLUMNS = list(df_intervals.columns)

    # ------------------------------------------------------------------
    # Pre-compute maximum observation count per patient
    # (used as a fast early-exit guard in the Sliding loop)
    # ------------------------------------------------------------------
    obs_count_cols = [
        c for c in df_intervals.columns if c.endswith("observation_count")
    ]
    if obs_count_cols:
        OBS_COUNT = (
            df_intervals
            .groupby("subject_id")[obs_count_cols]
            .max()
            .max(axis=1)
            .to_dict()
        )
    else:
        OBS_COUNT = {}

    # ------------------------------------------------------------------
    # Column selection helpers
    # ------------------------------------------------------------------

    def keep_column(col, train_labels):
        """
        Decide whether a column should be included in a Fixed training set.

        Single-interval columns are kept if they belong to at least one
        training interval.  Cross-interval columns (__to__) are kept only if
        ALL spanned intervals are inside the training window, preventing any
        look-ahead leakage.

        Parameters
        ----------
        col          : str        Column name.
        train_labels : list[str]  Training interval labels.

        Returns
        -------
        bool
        """
        intervals = COLUMN_INTERVAL_SET.get(col, set())

        if "__to__" not in col:
            # Standard single-interval column
            return any(lbl in intervals for lbl in train_labels)

        # Cross-interval column: all spanned intervals must be in the window
        return all(i in train_labels for i in intervals)

    def keep_column_sliding(col, train_labels):
        """
        Decide whether a column should be included in a Sliding training row.

        Equivalent logic to keep_column, but additionally requires that
        cross-interval columns span at least two distinct intervals to
        represent a genuine temporal relationship.

        Parameters
        ----------
        col          : str        Column name.
        train_labels : list[str]  Training interval labels for this window.

        Returns
        -------
        bool
        """
        intervals = COLUMN_INTERVAL_SET.get(col, set())

        if "__to__" not in col:
            return any(lbl in intervals for lbl in train_labels)

        # Must span >= 2 intervals AND all must be within the training window
        return len(intervals) >= 2 and all(i in train_labels for i in intervals)

    # ------------------------------------------------------------------
    # Stage 1: Fixed files
    # ------------------------------------------------------------------

    if generate_fixed:
        print("\nGenerating Fixed files...")

        interval_labels = [
            f"{i * interval_length}_{(i + 1) * interval_length}"
            for i in range(num_intervals)
        ]

        # One Fixed file per prediction horizon (i = 1, 2, ..., num_intervals-1)
        for i in tqdm(range(1, num_intervals), desc="Fixed files"):
            train_labels  = interval_labels[:i]
            target_label  = interval_labels[i]
            target_col_pattern = f"_{target_label}_Central"

            # Select feature columns strictly before the target interval
            train_cols = ["subject_id"] + [
                c for c in df_intervals.columns
                if keep_column(c, train_labels)
            ]

            # Identify the target column
            target_cols = [
                c for c in df_target.columns if target_col_pattern in c
            ]
            if not target_cols:
                print(f"No target column found for {target_label}.")
                end_month  = (i * interval_length) // 30
                output_path = os.path.join(output_dir, f"Fixed_{end_month}M.csv")
                fast_write_csv(pd.DataFrame(), output_path)
                continue
            target_col = target_cols[0]

            df_fixed = df_intervals[train_cols].copy()

            # Rename single-interval columns to T1, T2, ..., Th
            rename_map = {}
            for t_idx, lbl in enumerate(train_labels, start=1):
                for col in df_fixed.columns:
                    if lbl in COLUMN_INTERVAL_SET.get(col, set()):
                        rename_map[col] = re.sub(
                            rf"_{lbl}(_|$)",
                            f"_T{t_idx}\\1",
                            col
                        )

            # Rename cross-interval (__to__) columns
            for col in df_fixed.columns:
                if "__to__" in col:
                    intervals = COLUMN_INTERVALS[col]
                    if all(i in train_labels for i in intervals):
                        tpos = [
                            train_labels.index(i) + 1
                            for i in intervals
                        ]
                        prefix = col.split(intervals[0])[0]
                        suffix = col.split(intervals[-1])[-1]
                        rename_map[col] = (
                            f"{prefix}T{tpos[0]}__to__T{tpos[-1]}{suffix}"
                        )

            df_fixed.rename(columns=rename_map, inplace=True)

            # Remove any death indicator from features to prevent target leakage
            if 'DEA_Death_Days'    in df_fixed.columns:
                df_fixed = df_fixed.drop(columns=['DEA_Death_Days'])
            if 'DEA_Subject_Died'  in df_fixed.columns:
                df_fixed = df_fixed.drop(columns=['DEA_Subject_Died'])

            # Join target column from the target DataFrame
            df_fixed = df_fixed.merge(
                df_target[["subject_id", target_col]],
                on="subject_id",
                how="left"
            )

            # Remove rows without a target value
            df_fixed = df_fixed.dropna(subset=[target_col])
            df_fixed = df_fixed[df_fixed[target_col] != ""]

            # Remove patients with no feature information before the target
            df_fixed = df_fixed[
                ~df_fixed
                .drop(columns=["subject_id", target_col])
                .replace("", pd.NA)
                .isna()
                .all(axis=1)
            ]

            # Additional patient filtering for multi-table merges
            if merge_tables and merge_interval_options != 'union':
                if merge_interval_options == "intersection":
                    # Require data from every source table in the feature window
                    for table in merge_tables:
                        table_cols = [c for c in df_fixed.columns if c.startswith(table)]
                        df_fixed = df_fixed[
                            ~df_fixed[table_cols].replace("", pd.NA).isna().all(axis=1)
                        ]
                elif merge_interval_options == "left":
                    # Require data from the primary source table only
                    first_table      = merge_tables[0]
                    first_table_cols = [c for c in df_fixed.columns if c.startswith(first_table)]
                    df_fixed = df_fixed[
                        ~df_fixed[first_table_cols].replace("", pd.NA).isna().all(axis=1)
                    ]

            end_month   = (i * interval_length) // 30
            output_path = os.path.join(output_dir, f"Fixed_{end_month}M.csv")
            fast_write_csv(df_fixed, output_path)
            print(f"Saved: {output_path} ({df_fixed.shape[0]} rows, {df_fixed.shape[1]} cols)")

    # ------------------------------------------------------------------
    # Stage 2: Sliding files
    # ------------------------------------------------------------------
    #
    # While Fixed files use absolute time positions [0 -> t], Sliding files
    # use a sliding window over consecutive observed intervals so that the same
    # patient can contribute multiple training rows at different time positions.
    # All feature column names are renumbered T1, T2, ..., Th and the target
    # becomes Ti+1, making the dataset time-position-agnostic.

    if generate_sliding:
        print("\nGenerating Sliding files...")

        # horizon (in months) -> number of feature intervals required
        sliding_configs = {
            f"{(h * interval_length) // 30}M": h
            for h in range(1, num_intervals)
        }

        interval_labels = [
            f"{i * interval_length}_{(i + 1) * interval_length}"
            for i in range(num_intervals)
        ]

        SOURCE_COLUMNS = list(df_intervals.columns)

        # Fast target lookup: subject_id -> row of target values
        target_map = df_target.set_index("subject_id")

        # Pre-compute globally which intervals are filled for each patient
        df_filled = intervals_filled(
            df_intervals,
            interval_labels,
            COLUMNS_BY_INTERVAL,
            merge_tables=merge_tables,
            merge_interval_options=merge_interval_options
        )
        df_filled_global = (
            df_filled
            .groupby("subject_id")[interval_labels]
            .any()
        )

        # Indexed access to feature rows
        df_intervals_indexed = df_intervals.set_index("subject_id", drop=False)

        for label, horizon in tqdm(sliding_configs.items(), desc="Sliding files"):
            all_rows = []

            if stats_added:
                temp_files = []
                batch_size = 200
                batch_id   = 0

            required_obs = horizon  # minimum filled intervals needed for features

            for subject_id, df_patient in tqdm(
                df_intervals.groupby("subject_id"),
                desc=f"{label} patients",
                leave=False
            ):
                # Early exit: not enough observations to form even one window
                if OBS_COUNT.get(subject_id, 0) < required_obs:
                    continue

                # Early exit: no target values available for this patient
                if subject_id not in target_map.index:
                    continue

                df_patient = df_intervals_indexed.loc[[subject_id]]

                # Determine which intervals are actually filled for this patient
                filled_intervals = df_filled_global.loc[subject_id]
                filled = [
                    interval
                    for interval, val in filled_intervals.items()
                    if val
                ]

                # Need at least horizon + 1 filled intervals (h features + 1 target)
                if len(filled) < horizon + 1:
                    continue

                # Convert filled interval labels to their positional indices
                idxs = [interval_labels.index(i) for i in filled]

                # Identify consecutive runs of interval indices
                sequences = []
                current   = [idxs[0]]
                for prev, curr in zip(idxs, idxs[1:]):
                    if curr == prev + 1:
                        current.append(curr)
                    else:
                        if len(current) >= horizon + 1:
                            sequences.append(current)
                        current = [curr]
                if len(current) >= horizon + 1:
                    sequences.append(current)

                # Slide a window of size (horizon + 1) over each consecutive run
                for seq in sequences:
                    for start in range(len(seq) - horizon):
                        train_idxs = seq[start : start + horizon]
                        target_idx = seq[start + horizon]

                        train_labels = [interval_labels[i] for i in train_idxs]
                        target_label = interval_labels[target_idx]

                        # Feature columns for this window (death indicators excluded)
                        train_cols = [
                            c for c in SOURCE_COLUMNS
                            if (
                                c == "subject_id"
                                or keep_column_sliding(c, train_labels)
                            )
                            and c not in ('DEA_Death_Days', 'DEA_Subject_Died')
                        ]

                        # Skip windows with no usable features
                        if len(train_cols) == 1:  # only subject_id
                            continue

                        # Identify and validate the target value
                        target_col_pattern = f"_{target_label}_Central"
                        target_cols = [
                            c for c in df_target.columns
                            if target_col_pattern in c
                        ]
                        if not target_cols:
                            continue
                        target_col = target_cols[0]

                        try:
                            target_value = target_map.at[subject_id, target_col]
                        except KeyError:
                            continue

                        if pd.isna(target_value) or target_value == "":
                            continue

                        # Build the training row
                        row                = df_patient[train_cols].copy()
                        row[target_col]    = target_value

                        # Rename single-interval columns to T1, T2, ..., Th
                        rename_map = {}
                        for t_idx, lbl in enumerate(train_labels, start=1):
                            for col in row.columns:
                                if lbl in COLUMN_INTERVAL_SET.get(col, set()):
                                    rename_map[col] = re.sub(
                                        rf"_{lbl}(_|$)",
                                        f"_T{t_idx}\\1",
                                        col
                                    )

                        # Rename cross-interval (__to__) columns
                        for col in row.columns:
                            if "__to__" in col:
                                intervals = COLUMN_INTERVALS[col]
                                if all(i in train_labels for i in intervals):
                                    tpos   = [train_labels.index(i) + 1 for i in intervals]
                                    prefix = col.split(intervals[0])[0]
                                    suffix = col.split(intervals[-1])[-1]
                                    rename_map[col] = (
                                        f"{prefix}T{tpos[0]}__to__T{tpos[-1]}{suffix}"
                                    )

                        # Rename target column to T{i+1}
                        parts           = target_col.split("_")
                        prefix          = "_".join(parts[:-3])
                        rename_map[target_col] = f"{prefix}_Ti+1_Central"

                        row.rename(columns=rename_map, inplace=True)
                        all_rows.append(row)

                        # Flush to disk when batch is full (stats_added mode)
                        if stats_added and len(all_rows) >= batch_size:
                            temp_df   = pd.concat(all_rows, ignore_index=True)
                            temp_path = os.path.join(
                                output_dir, f"tmp_{label}_{batch_id}.csv"
                            )
                            fast_write_csv(temp_df, temp_path)
                            temp_files.append(temp_path)
                            print(f"Temporary batch written: {temp_path}")
                            del temp_df
                            all_rows.clear()
                            batch_id += 1

            # ------------------------------------------------------------------
            # Save the final Sliding file for this horizon
            # ------------------------------------------------------------------
            if stats_added:
                # Flush remaining rows
                if all_rows:
                    temp_df   = pd.concat(all_rows, ignore_index=True)
                    temp_path = os.path.join(
                        output_dir, f"tmp_{label}_{batch_id}.csv"
                    )
                    fast_write_csv(temp_df, temp_path)
                    temp_files.append(temp_path)
                    del temp_df
                    all_rows.clear()

                if not temp_files:
                    print(f"No valid windows found for Sliding_{label}")
                    output_path = os.path.join(output_dir, f"Sliding_{label}.csv")
                    fast_write_csv(pd.DataFrame(), output_path)
                    continue

                print(f"Merging temporary batches for {label}...")
                df_sliding = pd.concat(
                    (fast_read_csv(f) for f in temp_files),
                    ignore_index=True
                )
                output_path = os.path.join(output_dir, f"Sliding_{label}.csv")
                fast_write_csv(df_sliding, output_path)
                print(
                    f"Saved: {output_path} "
                    f"({df_sliding.shape[0]} rows, {df_sliding.shape[1]} cols)"
                )

                # Clean up temporary batch files
                for f in temp_files:
                    try:
                        os.remove(f)
                    except Exception:
                        pass

            else:
                if not all_rows:
                    print(f"No valid windows found for Sliding_{label}")
                    output_path = os.path.join(output_dir, f"Sliding_{label}.csv")
                    fast_write_csv(pd.DataFrame(), output_path)
                    continue

                df_sliding = pd.concat(all_rows, ignore_index=True)
                output_path = os.path.join(output_dir, f"Sliding_{label}.csv")
                fast_write_csv(df_sliding, output_path)
                print(
                    f"Saved: {output_path} "
                    f"({df_sliding.shape[0]} rows, {df_sliding.shape[1]} cols)"
                )

        print("\nGeneration complete.\n")





# ------------------------------------------------------------------
# Multi-table interval merge
# ------------------------------------------------------------------

def merge_interval_dataframes(
    dfs_interval,
    key="subject_id",
    how="union",
    sort=True,
    validate=None
):
    """
    Merge a list of interval feature DataFrames on subject_id.

    Each DataFrame in dfs_interval contains the interval-aggregated features
    for a single clinical table.  This function joins them horizontally so that
    all features are available in a single wide DataFrame for multi-table
    experiments.

    Merge strategies:
        'union'        -> outer join  (patient present in at least one table)
        'intersection' -> inner join  (patient present in all tables)
        (any other)    -> passed through directly to pandas merge 'how'

    Parameters
    ----------
    dfs_interval : list[pd.DataFrame]
        DataFrames to merge.  Must all share the key column.
    key          : str
        Join key column name (default: 'subject_id').
    how          : str
        Merge strategy: 'union', 'intersection', or a pandas how string.
    sort         : bool
        If True, sort by key and reset the index after merging.
    validate     : str | None
        Optional pandas merge validation (e.g. 'one_to_one').

    Returns
    -------
    pd.DataFrame
        Wide-format merged DataFrame.
    """
    if how == "union":
        how = "outer"
    elif how == "intersection":
        how = "inner"

    if not dfs_interval:
        raise ValueError("dfs_interval list is empty.")

    df_merged = dfs_interval[0]

    for df in dfs_interval[1:]:
        df_merged = df_merged.merge(
            df,
            on=key,
            how=how,
            validate=validate
        )

    if sort:
        df_merged = (
            df_merged
            .sort_values(by=key)
            .reset_index(drop=True)
        )

    return df_merged




# ------------------------------------------------------------------
# Non-temporal feature enrichment
# ------------------------------------------------------------------

def add_no_delta_to_prediction_files(
    input_dir,
    output_dir,
    df_no_delta,
    key="subject_id",
    how="left"
    # 'left' is used here because NODELTA is a union of all non-temporal tables
    # and therefore covers all 11,675 patients.  A left join ensures that no
    # prediction rows are lost, and no spurious new rows are added, which would
    # be the case with an outer or inner join respectively.
):
    """
    Append non-temporal (no Delta) features to every prediction file in a
    directory.

    Non-temporal tables (DEMOGRAPHICS, ALSHISTORY, EL ESCORIAL, etc.) contain
    patient-level baseline characteristics that do not vary over time.  After
    the interval prediction files are generated, this function enriches them
    with these static features by performing a left join on subject_id.

    The function iterates over all CSV files in input_dir, performs the merge,
    and writes the result to output_dir.  The target column (always the last
    column) is moved back to the last position after the merge.  Empty input
    files are copied as-is.

    If the source directory is a Death prediction directory, the DEA_Death_Days
    and DEA_Subject_Died columns are stripped from df_no_delta before joining
    to prevent target leakage.

    Parameters
    ----------
    input_dir   : str           Directory containing Fixed_*.csv / Sliding_*.csv.
    output_dir  : str           Destination directory.
    df_no_delta : pd.DataFrame  Non-temporal feature DataFrame (all patients).
    key         : str           Join key column name (default: 'subject_id').
    how         : str           Merge strategy (default: 'left').
    """
    if how == "union":
        how = "outer"
    elif how == "intersection":
        how = "inner"

    os.makedirs(output_dir, exist_ok=True)

    # Clean up stale files
    delete_count = 0
    for f in os.listdir(output_dir):
        file_path = os.path.join(output_dir, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
            delete_count += 1
    print(f"Output directory cleaned: {delete_count} file(s) removed")

    for file in os.listdir(input_dir):
        if not file.endswith(".csv"):
            continue

        input_path = os.path.join(input_dir, file)
        df_pred    = fast_read_csv(input_path)

        # Pass through empty files unchanged
        if df_pred.empty:
            output_path = os.path.join(output_dir, file)
            fast_write_csv(df_pred, output_path)
            print(f"Saved (empty): {output_path}")
            continue

        # Remove death target columns from NODELTA features when predicting death
        if "Death" in input_dir:
            cols_to_drop = [
                c for c in ('DEA_Death_Days', 'DEA_Subject_Died')
                if c in df_no_delta.columns
            ]
            df_no_delta = df_no_delta.drop(columns=cols_to_drop, errors='ignore')

        # The target is always the last column
        target_col = df_pred.columns[-1]

        df_merged = df_pred.merge(df_no_delta, on=key, how=how)

        # Remove rows whose target is missing after the join
        df_merged = df_merged.dropna(subset=[target_col])
        df_merged = df_merged[df_merged[target_col] != ""]

        # Restore target column to last position
        cols      = [c for c in df_merged.columns if c != target_col] + [target_col]
        df_merged = df_merged[cols]

        df_merged = (
            df_merged
            .sort_values(by=key)
            .reset_index(drop=True)
        )

        output_path = os.path.join(output_dir, file)
        fast_write_csv(df_merged, output_path)
        print(
            f"Saved: {output_path} "
            f"({df_merged.shape[0]} rows, {df_merged.shape[1]} cols)"
        )










def run(INTERVALS_FULL_PATH, INTERVALS_CUT_PATH, MERGE_NODELTA_PATH, FIRST_SYMPTOMS_PATH, PAPER_RESULTS_ONLY):

    print("\n" * 3)
    print("=" * 60)
    print("INTERVALS CUT PIPELINE")
    print("=" * 60)





    # ==============================================================================
    #
    # CALL SECTION
    #
    # The remainder of this file contains all `generate_interval_prediction_files`
    # calls, organised by table combination, prediction target, and temporal
    # reference frame.
    #
    # Structure:
    #   1.  Target variable extraction
    #   2.  Single-table experiments
    #   3.  Single-table + NODELTA enrichment
    #   4.  Full multi-table merges (ALL_DELTA) + NODELTA enrichment (ALL_DATA)
    #   5.  Pairwise table merges
    #   6.  Pairwise + NODELTA enrichment
    #   7.  Three-table and four-table merges
    #
    # For each table combination, three prediction tasks are covered:
    #   - ALSFRS study-inclusion aligned   (Fixed + Sliding)
    #   - ALSFRS first-symptom aligned     (Fixed only)
    #
    # ==============================================================================










    # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # ================================================== TARGET VARIABLE ==================================================
    # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # Study-inclusion aligned variables
    df_target = extract_target_als_variables(
        df_or_path=INTERVALS_FULL_PATH + "/PROACT_ALSFRS_INTERVALS.csv",
        variable_prefix="ALS_ALSFRS_R_Total",
        variable_suffix="Central"
    )
    fast_write_csv(df_target, INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv')

    # First-symptom aligned variables
    df_target_first_symptoms = extract_target_als_variables(
        df_or_path=INTERVALS_FULL_PATH + "/PROACT_ALSFRS_FIRST_SYMPTOMS_INTERVALS.csv",
        variable_prefix="ALS_ALSFRS_R_Total",
        variable_suffix="Central"
    )
    fast_write_csv(df_target_first_symptoms, INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv')










    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # ================================================== UNIQUE TABLES ==================================================
    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////



    # //////////////////////////////////////////////////////////
    # ------------------------- ALSFRS -------------------------
    # //////////////////////////////////////////////////////////

    # Study-inclusion aligned ALSFRS prediction
    generate_interval_prediction_files(
        df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_ALSFRS_INTERVALS.csv',
        df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
        output_dir=INTERVALS_CUT_PATH + '/ALSFRS/',
        interval_length=90,
        # num_intervals=5
    )

    # First-symptom aligned ALSFRS prediction
    generate_interval_prediction_files(
        df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_ALSFRS_FIRST_SYMPTOMS_INTERVALS.csv',
        df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
        output_dir=INTERVALS_CUT_PATH + '/ALSFRS_First_Symptoms/',
        interval_length=90,
        # num_intervals=17,
        generate_sliding=False,
    )





    # /////////////////////////////////////////////////////////////////////////
    # ------------------------- ALSFRS (central only) -------------------------
    # /////////////////////////////////////////////////////////////////////////

    # Study-inclusion aligned ALSFRS prediction (baseline)
    generate_interval_prediction_files(
        df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_ALSFRS_INTERVALS.csv',
        df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
        output_dir=INTERVALS_CUT_PATH + '/ALSFRS_Central_Only/',
        interval_length=90,
        # num_intervals=5,
        central_statistic_only=True,
    )

    # First-symptom aligned ALSFRS prediction (baseline)
    generate_interval_prediction_files(
        df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_ALSFRS_FIRST_SYMPTOMS_INTERVALS.csv',
        df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
        output_dir=INTERVALS_CUT_PATH + '/ALSFRS_Central_Only_First_Symptoms/',
        interval_length=90,
        central_statistic_only=True,
        # num_intervals=17,
        generate_sliding=False,
    )





    # ///////////////////////////////////////////////////////
    # ------------------------- FVC -------------------------
    # ///////////////////////////////////////////////////////

    # Study-inclusion aligned ALSFRS prediction
    generate_interval_prediction_files(
        df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_FVC_INTERVALS.csv',
        df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
        output_dir=INTERVALS_CUT_PATH + '/FVC/',
        interval_length=90,
        # num_intervals=5
    )

    # First-symptom aligned ALSFRS prediction
    generate_interval_prediction_files(
        df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_FVC_FIRST_SYMPTOMS_INTERVALS.csv',
        df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
        output_dir=INTERVALS_CUT_PATH + '/FVC_First_Symptoms/',
        interval_length=90,
        # num_intervals=17,
        generate_sliding=False,
    )





    # ////////////////////////////////////////////////////////////////////
    # ------------------------- HANDGRIPSTRENGTH -------------------------
    # ////////////////////////////////////////////////////////////////////

    # Study-inclusion aligned ALSFRS prediction
    generate_interval_prediction_files(
        df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_HANDGRIPSTRENGTH_INTERVALS.csv',
        df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
        output_dir=INTERVALS_CUT_PATH + '/HANDGRIPSTRENGTH/',
        interval_length=90,
        # num_intervals=5
    )

    # First-symptom aligned ALSFRS prediction
    generate_interval_prediction_files(
        df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_HANDGRIPSTRENGTH_FIRST_SYMPTOMS_INTERVALS.csv',
        df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
        output_dir=INTERVALS_CUT_PATH + '/HANDGRIPSTRENGTH_First_Symptoms/',
        interval_length=90,
        # num_intervals=17,
        generate_sliding=False,
    )





    # ////////////////////////////////////////////////////////
    # ------------------------- LABS -------------------------
    # ////////////////////////////////////////////////////////

    # Study-inclusion aligned ALSFRS prediction
    generate_interval_prediction_files(
        df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_LABS_INTERVALS.csv',
        df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
        output_dir=INTERVALS_CUT_PATH + '/LABS/',
        interval_length=90,
        # num_intervals=5
    )

    # First-symptom aligned ALSFRS prediction
    generate_interval_prediction_files(
        df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_LABS_FIRST_SYMPTOMS_INTERVALS.csv',
        df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
        output_dir=INTERVALS_CUT_PATH + '/LABS_First_Symptoms/',
        interval_length=90,
        # num_intervals=17,
        generate_sliding=False,
    )





    # //////////////////////////////////////////////////////////////////
    # ------------------------- MUSCLESTRENGTH -------------------------
    # //////////////////////////////////////////////////////////////////

    # Study-inclusion aligned ALSFRS prediction
    generate_interval_prediction_files(
        df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MUSCLESTRENGTH_INTERVALS.csv',
        df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
        output_dir=INTERVALS_CUT_PATH + '/MUSCLESTRENGTH/',
        interval_length=90,
        # num_intervals=5
    )

    # First-symptom aligned ALSFRS prediction
    generate_interval_prediction_files(
        df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MUSCLESTRENGTH_FIRST_SYMPTOMS_INTERVALS.csv',
        df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
        output_dir=INTERVALS_CUT_PATH + '/MUSCLESTRENGTH_First_Symptoms/',
        interval_length=90,
        # num_intervals=17,
        generate_sliding=False,
    )





    # ///////////////////////////////////////////////////////
    # ------------------------- SVC -------------------------
    # ///////////////////////////////////////////////////////

    # Study-inclusion aligned ALSFRS prediction
    generate_interval_prediction_files(
        df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_SVC_INTERVALS.csv',
        df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
        output_dir=INTERVALS_CUT_PATH + '/SVC/',
        interval_length=90,
        # num_intervals=5
    )

    # First-symptom aligned ALSFRS prediction
    generate_interval_prediction_files(
        df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_SVC_FIRST_SYMPTOMS_INTERVALS.csv',
        df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
        output_dir=INTERVALS_CUT_PATH + '/SVC_First_Symptoms/',
        interval_length=90,
        # num_intervals=17,
        generate_sliding=False,
    )





    # //////////////////////////////////////////////////////////////
    # ------------------------- VITALSIGNS -------------------------
    # //////////////////////////////////////////////////////////////

    # Study-inclusion aligned ALSFRS prediction
    generate_interval_prediction_files(
        df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_VITALSIGNS_INTERVALS.csv',
        df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
        output_dir=INTERVALS_CUT_PATH + '/VITALSIGNS/',
        interval_length=90,
        # num_intervals=5
    )

    # First-symptom aligned ALSFRS prediction
    generate_interval_prediction_files(
        df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_VITALSIGNS_FIRST_SYMPTOMS_INTERVALS.csv',
        df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
        output_dir=INTERVALS_CUT_PATH + '/VITALSIGNS_First_Symptoms/',
        interval_length=90,
        # num_intervals=17,
        generate_sliding=False,
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










    # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # ================================================== MERGE OF ALL TABLES ==================================================
    # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



    # ///////////////////////////////////////////////////////////////////
    # ------------------------- MERGE ALL DELTA -------------------------
    # ///////////////////////////////////////////////////////////////////

    # We merge all the data with DELTA into a single file
    df_ALSFRS = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_ALSFRS_INTERVALS.csv')
    df_FVC = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_FVC_INTERVALS.csv')
    df_LABS = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_LABS_INTERVALS.csv')
    df_HANDGRIPSTRENGTH = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_HANDGRIPSTRENGTH_INTERVALS.csv')
    df_MUSCLESTRENGTH = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_MUSCLESTRENGTH_INTERVALS.csv')
    df_SVC = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_SVC_INTERVALS.csv')
    df_VITALSIGNS = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_VITALSIGNS_INTERVALS.csv')

    # Merge (union)
    df_all_delta_merge = merge_interval_dataframes(
        dfs_interval=[
            df_ALSFRS,
            df_FVC,
            df_LABS,
            df_HANDGRIPSTRENGTH,
            df_MUSCLESTRENGTH,
            df_SVC,
            df_VITALSIGNS,
        ],
        how="union",
    )
    fast_write_csv(df_all_delta_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALL_DELTA_UNION_INTERVALS.csv')
    # We create the prediction files for the full DELTA merge
    generate_interval_prediction_files(
        df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALL_DELTA_UNION_INTERVALS.csv',
        df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
        output_dir=INTERVALS_CUT_PATH + '/ALL_DELTA_UNION/',
        interval_length=90,
        # num_intervals=5,
        merge_interval_options='union'
    )

    if PAPER_RESULTS_ONLY==False:
        # Merge (intersection)
        df_all_delta_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS,
                df_FVC,
                df_LABS,
                df_HANDGRIPSTRENGTH,
                df_MUSCLESTRENGTH,
                df_SVC,
                df_VITALSIGNS,
            ],
            how="intersection",
        )
        fast_write_csv(df_all_delta_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALL_DELTA_INTERSECTION_INTERVALS.csv')
        # We create the prediction files for the full DELTA merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALL_DELTA_INTERSECTION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/ALL_DELTA_INTERSECTION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='intersection'
        )

    if PAPER_RESULTS_ONLY==False:
        # Merge (left)
        df_all_delta_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS,
                df_FVC,
                df_LABS,
                df_HANDGRIPSTRENGTH,
                df_MUSCLESTRENGTH,
                df_SVC,
                df_VITALSIGNS,
            ],
            how="left",
        )
        fast_write_csv(df_all_delta_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALL_DELTA_LEFT_INTERVALS.csv')
        # We create the prediction files for the full DELTA merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALL_DELTA_LEFT_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/ALL_DELTA_LEFT/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='left'
        )



    # We merge all the data with DELTA into a single file (First Symptoms version)
    df_ALSFRS_First_Symptoms = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_ALSFRS_FIRST_SYMPTOMS_INTERVALS.csv')
    df_FVC_First_Symptoms = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_FVC_FIRST_SYMPTOMS_INTERVALS.csv')
    df_LABS_First_Symptoms = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_LABS_FIRST_SYMPTOMS_INTERVALS.csv')
    df_HANDGRIPSTRENGTH_First_Symptoms = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_HANDGRIPSTRENGTH_FIRST_SYMPTOMS_INTERVALS.csv')
    df_MUSCLESTRENGTH_First_Symptoms = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_MUSCLESTRENGTH_FIRST_SYMPTOMS_INTERVALS.csv')
    df_SVC_First_Symptoms = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_SVC_FIRST_SYMPTOMS_INTERVALS.csv')
    df_VITALSIGNS_First_Symptoms = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_VITALSIGNS_FIRST_SYMPTOMS_INTERVALS.csv')

    # Merge (union)
    df_all_delta_merge_first_symptoms = merge_interval_dataframes(
        dfs_interval=[
            df_ALSFRS_First_Symptoms,
            df_FVC_First_Symptoms,
            df_LABS_First_Symptoms,
            df_HANDGRIPSTRENGTH_First_Symptoms,
            df_MUSCLESTRENGTH_First_Symptoms,
            df_SVC_First_Symptoms,
            df_VITALSIGNS_First_Symptoms,
        ],
        how="union",
    )
    fast_write_csv(df_all_delta_merge_first_symptoms, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALL_DELTA_FIRST_SYMPTOMS_UNION_INTERVALS.csv')
    # We create the prediction files for the full DELTA merge (First Symptoms version)
    generate_interval_prediction_files(
        df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALL_DELTA_FIRST_SYMPTOMS_UNION_INTERVALS.csv',
        df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
        output_dir=INTERVALS_CUT_PATH + '/ALL_DELTA_First_Symptoms_UNION/',
        interval_length=90,
        # num_intervals=17,
        generate_sliding=False,
        merge_interval_options='union'
    )

    if PAPER_RESULTS_ONLY==False:
        # Merge (intersection)
        df_all_delta_merge_first_symptoms = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_First_Symptoms,
                df_FVC_First_Symptoms,
                df_LABS_First_Symptoms,
                df_HANDGRIPSTRENGTH_First_Symptoms,
                df_MUSCLESTRENGTH_First_Symptoms,
                df_SVC_First_Symptoms,
                df_VITALSIGNS_First_Symptoms,
            ],
            how="intersection",
        )
        fast_write_csv(df_all_delta_merge_first_symptoms, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALL_DELTA_FIRST_SYMPTOMS_INTERSECTION_INTERVALS.csv')
        # We create the prediction files for the full DELTA merge (First Symptoms version)
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALL_DELTA_FIRST_SYMPTOMS_INTERSECTION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
            output_dir=INTERVALS_CUT_PATH + '/ALL_DELTA_First_Symptoms_INTERSECTION/',
            interval_length=90,
            # num_intervals=17,
            generate_sliding=False,
            merge_interval_options='intersection'
        )

    if PAPER_RESULTS_ONLY==False:
        # Merge (left)
        df_all_delta_merge_first_symptoms = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_First_Symptoms,
                df_FVC_First_Symptoms,
                df_LABS_First_Symptoms,
                df_HANDGRIPSTRENGTH_First_Symptoms,
                df_MUSCLESTRENGTH_First_Symptoms,
                df_SVC_First_Symptoms,
                df_VITALSIGNS_First_Symptoms,
            ],
            how="left",
        )
        fast_write_csv(df_all_delta_merge_first_symptoms, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALL_DELTA_FIRST_SYMPTOMS_LEFT_INTERVALS.csv')
        # We create the prediction files for the full DELTA merge (First Symptoms version)
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALL_DELTA_FIRST_SYMPTOMS_LEFT_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
            output_dir=INTERVALS_CUT_PATH + '/ALL_DELTA_First_Symptoms_LEFT/',
            interval_length=90,
            # num_intervals=17,
            generate_sliding=False,
            merge_interval_options='left'
        )





    # //////////////////////////////////////////////////////////////////
    # ------------------------- MERGE ALL DATA -------------------------
    # //////////////////////////////////////////////////////////////////

    # Reading data without DELTA
    df_NODELTA_MERGE = fast_read_csv(MERGE_NODELTA_PATH + '/PROACT_MERGE_NODELTA_V2.csv')

    # Adding data without DELTA to prediction files
    add_no_delta_to_prediction_files(
        input_dir=INTERVALS_CUT_PATH + '/ALL_DELTA_UNION/',
        output_dir=INTERVALS_CUT_PATH + '/ALL_DATA_UNION/',
        df_no_delta=df_NODELTA_MERGE,
        key='subject_id'
    )
    if PAPER_RESULTS_ONLY==False:
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/ALL_DELTA_INTERSECTION/',
            output_dir=INTERVALS_CUT_PATH + '/ALL_DATA_INTERSECTION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )
    if PAPER_RESULTS_ONLY==False:
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/ALL_DELTA_LEFT/',
            output_dir=INTERVALS_CUT_PATH + '/ALL_DATA_LEFT/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



    # Reading data without DELTA (version First Symptoms)
    df_NODELTA_FIRST_SYMPTOMS_MERGE = fast_read_csv(FIRST_SYMPTOMS_PATH + '/PROACT_MERGE_NODELTA_FIRST_SYMPTOMS.csv')

    # Adding data without DELTA to prediction files
    add_no_delta_to_prediction_files(
        input_dir=INTERVALS_CUT_PATH + '/ALL_DELTA_First_Symptoms_UNION/',
        output_dir=INTERVALS_CUT_PATH + '/ALL_DATA_First_Symptoms_UNION/',
        df_no_delta=df_NODELTA_FIRST_SYMPTOMS_MERGE,
        key='subject_id'
    )
    if PAPER_RESULTS_ONLY==False:
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/ALL_DELTA_First_Symptoms_INTERSECTION/',
            output_dir=INTERVALS_CUT_PATH + '/ALL_DATA_First_Symptoms_INTERSECTION/',
            df_no_delta=df_NODELTA_FIRST_SYMPTOMS_MERGE,
            key='subject_id'
        )
    if PAPER_RESULTS_ONLY==False:
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/ALL_DELTA_First_Symptoms_LEFT/',
            output_dir=INTERVALS_CUT_PATH + '/ALL_DATA_First_Symptoms_LEFT/',
            df_no_delta=df_NODELTA_FIRST_SYMPTOMS_MERGE,
            key='subject_id'
        )










    if PAPER_RESULTS_ONLY==False:
        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # ================================================== UNIQUE TABLES + NODELTA ==================================================
        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



        # //////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS + NODELTA -------------------------
        # //////////////////////////////////////////////////////////////////////////

        # Reading data without DELTA
        df_NODELTA_MERGE = fast_read_csv(MERGE_NODELTA_PATH + '/PROACT_MERGE_NODELTA_V2.csv')
        # Adding data without DELTA to prediction files in ALSFRS
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/ALSFRS/',
            output_dir=INTERVALS_CUT_PATH + '/ALSFRS_NODELTA/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )

        # Reading data without DELTA (version First Symptoms)
        df_NODELTA_MERGE = fast_read_csv(FIRST_SYMPTOMS_PATH + '/PROACT_MERGE_NODELTA_FIRST_SYMPTOMS.csv')
        # Adding data without DELTA to prediction files in ALSFRS_First_Symptoms
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/ALSFRS_First_Symptoms/',
            output_dir=INTERVALS_CUT_PATH + '/ALSFRS_NODELTA_First_Symptoms/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





        # ///////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE FVC + NODELTA -------------------------
        # ///////////////////////////////////////////////////////////////////////

        # Reading data without DELTA
        df_NODELTA_MERGE = fast_read_csv(MERGE_NODELTA_PATH + '/PROACT_MERGE_NODELTA_V2.csv')
        # Adding data without DELTA to prediction files in FVC
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/FVC/',
            output_dir=INTERVALS_CUT_PATH + '/FVC_NODELTA/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )

        # Reading data without DELTA (version First Symptoms)
        df_NODELTA_MERGE = fast_read_csv(FIRST_SYMPTOMS_PATH + '/PROACT_MERGE_NODELTA_FIRST_SYMPTOMS.csv')
        # Adding data without DELTA to prediction files in FVC_First_Symptoms
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/FVC_First_Symptoms/',
            output_dir=INTERVALS_CUT_PATH + '/FVC_NODELTA_First_Symptoms/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





        # ////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE HANDGRIPSTRENGTH + NODELTA -------------------------
        # ////////////////////////////////////////////////////////////////////////////////////

        # Reading data without DELTA
        df_NODELTA_MERGE = fast_read_csv(MERGE_NODELTA_PATH + '/PROACT_MERGE_NODELTA_V2.csv')
        # Adding data without DELTA to prediction files in HANDGRIPSTRENGTH
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/HANDGRIPSTRENGTH/',
            output_dir=INTERVALS_CUT_PATH + '/HANDGRIPSTRENGTH_NODELTA/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )

        # Reading data without DELTA (version First Symptoms)
        df_NODELTA_MERGE = fast_read_csv(FIRST_SYMPTOMS_PATH + '/PROACT_MERGE_NODELTA_FIRST_SYMPTOMS.csv')
        # Adding data without DELTA to prediction files in HANDGRIPSTRENGTH_First_Symptoms
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/HANDGRIPSTRENGTH_First_Symptoms/',
            output_dir=INTERVALS_CUT_PATH + '/HANDGRIPSTRENGTH_NODELTA_First_Symptoms/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





        # ////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE LABS + NODELTA -------------------------
        # ////////////////////////////////////////////////////////////////////////

        # Reading data without DELTA
        df_NODELTA_MERGE = fast_read_csv(MERGE_NODELTA_PATH + '/PROACT_MERGE_NODELTA_V2.csv')
        # Adding data without DELTA to prediction files in LABS
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/LABS/',
            output_dir=INTERVALS_CUT_PATH + '/LABS_NODELTA/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )

        # Reading data without DELTA (version First Symptoms)
        df_NODELTA_MERGE = fast_read_csv(FIRST_SYMPTOMS_PATH + '/PROACT_MERGE_NODELTA_FIRST_SYMPTOMS.csv')
        # Adding data without DELTA to prediction files in LABS_First_Symptoms
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/LABS_First_Symptoms/',
            output_dir=INTERVALS_CUT_PATH + '/LABS_NODELTA_First_Symptoms/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





        # //////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE MUSCLESTRENGTH + NODELTA -------------------------
        # //////////////////////////////////////////////////////////////////////////////////

        # Reading data without DELTA
        df_NODELTA_MERGE = fast_read_csv(MERGE_NODELTA_PATH + '/PROACT_MERGE_NODELTA_V2.csv')
        # Adding data without DELTA to prediction files in MUSCLESTRENGTH
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MUSCLESTRENGTH/',
            output_dir=INTERVALS_CUT_PATH + '/MUSCLESTRENGTH_NODELTA/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )

        # Reading data without DELTA (version First Symptoms)
        df_NODELTA_MERGE = fast_read_csv(FIRST_SYMPTOMS_PATH + '/PROACT_MERGE_NODELTA_FIRST_SYMPTOMS.csv')
        # Adding data without DELTA to prediction files in MUSCLESTRENGTH_First_Symptoms
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MUSCLESTRENGTH_First_Symptoms/',
            output_dir=INTERVALS_CUT_PATH + '/MUSCLESTRENGTH_NODELTA_First_Symptoms/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





        # ///////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE SVC + NODELTA -------------------------
        # ///////////////////////////////////////////////////////////////////////

        # Reading data without DELTA
        df_NODELTA_MERGE = fast_read_csv(MERGE_NODELTA_PATH + '/PROACT_MERGE_NODELTA_V2.csv')
        # Adding data without DELTA to prediction files in SVC
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/SVC/',
            output_dir=INTERVALS_CUT_PATH + '/SVC_NODELTA/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )

        # Reading data without DELTA (version First Symptoms)
        df_NODELTA_MERGE = fast_read_csv(FIRST_SYMPTOMS_PATH + '/PROACT_MERGE_NODELTA_FIRST_SYMPTOMS.csv')
        # Adding data without DELTA to prediction files in SVC_First_Symptoms
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/SVC_First_Symptoms/',
            output_dir=INTERVALS_CUT_PATH + '/SVC_NODELTA_First_Symptoms/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





        # //////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE VITALSIGNS + NODELTA -------------------------
        # //////////////////////////////////////////////////////////////////////////////

        # Reading data without DELTA
        df_NODELTA_MERGE = fast_read_csv(MERGE_NODELTA_PATH + '/PROACT_MERGE_NODELTA_V2.csv')
        # Adding data without DELTA to prediction files in VITALSIGNS
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/VITALSIGNS/',
            output_dir=INTERVALS_CUT_PATH + '/VITALSIGNS_NODELTA/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )

        # Reading data without DELTA (version First Symptoms)
        df_NODELTA_MERGE = fast_read_csv(FIRST_SYMPTOMS_PATH + '/PROACT_MERGE_NODELTA_FIRST_SYMPTOMS.csv')
        # Adding data without DELTA to prediction files in VITALSIGNS_First_Symptoms
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/VITALSIGNS_First_Symptoms/',
            output_dir=INTERVALS_CUT_PATH + '/VITALSIGNS_NODELTA_First_Symptoms/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )










    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # ================================================== MERGE ALSFRS + NO DELTA + OTHERS ==================================================
    # ------------------------------------------ TARGET ALSFRS STUDY - FIXED AND SLIDING WINDOWS -------------------------------------------
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # The files are merged into the table, one by one, in separate files.
    df_ALSFRS = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_ALSFRS_INTERVALS.csv')
    df_FVC = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_FVC_INTERVALS.csv')
    df_HANDGRIPSTRENGTH = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_HANDGRIPSTRENGTH_INTERVALS.csv')
    df_SVC = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_SVC_INTERVALS.csv')

    if PAPER_RESULTS_ONLY==False:
        df_LABS = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_LABS_INTERVALS.csv')
        df_MUSCLESTRENGTH = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_MUSCLESTRENGTH_INTERVALS.csv')
        df_VITALSIGNS = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_VITALSIGNS_INTERVALS.csv')

    # Reading data without DELTA
    df_NODELTA_MERGE = fast_read_csv(MERGE_NODELTA_PATH + '/PROACT_MERGE_NODELTA_V2.csv')





    # /////////////////////////////////////////////////////////////////////////////////
    # ------------------------- MERGE ALSFRS + NO DELTA + FVC -------------------------
    # /////////////////////////////////////////////////////////////////////////////////

    # Merge ALSFRS + FVC (union)
    df_alsfrs_fvc_merge = merge_interval_dataframes(
        dfs_interval=[
            df_ALSFRS,
            df_FVC,
        ],
        how="union"
    )
    fast_write_csv(df_alsfrs_fvc_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_UNION_INTERVALS.csv')
    # We create the prediction files for the merge
    generate_interval_prediction_files(
        df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_UNION_INTERVALS.csv',
        df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
        output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_UNION/',
        interval_length=90,
        # num_intervals=5,
        merge_interval_options='union'
    )
    # Adding data without DELTA to prediction files in MERGE_ALSFRS_NODELTA_FVC_UNION
    add_no_delta_to_prediction_files(
        input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_UNION/',
        output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_FVC_UNION/',
        df_no_delta=df_NODELTA_MERGE,
        key='subject_id'
    )



    if PAPER_RESULTS_ONLY==False:
        # Merge ALSFRS + FVC (intersection)
        df_alsfrs_fvc_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS,
                df_FVC,
            ],
            how="intersection"
        )
        fast_write_csv(df_alsfrs_fvc_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_INTERSECTION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_INTERSECTION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_INTERSECTION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='intersection'
        )
        # Adding data without DELTA to prediction files in MERGE_ALSFRS_NODELTA_FVC_INTERSECTION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_INTERSECTION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_FVC_INTERSECTION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



    # Merge ALSFRS + FVC (left)
    df_alsfrs_fvc_merge = merge_interval_dataframes(
        dfs_interval=[
            df_ALSFRS,
            df_FVC,
        ],
        how="left"
    )
    fast_write_csv(df_alsfrs_fvc_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_LEFT_INTERVALS.csv')
    # We create the prediction files for the merge
    generate_interval_prediction_files(
        df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_LEFT_INTERVALS.csv',
        df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
        output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_LEFT/',
        interval_length=90,
        # num_intervals=5,
        merge_interval_options='left'
    )
    # Adding data without DELTA to prediction files in MERGE_ALSFRS_NODELTA_FVC_LEFT
    add_no_delta_to_prediction_files(
        input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_LEFT/',
        output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_FVC_LEFT/',
        df_no_delta=df_NODELTA_MERGE,
        key='subject_id'
    )





    if PAPER_RESULTS_ONLY==False:
        # //////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS + NO DELTA + LABS -------------------------
        # //////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS + LABS (union)
        df_alsfrs_labs_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS,
                df_LABS,
            ],
            how="union"
        )
        fast_write_csv(df_alsfrs_labs_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_LABS_UNION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_LABS_UNION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_LABS_UNION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='union'
        )
        # Adding data without DELTA to prediction files in MERGE_ALSFRS_LABS_UNION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_LABS_UNION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_LABS_UNION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



        # Merge ALSFRS + LABS (intersection)
        df_alsfrs_labs_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS,
                df_LABS,
            ],
            how="intersection"
        )
        fast_write_csv(df_alsfrs_labs_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_LABS_INTERSECTION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_LABS_INTERSECTION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_LABS_INTERSECTION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='intersection'
        )
        # Adding data without DELTA to prediction files in MERGE_ALSFRS_LABS_INTERSECTION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_LABS_INTERSECTION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_LABS_INTERSECTION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



        # Merge ALSFRS + LABS (left)
        df_alsfrs_labs_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS,
                df_LABS,
            ],
            how="left"
        )
        fast_write_csv(df_alsfrs_labs_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_LABS_LEFT_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_LABS_LEFT_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_LABS_LEFT/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='left'
        )
        # Adding data without DELTA to prediction files in MERGE_ALSFRS_LABS_LEFT
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_LABS_LEFT/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_LABS_LEFT/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





    # //////////////////////////////////////////////////////////////////////////////////////////////
    # ------------------------- MERGE ALSFRS + NO DELTA + HANDGRIPSTRENGTH -------------------------
    # //////////////////////////////////////////////////////////////////////////////////////////////

    # Merge ALSFRS + HANDGRIPSTRENGTH (union)
    df_alsfrs_handgrip_merge = merge_interval_dataframes(
        dfs_interval=[
            df_ALSFRS,
            df_HANDGRIPSTRENGTH,
        ],
        how="union"
    )
    fast_write_csv(df_alsfrs_handgrip_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_UNION_INTERVALS.csv')
    # We create the prediction files for the merge
    generate_interval_prediction_files(
        df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_UNION_INTERVALS.csv',
        df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
        output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_UNION/',
        interval_length=90,
        # num_intervals=5,
        merge_interval_options='union'
    )
    # Adding data without DELTA to prediction files in MERGE_ALSFRS_HANDGRIPSTRENGTH_UNION
    add_no_delta_to_prediction_files(
        input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_UNION/',
        output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_UNION/',
        df_no_delta=df_NODELTA_MERGE,
        key='subject_id'
    )



    if PAPER_RESULTS_ONLY==False:
        # Merge ALSFRS + HANDGRIPSTRENGTH (intersection)
        df_alsfrs_handgrip_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS,
                df_HANDGRIPSTRENGTH,
            ],
            how="intersection"
        )
        fast_write_csv(df_alsfrs_handgrip_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_INTERSECTION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_INTERSECTION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_INTERSECTION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='intersection'
        )
        # Adding data without DELTA to prediction files in MERGE_ALSFRS_HANDGRIPSTRENGTH_INTERSECTION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_INTERSECTION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_INTERSECTION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



    # Merge ALSFRS + HANDGRIPSTRENGTH (left)
    df_alsfrs_handgrip_merge = merge_interval_dataframes(
        dfs_interval=[
            df_ALSFRS,
            df_HANDGRIPSTRENGTH,
        ],
        how="left"
    )
    fast_write_csv(df_alsfrs_handgrip_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_LEFT_INTERVALS.csv')
    # We create the prediction files for the merge
    generate_interval_prediction_files(
        df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_LEFT_INTERVALS.csv',
        df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
        output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_LEFT/',
        interval_length=90,
        # num_intervals=5,
        merge_interval_options='left'
    )
    # Adding data without DELTA to prediction files in MERGE_ALSFRS_HANDGRIPSTRENGTH_LEFT
    add_no_delta_to_prediction_files(
        input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_LEFT/',
        output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_LEFT/',
        df_no_delta=df_NODELTA_MERGE,
        key='subject_id'
    )





    if PAPER_RESULTS_ONLY==False:
        # ////////////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS + NO DELTA + MUSCLESTRENGTH -------------------------
        # ////////////////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS + MUSCLESTRENGTH (union)
        df_alsfrs_muscle_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS,
                df_MUSCLESTRENGTH,
            ],
            how="union"
        )
        fast_write_csv(df_alsfrs_muscle_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_MUSCLESTRENGTH_UNION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_MUSCLESTRENGTH_UNION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_MUSCLESTRENGTH_UNION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='union'
        )
        # Adding data without DELTA to prediction files in MERGE_ALSFRS_MUSCLESTRENGTH_UNION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_MUSCLESTRENGTH_UNION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_MUSCLESTRENGTH_UNION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



        # Merge ALSFRS + MUSCLESTRENGTH (intersection)
        df_alsfrs_muscle_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS,
                df_MUSCLESTRENGTH,
            ],
            how="intersection"
        )
        fast_write_csv(df_alsfrs_muscle_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_MUSCLESTRENGTH_INTERSECTION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_MUSCLESTRENGTH_INTERSECTION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_MUSCLESTRENGTH_INTERSECTION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='intersection'
        )
        # Adding data without DELTA to prediction files in MERGE_ALSFRS_MUSCLESTRENGTH_INTERSECTION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_MUSCLESTRENGTH_INTERSECTION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_MUSCLESTRENGTH_INTERSECTION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



        # Merge ALSFRS + MUSCLESTRENGTH (left)
        df_alsfrs_muscle_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS,
                df_MUSCLESTRENGTH,
            ],
            how="left"
        )
        fast_write_csv(df_alsfrs_muscle_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_MUSCLESTRENGTH_LEFT_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_MUSCLESTRENGTH_LEFT_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_MUSCLESTRENGTH_LEFT/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='left'
        )
        # Adding data without DELTA to prediction files in MERGE_ALSFRS_MUSCLESTRENGTH_LEFT
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_MUSCLESTRENGTH_LEFT/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_MUSCLESTRENGTH_LEFT/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





    # /////////////////////////////////////////////////////////////////////////////////
    # ------------------------- MERGE ALSFRS + NO DELTA + SVC -------------------------
    # /////////////////////////////////////////////////////////////////////////////////

    if PAPER_RESULTS_ONLY==False:
        # Merge ALSFRS + SVC (union)
        df_alsfrs_svc_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS,
                df_SVC,
            ],
            how="union"
        )
        fast_write_csv(df_alsfrs_svc_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_UNION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_UNION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_UNION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='union'
        )
        # Adding data without DELTA to prediction files in MERGE_ALSFRS_SVC_UNION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_UNION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_SVC_UNION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



    # Merge ALSFRS + SVC (intersection)
    df_alsfrs_svc_merge = merge_interval_dataframes(
        dfs_interval=[
            df_ALSFRS,
            df_SVC,
        ],
        how="intersection"
    )
    fast_write_csv(df_alsfrs_svc_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_INTERSECTION_INTERVALS.csv')
    # We create the prediction files for the merge
    generate_interval_prediction_files(
        df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_INTERSECTION_INTERVALS.csv',
        df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
        output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_INTERSECTION/',
        interval_length=90,
        # num_intervals=5,
        merge_interval_options='intersection'
    )
    # Adding data without DELTA to prediction files in MERGE_ALSFRS_SVC_INTERSECTION
    add_no_delta_to_prediction_files(
        input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_INTERSECTION/',
        output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_SVC_INTERSECTION/',
        df_no_delta=df_NODELTA_MERGE,
        key='subject_id'
    )



    if PAPER_RESULTS_ONLY==False:
        # Merge ALSFRS + SVC (left)
        df_alsfrs_svc_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS,
                df_SVC,
            ],
            how="left"
        )
        fast_write_csv(df_alsfrs_svc_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_LEFT_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_LEFT_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_LEFT/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='left'
        )
        # Adding data without DELTA to prediction files in MERGE_ALSFRS_SVC_LEFT
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_LEFT/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_SVC_LEFT/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





    if PAPER_RESULTS_ONLY==False:
        # ////////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS + NO DELTA + VITALSIGNS -------------------------
        # ////////////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS + VITALSIGNS (union)
        df_alsfrs_vitalsigns_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS,
                df_VITALSIGNS,
            ],
            how='union'
        )
        fast_write_csv(df_alsfrs_vitalsigns_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_VITALSIGNS_UNION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_VITALSIGNS_UNION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_VITALSIGNS_UNION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='union'
        )
        # Adding data without DELTA to prediction files in MERGE_ALSFRS_VITALSIGNS_UNION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_VITALSIGNS_UNION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_VITALSIGNS_UNION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



        # Merge ALSFRS + VITALSIGNS (intersection)
        df_alsfrs_vitalsigns_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS,
                df_VITALSIGNS,
            ],
            how='intersection'
        )
        fast_write_csv(df_alsfrs_vitalsigns_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_VITALSIGNS_INTERSECTION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_VITALSIGNS_INTERSECTION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_VITALSIGNS_INTERSECTION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='intersection'
        )
        # Adding data without DELTA to prediction files in MERGE_ALSFRS_VITALSIGNS_INTERSECTION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_VITALSIGNS_INTERSECTION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_VITALSIGNS_INTERSECTION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



        # Merge ALSFRS + VITALSIGNS (left)
        df_alsfrs_vitalsigns_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS,
                df_VITALSIGNS,
            ],
            how='left'
        )
        fast_write_csv(df_alsfrs_vitalsigns_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_VITALSIGNS_LEFT_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_VITALSIGNS_LEFT_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_VITALSIGNS_LEFT/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='left'
        )
        # Adding data without DELTA to prediction files in MERGE_ALSFRS_VITALSIGNS_LEFT
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_VITALSIGNS_LEFT/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_VITALSIGNS_LEFT/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )










    # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # ================================================== MERGE ALSFRS + NO DELTA + HANDGRIPSTRENGTH + OTHERS : UNION & LEFT ==================================================
    # ----------------------------------------------------------------- TARGET ALSFRS STUDY - FIXED WINDOWS ------------------------------------------------------------------
    # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # We merge the files, one by one, into separate files
    df_ALSFRS_HANDGRIP_UNION = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_UNION_INTERVALS.csv')
    df_ALSFRS_HANDGRIP_LEFT = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_LEFT_INTERVALS.csv')
    df_FVC = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_FVC_INTERVALS.csv')
    
    if PAPER_RESULTS_ONLY==False:
        df_LABS = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_LABS_INTERVALS.csv')
        df_MUSCLESTRENGTH = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_MUSCLESTRENGTH_INTERVALS.csv')
        df_SVC = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_SVC_INTERVALS.csv')
        df_VITALSIGNS = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_VITALSIGNS_INTERVALS.csv')

    # Reading data without DELTA
    df_NODELTA_MERGE = fast_read_csv(MERGE_NODELTA_PATH + '/PROACT_MERGE_NODELTA_V2.csv')





    # //////////////////////////////////////////////////////////////////////////////////////////////////
    # ------------------------- MERGE ALSFRS HANDGRIPSTRENGTH + NO DELTA + FVC -------------------------
    # //////////////////////////////////////////////////////////////////////////////////////////////////

    # Merge ALSFRS HANDGRIPSTRENGTH + FVC (union)
    df_alsfrs_handgrip_fvc_merge = merge_interval_dataframes(
        dfs_interval=[
            df_ALSFRS_HANDGRIP_UNION,
            df_FVC,
        ],
        how="union"
    )
    fast_write_csv(df_alsfrs_handgrip_fvc_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_UNION_INTERVALS.csv')
    # We create the prediction files for the merge
    generate_interval_prediction_files(
        df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_UNION_INTERVALS.csv',
        df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
        output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_UNION/',
        interval_length=90,
        # num_intervals=5,
        merge_interval_options='union',
        generate_sliding=False
    )
    # Adding data without DELTA to prediction files in MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_FVC_UNION
    add_no_delta_to_prediction_files(
        input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_UNION/',
        output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_FVC_UNION/',
        df_no_delta=df_NODELTA_MERGE,
        key='subject_id'
    )



    # Merge ALSFRS HANDGRIPSTRENGTH + FVC (left)
    df_alsfrs_handgrip_fvc_merge = merge_interval_dataframes(
        dfs_interval=[
            df_ALSFRS_HANDGRIP_LEFT,
            df_FVC,
        ],
        how="left"
    )
    fast_write_csv(df_alsfrs_handgrip_fvc_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_LEFT_INTERVALS.csv')
    # We create the prediction files for the merge
    generate_interval_prediction_files(
        df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_LEFT_INTERVALS.csv',
        df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
        output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_LEFT/',
        interval_length=90,
        # num_intervals=5,
        merge_interval_options='left',
        generate_sliding=False
    )
    # Adding data without DELTA to prediction files in MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_FVC_LEFT
    add_no_delta_to_prediction_files(
        input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_LEFT/',
        output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_FVC_LEFT/',
        df_no_delta=df_NODELTA_MERGE,
        key='subject_id'
    )





    if PAPER_RESULTS_ONLY==False:
        # ///////////////////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS HANDGRIPSTRENGTH + NO DELTA + LABS -------------------------
        # ///////////////////////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS HANDGRIPSTRENGTH + LABS (union)
        df_alsfrs_handgrip_labs_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_HANDGRIP_UNION,
                df_LABS,
            ],
            how="union"
        )
        fast_write_csv(df_alsfrs_handgrip_labs_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_LABS_UNION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_LABS_UNION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_LABS_UNION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='union',
            generate_sliding=False
        )
        # Adding data without DELTA to prediction files in MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_LABS_UNION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_LABS_UNION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_LABS_UNION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



        # Merge ALSFRS HANDGRIPSTRENGTH + LABS (left)
        df_alsfrs_handgrip_labs_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_HANDGRIP_LEFT,
                df_LABS,
            ],
            how="left"
        )
        fast_write_csv(df_alsfrs_handgrip_labs_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_LABS_LEFT_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_LABS_LEFT_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_LABS_LEFT/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='left',
            generate_sliding=False
        )
        # Adding data without DELTA to prediction files in MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_LABS_LEFT
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_LABS_LEFT/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_LABS_LEFT/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





    if PAPER_RESULTS_ONLY==False:
        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS HANDGRIPSTRENGTH + NO DELTA + MUSCLESTRENGTH -------------------------
        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS HANDGRIPSTRENGTH + MUSCLESTRENGTH (union)
        df_alsfrs_handgrip_muscle_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_HANDGRIP_UNION,
                df_MUSCLESTRENGTH,
            ],
            how="union"
        )
        fast_write_csv(df_alsfrs_handgrip_muscle_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_MUSCLESTRENGTH_UNION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_MUSCLESTRENGTH_UNION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_MUSCLESTRENGTH_UNION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='union',
            generate_sliding=False
        )
        # Adding data without DELTA to prediction files in MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_MUSCLESTRENGTH_UNION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_MUSCLESTRENGTH_UNION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_MUSCLESTRENGTH_UNION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



        # Merge ALSFRS HANDGRIPSTRENGTH + MUSCLESTRENGTH (left)
        df_alsfrs_handgrip_muscle_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_HANDGRIP_LEFT,
                df_MUSCLESTRENGTH,
            ],
            how="left"
        )
        fast_write_csv(df_alsfrs_handgrip_muscle_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_MUSCLESTRENGTH_LEFT_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_MUSCLESTRENGTH_LEFT_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_MUSCLESTRENGTH_LEFT/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='left',
            generate_sliding=False
        )
        # Adding data without DELTA to prediction files in MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_MUSCLESTRENGTH_LEFT
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_MUSCLESTRENGTH_LEFT/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_MUSCLESTRENGTH_LEFT/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





    if PAPER_RESULTS_ONLY==False:
        # //////////////////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS HANDGRIPSTRENGTH + NO DELTA + SVC -------------------------
        # //////////////////////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS HANDGRIPSTRENGTH + SVC (union)
        df_alsfrs_handgrip_svc_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_HANDGRIP_UNION,
                df_SVC,
            ],
            how="union"
        )
        fast_write_csv(df_alsfrs_handgrip_svc_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_SVC_UNION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_SVC_UNION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_SVC_UNION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='union',
            generate_sliding=False
        )
        # Adding data without DELTA to prediction files in MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_SVC_UNION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_SVC_UNION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_SVC_UNION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



        # Merge ALSFRS HANDGRIPSTRENGTH + SVC (left)
        df_alsfrs_handgrip_svc_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_HANDGRIP_LEFT,
                df_SVC,
            ],
            how="left"
        )
        fast_write_csv(df_alsfrs_handgrip_svc_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_SVC_LEFT_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_SVC_LEFT_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_SVC_LEFT/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='left',
            generate_sliding=False
        )
        # Adding data without DELTA to prediction files in MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_SVC_LEFT
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_SVC_LEFT/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_SVC_LEFT/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





    
    if PAPER_RESULTS_ONLY==False:
        # /////////////////////////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS HANDGRIPSTRENGTH + NO DELTA + VITALSIGNS -------------------------
        # /////////////////////////////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS HANDGRIPSTRENGTH + VITALSIGNS (union)
        df_alsfrs_handgrip_vitalsigns_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_HANDGRIP_UNION,
                df_VITALSIGNS,
            ],
            how='union'
        )
        fast_write_csv(df_alsfrs_handgrip_vitalsigns_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_VITALSIGNS_UNION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_VITALSIGNS_UNION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_VITALSIGNS_UNION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='union',
            generate_sliding=False
        )
        # Adding data without DELTA to prediction files in MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_VITALSIGNS_UNION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_VITALSIGNS_UNION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_VITALSIGNS_UNION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



        # Merge ALSFRS HANDGRIPSTRENGTH + VITALSIGNS (left)
        df_alsfrs_handgrip_vitalsigns_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_HANDGRIP_LEFT,
                df_VITALSIGNS,
            ],
            how='left'
        )
        fast_write_csv(df_alsfrs_handgrip_vitalsigns_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_VITALSIGNS_LEFT_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_VITALSIGNS_LEFT_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_VITALSIGNS_LEFT/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='left',
            generate_sliding=False
        )
        # Adding data without DELTA to prediction files in MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_VITALSIGNS_LEFT
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_VITALSIGNS_LEFT/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_VITALSIGNS_LEFT/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )










    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # ================================================== MERGE ALSFRS + NO DELTA + SVC + OTHERS : INTERSECTION ==================================================
    # ----------------------------------------------------- TARGET ALSFRS STUDY - FIXED AND SLIDING WINDOWS -----------------------------------------------------
    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # We merge the files, one by one, into separate files
    df_ALSFRS_SVC_INTERSECTION = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_INTERSECTION_INTERVALS.csv')
    df_LABS = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_LABS_INTERVALS.csv')

    if PAPER_RESULTS_ONLY==False:
        df_FVC = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_FVC_INTERVALS.csv')
        df_HANDGRIPSTRENGTH = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_HANDGRIPSTRENGTH_INTERVALS.csv')
        df_MUSCLESTRENGTH = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_MUSCLESTRENGTH_INTERVALS.csv')
        df_VITALSIGNS = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_VITALSIGNS_INTERVALS.csv')

    # Reading data without DELTA
    df_NODELTA_MERGE = fast_read_csv(MERGE_NODELTA_PATH + '/PROACT_MERGE_NODELTA_V2.csv')





    if PAPER_RESULTS_ONLY==False:
        # /////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS SVC + NO DELTA + FVC -------------------------
        # /////////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS SVC + FVC (intersection)
        df_alsfrs_svc_fvc_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_SVC_INTERSECTION,
                df_FVC,
            ],
            how="intersection"
        )
        fast_write_csv(df_alsfrs_svc_fvc_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_FVC_INTERSECTION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_FVC_INTERSECTION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_FVC_INTERSECTION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='intersection'
        )
        # Adding data without DELTA to prediction files in MERGE_ALSFRS_NODELTA_FVC_INTERSECTION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_FVC_INTERSECTION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_SVC_FVC_INTERSECTION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





    # //////////////////////////////////////////////////////////////////////////////////////
    # ------------------------- MERGE ALSFRS SVC + NO DELTA + LABS -------------------------
    # //////////////////////////////////////////////////////////////////////////////////////

    # Merge ALSFRS SVC + LABS (intersection)
    df_alsfrs_svc_labs_merge = merge_interval_dataframes(
        dfs_interval=[
            df_ALSFRS_SVC_INTERSECTION,
            df_LABS,
        ],
        how="intersection"
    )
    fast_write_csv(df_alsfrs_svc_labs_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_LABS_INTERSECTION_INTERVALS.csv')
    # We create the prediction files for the merge
    generate_interval_prediction_files(
        df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_LABS_INTERSECTION_INTERVALS.csv',
        df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
        output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_LABS_INTERSECTION/',
        interval_length=90,
        # num_intervals=5,
        merge_interval_options='intersection'
    )
    # Adding data without DELTA to prediction files in MERGE_ALSFRS_NODELTA_SVC_LABS_INTERSECTION
    add_no_delta_to_prediction_files(
        input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_LABS_INTERSECTION/',
        output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_SVC_LABS_INTERSECTION/',
        df_no_delta=df_NODELTA_MERGE,
        key='subject_id'
    )





    if PAPER_RESULTS_ONLY==False:
        # //////////////////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS SVC + NO DELTA + HANDGRIPSTRENGTH -------------------------
        # //////////////////////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS SVC + HANDGRIPSTRENGTH (intersection)
        df_alsfrs_svc_handgrip_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_SVC_INTERSECTION,
                df_HANDGRIPSTRENGTH,
            ],
            how="intersection"
        )
        fast_write_csv(df_alsfrs_svc_handgrip_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_HANDGRIPSTRENGTH_INTERSECTION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_HANDGRIPSTRENGTH_INTERSECTION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_HANDGRIPSTRENGTH_INTERSECTION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='intersection'
        )
        # Adding data without DELTA to prediction files in MERGE_ALSFRS_NODELTA_SVC_HANDGRIPSTRENGTH_INTERSECTION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_HANDGRIPSTRENGTH_INTERSECTION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_SVC_HANDGRIPSTRENGTH_INTERSECTION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





    if PAPER_RESULTS_ONLY==False:
        # ////////////////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS SVC + NO DELTA + MUSCLESTRENGTH -------------------------
        # ////////////////////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS SVC + MUSCLESTRENGTH (intersection)
        df_alsfrs_svc_muscle_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_SVC_INTERSECTION,
                df_MUSCLESTRENGTH,
            ],
            how="intersection"
        )
        fast_write_csv(df_alsfrs_svc_muscle_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_MUSCLESTRENGTH_INTERSECTION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_MUSCLESTRENGTH_INTERSECTION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_MUSCLESTRENGTH_INTERSECTION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='intersection'
        )
        # Adding data without DELTA to prediction files in MERGE_ALSFRS_NODELTA_SVC_MUSCLESTRENGTH_INTERSECTION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_MUSCLESTRENGTH_INTERSECTION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_SVC_MUSCLESTRENGTH_INTERSECTION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





    if PAPER_RESULTS_ONLY==False:
        # ////////////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS SVC + NO DELTA + VITALSIGNS -------------------------
        # ////////////////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS SVC + VITALSIGNS (intersection)
        df_alsfrs_svc_vitalsigns_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_SVC_INTERSECTION,
                df_VITALSIGNS,
            ],
            how='intersection'
        )
        fast_write_csv(df_alsfrs_svc_vitalsigns_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_VITALSIGNS_INTERSECTION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_VITALSIGNS_INTERSECTION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_VITALSIGNS_INTERSECTION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='intersection'
        )
        # Adding data without DELTA to prediction files in MERGE_ALSFRS_NODELTA_SVC_VITALSIGNS_INTERSECTION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_VITALSIGNS_INTERSECTION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_SVC_VITALSIGNS_INTERSECTION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )










    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # ================================================== MERGE ALSFRS + NO DELTA + FVC + OTHERS : UNION & LEFT ==================================================
    # ---------------------------------------------------------- TARGET ALSFRS STUDY - SLIDING WINDOWS ----------------------------------------------------------
    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # We merge the files, one by one, into separate files
    df_ALSFRS_FVC_UNION = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_UNION_INTERVALS.csv')
    df_ALSFRS_FVC_LEFT = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_LEFT_INTERVALS.csv')
    df_SVC = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_SVC_INTERVALS.csv')

    if PAPER_RESULTS_ONLY==False:
        df_LABS = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_LABS_INTERVALS.csv')
        df_HANDGRIPSTRENGTH = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_HANDGRIPSTRENGTH_INTERVALS.csv')
        df_MUSCLESTRENGTH = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_MUSCLESTRENGTH_INTERVALS.csv')
        df_VITALSIGNS = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_VITALSIGNS_INTERVALS.csv')

    # Reading data without DELTA
    df_NODELTA_MERGE = fast_read_csv(MERGE_NODELTA_PATH + '/PROACT_MERGE_NODELTA_V2.csv')





    if PAPER_RESULTS_ONLY==False:
        # //////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS FVC + NO DELTA + LABS -------------------------
        # //////////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS FVC + LABS (union)
        df_alsfrs_fvc_labs_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_FVC_UNION,
                df_LABS,
            ],
            how="union"
        )
        fast_write_csv(df_alsfrs_fvc_labs_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_LABS_UNION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_LABS_UNION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_LABS_UNION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='union',
            generate_fixed=False
        )
        # Adding data without DELTA to prediction files in MERGE_ALSFRS_NODELTA_FVC_LABS_UNION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_LABS_UNION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_FVC_LABS_UNION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



        # Merge ALSFRS FVC + LABS (left)
        df_alsfrs_fvc_labs_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_FVC_LEFT,
                df_LABS,
            ],
            how="left"
        )
        fast_write_csv(df_alsfrs_fvc_labs_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_LABS_LEFT_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_LABS_LEFT_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_LABS_LEFT/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='left',
            generate_fixed=False
        )
        # Adding data without DELTA to prediction files in MERGE_ALSFRS_NODELTA_FVC_LABS_LEFT
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_LABS_LEFT/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_FVC_LABS_LEFT/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





    if PAPER_RESULTS_ONLY==False:
        # //////////////////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS FVC + NO DELTA + HANDGRIPSTRENGTH -------------------------
        # //////////////////////////////////////////////////////////////////////////////////////////////////



        # Merge ALSFRS FVC + HANDGRIPSTRENGTH (union)
        df_alsfrs_fvc_handgrip_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_FVC_UNION,
                df_HANDGRIPSTRENGTH,
            ],
            how="union"
        )
        fast_write_csv(df_alsfrs_fvc_handgrip_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_HANDGRIPSTRENGTH_UNION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_HANDGRIPSTRENGTH_UNION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_HANDGRIPSTRENGTH_UNION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='union',
            generate_fixed=False
        )
        # Adding data without DELTA to prediction files in MERGE_ALSFRS_NODELTA_FVC_HANDGRIPSTRENGTH_UNION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_HANDGRIPSTRENGTH_UNION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_FVC_HANDGRIPSTRENGTH_UNION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



        # Merge ALSFRS FVC + HANDGRIPSTRENGTH (left)
        df_alsfrs_fvc_handgrip_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_FVC_LEFT,
                df_HANDGRIPSTRENGTH,
            ],
            how="left"
        )
        fast_write_csv(df_alsfrs_fvc_handgrip_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_HANDGRIPSTRENGTH_LEFT_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_HANDGRIPSTRENGTH_LEFT_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_HANDGRIPSTRENGTH_LEFT/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='left',
            generate_fixed=False
        )
        # Adding data without DELTA to prediction files in MERGE_ALSFRS_NODELTA_FVC_HANDGRIPSTRENGTH_LEFT
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_HANDGRIPSTRENGTH_LEFT/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_FVC_HANDGRIPSTRENGTH_LEFT/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





    if PAPER_RESULTS_ONLY==False:
        # ////////////////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS FVC + NO DELTA + MUSCLESTRENGTH -------------------------
        # ////////////////////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS FVC + MUSCLESTRENGTH (union)
        df_alsfrs_fvc_muscle_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_FVC_UNION,
                df_MUSCLESTRENGTH,
            ],
            how="union"
        )
        fast_write_csv(df_alsfrs_fvc_muscle_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_MUSCLESTRENGTH_UNION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_MUSCLESTRENGTH_UNION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_MUSCLESTRENGTH_UNION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='union',
            generate_fixed=False
        )
        # Adding data without DELTA to prediction files in MERGE_ALSFRS_NODELTA_FVC_MUSCLESTRENGTH_UNION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_MUSCLESTRENGTH_UNION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_FVC_MUSCLESTRENGTH_UNION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



        # Merge ALSFRS FVC + MUSCLESTRENGTH (left)
        df_alsfrs_fvc_muscle_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_FVC_LEFT,
                df_MUSCLESTRENGTH,
            ],
            how="left"
        )
        fast_write_csv(df_alsfrs_fvc_muscle_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_MUSCLESTRENGTH_LEFT_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_MUSCLESTRENGTH_LEFT_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_MUSCLESTRENGTH_LEFT/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='left',
            generate_fixed=False
        )
        # Adding data without DELTA to prediction files in MERGE_ALSFRS_NODELTA_FVC_MUSCLESTRENGTH_LEFT
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_MUSCLESTRENGTH_LEFT/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_FVC_MUSCLESTRENGTH_LEFT/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





    # /////////////////////////////////////////////////////////////////////////////////////
    # ------------------------- MERGE ALSFRS FVC + NO DELTA + SVC -------------------------
    # /////////////////////////////////////////////////////////////////////////////////////

    # Merge ALSFRS FVC + SVC (union)
    df_alsfrs_fvc_svc_merge = merge_interval_dataframes(
        dfs_interval=[
            df_ALSFRS_FVC_UNION,
            df_SVC,
        ],
        how="union"
    )
    fast_write_csv(df_alsfrs_fvc_svc_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_SVC_UNION_INTERVALS.csv')
    # We create the prediction files for the merge
    generate_interval_prediction_files(
        df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_SVC_UNION_INTERVALS.csv',
        df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
        output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_SVC_UNION/',
        interval_length=90,
        # num_intervals=5,
        merge_interval_options='union',
        generate_fixed=False
    )
    # Adding data without DELTA to prediction files in MERGE_ALSFRS_NODELTA_FVC_SVC_UNION
    add_no_delta_to_prediction_files(
        input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_SVC_UNION/',
        output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_FVC_SVC_UNION/',
        df_no_delta=df_NODELTA_MERGE,
        key='subject_id'
    )



    # Merge ALSFRS FVC + SVC (left)
    df_alsfrs_fvc_svc_merge = merge_interval_dataframes(
        dfs_interval=[
            df_ALSFRS_FVC_LEFT,
            df_SVC,
        ],
        how="left"
    )
    fast_write_csv(df_alsfrs_fvc_svc_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_SVC_LEFT_INTERVALS.csv')
    # We create the prediction files for the merge
    generate_interval_prediction_files(
        df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_SVC_LEFT_INTERVALS.csv',
        df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
        output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_SVC_LEFT/',
        interval_length=90,
        # num_intervals=5,
        merge_interval_options='left',
        generate_fixed=False
    )
    # Adding data without DELTA to prediction files in MERGE_ALSFRS_NODELTA_FVC_SVC_LEFT
    add_no_delta_to_prediction_files(
        input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_SVC_LEFT/',
        output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_FVC_SVC_LEFT/',
        df_no_delta=df_NODELTA_MERGE,
        key='subject_id'
    )





    if PAPER_RESULTS_ONLY==False:
        # ////////////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS FVC + NO DELTA + VITALSIGNS -------------------------
        # ////////////////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS FVC + VITALSIGNS (union)
        df_alsfrs_fvc_vitalsigns_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_FVC_UNION,
                df_VITALSIGNS,
            ],
            how='union'
        )
        fast_write_csv(df_alsfrs_fvc_vitalsigns_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_VITALSIGNS_UNION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_VITALSIGNS_UNION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_VITALSIGNS_UNION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='union',
            generate_fixed=False
        )
        # Adding data without DELTA to prediction files in MERGE_ALSFRS_NODELTA_FVC_VITALSIGNS_UNION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_VITALSIGNS_UNION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_FVC_VITALSIGNS_UNION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



        # Merge ALSFRS FVC + VITALSIGNS (left)
        df_alsfrs_fvc_vitalsigns_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_FVC_LEFT,
                df_VITALSIGNS,
            ],
            how='left'
        )
        fast_write_csv(df_alsfrs_fvc_vitalsigns_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_VITALSIGNS_LEFT_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_VITALSIGNS_LEFT_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_VITALSIGNS_LEFT/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='left',
            generate_fixed=False
        )
        # Adding data without DELTA to prediction files in MERGE_ALSFRS_NODELTA_FVC_VITALSIGNS_LEFT
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_VITALSIGNS_LEFT/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_FVC_VITALSIGNS_LEFT/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )










    if PAPER_RESULTS_ONLY==False:
        # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # ================================================== MERGE ALSFRS + NO DELTA + HANDGRIPSTRENGTH + FVC + OTHERS : UNION & LEFT ==================================================
        # -------------------------------------------------------------------- TARGET ALSFRS STUDY - FIXED WINDOWS ---------------------------------------------------------------------
        # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # We merge the files, one by one, into separate files
        df_ALSFRS_HANDGRIP_FVC_UNION = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_UNION_INTERVALS.csv')
        df_ALSFRS_HANDGRIP_FVC_LEFT = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_LEFT_INTERVALS.csv')
        df_LABS = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_LABS_INTERVALS.csv')
        df_MUSCLESTRENGTH = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_MUSCLESTRENGTH_INTERVALS.csv')
        df_SVC = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_SVC_INTERVALS.csv')
        df_VITALSIGNS = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_VITALSIGNS_INTERVALS.csv')

        # Reading data without DELTA
        df_NODELTA_MERGE = fast_read_csv(MERGE_NODELTA_PATH + '/PROACT_MERGE_NODELTA_V2.csv')





        # ///////////////////////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS HANDGRIPSTRENGTH FVC + NO DELTA + LABS -------------------------
        # ///////////////////////////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS HANDGRIPSTRENGTH FVC + LABS (union)
        df_alsfrs_handgrip_fvc_labs_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_HANDGRIP_FVC_UNION,
                df_LABS,
            ],
            how="union"
        )
        fast_write_csv(df_alsfrs_handgrip_fvc_labs_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_LABS_UNION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_LABS_UNION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_LABS_UNION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='union',
            generate_sliding=False
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_FVC_LABS_UNION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_LABS_UNION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_FVC_LABS_UNION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



        # Merge ALSFRS HANDGRIPSTRENGTH FVC + LABS (left)
        df_alsfrs_handgrip_fvc_labs_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_HANDGRIP_FVC_LEFT,
                df_LABS,
            ],
            how="left"
        )
        fast_write_csv(df_alsfrs_handgrip_fvc_labs_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_LABS_LEFT_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_LABS_LEFT_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_LABS_LEFT/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='left',
            generate_sliding=False
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_FVC_LABS_LEFT
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_LABS_LEFT/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_FVC_LABS_LEFT/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS HANDGRIPSTRENGTH FVC + NO DELTA + MUSCLESTRENGTH -------------------------
        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS HANDGRIPSTRENGTH FVC + MUSCLESTRENGTH (union)
        df_alsfrs_handgrip_fvc_muscle_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_HANDGRIP_FVC_UNION,
                df_MUSCLESTRENGTH,
            ],
            how="union"
        )
        fast_write_csv(df_alsfrs_handgrip_fvc_muscle_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_MUSCLESTRENGTH_UNION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_MUSCLESTRENGTH_UNION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_MUSCLESTRENGTH_UNION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='union',
            generate_sliding=False
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_FVC_MUSCLESTRENGTH_UNION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_MUSCLESTRENGTH_UNION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_FVC_MUSCLESTRENGTH_UNION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



        # Merge ALSFRS HANDGRIPSTRENGTH FVC + MUSCLESTRENGTH (left)
        df_alsfrs_handgrip_fvc_muscle_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_HANDGRIP_FVC_LEFT,
                df_MUSCLESTRENGTH,
            ],
            how="left"
        )
        fast_write_csv(df_alsfrs_handgrip_fvc_muscle_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_MUSCLESTRENGTH_LEFT_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_MUSCLESTRENGTH_LEFT_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_MUSCLESTRENGTH_LEFT/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='left',
            generate_sliding=False
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_FVC_MUSCLESTRENGTH_LEFT
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_MUSCLESTRENGTH_LEFT/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_FVC_MUSCLESTRENGTH_LEFT/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





        # //////////////////////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS HANDGRIPSTRENGTH FVC + NO DELTA + SVC -------------------------
        # //////////////////////////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS HANDGRIPSTRENGTH FVC + SVC (union)
        df_alsfrs_handgrip_fvc_svc_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_HANDGRIP_FVC_UNION,
                df_SVC,
            ],
            how="union"
        )
        fast_write_csv(df_alsfrs_handgrip_fvc_svc_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_SVC_UNION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_SVC_UNION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_SVC_UNION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='union',
            generate_sliding=False
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_FVC_SVC_UNION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_SVC_UNION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_FVC_SVC_UNION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



        # Merge ALSFRS HANDGRIPSTRENGTH FVC + SVC (left)
        df_alsfrs_handgrip_fvc_svc_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_HANDGRIP_FVC_LEFT,
                df_SVC,
            ],
            how="left"
        )
        fast_write_csv(df_alsfrs_handgrip_fvc_svc_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_SVC_LEFT_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_SVC_LEFT_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_SVC_LEFT/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='left',
            generate_sliding=False
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_FVC_SVC_LEFT
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_SVC_LEFT/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_FVC_SVC_LEFT/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS HANDGRIPSTRENGTH FVC + NO DELTA + VITALSIGNS -------------------------
        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS HANDGRIPSTRENGTH FVC + VITALSIGNS (union)
        df_alsfrs_handgrip_fvc_vitalsigns_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_HANDGRIP_FVC_UNION,
                df_VITALSIGNS,
            ],
            how='union'
        )
        fast_write_csv(df_alsfrs_handgrip_fvc_vitalsigns_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_VITALSIGNS_UNION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_VITALSIGNS_UNION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_VITALSIGNS_UNION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='union',
            generate_sliding=False
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_FVC_VITALSIGNS_UNION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_VITALSIGNS_UNION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_FVC_VITALSIGNS_UNION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



        # Merge ALSFRS HANDGRIPSTRENGTH FVC + VITALSIGNS (left)
        df_alsfrs_handgrip_fvc_vitalsigns_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_HANDGRIP_FVC_LEFT,
                df_VITALSIGNS,
            ],
            how='left'
        )
        fast_write_csv(df_alsfrs_handgrip_fvc_vitalsigns_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_VITALSIGNS_LEFT_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_VITALSIGNS_LEFT_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_VITALSIGNS_LEFT/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='left',
            generate_sliding=False
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_FVC_VITALSIGNS_LEFT
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_FVC_VITALSIGNS_LEFT/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_FVC_VITALSIGNS_LEFT/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )










    if PAPER_RESULTS_ONLY==False:
        # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # ================================================== MERGE ALSFRS + NO DELTA + SVC + LABS + OTHERS : INTERSECTION ==================================================
        # ------------------------------------------------------------- TARGET ALSFRS STUDY - SLIDING WINDOWS --------------------------------------------------------------
        # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # We merge the interval dataframes, one by one, into separate files
        df_ALSFRS_SVC_LABS_INTERSECTION = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_LABS_INTERSECTION_INTERVALS.csv')
        df_FVC = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_FVC_INTERVALS.csv')
        df_HANDGRIPSTRENGTH = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_HANDGRIPSTRENGTH_INTERVALS.csv')
        df_MUSCLESTRENGTH = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_MUSCLESTRENGTH_INTERVALS.csv')
        df_VITALSIGNS = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_VITALSIGNS_INTERVALS.csv')

        # Reading data without DELTA
        df_NODELTA_MERGE = fast_read_csv(MERGE_NODELTA_PATH + '/PROACT_MERGE_NODELTA_V2.csv')





        # //////////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS SVC LABS + NO DELTA + FVC -------------------------
        # //////////////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS SVC LABS + FVC (intersection)
        df_alsfrs_svc_labs_fvc_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_SVC_LABS_INTERSECTION,
                df_FVC,
            ],
            how="intersection"
        )
        fast_write_csv(df_alsfrs_svc_labs_fvc_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_LABS_FVC_INTERSECTION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_LABS_FVC_INTERSECTION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_LABS_FVC_INTERSECTION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='intersection',
            generate_fixed=False,
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_NODELTA_SVC_LABS_FVC_INTERSECTION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_LABS_FVC_INTERSECTION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_SVC_LABS_FVC_INTERSECTION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





        # ///////////////////////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS SVC LABS + NO DELTA + HANDGRIPSTRENGTH -------------------------
        # ///////////////////////////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS SVC LABS + HANDGRIPSTRENGTH (intersection)
        df_alsfrs_svc_labs_handgrip_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_SVC_LABS_INTERSECTION,
                df_HANDGRIPSTRENGTH,
            ],
            how="intersection"
        )
        fast_write_csv(df_alsfrs_svc_labs_handgrip_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_LABS_HANDGRIPSTRENGTH_INTERSECTION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_LABS_HANDGRIPSTRENGTH_INTERSECTION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_LABS_HANDGRIPSTRENGTH_INTERSECTION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='intersection',
            generate_fixed=False,
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_NODELTA_SVC_LABS_HANDGRIPSTRENGTH_INTERSECTION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_LABS_HANDGRIPSTRENGTH_INTERSECTION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_SVC_LABS_HANDGRIPSTRENGTH_INTERSECTION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





        # /////////////////////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS SVC LABS + NO DELTA + MUSCLESTRENGTH -------------------------
        # /////////////////////////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS SVC LABS + MUSCLESTRENGTH (intersection)
        df_alsfrs_svc_labs_muscle_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_SVC_LABS_INTERSECTION,
                df_MUSCLESTRENGTH,
            ],
            how="intersection"
        )
        fast_write_csv(df_alsfrs_svc_labs_muscle_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_LABS_MUSCLESTRENGTH_INTERSECTION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_LABS_MUSCLESTRENGTH_INTERSECTION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_LABS_MUSCLESTRENGTH_INTERSECTION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='intersection',
            generate_fixed=False,
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_NODELTA_SVC_LABS_MUSCLESTRENGTH_INTERSECTION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_LABS_MUSCLESTRENGTH_INTERSECTION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_SVC_LABS_MUSCLESTRENGTH_INTERSECTION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





        # /////////////////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS SVC LABS + NO DELTA + VITALSIGNS -------------------------
        # /////////////////////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS SVC LABS + VITALSIGNS (intersection)
        df_alsfrs_svc_labs_vitalsigns_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_SVC_LABS_INTERSECTION,
                df_VITALSIGNS,
            ],
            how='intersection'
        )
        fast_write_csv(df_alsfrs_svc_labs_vitalsigns_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_LABS_VITALSIGNS_INTERSECTION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_LABS_VITALSIGNS_INTERSECTION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_LABS_VITALSIGNS_INTERSECTION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='intersection',
            generate_fixed=False,
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_NODELTA_SVC_LABS_VITALSIGNS_INTERSECTION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_LABS_VITALSIGNS_INTERSECTION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_SVC_LABS_VITALSIGNS_INTERSECTION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )










    # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # ================================================== MERGE ALSFRS + NO DELTA + FVC + SVC + OTHERS : UNION & LEFT ==================================================
    # ------------------------------------------------------------ TARGET ALSFRS STUDY - SLIDING WINDOWS --------------------------------------------------------------
    # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # We merge the files, one by one, into separate files
    df_ALSFRS_FVC_SVC_UNION = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_SVC_UNION_INTERVALS.csv')
    df_ALSFRS_FVC_SVC_LEFT = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_SVC_LEFT_INTERVALS.csv')
    df_HANDGRIPSTRENGTH = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_HANDGRIPSTRENGTH_INTERVALS.csv')

    if PAPER_RESULTS_ONLY==False:
        df_LABS = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_LABS_INTERVALS.csv')
        df_MUSCLESTRENGTH = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_MUSCLESTRENGTH_INTERVALS.csv')
        df_VITALSIGNS = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_VITALSIGNS_INTERVALS.csv')

    # We read the data without DELTA
    df_NODELTA_MERGE = fast_read_csv(MERGE_NODELTA_PATH + '/PROACT_MERGE_NODELTA_V2.csv')





    if PAPER_RESULTS_ONLY==False:
        # //////////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS FVC SVC + NO DELTA + LABS -------------------------
        # //////////////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS FVC SVC + LABS (union)
        df_alsfrs_fvc_svc_labs_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_FVC_SVC_UNION,
                df_LABS,
            ],
            how="union"
        )
        fast_write_csv(df_alsfrs_fvc_svc_labs_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_SVC_LABS_UNION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_SVC_LABS_UNION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_SVC_LABS_UNION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='union',
            generate_fixed=False
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_NODELTA_FVC_SVC_LABS_UNION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_SVC_LABS_UNION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_FVC_SVC_LABS_UNION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



        # Merge ALSFRS FVC SVC + LABS (left)
        df_alsfrs_fvc_svc_labs_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_FVC_SVC_LEFT,
                df_LABS,
            ],
            how="left"
        )
        fast_write_csv(df_alsfrs_fvc_svc_labs_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_SVC_LABS_LEFT_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_SVC_LABS_LEFT_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_SVC_LABS_LEFT/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='left',
            generate_fixed=False
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_NODELTA_FVC_SVC_LABS_LEFT
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_SVC_LABS_LEFT/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_FVC_SVC_LABS_LEFT/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





    # //////////////////////////////////////////////////////////////////////////////////////////////////////
    # ------------------------- MERGE ALSFRS FVC SVC + NO DELTA + HANDGRIPSTRENGTH -------------------------
    # //////////////////////////////////////////////////////////////////////////////////////////////////////

    if PAPER_RESULTS_ONLY==False:
        # Merge ALSFRS FVC SVC + HANDGRIPSTRENGTH (union)
        df_alsfrs_fvc_svc_handgrip_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_FVC_SVC_UNION,
                df_HANDGRIPSTRENGTH,
            ],
            how="union"
        )
        fast_write_csv(df_alsfrs_fvc_svc_handgrip_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_SVC_HANDGRIPSTRENGTH_UNION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_SVC_HANDGRIPSTRENGTH_UNION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_SVC_HANDGRIPSTRENGTH_UNION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='union',
            generate_fixed=False
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_NODELTA_FVC_SVC_HANDGRIPSTRENGTH_UNION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_SVC_HANDGRIPSTRENGTH_UNION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_FVC_SVC_HANDGRIPSTRENGTH_UNION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



    # Merge ALSFRS FVC SVC + HANDGRIPSTRENGTH (left)
    df_alsfrs_fvc_svc_handgrip_merge = merge_interval_dataframes(
        dfs_interval=[
            df_ALSFRS_FVC_SVC_LEFT,
            df_HANDGRIPSTRENGTH,
        ],
        how="left"
    )
    fast_write_csv(df_alsfrs_fvc_svc_handgrip_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_SVC_HANDGRIPSTRENGTH_LEFT_INTERVALS.csv')
    # We create the prediction files for the merge
    generate_interval_prediction_files(
        df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_SVC_HANDGRIPSTRENGTH_LEFT_INTERVALS.csv',
        df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
        output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_SVC_HANDGRIPSTRENGTH_LEFT/',
        interval_length=90,
        # num_intervals=5,
        merge_interval_options='left',
        generate_fixed=False
    )
    # We add the data without DELTA to the prediction files in MERGE_ALSFRS_NODELTA_FVC_SVC_HANDGRIPSTRENGTH_LEFT
    add_no_delta_to_prediction_files(
        input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_SVC_HANDGRIPSTRENGTH_LEFT/',
        output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_FVC_SVC_HANDGRIPSTRENGTH_LEFT/',
        df_no_delta=df_NODELTA_MERGE,
        key='subject_id'
    )





    if PAPER_RESULTS_ONLY==False:
        # ////////////////////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS FVC SVC + NO DELTA + MUSCLESTRENGTH -------------------------
        # ////////////////////////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS FVC SVC + MUSCLESTRENGTH (union)
        df_alsfrs_fvc_svc_muscle_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_FVC_SVC_UNION,
                df_MUSCLESTRENGTH,
            ],
            how="union"
        )
        fast_write_csv(df_alsfrs_fvc_svc_muscle_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_SVC_MUSCLESTRENGTH_UNION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_SVC_MUSCLESTRENGTH_UNION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_SVC_MUSCLESTRENGTH_UNION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='union',
            generate_fixed=False
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_NODELTA_FVC_SVC_MUSCLESTRENGTH_UNION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_SVC_MUSCLESTRENGTH_UNION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_FVC_SVC_MUSCLESTRENGTH_UNION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



        # Merge ALSFRS FVC SVC + MUSCLESTRENGTH (left)
        df_alsfrs_fvc_svc_muscle_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_FVC_SVC_LEFT,
                df_MUSCLESTRENGTH,
            ],
            how="left"
        )
        fast_write_csv(df_alsfrs_fvc_svc_muscle_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_SVC_MUSCLESTRENGTH_LEFT_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_SVC_MUSCLESTRENGTH_LEFT_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_SVC_MUSCLESTRENGTH_LEFT/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='left',
            generate_fixed=False
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_NODELTA_FVC_SVC_MUSCLESTRENGTH_LEFT
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_SVC_MUSCLESTRENGTH_LEFT/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_FVC_SVC_MUSCLESTRENGTH_LEFT/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





    if PAPER_RESULTS_ONLY==False:
        # ////////////////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS FVC SVC + NO DELTA + VITALSIGNS -------------------------
        # ////////////////////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS FVC SVC + VITALSIGNS (union)
        df_alsfrs_fvc_svc_vitalsigns_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_FVC_SVC_UNION,
                df_VITALSIGNS,
            ],
            how='union'
        )
        fast_write_csv(df_alsfrs_fvc_svc_vitalsigns_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_SVC_VITALSIGNS_UNION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_SVC_VITALSIGNS_UNION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_SVC_VITALSIGNS_UNION/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='union',
            generate_fixed=False
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_NODELTA_FVC_SVC_VITALSIGNS_UNION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_SVC_VITALSIGNS_UNION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_FVC_SVC_VITALSIGNS_UNION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



        # Merge ALSFRS FVC SVC + VITALSIGNS (left)
        df_alsfrs_fvc_svc_vitalsigns_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_FVC_SVC_LEFT,
                df_VITALSIGNS,
            ],
            how='left'
        )
        fast_write_csv(df_alsfrs_fvc_svc_vitalsigns_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_SVC_VITALSIGNS_LEFT_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_SVC_VITALSIGNS_LEFT_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_SVC_VITALSIGNS_LEFT/',
            interval_length=90,
            # num_intervals=5,
            merge_interval_options='left',
            generate_fixed=False
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_NODELTA_FVC_SVC_VITALSIGNS_LEFT
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_SVC_VITALSIGNS_LEFT/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_FVC_SVC_VITALSIGNS_LEFT/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )










    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # ================================================== MERGE ALSFRS + NO DELTA + OTHERS ==================================================
    # -------------------------------------------- TARGET ALSFRS FIRST SYMPTOMS - FIXED WINDOWS --------------------------------------------
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # We merge the files, one by one, into separate files (First Symptoms version)
    df_ALSFRS_First_Symptoms = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_ALSFRS_FIRST_SYMPTOMS_INTERVALS.csv')
    df_SVC_First_Symptoms = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_SVC_FIRST_SYMPTOMS_INTERVALS.csv')

    if PAPER_RESULTS_ONLY==False:
        df_FVC_First_Symptoms = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_FVC_FIRST_SYMPTOMS_INTERVALS.csv')
        df_LABS_First_Symptoms = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_LABS_FIRST_SYMPTOMS_INTERVALS.csv')
        df_HANDGRIPSTRENGTH_First_Symptoms = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_HANDGRIPSTRENGTH_FIRST_SYMPTOMS_INTERVALS.csv')
        df_MUSCLESTRENGTH_First_Symptoms = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_MUSCLESTRENGTH_FIRST_SYMPTOMS_INTERVALS.csv')
        df_VITALSIGNS_First_Symptoms = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_VITALSIGNS_FIRST_SYMPTOMS_INTERVALS.csv')

    # We read the data without DELTA (First Symptoms version)
    df_NODELTA_MERGE = fast_read_csv(FIRST_SYMPTOMS_PATH + '/PROACT_MERGE_NODELTA_FIRST_SYMPTOMS.csv')





    if PAPER_RESULTS_ONLY==False:
        # /////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS + NO DELTA + FVC -------------------------
        # /////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS + FVC (union)
        df_alsfrs_fvc_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_First_Symptoms,
                df_FVC_First_Symptoms,
            ],
            how="union"
        )
        fast_write_csv(df_alsfrs_fvc_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_FIRST_SYMPTOMS_UNION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_FIRST_SYMPTOMS_UNION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_First_Symptoms_UNION/',
            interval_length=90,
            # num_intervals=17,
            generate_sliding=False,
            merge_interval_options='union'
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_NODELTA_FVC_First_Symptoms_UNION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_First_Symptoms_UNION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_FVC_First_Symptoms_UNION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



        # Merge ALSFRS + FVC (intersection)
        df_alsfrs_fvc_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_First_Symptoms,
                df_FVC_First_Symptoms,
            ],
            how="intersection"
        )
        fast_write_csv(df_alsfrs_fvc_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_FIRST_SYMPTOMS_INTERSECTION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_FIRST_SYMPTOMS_INTERSECTION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_First_Symptoms_INTERSECTION/',
            interval_length=90,
            # num_intervals=17,
            generate_sliding=False,
            merge_interval_options='intersection'
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_NODELTA_FVC_First_Symptoms_INTERSECTION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_First_Symptoms_INTERSECTION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_FVC_First_Symptoms_INTERSECTION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



        # Merge ALSFRS + FVC (left)
        df_alsfrs_fvc_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_First_Symptoms,
                df_FVC_First_Symptoms,
            ],
            how="left"
        )
        fast_write_csv(df_alsfrs_fvc_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_FIRST_SYMPTOMS_LEFT_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_FVC_FIRST_SYMPTOMS_LEFT_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_First_Symptoms_LEFT/',
            interval_length=90,
            # num_intervals=17,
            generate_sliding=False,
            merge_interval_options='left'
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_NODELTA_FVC_First_Symptoms_LEFT
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_FVC_First_Symptoms_LEFT/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_FVC_First_Symptoms_LEFT/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





    if PAPER_RESULTS_ONLY==False:
        # //////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS + NO DELTA + LABS -------------------------
        # //////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS + LABS (union)
        df_alsfrs_labs_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_First_Symptoms,
                df_LABS_First_Symptoms,
            ],
            how="union"
        )
        fast_write_csv(df_alsfrs_labs_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_LABS_FIRST_SYMPTOMS_UNION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_LABS_FIRST_SYMPTOMS_UNION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_LABS_First_Symptoms_UNION/',
            interval_length=90,
            # num_intervals=17,
            generate_sliding=False,
            merge_interval_options='union'
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_LABS_First_Symptoms_UNION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_LABS_First_Symptoms_UNION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_LABS_First_Symptoms_UNION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



        # Merge ALSFRS + LABS (intersection)
        df_alsfrs_labs_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_First_Symptoms,
                df_LABS_First_Symptoms,
            ],
            how="intersection"
        )
        fast_write_csv(df_alsfrs_labs_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_LABS_FIRST_SYMPTOMS_INTERSECTION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_LABS_FIRST_SYMPTOMS_INTERSECTION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_LABS_First_Symptoms_INTERSECTION/',
            interval_length=90,
            # num_intervals=17,
            generate_sliding=False,
            merge_interval_options='intersection'
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_LABS_First_Symptoms_INTERSECTION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_LABS_First_Symptoms_INTERSECTION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_LABS_First_Symptoms_INTERSECTION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



        # Merge ALSFRS + LABS (left)
        df_alsfrs_labs_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_First_Symptoms,
                df_LABS_First_Symptoms,
            ],
            how="left"
        )
        fast_write_csv(df_alsfrs_labs_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_LABS_FIRST_SYMPTOMS_LEFT_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_LABS_FIRST_SYMPTOMS_LEFT_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_LABS_First_Symptoms_LEFT/',
            interval_length=90,
            # num_intervals=17,
            generate_sliding=False,
            merge_interval_options='left'
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_LABS_First_Symptoms_LEFT
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_LABS_First_Symptoms_LEFT/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_LABS_First_Symptoms_LEFT/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





    if PAPER_RESULTS_ONLY==False:
        # //////////////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS + NO DELTA + HANDGRIPSTRENGTH -------------------------
        # //////////////////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS + HANDGRIPSTRENGTH (union)
        df_alsfrs_handgrip_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_First_Symptoms,
                df_HANDGRIPSTRENGTH_First_Symptoms,
            ],
            how='union'
        )
        fast_write_csv(df_alsfrs_handgrip_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_FIRST_SYMPTOMS_UNION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_FIRST_SYMPTOMS_UNION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_First_Symptoms_UNION/',
            interval_length=90,
            # num_intervals=17,
            generate_sliding=False,
            merge_interval_options='union'
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_HANDGRIPSTRENGTH_First_Symptoms_UNION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_First_Symptoms_UNION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_First_Symptoms_UNION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



        # Merge ALSFRS + HANDGRIPSTRENGTH (intersection)
        df_alsfrs_handgrip_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_First_Symptoms,
                df_HANDGRIPSTRENGTH_First_Symptoms,
            ],
            how='intersection'
        )
        fast_write_csv(df_alsfrs_handgrip_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_FIRST_SYMPTOMS_INTERSECTION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_FIRST_SYMPTOMS_INTERSECTION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_First_Symptoms_INTERSECTION/',
            interval_length=90,
            # num_intervals=17,
            generate_sliding=False,
            merge_interval_options='intersection'
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_HANDGRIPSTRENGTH_First_Symptoms_INTERSECTION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_First_Symptoms_INTERSECTION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_First_Symptoms_INTERSECTION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



        # Merge ALSFRS + HANDGRIPSTRENGTH (left)
        df_alsfrs_handgrip_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_First_Symptoms,
                df_HANDGRIPSTRENGTH_First_Symptoms,
            ],
            how='left'
        )
        fast_write_csv(df_alsfrs_handgrip_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_FIRST_SYMPTOMS_LEFT_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_HANDGRIPSTRENGTH_FIRST_SYMPTOMS_LEFT_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_First_Symptoms_LEFT/',
            interval_length=90,
            # num_intervals=17,
            generate_sliding=False,
            merge_interval_options='left'
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_HANDGRIPSTRENGTH_First_Symptoms_LEFT
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_HANDGRIPSTRENGTH_First_Symptoms_LEFT/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_HANDGRIPSTRENGTH_First_Symptoms_LEFT/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





    if PAPER_RESULTS_ONLY==False:
        # ////////////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS + NO DELTA + MUSCLESTRENGTH -------------------------
        # ////////////////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS + MUSCLESTRENGTH (union)
        df_alsfrs_muscle_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_First_Symptoms,
                df_MUSCLESTRENGTH_First_Symptoms,
            ],
            how='union'
        )
        fast_write_csv(df_alsfrs_muscle_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_MUSCLESTRENGTH_FIRST_SYMPTOMS_UNION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_MUSCLESTRENGTH_FIRST_SYMPTOMS_UNION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_MUSCLESTRENGTH_First_Symptoms_UNION/',
            interval_length=90,
            # num_intervals=17,
            generate_sliding=False,
            merge_interval_options='union'
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_MUSCLESTRENGTH_First_Symptoms_UNION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_MUSCLESTRENGTH_First_Symptoms_UNION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_MUSCLESTRENGTH_First_Symptoms_UNION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



        # Merge ALSFRS + MUSCLESTRENGTH (intersection)
        df_alsfrs_muscle_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_First_Symptoms,
                df_MUSCLESTRENGTH_First_Symptoms,
            ],
            how='intersection'
        )
        fast_write_csv(df_alsfrs_muscle_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_MUSCLESTRENGTH_FIRST_SYMPTOMS_INTERSECTION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_MUSCLESTRENGTH_FIRST_SYMPTOMS_INTERSECTION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_MUSCLESTRENGTH_First_Symptoms_INTERSECTION/',
            interval_length=90,
            # num_intervals=17,
            generate_sliding=False,
            merge_interval_options='intersection'
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_MUSCLESTRENGTH_First_Symptoms_INTERSECTION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_MUSCLESTRENGTH_First_Symptoms_INTERSECTION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_MUSCLESTRENGTH_First_Symptoms_INTERSECTION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



        # Merge ALSFRS + MUSCLESTRENGTH (left)
        df_alsfrs_muscle_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_First_Symptoms,
                df_MUSCLESTRENGTH_First_Symptoms,
            ],
            how='left'
        )
        fast_write_csv(df_alsfrs_muscle_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_MUSCLESTRENGTH_FIRST_SYMPTOMS_LEFT_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_MUSCLESTRENGTH_FIRST_SYMPTOMS_LEFT_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_MUSCLESTRENGTH_First_Symptoms_LEFT/',
            interval_length=90,
            # num_intervals=17,
            generate_sliding=False,
            merge_interval_options='left'
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_MUSCLESTRENGTH_First_Symptoms_LEFT
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_MUSCLESTRENGTH_First_Symptoms_LEFT/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_MUSCLESTRENGTH_First_Symptoms_LEFT/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





    # /////////////////////////////////////////////////////////////////////////////////
    # ------------------------- MERGE ALSFRS + NO DELTA + SVC -------------------------
    # /////////////////////////////////////////////////////////////////////////////////



    if PAPER_RESULTS_ONLY==False:
        # Merge ALSFRS + SVC (union)
        df_alsfrs_svc_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_First_Symptoms,
                df_SVC_First_Symptoms,
            ],
            how='union'
        )
        fast_write_csv(df_alsfrs_svc_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_FIRST_SYMPTOMS_UNION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_FIRST_SYMPTOMS_UNION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_First_Symptoms_UNION/',
            interval_length=90,
            # num_intervals=17,
            generate_sliding=False,
            merge_interval_options='union'
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_SVC_First_Symptoms_UNION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_First_Symptoms_UNION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_SVC_First_Symptoms_UNION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



    # Merge ALSFRS + SVC (intersection)
    df_alsfrs_svc_merge = merge_interval_dataframes(
        dfs_interval=[
            df_ALSFRS_First_Symptoms,
            df_SVC_First_Symptoms,
        ],
        how='intersection'
    )
    fast_write_csv(df_alsfrs_svc_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_FIRST_SYMPTOMS_INTERSECTION_INTERVALS.csv')
    # We create the prediction files for the merge
    generate_interval_prediction_files(
        df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_FIRST_SYMPTOMS_INTERSECTION_INTERVALS.csv',
        df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
        output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_First_Symptoms_INTERSECTION/',
        interval_length=90,
        # num_intervals=17,
        generate_sliding=False,
        merge_interval_options='intersection'
    )
    # We add the data without DELTA to the prediction files in MERGE_ALSFRS_SVC_First_Symptoms_INTERSECTION
    add_no_delta_to_prediction_files(
        input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_First_Symptoms_INTERSECTION/',
        output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_SVC_First_Symptoms_INTERSECTION/',
        df_no_delta=df_NODELTA_MERGE,
        key='subject_id'
    )



    if PAPER_RESULTS_ONLY==False:
        # Merge ALSFRS + SVC (left)
        df_alsfrs_svc_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_First_Symptoms,
                df_SVC_First_Symptoms,
            ],
            how='left'
        )
        fast_write_csv(df_alsfrs_svc_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_FIRST_SYMPTOMS_LEFT_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_FIRST_SYMPTOMS_LEFT_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_First_Symptoms_LEFT/',
            interval_length=90,
            # num_intervals=17,
            generate_sliding=False,
            merge_interval_options='left'
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_SVC_First_Symptoms_LEFT
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_First_Symptoms_LEFT/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_SVC_First_Symptoms_LEFT/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





    if PAPER_RESULTS_ONLY==False:
        # ////////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS + NO DELTA + VITALSIGNS -------------------------
        # ////////////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS + VITALSIGNS (union)
        df_alsfrs_vitalsigns_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_First_Symptoms,
                df_VITALSIGNS_First_Symptoms,
            ],
            how='union'
        )
        fast_write_csv(df_alsfrs_vitalsigns_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_VITALSIGNS_FIRST_SYMPTOMS_UNION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_VITALSIGNS_FIRST_SYMPTOMS_UNION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_VITALSIGNS_First_Symptoms_UNION/',
            interval_length=90,
            # num_intervals=17,
            generate_sliding=False,
            merge_interval_options='union'
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_VITALSIGNS_First_Symptoms_UNION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_VITALSIGNS_First_Symptoms_UNION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_VITALSIGNS_First_Symptoms_UNION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



        # Merge ALSFRS + VITALSIGNS (intersection)
        df_alsfrs_vitalsigns_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_First_Symptoms,
                df_VITALSIGNS_First_Symptoms,
            ],
            how='intersection'
        )
        fast_write_csv(df_alsfrs_vitalsigns_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_VITALSIGNS_FIRST_SYMPTOMS_INTERSECTION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_VITALSIGNS_FIRST_SYMPTOMS_INTERSECTION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_VITALSIGNS_First_Symptoms_INTERSECTION/',
            interval_length=90,
            # num_intervals=17,
            generate_sliding=False,
            merge_interval_options='intersection'
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_VITALSIGNS_First_Symptoms_INTERSECTION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_VITALSIGNS_First_Symptoms_INTERSECTION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_VITALSIGNS_First_Symptoms_INTERSECTION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )



        # Merge ALSFRS + VITALSIGNS (left)
        df_alsfrs_vitalsigns_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_First_Symptoms,
                df_VITALSIGNS_First_Symptoms,
            ],
            how='left'
        )
        fast_write_csv(df_alsfrs_vitalsigns_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_VITALSIGNS_FIRST_SYMPTOMS_LEFT_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_VITALSIGNS_FIRST_SYMPTOMS_LEFT_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_VITALSIGNS_First_Symptoms_LEFT/',
            interval_length=90,
            # num_intervals=17,
            generate_sliding=False,
            merge_interval_options='left'
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_VITALSIGNS_First_Symptoms_LEFT
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_VITALSIGNS_First_Symptoms_LEFT/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_VITALSIGNS_First_Symptoms_LEFT/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )










    if PAPER_RESULTS_ONLY==False:
        # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # ================================================== MERGE ALSFRS + NO DELTA + SVC + OTHERS : INTERSECTION ==================================================
        # ------------------------------------------------------ TARGET ALSFRS FIRST SYMPTOMS - FIXED WINDOWS -------------------------------------------------------
        # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # We merge the files into a single table (version First Symptoms)
        df_ALSFRS_SVC_First_Symptoms_INTERSECTION = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_FIRST_SYMPTOMS_INTERSECTION_INTERVALS.csv')
        df_FVC_First_Symptoms = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_FVC_FIRST_SYMPTOMS_INTERVALS.csv')
        df_LABS_First_Symptoms = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_LABS_FIRST_SYMPTOMS_INTERVALS.csv')
        df_HANDGRIPSTRENGTH_First_Symptoms = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_HANDGRIPSTRENGTH_FIRST_SYMPTOMS_INTERVALS.csv')
        df_MUSCLESTRENGTH_First_Symptoms = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_MUSCLESTRENGTH_FIRST_SYMPTOMS_INTERVALS.csv')
        df_VITALSIGNS_First_Symptoms = fast_read_csv(INTERVALS_FULL_PATH + '/PROACT_VITALSIGNS_FIRST_SYMPTOMS_INTERVALS.csv')

        # We read the data without DELTA (version First Symptoms)
        df_NODELTA_MERGE = fast_read_csv(FIRST_SYMPTOMS_PATH + '/PROACT_MERGE_NODELTA_FIRST_SYMPTOMS.csv')





        # /////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS SVC + NO DELTA + FVC -------------------------
        # /////////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS SVC + FVC (intersection)
        df_alsfrs_svc_fvc_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_SVC_First_Symptoms_INTERSECTION,
                df_FVC_First_Symptoms,
            ],
            how="intersection"
        )
        fast_write_csv(df_alsfrs_svc_fvc_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_FVC_FIRST_SYMPTOMS_INTERSECTION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_FVC_FIRST_SYMPTOMS_INTERSECTION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_FVC_First_Symptoms_INTERSECTION/',
            interval_length=90,
            # num_intervals=17,
            generate_sliding=False,
            merge_interval_options='intersection'
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_NODELTA_SVC_FVC_First_Symptoms_INTERSECTION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_FVC_First_Symptoms_INTERSECTION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_SVC_FVC_First_Symptoms_INTERSECTION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





        # //////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS SVC + NO DELTA + LABS -------------------------
        # //////////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS SVC + LABS (intersection)
        df_alsfrs_svc_labs_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_SVC_First_Symptoms_INTERSECTION,
                df_LABS_First_Symptoms,
            ],
            how="intersection"
        )
        fast_write_csv(df_alsfrs_svc_labs_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_LABS_FIRST_SYMPTOMS_INTERSECTION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_LABS_FIRST_SYMPTOMS_INTERSECTION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_LABS_First_Symptoms_INTERSECTION/',
            interval_length=90,
            # num_intervals=17,
            generate_sliding=False,
            merge_interval_options='intersection'
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_NODELTA_SVC_LABS_First_Symptoms_INTERSECTION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_LABS_First_Symptoms_INTERSECTION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_SVC_LABS_First_Symptoms_INTERSECTION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





        # //////////////////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS SVC + NO DELTA + HANDGRIPSTRENGTH -------------------------
        # //////////////////////////////////////////////////////////////////////////////////////////////////



        # Merge ALSFRS SVC + HANDGRIPSTRENGTH (intersection)
        df_alsfrs_svc_handgrip_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_SVC_First_Symptoms_INTERSECTION,
                df_HANDGRIPSTRENGTH_First_Symptoms,
            ],
            how='intersection'
        )
        fast_write_csv(df_alsfrs_svc_handgrip_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_HANDGRIPSTRENGTH_FIRST_SYMPTOMS_INTERSECTION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_HANDGRIPSTRENGTH_FIRST_SYMPTOMS_INTERSECTION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_HANDGRIPSTRENGTH_First_Symptoms_INTERSECTION/',
            interval_length=90,
            # num_intervals=17,
            generate_sliding=False,
            merge_interval_options='intersection'
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_NODELTA_SVC_HANDGRIPSTRENGTH_First_Symptoms_INTERSECTION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_HANDGRIPSTRENGTH_First_Symptoms_INTERSECTION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_SVC_HANDGRIPSTRENGTH_First_Symptoms_INTERSECTION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





        # ////////////////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS SVC + NO DELTA + MUSCLESTRENGTH -------------------------
        # ////////////////////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS SVC + MUSCLESTRENGTH (intersection)
        df_alsfrs_svc_muscle_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_SVC_First_Symptoms_INTERSECTION,
                df_MUSCLESTRENGTH_First_Symptoms,
            ],
            how='intersection'
        )
        fast_write_csv(df_alsfrs_svc_muscle_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_MUSCLESTRENGTH_FIRST_SYMPTOMS_INTERSECTION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_MUSCLESTRENGTH_FIRST_SYMPTOMS_INTERSECTION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_MUSCLESTRENGTH_First_Symptoms_INTERSECTION/',
            interval_length=90,
            # num_intervals=17,
            generate_sliding=False,
            merge_interval_options='intersection'
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_NODELTA_SVC_MUSCLESTRENGTH_First_Symptoms_INTERSECTION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_MUSCLESTRENGTH_First_Symptoms_INTERSECTION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_SVC_MUSCLESTRENGTH_First_Symptoms_INTERSECTION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )





        # ////////////////////////////////////////////////////////////////////////////////////////////
        # ------------------------- MERGE ALSFRS SVC + NO DELTA + VITALSIGNS -------------------------
        # ////////////////////////////////////////////////////////////////////////////////////////////

        # Merge ALSFRS SVC + VITALSIGNS (intersection)
        df_alsfrs_svc_vitalsigns_merge = merge_interval_dataframes(
            dfs_interval=[
                df_ALSFRS_SVC_First_Symptoms_INTERSECTION,
                df_VITALSIGNS_First_Symptoms,
            ],
            how='intersection'
        )
        fast_write_csv(df_alsfrs_svc_vitalsigns_merge, INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_VITALSIGNS_FIRST_SYMPTOMS_INTERSECTION_INTERVALS.csv')
        # We create the prediction files for the merge
        generate_interval_prediction_files(
            df_intervals_path=INTERVALS_FULL_PATH + '/PROACT_MERGE_ALSFRS_SVC_VITALSIGNS_FIRST_SYMPTOMS_INTERSECTION_INTERVALS.csv',
            df_target_path=INTERVALS_CUT_PATH + '/PROACT_Target_Variables_ALSFRS_First_Symptoms.csv',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_VITALSIGNS_First_Symptoms_INTERSECTION/',
            interval_length=90,
            # num_intervals=17,
            generate_sliding=False,
            merge_interval_options='intersection'
        )
        # We add the data without DELTA to the prediction files in MERGE_ALSFRS_SVC_VITALSIGNS_First_Symptoms_INTERSECTION
        add_no_delta_to_prediction_files(
            input_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_SVC_VITALSIGNS_First_Symptoms_INTERSECTION/',
            output_dir=INTERVALS_CUT_PATH + '/MERGE_ALSFRS_NODELTA_SVC_VITALSIGNS_First_Symptoms_INTERSECTION/',
            df_no_delta=df_NODELTA_MERGE,
            key='subject_id'
        )











