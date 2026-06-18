"""
PROACT Temporal Observation Distribution Analysis
==================================================
This script analyses the distribution of patient observations across 90-day
intervals for each temporal PROACT table, under both temporal reference frames:
study-inclusion alignment (δ relative to trial entry) and first-symptom
alignment (δ relative to symptom onset, produced by the first symptoms
alignment pipeline).

For each temporal table, two outputs are produced per reference frame:
    - A CSV counting the number of observations per patient per 90-day interval
    - A bar chart showing, for each interval, the percentage of patients who
      have at least one observation, relative to both the table cohort and the
      full PROACT dataset (11,675 patients)

An additional ALSFRS-specific analysis breaks down, per interval, how many
patients have only ALSFRS (original), only ALSFRS-R (revised), or both
versions recorded. This is used to inform the score imputation decisions
made in the ALSFRS preprocessing pipeline.

All outputs are written to two subdirectories:
    Intervals/Count/Study  - study-inclusion-aligned counts
    Intervals/Count/Onset  - first-symptom-aligned counts

Tables processed:
    ALSFRS, FVC, HANDGRIPSTRENGTH, LABS, MUSCLESTRENGTH, SVC, VITALSIGNS

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
import os
import matplotlib.pyplot as plt





# ------------------------------------------------------------------
# Generic interval observation counter
# ------------------------------------------------------------------

def generic_observation_count(
    file_path,
    prefix,
    observation_count_col,
    delta_suffix,
    output_csv,
    output_plot,
    interval_length=90,
    col_suffix="_Observation_Count",
    plot_title="Percentage of Patients with Observations per Interval",
    dataset_total=11675
):
    """
    Count the number of observations per patient per 90-day interval and
    produce a distribution plot showing cohort coverage over time.

    For each patient, every recorded visit (indexed by its Delta value) is
    assigned to the 90-day interval it falls into:

        interval_index = floor(Delta / interval_length)

    The resulting count columns are named:
        {start}_{end}{col_suffix}
        e.g. '0_90_ALS_Observation_Count', '90_180_ALS_Observation_Count'

    The bar chart displays two overlapping series:
        - Blue:   percentage of patients in the table with at least one
                  observation in each interval (relative to table cohort size)
        - Orange: same percentage relative to the full PROACT dataset
                  (dataset_total = 11,675 patients)
    A red dashed horizontal line marks the fraction of PROACT patients
    represented in this table (table cohort / dataset_total), providing
    a visual reference for overall coverage.

    Parameters
    ----------
    file_path            : str   Path to the input wide-format CSV file.
    prefix               : str   Three-letter table prefix (e.g. 'ALS', 'FVC').
    observation_count_col: str   Name of the per-patient observation count column
                                 (e.g. 'ALS_observation_count').
    delta_suffix         : str   Suffix identifying the Delta column within each
                                 visit block (e.g. 'ALSFRS_Delta').
    output_csv           : str   Path for the output interval count CSV.
    output_plot          : str   Path for the output distribution PNG.
    interval_length      : int   Width of each interval in days (default: 90).
    col_suffix           : str   Suffix appended to interval column names.
    plot_title           : str   Title shown on the distribution chart.
    dataset_total        : int   Total number of patients in the full PROACT
                                 dataset used as the percentage denominator
                                 for the orange series (default: 11,675).

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with one row per patient and one column per
        interval, containing the number of observations in that interval.
        Also saved to output_csv.
    """
    print(f"Processing: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    records = []

    for _, row in df.iterrows():
        subject_id      = row['subject_id']
        observation_count = int(row[observation_count_col])
        record = {"subject_id": subject_id}

        # Assign each visit to its 90-day interval bucket
        for i in range(1, observation_count + 1):
            delta_col = f"{prefix}_{i}_{delta_suffix}"
            if pd.notna(row[delta_col]):
                delta          = int(row[delta_col])
                interval_index = delta // interval_length
                col_name       = (
                    f"{interval_index * interval_length}"
                    f"_{(interval_index + 1) * interval_length}"
                    f"{col_suffix}"
                )
                record[col_name] = record.get(col_name, 0) + 1

        records.append(record)

        # Recompute observation count as the sum across all interval columns
        # (used as a consistency check against the original observation_count)
        record[f"initial_{observation_count_col}"] = observation_count
        record[observation_count_col] = sum(
            v for k, v in record.items() if k != "subject_id"
        )

    result = pd.DataFrame.from_records(records)

    # Sort interval columns chronologically
    cols          = result.columns.tolist()
    interval_cols = [
        c for c in cols
        if c != "subject_id"
        and not c.startswith("initial_")
        and c != observation_count_col
    ]
    interval_cols = sorted(interval_cols, key=lambda x: int(x.split("_")[0]))
    result        = result[["subject_id"] + interval_cols]

    # Recompute summary columns from the sorted interval columns
    result[f"initial_{observation_count_col}"] = result[interval_cols].apply(
        lambda row: row.notna().sum(), axis=1
    )
    result[observation_count_col] = result[interval_cols].apply(
        lambda row: row[row.notna()].sum(), axis=1
    )

    # ------------------------------------------------------------------
    # Distribution plot
    # ------------------------------------------------------------------
    def plot_distribution(df):
        """
        Plot the percentage of patients with at least one observation per
        interval, with two overlapping series (table-relative and
        dataset-relative) and a red reference line for table coverage.
        """
        plot_cols = [
            c for c in df.columns
            if c not in ("subject_id", f"initial_{observation_count_col}", observation_count_col)
        ]

        # Center the visible window around interval 0 (the study baseline or
        # symptom onset, depending on the temporal reference used)
        zero_col    = f"0_{interval_length}{col_suffix}"
        zero_index  = plot_cols.index(zero_col) if zero_col in plot_cols else 0
        start_index = max(0, zero_index - 5)
        end_index   = min(len(plot_cols), zero_index + 10000)
        plot_cols   = plot_cols[start_index:end_index]

        interval_sums             = df[plot_cols].gt(0).sum()
        total_patients            = df.shape[0]
        percentages               = (interval_sums / total_patients) * 100
        percentages_all_datasets  = (interval_sums / dataset_total) * 100

        # Strip the col_suffix from tick labels for readability
        percentages.index              = [c.replace(col_suffix, "") for c in percentages.index]
        percentages_all_datasets.index = [c.replace(col_suffix, "") for c in percentages_all_datasets.index]

        plt.figure(figsize=(12, 6))
        plt.bar(percentages.index, percentages.values, label=f"% patients in table ({total_patients})")
        plt.bar(percentages_all_datasets.index, percentages_all_datasets.values,
                color="orange", label=f"% patients in full dataset ({dataset_total})")

        plt.xlabel("Intervals (days)")
        plt.ylabel("Percentage of Patients (%)")
        plt.title(plot_title)
        plt.xticks(rotation=90)
        plt.tight_layout()

        # Reference line: fraction of total PROACT patients in this table
        coverage_pct = (total_patients / dataset_total) * 100
        plt.axhline(y=coverage_pct, color="red", linestyle="--", linewidth=1)
        plt.text(
            len(plot_cols) - 1, coverage_pct + 1,
            f"Table / dataset coverage: {coverage_pct:.1f}%",
            color="red", ha="right"
        )

        # Horizontal grid lines every 10%
        for y in range(10, 101, 10):
            plt.axhline(y=y, color="black", linestyle="--", linewidth=0.5)

        plt.legend(loc="upper right")
        plt.savefig(output_plot)
        plt.close()

    plot_distribution(result)
    result.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")
    return result





# ------------------------------------------------------------------
# Stacked line chart of ALSFRS type distribution
# ------------------------------------------------------------------

def plot_alsfrs_types_distribution(df, plot_output):
    """
    Plot one line per ALSFRS category (Only_ALSFRS_Total,
    Only_ALSFRS_R_Total, Both) plus a dashed black line for the total,
    as a percentage of the table cohort.
    """
    df = df.sort_values("Start")

    df_melted = df.melt(
        id_vars=["Interval", "Total_Patients", "Start"],
        value_vars=["Only_ALSFRS_Total", "Only_ALSFRS_R_Total", "Both"],
        var_name="Type",
        value_name="Count"
    )
    df_melted["Percentage"] = (df_melted["Count"] / df_melted["Total_Patients"]) * 100

    plt.figure(figsize=(12, 6))
    for t in df_melted["Type"].unique():
        subset = df_melted[df_melted["Type"] == t]
        plt.plot(subset["Start"], subset["Percentage"], marker='o', label=t)

    # Total line (sum of all three categories)
    df_sum = (
        df_melted.groupby(["Start", "Interval"])["Count"]
        .sum()
        .reset_index()
        .sort_values("Start")
    )
    df_sum["Percentage"] = (
        df_sum["Count"] / df_melted["Total_Patients"].iloc[0]
    ) * 100
    plt.plot(df_sum["Start"], df_sum["Percentage"],
                marker='o', color="black", linestyle="--", label="Total (%)")

    plt.xlabel("Intervals (days)")
    plt.ylabel("Percentage of Patients (%)")
    plt.title("ALSFRS Score Type Distribution per Interval")
    plt.xticks(df_sum["Start"], df_sum["Interval"], rotation=90)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_output)
    plt.close()





# ------------------------------------------------------------------
# ALSFRS-specific: breakdown of score type per interval
# ------------------------------------------------------------------

def count_alsfrs_types_per_interval(file_path, plot_output):
    """
    For each 90-day interval, classify patients into three mutually exclusive
    categories based on which version of the ALSFRS score they have recorded:

        Only_ALSFRS_Total    - patient has ALSFRS (original) but not ALSFRS-R
        Only_ALSFRS_R_Total  - patient has ALSFRS-R (revised) but not ALSFRS
        Both                 - patient has both versions in this interval

    A patient is considered to have a score in an interval if they have at
    least one non-zero value in the corresponding score column AND at least
    one ALSFRS_Delta value falling within [start, end) for that interval.

    This analysis is used to understand how many patients have each scale
    version available at each disease stage, informing the cross-scale
    imputation decisions described in the ALSFRS preprocessing pipeline.

    Two reference frames are analysed separately (study-inclusion aligned and
    first-symptoms aligned) by calling this function with the respective file.

    Parameters
    ----------
    file_path   : str  Path to the wide-format ALSFRS CSV (study or onset).
    plot_output : str  Path for the output stacked line chart PNG.

    Returns
    -------
    pd.DataFrame
        One row per 90-day interval with columns [Interval, Start,
        Only_ALSFRS_Total, Only_ALSFRS_R_Total, Both, Total_Patients].
        Also saved to a CSV by the caller.
    """
    df_alsfrs       = pd.read_csv(file_path, low_memory=False)
    interval_length = 90
    results         = []

    print(f"Counting ALSFRS types per interval for: {file_path}")

    for interval_index in range(25):  # 25 intervals of 90 days = 2250 days (~6.2 years)
        start = interval_index * interval_length
        end   = (interval_index + 1) * interval_length

        # Patients with at least one non-zero ALSFRS_Total AND a visit in this interval
        has_alsfrs_total = (
            df_alsfrs.filter(like="_ALSFRS_Total").gt(0).any(axis=1)
            & df_alsfrs.filter(like="_ALSFRS_Delta").apply(
                lambda x: x.between(start, end).any(), axis=1
            )
        )
        # Patients with at least one non-zero ALSFRS_R_Total AND a visit in this interval
        has_alsfrs_r_total = (
            df_alsfrs.filter(like="_ALSFRS_R_Total").gt(0).any(axis=1)
            & df_alsfrs.filter(like="_ALSFRS_Delta").apply(
                lambda x: x.between(start, end).any(), axis=1
            )
        )

        only_alsfrs_total   = has_alsfrs_total  & ~has_alsfrs_r_total
        only_alsfrs_r_total = has_alsfrs_r_total & ~has_alsfrs_total
        both                = has_alsfrs_total  & has_alsfrs_r_total

        results.append({
            "Interval":           f"{start}_{end}",
            "Start":              start,
            "Only_ALSFRS_Total":  only_alsfrs_total.sum(),
            "Only_ALSFRS_R_Total": only_alsfrs_r_total.sum(),
            "Both":               both.sum(),
            "Total_Patients":     df_alsfrs.shape[0],
        })

    df_results = pd.DataFrame(results)
    plot_alsfrs_types_distribution(df_results, plot_output)
    print(f"Results saved to: {plot_output}")
    return df_results










def run(DATA_PATH, FIRST_SYMPTOMS_PATH, INTERVALS_COUNT_PATH):

    print("\n" * 3)
    print("=" * 60)
    print("INTERVALS COUNT PIPELINE")
    print("=" * 60)



    # ------------------------------------------------------------------
    # Path configuration
    # ------------------------------------------------------------------

    # Create output subdirectories for study-aligned and onset-aligned counts
    for path in [
        INTERVALS_COUNT_PATH + '/Study',
        INTERVALS_COUNT_PATH + '/Onset',
    ]:
        if not os.path.exists(path):
            os.makedirs(path)
    




    # //////////////////////////////////////////////////////////
    # ------------------------- ALSFRS -------------------------
    # //////////////////////////////////////////////////////////

    # Study-inclusion-aligned observation counts
    df_alsfrs = generic_observation_count(
        file_path             = DATA_PATH + "/PROACT_ALSFRS_v8.csv",
        prefix                = "ALS",
        observation_count_col = "ALS_observation_count",
        delta_suffix          = "ALSFRS_Delta",
        output_csv            = INTERVALS_COUNT_PATH + "/Study/PROACT_ALSFRS_INTERVALS_COUNT.csv",
        output_plot           = INTERVALS_COUNT_PATH + "/Study/PROACT_ALSFRS_INTERVALS_DISTRIBUTION.png",
        col_suffix            = "_ALS_Observation_Count",
        plot_title            = "ALSFRS - Percentage of Patients with Observations per Interval",
    )

    # First-symptom-aligned observation counts
    df_alsfrs_onset = generic_observation_count(
        file_path             = FIRST_SYMPTOMS_PATH + "/PROACT_ALSFRS_FIRST_SYMPTOMS.csv",
        prefix                = "ALS",
        observation_count_col = "ALS_observation_count",
        delta_suffix          = "ALSFRS_Delta",
        output_csv            = INTERVALS_COUNT_PATH + "/Onset/PROACT_ALSFRS_FIRST_SYMPTOMS_INTERVALS_COUNT.csv",
        output_plot           = INTERVALS_COUNT_PATH + "/Onset/PROACT_ALSFRS_FIRST_SYMPTOMS_INTERVALS_DISTRIBUTION.png",
        col_suffix            = "_ALS_Observation_Count",
        plot_title            = "ALSFRS (First Symptoms Aligned) - Percentage of Patients with Observations per Interval",
    )

    # ALSFRS scale version breakdown per interval (study-aligned and onset-aligned)
    df_alsfrs_count = count_alsfrs_types_per_interval(
        DATA_PATH + "/PROACT_ALSFRS_v8.csv",
        INTERVALS_COUNT_PATH + "/Study/PROACT_ALSFRS_TYPES_PER_INTERVAL_DISTRIBUTION.png"
    )
    df_alsfrs_count.to_csv(
        INTERVALS_COUNT_PATH + "/Study/PROACT_ALSFRS_TYPES_PER_INTERVAL_COUNT.csv", index=False
    )

    df_alsfrs_count_onset = count_alsfrs_types_per_interval(
        FIRST_SYMPTOMS_PATH + "/PROACT_ALSFRS_FIRST_SYMPTOMS.csv",
        INTERVALS_COUNT_PATH + "/Onset/PROACT_ALSFRS_FIRST_SYMPTOMS_TYPES_PER_INTERVAL_DISTRIBUTION.png"
    )
    df_alsfrs_count_onset.to_csv(
        INTERVALS_COUNT_PATH + "/Onset/PROACT_ALSFRS_FIRST_SYMPTOMS_TYPES_PER_INTERVAL_COUNT.csv", index=False
    )





    # ///////////////////////////////////////////////////////
    # ------------------------- FVC -------------------------
    # ///////////////////////////////////////////////////////

    df_fvc = generic_observation_count(
        file_path             = DATA_PATH + "/PROACT_FVC_v7.csv",
        prefix                = "FVC",
        observation_count_col = "FVC_observation_count",
        delta_suffix          = "Forced_Vital_Capacity_Delta",
        output_csv            = INTERVALS_COUNT_PATH + "/Study/PROACT_FVC_INTERVALS_COUNT.csv",
        output_plot           = INTERVALS_COUNT_PATH + "/Study/PROACT_FVC_INTERVALS_DISTRIBUTION.png",
        col_suffix            = "_FVC_Observation_Count",
        plot_title            = "FVC - Percentage of Patients with Observations per Interval",
    )

    df_fvc_onset = generic_observation_count(
        file_path             = FIRST_SYMPTOMS_PATH + "/PROACT_FVC_FIRST_SYMPTOMS.csv",
        prefix                = "FVC",
        observation_count_col = "FVC_observation_count",
        delta_suffix          = "Forced_Vital_Capacity_Delta",
        output_csv            = INTERVALS_COUNT_PATH + "/Onset/PROACT_FVC_FIRST_SYMPTOMS_INTERVALS_COUNT.csv",
        output_plot           = INTERVALS_COUNT_PATH + "/Onset/PROACT_FVC_FIRST_SYMPTOMS_INTERVALS_DISTRIBUTION.png",
        col_suffix            = "_FVC_Observation_Count",
        plot_title            = "FVC (First Symptoms Aligned) - Percentage of Patients with Observations per Interval",
    )





    # ////////////////////////////////////////////////////////////////////
    # ------------------------- HANDGRIPSTRENGTH -------------------------
    # ////////////////////////////////////////////////////////////////////

    df_handgripstrength = generic_observation_count(
        file_path             = DATA_PATH + "/PROACT_HANDGRIPSTRENGTH_v8.csv",
        prefix                = "HAN",
        observation_count_col = "HAN_observation_count",
        delta_suffix          = "MS_Delta",
        output_csv            = INTERVALS_COUNT_PATH + "/Study/PROACT_HANDGRIPSTRENGTH_INTERVALS_COUNT.csv",
        output_plot           = INTERVALS_COUNT_PATH + "/Study/PROACT_HANDGRIPSTRENGTH_INTERVALS_DISTRIBUTION.png",
        col_suffix            = "_HAN_Observation_Count",
        plot_title            = "HANDGRIPSTRENGTH - Percentage of Patients with Observations per Interval",
    )

    df_handgripstrength_onset = generic_observation_count(
        file_path             = FIRST_SYMPTOMS_PATH + "/PROACT_HANDGRIPSTRENGTH_FIRST_SYMPTOMS.csv",
        prefix                = "HAN",
        observation_count_col = "HAN_observation_count",
        delta_suffix          = "MS_Delta",
        output_csv            = INTERVALS_COUNT_PATH + "/Onset/PROACT_HANDGRIPSTRENGTH_FIRST_SYMPTOMS_INTERVALS_COUNT.csv",
        output_plot           = INTERVALS_COUNT_PATH + "/Onset/PROACT_HANDGRIPSTRENGTH_FIRST_SYMPTOMS_INTERVALS_DISTRIBUTION.png",
        col_suffix            = "_HAN_Observation_Count",
        plot_title            = "HANDGRIPSTRENGTH (First Symptoms Aligned) - Percentage of Patients with Observations per Interval",
    )





    # ////////////////////////////////////////////////////////
    # ------------------------- LABS -------------------------
    # ////////////////////////////////////////////////////////

    df_labs = generic_observation_count(
        file_path             = DATA_PATH + "/PROACT_LABS_v10.csv",
        prefix                = "LAB",
        observation_count_col = "LAB_observation_count",
        delta_suffix          = "Laboratory_Delta",
        output_csv            = INTERVALS_COUNT_PATH + "/Study/PROACT_LABS_INTERVALS_COUNT.csv",
        output_plot           = INTERVALS_COUNT_PATH + "/Study/PROACT_LABS_INTERVALS_DISTRIBUTION.png",
        col_suffix            = "_LAB_Observation_Count",
        plot_title            = "LABS - Percentage of Patients with Observations per Interval",
    )

    df_labs_onset = generic_observation_count(
        file_path             = FIRST_SYMPTOMS_PATH + "/PROACT_LABS_FIRST_SYMPTOMS.csv",
        prefix                = "LAB",
        observation_count_col = "LAB_observation_count",
        delta_suffix          = "Laboratory_Delta",
        output_csv            = INTERVALS_COUNT_PATH + "/Onset/PROACT_LABS_FIRST_SYMPTOMS_INTERVALS_COUNT.csv",
        output_plot           = INTERVALS_COUNT_PATH + "/Onset/PROACT_LABS_FIRST_SYMPTOMS_INTERVALS_DISTRIBUTION.png",
        col_suffix            = "_LAB_Observation_Count",
        plot_title            = "LABS (First Symptoms Aligned) - Percentage of Patients with Observations per Interval",
    )





    # //////////////////////////////////////////////////////////////////
    # ------------------------- MUSCLESTRENGTH -------------------------
    # //////////////////////////////////////////////////////////////////

    df_musclestrength = generic_observation_count(
        file_path             = DATA_PATH + "/PROACT_MUSCLESTRENGTH_v8.csv",
        prefix                = "MUS",
        observation_count_col = "MUS_observation_count",
        delta_suffix          = "MS_Delta",
        output_csv            = INTERVALS_COUNT_PATH + "/Study/PROACT_MUSCLESTRENGTH_INTERVALS_COUNT.csv",
        output_plot           = INTERVALS_COUNT_PATH + "/Study/PROACT_MUSCLESTRENGTH_INTERVALS_DISTRIBUTION.png",
        col_suffix            = "_MUS_Observation_Count",
        plot_title            = "MUSCLESTRENGTH - Percentage of Patients with Observations per Interval",
    )

    df_musclestrength_onset = generic_observation_count(
        file_path             = FIRST_SYMPTOMS_PATH + "/PROACT_MUSCLESTRENGTH_FIRST_SYMPTOMS.csv",
        prefix                = "MUS",
        observation_count_col = "MUS_observation_count",
        delta_suffix          = "MS_Delta",
        output_csv            = INTERVALS_COUNT_PATH + "/Onset/PROACT_MUSCLESTRENGTH_FIRST_SYMPTOMS_INTERVALS_COUNT.csv",
        output_plot           = INTERVALS_COUNT_PATH + "/Onset/PROACT_MUSCLESTRENGTH_FIRST_SYMPTOMS_INTERVALS_DISTRIBUTION.png",
        col_suffix            = "_MUS_Observation_Count",
        plot_title            = "MUSCLESTRENGTH (First Symptoms Aligned) - Percentage of Patients with Observations per Interval",
    )





    # ///////////////////////////////////////////////////////
    # ------------------------- SVC -------------------------
    # ///////////////////////////////////////////////////////

    df_svc = generic_observation_count(
        file_path             = DATA_PATH + "/PROACT_SVC_v7.csv",
        prefix                = "SVC",
        observation_count_col = "SVC_observation_count",
        delta_suffix          = "Slow_Vital_Capacity_Delta",
        output_csv            = INTERVALS_COUNT_PATH + "/Study/PROACT_SVC_INTERVALS_COUNT.csv",
        output_plot           = INTERVALS_COUNT_PATH + "/Study/PROACT_SVC_INTERVALS_DISTRIBUTION.png",
        col_suffix            = "_SVC_Observation_Count",
        plot_title            = "SVC - Percentage of Patients with Observations per Interval",
    )

    df_svc_onset = generic_observation_count(
        file_path             = FIRST_SYMPTOMS_PATH + "/PROACT_SVC_FIRST_SYMPTOMS.csv",
        prefix                = "SVC",
        observation_count_col = "SVC_observation_count",
        delta_suffix          = "Slow_Vital_Capacity_Delta",
        output_csv            = INTERVALS_COUNT_PATH + "/Onset/PROACT_SVC_FIRST_SYMPTOMS_INTERVALS_COUNT.csv",
        output_plot           = INTERVALS_COUNT_PATH + "/Onset/PROACT_SVC_FIRST_SYMPTOMS_INTERVALS_DISTRIBUTION.png",
        col_suffix            = "_SVC_Observation_Count",
        plot_title            = "SVC (First Symptoms Aligned) - Percentage of Patients with Observations per Interval",
    )





    # //////////////////////////////////////////////////////////////
    # ------------------------- VITALSIGNS -------------------------
    # //////////////////////////////////////////////////////////////

    df_vitalsigns = generic_observation_count(
        file_path             = DATA_PATH + "/PROACT_VITALSIGNS_v7.csv",
        prefix                = "VIT",
        observation_count_col = "VIT_observation_count",
        delta_suffix          = "Vital_Signs_Delta",
        output_csv            = INTERVALS_COUNT_PATH + "/Study/PROACT_VITALSIGNS_INTERVALS_COUNT.csv",
        output_plot           = INTERVALS_COUNT_PATH + "/Study/PROACT_VITALSIGNS_INTERVALS_DISTRIBUTION.png",
        col_suffix            = "_VIT_Observation_Count",
        plot_title            = "VITALSIGNS - Percentage of Patients with Observations per Interval",
    )

    df_vitalsigns_onset = generic_observation_count(
        file_path             = FIRST_SYMPTOMS_PATH + "/PROACT_VITALSIGNS_FIRST_SYMPTOMS.csv",
        prefix                = "VIT",
        observation_count_col = "VIT_observation_count",
        delta_suffix          = "Vital_Signs_Delta",
        output_csv            = INTERVALS_COUNT_PATH + "/Onset/PROACT_VITALSIGNS_FIRST_SYMPTOMS_INTERVALS_COUNT.csv",
        output_plot           = INTERVALS_COUNT_PATH + "/Onset/PROACT_VITALSIGNS_FIRST_SYMPTOMS_INTERVALS_DISTRIBUTION.png",
        col_suffix            = "_VIT_Observation_Count",
        plot_title            = "VITALSIGNS (First Symptoms Aligned) - Percentage of Patients with Observations per Interval",
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