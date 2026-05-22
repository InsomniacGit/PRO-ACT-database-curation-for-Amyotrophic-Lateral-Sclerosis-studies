"""
PROACT Laboratory Results Processing Pipeline
=============================================
This script processes the PROACT (PRO-ACT ALS) laboratory results dataset
through a sequential validation, cleaning, imputation, and reshaping pipeline.
It produces ten intermediate CSV files, culminating in a wide-format
one-row-per-patient matrix with visits stored as sequentially prefixed columns.

The lab dataset contains results from a large number of distinct tests. Two
independent expert validation lists (Dr. n°1 and Dr. n°2) are used to restrict 
the analysis to clinically relevant tests. Tests are further filtered by a 
minimum patient prevalence threshold before reshaping.

Pipeline stages:
    Reference   - List all unique test names and cross-reference against two
                  expert validation lists (Dr. n°1, Dr. n°2)
    v2          - Normalise Test_Name casing; harmonise units; clean numeric
                  Test_Result values; standardise Urine Color labels
    v3          - Retain only tests validated by at least one expert list
    v4          - Retain only tests with patient prevalence >= threshold
    v5          - Pivot to semi-wide format: one row per (subject_id, Laboratory_Delta),
                  one column pair (Test_Result, Test_Unit) per test name
    v6          - Resolve residual multi-unit conflicts; drop irrecoverable tests
    v7          - Replace non-numeric and sentinel result values with NaN;
                  fix comma-formatted numbers; drop low-quality test columns
    v8          - Add per-patient observation count
    v9          - Reshape to fully wide format (one row per patient, visits
                  as sequentially prefixed column blocks)
    v10         - Prefix all feature columns with 'LAB_' for downstream merging

Author: Bouclier Lucas
Data:   PROACT dataset (2022-07-29 release)
"""



import pandas as pd
import numpy as np





# ////////////////////////////////////////////////////////
# ------------------------- LABS -------------------------
# ////////////////////////////////////////////////////////



# ------------------------------------------------------------------
# Reference - List unique test names and cross-validate against expert lists
# ------------------------------------------------------------------

def list_unique_test_names(file_path, validation_path):
    """
    Build a reference table of all unique lab test names, enriched with two
    independent expert validation flags and patient prevalence statistics.

    The two validation sources are:
        Valid_Dr1 - tests selected by Dr. n°1, loaded from
                           an external CSV (column index 4 of that file)
        Valid_Dr2 - tests selected by Dr. n°2, hardcoded from
                           the published list

    The resulting table is used by filter_validated_tests() to decide which
    tests to retain in the pipeline. It is also saved as an audit file for
    manual inspection.

    Parameters
    ----------
    file_path       : str  Path to the raw PROACT_LABS.csv file.
    validation_path : str  Path to PROACT_LABS_Validation_Dr1.csv
                           (semicolon-separated).

    Returns
    -------
    pd.DataFrame
        Summary table with columns [Test_Name, Unique_Subject_ID_Count,
        Percentage_of_Total_Subjects, Valid_Dr1, Valid_Dr2], sorted by 
        Unique_Subject_ID_Count descending
        (-> saved as PROACT_LABS_Test_Names.csv).
    """
    df            = pd.read_csv(file_path,       low_memory=False)
    df_validation = pd.read_csv(validation_path, low_memory=False, sep=';')

    # Normalise test names in both sources before merging
    df['Test_Name']            = df['Test_Name'].str.strip().str.title()
    df_validation['Test_Name'] = df_validation['Test_Name'].str.strip().str.title()

    # Count distinct patients per test name
    total_subjects   = df['subject_id'].nunique()
    test_name_counts = (
        df.groupby('Test_Name')['subject_id']
        .nunique()
        .reset_index()
        .rename(columns={'subject_id': 'Unique_Subject_ID_Count'})
    )

    # Merge the Dr. n°1 validation flag (column index 4 of the validation file)
    test_name_counts = test_name_counts.merge(
        df_validation[['Test_Name', df_validation.columns[4]]],
        on='Test_Name', how='left'
    )
    test_name_counts.rename(columns={df_validation.columns[4]: 'Valid_Dr1'}, inplace=True)
    # Convert 'x' markers to True and missing values to False
    test_name_counts['Valid_Dr1'] = test_name_counts['Valid_Dr1'].apply(
        lambda x: True if x == 'x' else False
    )

    # Encode the Dr. n°2 validation flag from the published test list
    dr2_tests = [
        'Creatine Kinase', 'Lymphocytes', 'Neutrophils',
        'Absolute Lymphocyte Count', 'Absolute Neutrophil Count',
        'Monocytes', 'Absolute Monocyte Count', 'Urine Ketones',
    ]
    test_name_counts['Valid_Dr2'] = test_name_counts['Test_Name'].apply(
        lambda x: True if x in dr2_tests else False
    )

    test_name_counts['Percentage_of_Total_Subjects'] = (
        test_name_counts['Unique_Subject_ID_Count'] / total_subjects * 100
    ).round(2)

    test_name_counts = test_name_counts[[
        'Test_Name', 'Unique_Subject_ID_Count',
        'Percentage_of_Total_Subjects', 'Valid_Dr1', 'Valid_Dr2',
    ]]
    test_name_counts = test_name_counts.sort_values(
        by='Unique_Subject_ID_Count', ascending=False
    )

    return test_name_counts





# ------------------------------------------------------------------
# Stage v2 - Normalise test names, units, and result values
# ------------------------------------------------------------------

def filter_labs(file_path):
    """
    Apply a series of normalisation steps to the raw lab data:

    1. Unit consistency check: for each test, verify that only one unit is
       used across all records. Tests with multiple units are reported and the
       most common harmonisation ('IU/L' -> 'U/L') is applied.

    2. Test_Result numeric cleaning:
       - Leading decimal points are fixed (e.g. '.5' -> '0.5')
       - Trailing '.00' and '.0' are stripped for cleaner display

    3. Urine Color normalisation: the many free-text colour variants are
       mapped to a controlled vocabulary of seven canonical labels
       (Colourless, Light Yellow, Yellow, Dark Yellow, Amber, Brown, Other).

    Parameters
    ----------
    file_path : str
        Path to the raw PROACT_LABS.csv file.

    Returns
    -------
    pd.DataFrame
        Normalised DataFrame sorted by subject_id, Test_Name, Laboratory_Delta
        (-> saved as PROACT_LABS_v2.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)
    print("Patients:", df["subject_id"].nunique())

    df['Test_Name'] = df['Test_Name'].str.strip().str.title()
    print("Unique test names:", df["Test_Name"].nunique())

    # ------------------------------------------------------------------
    # Step 1: Report and harmonise tests with multiple measurement units
    # ------------------------------------------------------------------
    plural_list = []
    plural_unit = 0
    single_unit = 0
    no_unit     = 0

    for test_name in df["Test_Name"].unique():
        test_units = df.loc[df["Test_Name"] == test_name, "Test_Unit"].unique()
        test_units = [u if pd.notna(u) else "No unit" for u in test_units]

        if len(test_units) > 1:
            print(f"Test_Name: {test_name} => Test_Unit: {', '.join(test_units)}")
            plural_unit += 1
            plural_list.append(test_name)
        elif test_units[0] == "No unit":
            no_unit += 1
        else:
            single_unit += 1

    print(f"Tests with multiple units: {plural_unit}")
    print(f"Tests with a single unit:  {single_unit}")
    print(f"Tests with no unit:        {no_unit}")

    # Harmonise the only known multi-unit case: IU/L and U/L are equivalent
    for test_name in plural_list:
        df.loc[df["Test_Name"] == test_name, "Test_Unit"] = (
            df.loc[df["Test_Name"] == test_name, "Test_Unit"].replace("IU/L", "U/L")
        )

    # ------------------------------------------------------------------
    # Step 2: Clean numeric result strings
    # ------------------------------------------------------------------
    df["Test_Result"] = (
        df["Test_Result"]
        .astype(str)
        .replace(r"^\.", "0.", regex=True)  # .5 -> 0.5
        .replace(r"\.00$", "", regex=True)  # 12.00 -> 12
        .replace(r"\.0$",  "", regex=True)  # 12.0  -> 12
    )

    # ------------------------------------------------------------------
    # Step 3: Standardise Urine Color free-text to a controlled vocabulary
    # ------------------------------------------------------------------
    color_mapping = {
        # Light Yellow variants
        "LIGHT YELLOW": "Light Yellow", "LIGHT Yellow": "Light Yellow",
        "Light Yellow": "Light Yellow", "Light yellow": "Light Yellow",
        "Light-Yellow": "Light Yellow", "Slightly yellow": "Light Yellow",
        "pale yellow":  "Light Yellow", "Pale yellow": "Light Yellow",
        "PALE YELLOW":  "Light Yellow", "Straw": "Light Yellow",
        "Straw yellow": "Light Yellow",
        # Dark Yellow variants
        "DARK YELLOW": "Dark Yellow", "Dark Yellow": "Dark Yellow",
        "dark yellow": "Dark Yellow", "Dark": "Dark Yellow",
        "Dk Yellow":   "Dark Yellow",
        # Yellow variants
        "YELLOW": "Yellow", "Yellow": "Yellow",
        "yellow": "Yellow", "medium yellow": "Yellow",
        # Colourless variants
        "COLOURLESS": "Colourless", "Colourless": "Colourless",
        "Colorless":  "Colourless", "colourless":  "Colourless",
        "Water Wht":  "Colourless",
        # Other single-label colours
        "BROWN": "Brown", "Brown": "Brown",
        "Amber": "Amber", "Orange": "Orange",
        "Red":   "Red",   "Green":  "Green",
        # Non-specific value
        "Normal": "Other",
    }
    df.loc[df["Test_Name"] == "Urine Color", "Test_Result"] = (
        df.loc[df["Test_Name"] == "Urine Color", "Test_Result"].replace(color_mapping)
    )

    df = df.sort_values(by=["subject_id", "Test_Name", "Laboratory_Delta"])

    return df





# ------------------------------------------------------------------
# Stage v3 - Retain only expert-validated tests
# ------------------------------------------------------------------

def filter_validated_tests(file_path, data_path):
    """
    Remove all tests that are not validated by at least one expert source.

    A test is retained if either Valid_Dr1 or Valid_Dr2 is True in the 
    reference table produced by list_unique_test_names(). Tests not present 
    in either list are discarded as clinically uninformative for this study.

    Parameters
    ----------
    file_path : str
        Path to PROACT_LABS_v2.csv.
    data_path : str   
        Path to the Root directory for all processed outputs

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only validated tests
        (-> saved as PROACT_LABS_v3.csv).
    """
    df             = pd.read_csv(file_path,                                 low_memory=False)
    df_valid_tests = pd.read_csv(data_path + '/PROACT_LABS_Test_Names.csv', low_memory=False)

    print("Patients before filtering:", df["subject_id"].nunique())
    print("Tests before filtering:",    df["Test_Name"].nunique())

    valid_tests = df_valid_tests[
        (df_valid_tests['Valid_Dr1'] == True) |
        (df_valid_tests['Valid_Dr2']    == True)
    ]['Test_Name'].tolist()

    df_filtered = df[df['Test_Name'].isin(valid_tests)].copy()

    print("Patients after filtering:", df_filtered["subject_id"].nunique())
    print("Tests after filtering:",    df_filtered["Test_Name"].nunique())

    return df_filtered



def list_valid_tests(file_path):
    """
    Export the subset of the reference table corresponding to validated tests.

    This audit file provides a concise view of the tests retained after the
    expert-validation filter, with their prevalence statistics, for reporting
    in the methods section of the paper.

    Parameters
    ----------
    file_path : str
        Path to PROACT_LABS_Test_Names.csv.

    Returns
    -------
    pd.DataFrame
        Filtered reference table (-> saved as PROACT_LABS_Test_Names_Validated.csv).
    """
    df_valid_tests = pd.read_csv(file_path, low_memory=False)
    return df_valid_tests[
        (df_valid_tests['Valid_Dr1'] == True) |
        (df_valid_tests['Valid_Dr2']    == True)
    ][['Test_Name', 'Unique_Subject_ID_Count', 'Percentage_of_Total_Subjects',
       'Valid_Dr1', 'Valid_Dr2']]





# ------------------------------------------------------------------
# Stage v4 - Filter tests below the prevalence threshold
# ------------------------------------------------------------------

def filter_by_percentage(file_path, threshold):
    """
    Remove tests reported by fewer than `threshold` percent of the total
    patient cohort.

    This step ensures that the final feature matrix does not contain columns
    that are too sparse to be statistically informative. Prevalence is
    recomputed directly from the filtered dataset (not from the reference
    table) to reflect the cohort size after validation filtering.

    Parameters
    ----------
    file_path : str   Path to PROACT_LABS_v3.csv.
    threshold : float Minimum patient prevalence (%) required to keep a test.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only sufficiently prevalent tests
        (-> saved as PROACT_LABS_v4.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)

    print("Patients before filtering:", df["subject_id"].nunique())
    print("Tests before filtering:",    df["Test_Name"].nunique())

    total_subjects = df["subject_id"].nunique()
    test_counts    = df.groupby('Test_Name')['subject_id'].nunique().reset_index()
    test_counts['Percentage_of_Total_Subjects'] = (
        test_counts['subject_id'] / total_subjects * 100
    ).round(2)

    valid_tests = test_counts[
        test_counts['Percentage_of_Total_Subjects'] >= threshold
    ]['Test_Name'].tolist()

    df_filtered = df[df['Test_Name'].isin(valid_tests)].copy()

    print("Patients after filtering:", df_filtered["subject_id"].nunique())
    print("Tests after filtering:",    df_filtered["Test_Name"].nunique())

    return df_filtered





# ------------------------------------------------------------------
# Stage v5 - Pivot to semi-wide format (one row per subject/visit)
# ------------------------------------------------------------------

def reshape_to_semi_wide_format(csv_file):
    """
    Reshape the long-format lab data into a semi-wide format with one row per
    (subject_id, Laboratory_Delta) visit, where each test is represented by
    two columns: {TestName}_Test_Result and {TestName}_Test_Unit.

    Test names are cleaned by removing spaces before being used as column name
    prefixes (e.g. 'Creatine Kinase' -> 'CreatineKinase'). Other special
    characters (hyphens, parentheses, slashes) are intentionally preserved
    at this stage; they can be removed by uncommenting the relevant lines in
    clean_name() if needed.

    Columns are sorted alphabetically by test name within each visit, and the
    final column order groups all columns for the same test together.

    A progress message is printed every 2000 groups to monitor execution.

    Parameters
    ----------
    csv_file : str
        Path to PROACT_LABS_v4.csv.

    Returns
    -------
    pd.DataFrame
        Semi-wide DataFrame with one row per (subject_id, Laboratory_Delta)
        (-> saved as PROACT_LABS_v5.csv).
    """
    df = pd.read_csv(csv_file, low_memory=False)

    # Columns to carry per test per visit (Laboratory_Delta is the row key)
    colonnes = ['Laboratory_Delta', 'Test_Result', 'Test_Unit']

    def clean_name(name):
        """Remove spaces from test names to produce valid column name prefixes."""
        return name.replace(" ", "")
        # Additional characters can be removed if needed:
        # .replace("-", "").replace("(", "").replace(")", "")
        # .replace("/", "").replace(":", "").replace(".", "").replace(",", "")

    df = df.sort_values(by=['subject_id', 'Laboratory_Delta', 'Test_Name'])

    grouped     = df.groupby(['subject_id', 'Laboratory_Delta'])
    rows        = []
    total_groups = len(grouped)
    counter    = 0

    for (subject_id, lab_delta), group in grouped:
        counter += 1
        if counter % 2000 == 0:
            print(f"Processed {counter} groups out of {total_groups}...")

        row_data = {
            'subject_id':       subject_id,
            'Laboratory_Delta': lab_delta,
        }

        # One column pair per test present at this visit
        for _, test_row in group.iterrows():
            test_name = clean_name(test_row['Test_Name'])
            for col in colonnes:
                if col == 'Laboratory_Delta':
                    continue
                row_data[f"{test_name}_{col}"] = test_row[col]

        rows.append(row_data)

    df_wide = pd.DataFrame(rows)

    # Sort columns: subject_id and Laboratory_Delta first, then alphabetically
    # by test name, with _Test_Result before _Test_Unit within each test
    test_names_sorted = sorted(df['Test_Name'].drop_duplicates().tolist())

    final_cols = ['subject_id', 'Laboratory_Delta']
    for test_name in test_names_sorted:
        cn = clean_name(test_name)
        for col in colonnes:
            if col != 'Laboratory_Delta':
                final_cols.append(f"{cn}_{col}")

    df_wide = df_wide.reindex(columns=final_cols)

    return df_wide





# ------------------------------------------------------------------
# Diagnostic - Check for residual multi-unit columns
# ------------------------------------------------------------------

def check_multiple_units(file_path):
    """
    Report any _Test_Unit columns that still contain more than one distinct
    unit after the harmonisation applied in stage v2.

    For each multi-unit column, the number of patients associated with each
    unit is printed to inform the manual resolution applied in stage v6.

    Parameters
    ----------
    file_path : str
        Path to PROACT_LABS_v5.csv.
    """
    df       = pd.read_csv(file_path, low_memory=False)
    df_units = df[[col for col in df.columns if col.endswith('_Test_Unit')]]

    plural_unit = 0

    for col in df_units.columns:
        test_units = [u if pd.notna(u) else "No unit" for u in df_units[col].dropna().unique()]

        if len(test_units) > 1:
            print(f"Column: {col} => Units: {', '.join(test_units)}")
            plural_unit += 1
            for unit in test_units:
                if unit != "No unit":
                    patient_count = df[df[col] == unit]['subject_id'].nunique()
                    print(f"\t{unit}: {patient_count} patients")





# ------------------------------------------------------------------
# Stage v6 - Resolve residual multi-unit conflicts
# ------------------------------------------------------------------

def remove_plural_units(file_path):
    """
    Manually resolve the two residual multi-unit conflicts identified by
    check_multiple_units().

    Albumin:
        A small number of records use '%' as the unit instead of 'g/dL'.
        These represent a different measurement type and cannot be converted;
        both the result and the unit are set to NaN.

    Mean Corpuscular Hemoglobin Concentration (MCHC):
        This test has irreconcilable unit inconsistencies across the dataset.
        All MCHC columns are dropped entirely rather than retaining
        unreliable measurements.

    Parameters
    ----------
    file_path : str
        Path to PROACT_LABS_v5.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with unit conflicts resolved
        (-> saved as PROACT_LABS_v6.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)

    # Albumin in '%': incompatible unit, nullify result and unit
    condition = df['Albumin_Test_Unit'] == '%'
    df.loc[condition, 'Albumin_Test_Unit']   = np.nan
    df.loc[condition, 'Albumin_Test_Result'] = np.nan

    # MCHC: irreconcilable unit conflicts across records - drop entirely
    cols_to_remove = [col for col in df.columns if col.startswith('MeanCorpuscularHemoglobinConcentration')]
    df = df.drop(columns=cols_to_remove)

    return df





# ------------------------------------------------------------------
# Diagnostic - Identify non-numeric result values (run before stage v7)
# ------------------------------------------------------------------

def check_textual_results(file_path):
    """
    Report all non-numeric values present in _Test_Result columns, along with
    the number of patients affected by each.

    This diagnostic is run manually before stage v7 to identify which textual
    values need to be resolved (converted, nullified, or left as-is). It is
    commented out in normal pipeline execution.

    Parameters
    ----------
    file_path : str
        Path to PROACT_LABS_v6.csv (or v7.csv for post-correction verification).
    """
    df         = pd.read_csv(file_path, low_memory=False)
    df_results = df[[col for col in df.columns if col.endswith('_Test_Result')]]

    def is_number(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    for col in df_results.columns:
        non_numeric = [
            val for val in df_results[col].dropna().astype(str).unique()
            if not is_number(val)
        ]
        if non_numeric:
            print(f"Column: {col}")
            for entry in non_numeric:
                count = df[df[col].astype(str) == entry]['subject_id'].nunique()
                print(f"\t'{entry}': {count} patients")
            print()





# ------------------------------------------------------------------
# Stage v7 - Clean non-numeric and sentinel result values
# ------------------------------------------------------------------

def remove_unusable_tests_units(file_path):
    """
    Replace non-numeric, sentinel, and formatting-error result values with
    NaN or their correct numeric equivalents, and drop columns that cannot
    be recovered.

    Operations applied per test (identified by check_textual_results()):

        AbsoluteBasophilCount     - 'Normal', 'Trace' -> NaN
        AbsoluteEosinophilCount   - 'Normal' -> NaN
        AbsoluteMonocyteCount     - '4.9e-05' -> 0.000049 (scientific notation fix)
        AlkalinePhosphatase       - '<5' -> NaN (below-detection sentinel)
        Alt(Sgpt)                 - 'Normal' -> NaN
        Basophils                 - 'Normal' -> NaN
        Bilirubin(Total)          - '<3.42', '<0.2', '<3', '-', '<2.5' -> NaN
        CreatineKinase            - '<18', '<23', 'Normal', '<7' -> NaN
        Creatinine                - '<18' -> NaN
        Eosinophils               - 'Normal' -> NaN
        Gamma-Glutamyltransferase - '<4' -> NaN
        Hba1C(GlycatedHemoglobin) - 'Normal' -> NaN
        Platelets                 - '229,000' -> 229000, '303,000' -> 303000
        Protein                   - 'Trace', '-', '1+', '2+', '4+' -> NaN
                                    (semi-quantitative dipstick results)
        RedBloodCells(Rbc)        - '-', 'Very Large' -> NaN;
                                    '4750,000' -> 4750000, '4820,000' -> 4820000
        WhiteBloodCell(Wbc)       - '-', 'Very Large' -> NaN;
                                    '6,900' -> 6900, '7,900' -> 7900

    Three tests are dropped entirely because they contain predominantly
    non-numeric values and fall below the prevalence threshold in practice:
        UrineBlood, UrineGlucose, UrineKetones

    Parameters
    ----------
    file_path : str
        Path to PROACT_LABS_v6.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with all result values either numeric or NaN
        (-> saved as PROACT_LABS_v7.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)

    # Helper to nullify both result and unit for a given condition and test prefix
    def nullify(col_result, condition):
        col_unit = col_result.replace('_Test_Result', '_Test_Unit')
        df.loc[condition, col_result] = np.nan
        df.loc[condition, col_unit]   = np.nan

    nullify('AbsoluteBasophilCount_Test_Result',
            df['AbsoluteBasophilCount_Test_Result'].isin(['Normal', 'Trace']))

    nullify('AbsoluteEosinophilCount_Test_Result',
            df['AbsoluteEosinophilCount_Test_Result'] == 'Normal')

    # Scientific notation string -> float
    df.loc[df['AbsoluteMonocyteCount_Test_Result'] == '4.9e-05',
           'AbsoluteMonocyteCount_Test_Result'] = 0.000049

    nullify('AlkalinePhosphatase_Test_Result',
            df['AlkalinePhosphatase_Test_Result'] == '<5')

    nullify('Alt(Sgpt)_Test_Result',
            df['Alt(Sgpt)_Test_Result'] == 'Normal')

    nullify('Basophils_Test_Result',
            df['Basophils_Test_Result'] == 'Normal')

    nullify('Bilirubin(Total)_Test_Result',
            df['Bilirubin(Total)_Test_Result'].isin(['<3.42', '<0.2', '<3', '-', '<2.5']))

    nullify('CreatineKinase_Test_Result',
            df['CreatineKinase_Test_Result'].isin(['<18', '<23', 'Normal', '<7']))

    nullify('Creatinine_Test_Result',
            df['Creatinine_Test_Result'] == '<18')

    nullify('Eosinophils_Test_Result',
            df['Eosinophils_Test_Result'] == 'Normal')

    nullify('Gamma-Glutamyltransferase_Test_Result',
            df['Gamma-Glutamyltransferase_Test_Result'] == '<4')

    nullify('Hba1C(GlycatedHemoglobin)_Test_Result',
            df['Hba1C(GlycatedHemoglobin)_Test_Result'] == 'Normal')

    # Comma-formatted integers -> numeric
    df.loc[df['Platelets_Test_Result'] == '229,000', 'Platelets_Test_Result'] = 229000
    df.loc[df['Platelets_Test_Result'] == '303,000', 'Platelets_Test_Result'] = 303000

    nullify('Protein_Test_Result',
            df['Protein_Test_Result'].isin(['Trace', '-', '1+', '2+', '4+']))

    nullify('RedBloodCells(Rbc)_Test_Result',
            df['RedBloodCells(Rbc)_Test_Result'].isin(['-', 'Very Large']))
    df.loc[df['RedBloodCells(Rbc)_Test_Result'] == '4750,000', 'RedBloodCells(Rbc)_Test_Result'] = 4750000
    df.loc[df['RedBloodCells(Rbc)_Test_Result'] == '4820,000', 'RedBloodCells(Rbc)_Test_Result'] = 4820000

    nullify('WhiteBloodCell(Wbc)_Test_Result',
            df['WhiteBloodCell(Wbc)_Test_Result'].isin(['-', 'Very Large']))
    df.loc[df['WhiteBloodCell(Wbc)_Test_Result'] == '6,900', 'WhiteBloodCell(Wbc)_Test_Result'] = 6900
    df.loc[df['WhiteBloodCell(Wbc)_Test_Result'] == '7,900', 'WhiteBloodCell(Wbc)_Test_Result'] = 7900

    # Drop tests that are predominantly non-numeric and too sparse to recover
    # (UrineBlood: 31%, UrineGlucose: 39%, UrineKetones: 24% patient prevalence)
    df = df.drop(columns=[
        'UrineBlood_Test_Result',    'UrineBlood_Test_Unit',
        'UrineGlucose_Test_Result',  'UrineGlucose_Test_Unit',
        'UrineKetones_Test_Result',  'UrineKetones_Test_Unit',
    ])

    return df



# Audit: compute per-column filling rates after all cleaning steps
def calculate_filling_rate(file_path):
    """
    Compute the percentage of patients with at least one non-null value in
    each _Test_Result column, after all cleaning and filtering steps.

    This audit is used to verify that no important test was inadvertently
    emptied by the cleaning operations and to report final data completeness
    in the paper.

    Parameters
    ----------
    file_path : str
        Path to PROACT_LABS_v7.csv.

    Returns
    -------
    pd.DataFrame
        Table with columns [Test_Result_Column, Filling_Rate_Percentage],
        sorted by Filling_Rate_Percentage descending
        (-> saved as PROACT_LABS_Filling_Rates.csv).
    """
    df         = pd.read_csv(file_path, low_memory=False)
    df_results = df[[col for col in df.columns if col.endswith('_Test_Result')]]

    total_subjects = df['subject_id'].nunique()
    filling_rates  = {}

    for col in df_results.columns:
        non_null_count   = df[df_results[col].notna()]['subject_id'].nunique()
        filling_rates[col] = round((non_null_count / total_subjects) * 100, 2)

    filling_rate_df = pd.DataFrame(
        list(filling_rates.items()),
        columns=['Test_Result_Column', 'Filling_Rate_Percentage']
    )
    filling_rate_df = filling_rate_df.sort_values(
        by='Filling_Rate_Percentage', ascending=False
    ).reset_index(drop=True)

    return filling_rate_df





# ------------------------------------------------------------------
# Stage v8 - Add per-patient observation count
# ------------------------------------------------------------------

def observation_counter_labs(file_path):
    """
    Add an `observation_count` column recording how many visit rows exist
    per patient in the semi-wide dataset.

    Each row represents one Laboratory_Delta timepoint for a patient.

    Parameters
    ----------
    file_path : str
        Path to PROACT_LABS_v7.csv.

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
# Stage v9 - Reshape to fully wide format
# ------------------------------------------------------------------

def reshape_to_fully_wide_format(csv_file):
    """
    Reshape the semi-wide visit-level data (one row per Laboratory_Delta) into
    a fully wide patient-level DataFrame (one row per patient), where each
    visit's values are stored in sequentially prefixed column blocks.

    Within each visit block, Laboratory_Delta is placed first, followed by
    all test result/unit pairs in alphabetical order by test name. The column
    sort key ensures that Laboratory_Delta always appears at the start of each
    numbered block (e.g. 1_Laboratory_Delta, 1_Albumin_Test_Result, ...,
    2_Laboratory_Delta, 2_Albumin_Test_Result, ...).

    All test columns from the global schema are included for every visit,
    even if a given patient had no result for that test at that visit (NaN).
    This produces a complete rectangular matrix regardless of which tests
    were actually performed.

    A progress message is printed every 500 patients to monitor execution.

    Parameters
    ----------
    csv_file : str
        Path to PROACT_LABS_v8.csv.

    Returns
    -------
    pd.DataFrame
        Wide-format patient-level DataFrame
        (-> saved as PROACT_LABS_v9.csv).
    """
    df = pd.read_csv(csv_file, low_memory=False)

    df = df.sort_values(by=['subject_id', 'Laboratory_Delta'])

    fixed_cols  = ['subject_id', 'observation_count']
    value_cols  = [col for col in df.columns if col not in fixed_cols]
    test_columns = sorted([col for col in value_cols if col != 'Laboratory_Delta'])

    # All unique test prefixes (e.g. 'Albumin', 'CreatineKinase', ...)
    all_tests = sorted(list({c.rsplit('_', 2)[0] for c in test_columns}))

    grouped_rows   = []
    total_subjects = df['subject_id'].nunique()
    counter       = 0

    for subject_id, group in df.groupby('subject_id'):
        counter += 1
        if counter % 500 == 0:
            print(f"Processed {counter} patients out of {total_subjects}...")

        group     = group.reset_index(drop=True)
        obs_count = int(group['observation_count'].iloc[0])

        row_data = {
            'subject_id':       subject_id,
            'observation_count': obs_count,
        }

        for i, (_, obs) in enumerate(group.iterrows(), start=1):
            # Visit anchor: time delta
            row_data[f"{i}_Laboratory_Delta"] = obs.get('Laboratory_Delta', np.nan)

            # All test columns for this visit (NaN if the test was not performed)
            for test in all_tests:
                row_data[f"{i}_{test}_Test_Result"] = obs.get(f"{test}_Test_Result", np.nan)
                row_data[f"{i}_{test}_Test_Unit"]   = obs.get(f"{test}_Test_Unit",   np.nan)

        grouped_rows.append(row_data)

    df_grouped = pd.DataFrame(grouped_rows)

    # Sort columns: fixed headers first, then visit blocks in numeric order,
    # with Laboratory_Delta as the first column within each block
    def sort_key(col):
        if col in fixed_cols:
            return (-1, col)
        parts = col.split('_', 1)
        if len(parts) < 2:
            return (9999, col)
        num = int(parts[0]) if parts[0].isdigit() else 9999
        # Force Laboratory_Delta to sort before all test columns in its block
        return (num, "0000") if "_Laboratory_Delta" in col else (num, col)

    ordered_cols = fixed_cols + sorted(
        [c for c in df_grouped.columns if c not in fixed_cols], key=sort_key
    )
    df_grouped = df_grouped[ordered_cols]

    return df_grouped





# ------------------------------------------------------------------
# Stage v10 - Add 'LAB_' prefix to all feature columns
# ------------------------------------------------------------------

def rename_all_columns(file_path):
    """
    Prefix every feature column with 'LAB_' to namespace the laboratory
    variables when merging with other PROACT sub-datasets.

    `subject_id` is the join key and is left unchanged.

    Parameters
    ----------
    file_path : str
        Path to PROACT_LABS_v9.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns
        (-> saved as PROACT_LABS_v10.csv).
    """
    df = pd.read_csv(file_path, low_memory=False)
    df = df.rename(columns={col: f'LAB_{col}' for col in df.columns if col != 'subject_id'})
    return df










# ==================================================================
# ------------------------- PIPELINE EXECUTION ---------------------
# ==================================================================

def run(DATA_PATH, PROACT_PATH, VALIDATION_PATH):

    print("\n" * 3)
    print("=" * 60)
    print("LABS PIPELINE")
    print("=" * 60)

    

    # Reference - List unique test names and cross-validate against expert lists
    test_names = list_unique_test_names(
        PROACT_PATH + '/PROACT_LABS.csv',
        VALIDATION_PATH + '/PROACT_LABS_Validation_Dr1.csv'
    )
    test_names.to_csv(DATA_PATH + '/PROACT_LABS_Test_Names.csv', index=False)



    # Stage v2 - Normalise test names, units, and result values
    df = filter_labs(PROACT_PATH + '/PROACT_LABS.csv')
    df.to_csv(DATA_PATH + '/PROACT_LABS_v2.csv', index=False)



    # Stage v3 - Retain only expert-validated tests
    df_labs = filter_validated_tests(DATA_PATH + '/PROACT_LABS_v2.csv', data_path=DATA_PATH)
    df_labs.to_csv(DATA_PATH + '/PROACT_LABS_v3.csv', index=False)

    df_valid_tests = list_valid_tests(DATA_PATH + '/PROACT_LABS_Test_Names.csv')
    df_valid_tests.to_csv(DATA_PATH + '/PROACT_LABS_Test_Names_Validated.csv', index=False)



    # Stage v4 - Filter tests below the prevalence threshold
    df_labs = filter_by_percentage(DATA_PATH + '/PROACT_LABS_v3.csv', threshold=20.0)
    df_labs.to_csv(DATA_PATH + '/PROACT_LABS_v4.csv', index=False)



    # Stage v5 - Pivot to semi-wide format (one row per subject/visit)
    df_labs = reshape_to_semi_wide_format(DATA_PATH + '/PROACT_LABS_v4.csv')
    df_labs.to_csv(DATA_PATH + '/PROACT_LABS_v5.csv', index=False)



    # Diagnostic - Check for residual multi-unit columns
    check_multiple_units(DATA_PATH + '/PROACT_LABS_v5.csv')



    # Stage v6 - Resolve residual multi-unit conflicts
    df_labs = remove_plural_units(DATA_PATH + '/PROACT_LABS_v5.csv')
    df_labs.to_csv(DATA_PATH + '/PROACT_LABS_v6.csv', index=False)



    # Diagnostic - Identify non-numeric result values (run before stage v7)
    # check_textual_results(DATA_PATH + '/PROACT_LABS_v6.csv')



    # Stage v7 - Clean non-numeric and sentinel result values
    df_labs = remove_unusable_tests_units(DATA_PATH + '/PROACT_LABS_v6.csv')
    df_labs.to_csv(DATA_PATH + '/PROACT_LABS_v7.csv', index=False)
    # check_textual_results(DATA_PATH + '/PROACT_LABS_v7.csv')



    # Audit: compute per-column filling rates after all cleaning steps
    filling_rates = calculate_filling_rate(DATA_PATH + '/PROACT_LABS_v7.csv')
    filling_rates.to_csv(DATA_PATH + '/PROACT_LABS_Filling_Rates.csv', index=False)



    # Stage v8 - Add per-patient observation count
    df_labs = observation_counter_labs(DATA_PATH + '/PROACT_LABS_v7.csv')
    df_labs.to_csv(DATA_PATH + '/PROACT_LABS_v8.csv', index=False)



    # Stage v9 - Reshape to fully wide format
    df_labs = reshape_to_fully_wide_format(DATA_PATH + '/PROACT_LABS_v8.csv')
    df_labs.to_csv(DATA_PATH + '/PROACT_LABS_v9.csv', index=False)



    # Stage v10 - Add 'LAB_' prefix to all feature columns
    df_renamed = rename_all_columns(DATA_PATH + '/PROACT_LABS_v9.csv')
    df_renamed.to_csv(DATA_PATH + '/PROACT_LABS_v10.csv', index=False)