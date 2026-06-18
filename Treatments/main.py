"""
PROACT Global Pipeline - Centralised Entry Point
=================================================
This script is the single entry point for the complete PROACT data-processing
and machine-learning pipeline. It sequentially executes all preprocessing,
merging, temporal alignment, interval generation, prediction, and result
formatting modules in the correct dependency order.

The pipeline transforms raw PROACT clinical trial data into interval-based
datasets, trains Random Forest prediction models, and produces formatted
result summaries for analysis.

Execution order
---------------
Stage A - Per-table preprocessing (16 tables, run independently):
    ADVERSEEVENTS, ALSFRS, ALSHISTORY, CONMEDS, DEATHDATA, DEMOGRAPHICS,
    ELESCORIAL, FAMILYHISTORY, FVC, HANDGRIPSTRENGTH, LABS, MUSCLESTRENGTH,
    RILUZOLE, SVC, TREATMENT, VITALSIGNS

Stage B - Non-temporal union merge:
    MERGE_NODELTA  
        (joins all non-temporal tables into a single patient-level baseline 
         feature matrix)

Stage C - First-symptom temporal alignment:
    ALIGNMENT_FIRST_SYMPTOMS  
        (re-expresses all Delta columns relative to HIS_Onset_Delta across 
         every temporal table)

Stage D - Interval dataset generation:
    INTERVALS_COUNT, INTERVALS_ALL, INTERVALS_CUT
        (build 90-day interval representations, generate interval-level
         statistics and distributions, and create prediction-ready datasets)

Stage E - Random Forest prediction:
    RF_PREDICTION
        (trains and evaluates Random Forest models on interval datasets)

Stage F - Result parsing:
    RESULT_TEXT_TO_EXCEL
        (converts raw experiment outputs into formatted Excel summaries)

To run the full pipeline:
    python main.py

Author: Bouclier Lucas
Data:   PROACT dataset (2022-07-29 release)
"""



from pathlib import Path
import os

from A_Tables_Preprocessing import ADVERSEEVENTS
from A_Tables_Preprocessing import ALSFRS
from A_Tables_Preprocessing import ALSHISTORY
from A_Tables_Preprocessing import CONMEDS
from A_Tables_Preprocessing import DEATHDATA
from A_Tables_Preprocessing import DEMOGRAPHICS
from A_Tables_Preprocessing import ELESCORIAL
from A_Tables_Preprocessing import FAMILYHISTORY
from A_Tables_Preprocessing import FVC
from A_Tables_Preprocessing import HANDGRIPSTRENGTH
from A_Tables_Preprocessing import LABS
from A_Tables_Preprocessing import MUSCLESTRENGTH
from A_Tables_Preprocessing import RILUZOLE
from A_Tables_Preprocessing import SVC
from A_Tables_Preprocessing import TREATMENT
from A_Tables_Preprocessing import VITALSIGNS

from B_Merge_Nodelta import MERGE_NODELTA

from C_Alignment_First_Symptoms import ALIGNMENT_FIRST_SYMPTOMS

from D_Intervals import INTERVALS_COUNT
from D_Intervals import INTERVALS_ALL
from D_Intervals import INTERVALS_CUT

from E_Random_Forest import RF_PREDICTION

from F_Result_Parser import RESULT_TEXT_TO_EXCEL



# ------------------------------------------------------------------
# Path configuration
# ------------------------------------------------------------------

# Root directory containing raw PROACT CSV exports (2022-07-29 release)
PROACT_PATH = str(
    Path.home() / "Desktop" / "DATA_PROACT" / "2022_07_29_PROACT_ALL_FORMS"
)

# Root directory containing external expert validation files
VALIDATION_PATH = str(
    Path.home() / "Desktop" / "DATA_PROACT" / "VALIDATION"
)

# Root directory for per-table preprocessed CSV outputs (Stage A)
DATA_PATH = str(
    Path.home() / "Desktop" / "DATA_PROACT" / "Preprocessed_Tables"
)

# Root directory for the non-temporal union merge output (Stage B)
MERGE_NODELTA_PATH = str(
    Path.home() / "Desktop" / "DATA_PROACT" / "Merge"
)

# Root directory for first-symptom-aligned datasets (Stage C)
FIRST_SYMPTOMS_PATH = str(
    Path.home() / "Desktop" / "DATA_PROACT" / "First_Symptoms"
)

# Root directory for interval-based supervised learning datasets (Stage D)
INTERVALS_PATH = str(
    Path.home() / "Desktop" / "DATA_PROACT" / "Intervals"
)

INTERVALS_COUNT_PATH = str(
    Path.home() / "Desktop" / "DATA_PROACT" / "Intervals" / "Count"
)

INTERVALS_FULL_PATH = str(
    Path.home() / "Desktop" / "DATA_PROACT" / "Intervals" / "CSV"
)

INTERVALS_CUT_PATH = str(
    Path.home() / "Desktop" / "DATA_PROACT" / "Intervals" / "Cut"
)

RESULT_PATH = str(
    Path.home() / "Desktop" / "DATA_PROACT" / "Results"
)

FEATURE_IMPORTANCE_PATH = str(
    Path.home() / "Desktop" / "DATA_PROACT" / "Results" / "Feature Importance"
)

# Create output directories if they do not already exist
for path in (DATA_PATH, MERGE_NODELTA_PATH, FIRST_SYMPTOMS_PATH, INTERVALS_PATH, INTERVALS_COUNT_PATH, INTERVALS_FULL_PATH, INTERVALS_CUT_PATH, FEATURE_IMPORTANCE_PATH):
    if not os.path.exists(path):
        os.makedirs(path)



# ------------------------------------------------------------------
# Global configuration
# ------------------------------------------------------------------

PAPER_RESULTS_ONLY = True  # Set to False to generate all intermediate datasets, not just the final results used in the paper





# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

def main():
    """
    Run the complete PROACT curation pipeline in the correct dependency order.

    Stage A modules are independent of each other and can in principle be run 
    in any order. Stages B, C, D, E and F depend on previous Stage outputs and 
    mustrun after all previous stages are completed.
    """

    print("=" * 60)
    print("PROACT GLOBAL PIPELINE")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Stage A - Per-table preprocessing
    # ------------------------------------------------------------------
    # Each call reads raw PROACT CSV files from PROACT_PATH, applies the
    # full cleaning and feature-engineering pipeline described in the paper,
    # and writes the final versioned CSV to DATA_PATH.
    # LABS additionally requires VALIDATION_PATH for the expert test list.

    ADVERSEEVENTS.run(DATA_PATH, PROACT_PATH)
    ALSFRS.run(DATA_PATH, PROACT_PATH)
    ALSHISTORY.run(DATA_PATH, PROACT_PATH)
    CONMEDS.run(DATA_PATH, PROACT_PATH)
    DEATHDATA.run(DATA_PATH, PROACT_PATH)
    DEMOGRAPHICS.run(DATA_PATH, PROACT_PATH)
    ELESCORIAL.run(DATA_PATH, PROACT_PATH)
    FAMILYHISTORY.run(DATA_PATH, PROACT_PATH)
    FVC.run(DATA_PATH, PROACT_PATH)
    HANDGRIPSTRENGTH.run(DATA_PATH, PROACT_PATH)
    LABS.run(DATA_PATH, PROACT_PATH, VALIDATION_PATH)  # requires external validation file
    MUSCLESTRENGTH.run(DATA_PATH, PROACT_PATH)
    RILUZOLE.run(DATA_PATH, PROACT_PATH)
    SVC.run(DATA_PATH, PROACT_PATH)
    TREATMENT.run(DATA_PATH, PROACT_PATH)
    VITALSIGNS.run(DATA_PATH, PROACT_PATH)

    # ------------------------------------------------------------------
    # Stage B - Non-temporal union merge
    # ------------------------------------------------------------------
    # Joins all non-temporal (no Delta column) tables into a single
    # patient-level baseline feature matrix saved to MERGE_NODELTA_PATH.

    MERGE_NODELTA.run(DATA_PATH, MERGE_NODELTA_PATH)

    # ------------------------------------------------------------------
    # Stage C - First-symptom temporal alignment
    # ------------------------------------------------------------------
    # Re-expresses all Delta columns relative to the date of first symptom
    # onset (HIS_Onset_Delta) across every temporal table, producing aligned
    # versions saved to FIRST_SYMPTOMS_PATH.

    ALIGNMENT_FIRST_SYMPTOMS.run(DATA_PATH, MERGE_NODELTA_PATH, FIRST_SYMPTOMS_PATH)

    # ------------------------------------------------------------------
    # Stage D - Interval-based supervised learning datasets
    # ------------------------------------------------------------------

    INTERVALS_COUNT.run(DATA_PATH, FIRST_SYMPTOMS_PATH, INTERVALS_COUNT_PATH)
    INTERVALS_ALL.run(DATA_PATH, FIRST_SYMPTOMS_PATH, INTERVALS_FULL_PATH)
    INTERVALS_CUT.run(INTERVALS_FULL_PATH, INTERVALS_CUT_PATH, MERGE_NODELTA_PATH, FIRST_SYMPTOMS_PATH, PAPER_RESULTS_ONLY)     # It takes a while to run

    # ------------------------------------------------------------------
    # Stage E - Random Forest prediction
    # ------------------------------------------------------------------

    RF_PREDICTION.run(INTERVALS_CUT_PATH, RESULT_PATH, FEATURE_IMPORTANCE_PATH, PAPER_RESULTS_ONLY)     # It takes a while to run

    # ------------------------------------------------------------------
    # Stage F - Result parsing and Excel formatting
    # ------------------------------------------------------------------

    RESULT_TEXT_TO_EXCEL.run(RESULT_PATH)

    print("\nAll pipelines completed.")



if __name__ == "__main__":
    main()