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





# ------------------------------------------------------------------
# Path configuration
# ------------------------------------------------------------------

# Root directory containing raw PROACT CSV exports
PROACT_PATH = str(
    Path.home() / "Desktop" / "DATA_PROACT" / "2022_07_29_PROACT_ALL_FORMS"
)

# Root directory for all processed outputs
DATA_PATH = str(
    Path.home() / "Desktop" / "DATA_PROACT" / "Preprocessed_Tables"
)

# Root directory containing external validation files (e.g. Desnuelles test list)
VALIDATION_PATH = str(
    Path.home() / "Desktop" / "DATA_PROACT" / "VALIDATION"
)

# Root directory for merged (multi-table) datasets
MERGE_PATH = str(
    Path.home() / "Desktop" / "DATA_PROACT" / "Merge"
)

# Root directory for first-symptoms-aligned output datasets
FIRST_SYMPTOMS_PATH = str(
    Path.home() / "Desktop" / "DATA_PROACT" / "First_Symptoms"
)

# # Root directory for interval-based supervised learning datasets
# INTERVAL_PATH = str(
#     Path.home() / "Desktop" / "DATA_PROACT" / "Intervals" / "Cut"
# )



# Create the output subdirectories if they do not already exist
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

if not os.path.exists(MERGE_PATH):
    os.makedirs(MERGE_PATH)

if not os.path.exists(FIRST_SYMPTOMS_PATH):
    os.makedirs(FIRST_SYMPTOMS_PATH)

# if not os.path.exists(INTERVAL_PATH):
#     os.makedirs(INTERVAL_PATH)





def main():

    print("=" * 60)
    print("PROACT GLOBAL PIPELINE")
    print("=" * 60)

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
    LABS.run(DATA_PATH, PROACT_PATH, VALIDATION_PATH)
    MUSCLESTRENGTH.run(DATA_PATH, PROACT_PATH)
    RILUZOLE.run(DATA_PATH, PROACT_PATH)
    SVC.run(DATA_PATH, PROACT_PATH)
    TREATMENT.run(DATA_PATH, PROACT_PATH)
    VITALSIGNS.run(DATA_PATH, PROACT_PATH)

    MERGE_NODELTA.run(DATA_PATH, MERGE_PATH)

    ALIGNMENT_FIRST_SYMPTOMS.run(DATA_PATH, MERGE_PATH, FIRST_SYMPTOMS_PATH)

    print("\nAll pipelines completed.")

if __name__ == "__main__":
    main()