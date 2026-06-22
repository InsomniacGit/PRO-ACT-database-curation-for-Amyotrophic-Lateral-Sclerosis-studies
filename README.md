# PRO-ACT Database Curation for Amyotrophic Lateral Sclerosis Studies

This repository contains the full data curation and machine learning pipeline described in:

> **PRO-ACT database curation for Amyotrophic Lateral Sclerosis studies**  
> Lucas Bouclier, Christelle Dartigues-Pallez, Johan Montagnat  
> Université Côte d'Azur — I3S Laboratory
> 
> *Manuscript currently under review and not yet formally published.*

The pipeline transforms the raw PROACT clinical dataset into structured, analysis-ready supervised learning files and evaluates them with a Random Forest regression baseline.

---

## Table of contents

1. [Overview](#overview)
2. [Repository structure](#repository-structure)
3. [Data access](#data-access)
4. [Installation](#installation)
5. [Running the pipeline](#running-the-pipeline)
6. [Script reference](#script-reference)
7. [Citation](#citation)
8. [Authors](#authors)
7. [Licence](#licence)

---

## Overview

The **PRO-ACT** (Pooled Resource Open-Access ALS Clinical Trials) database aggregates longitudinal clinical data from 23 ALS trials covering 11,675 patients across 16 clinical tables.

This codebase implements a six-stage pipeline:

| Stage | Description |
|---|---|
| **A** | Per-table preprocessing — cleaning, unit harmonisation, feature engineering |
| **B** | Non-temporal merge — union join of all static (no-Delta) tables |
| **C** | First-symptom temporal alignment — re-express all time deltas relative to symptom onset |
| **D** | Interval feature extraction and supervised learning file generation |
| **E** | Random Forest prediction and evaluation |
| **F** | Parsing and formatting of machine-learning results into Excel summaries |

---

## Repository structure

```
├── environment.yml                           # Conda reproducibility environment
│
├── Resources/
│   └── VALIDATION/
│       └── PROACT_LABS_Validation_Dr1.csv    # Expert laboratory-test validation file
│
└── Treatments/
    ├── main.py                               # Complete pipeline entry point (Stages A–F)
    │
    ├── A_Tables_Preprocessing/               # All single table preprocessing
    │   ├── ADVERSEEVENTS.py
    │   ├── ALSFRS.py
    │   ├── ALSHISTORY.py
    │   ├── CONMEDS.py
    │   ├── DEATHDATA.py
    │   ├── DEMOGRAPHICS.py
    │   ├── ELESCORIAL.py
    │   ├── FAMILYHISTORY.py
    │   ├── FVC.py
    │   ├── HANDGRIPSTRENGTH.py
    │   ├── LABS.py
    │   ├── MUSCLESTRENGTH.py
    │   ├── RILUZOLE.py
    │   ├── SVC.py
    │   ├── TREATMENT.py
    │   └── VITALSIGNS.py
    │
    ├── B_Merge_Nodelta/
    │   └── MERGE_NODELTA.py                # Merge of all non-temporal tables
    │
    ├── C_Alignment_First_Symptoms/
    │   └── ALIGNMENT_FIRST_SYMPTOMS.py     # Alignment relative to symptom onset
    │
    ├── D_Intervals/
    │   ├── INTERVALS_COUNT.py              # Interval observation counts
    │   ├── INTERVALS_ALL.py                # Interval statistical feature extraction
    │   └── INTERVALS_CUT.py                # Supervised learning file generation
    │
    ├── E_Random_Forest/
    │   └── RF_PREDICTION.py                # Random Forest regression evaluation
    │
    └── F_Result_Parser/
        └── RESULT_TEXT_TO_EXCEL.py         # Conversion of result logs to Excel summaries
```

**Expected data directory layout** (configured via path variables in `main.py` script):

```
DATA_PROACT/
├── 2022_07_29_PROACT_ALL_FORMS/    # Raw PROACT CSV exports (2022-07-29 release)
│
├── VALIDATION/                     # External lab-test validation files (given in this repository)
│   └── PROACT_LABS_Validation_Dr1.csv
│
├── Preprocessed_Tables/            # Stage A outputs
│
├── Merge/                          # Stage B outputs
│
├── First_Symptoms/                 # Stage C outputs
│
├── Intervals/                      # Stage D outputs
│   ├── Count/
│   ├── CSV/
│   └── Cut/
│
└── Results/
    ├── Feature Importance/         # Stage E outputs
    ├── PROACT - RF Results.txt     # Stage E outputs
    └── PROACT - RF Results.xlsx    # Stage F outputs
```

---

## Data access

The PRO-ACT dataset can be obtained after registration through the official request portal:

https://ncri1.partners.org/ProACT/Account/Register

This repository was developed against the **2022-07-29 release**.  
Once downloaded, point the path variables at the top of `main.py` to your local copy.

The external laboratory validation file used by the LABS preprocessing step must be placed in the `VALIDATION/` directory.

---

## Installation

All dependencies are pinned in `environment.yml`.  [Conda](https://docs.conda.io/) or [Miniconda](https://docs.anaconda.com/miniconda/) is required.

```bash
# 1. Clone the repository
git clone https://github.com/InsomniacGit/PRO-ACT-database-curation-for-Amyotrophic-Lateral-Sclerosis-studies.git
cd PRO-ACT-database-curation-for-Amyotrophic-Lateral-Sclerosis-studies

# 2. Create and activate the environment
conda env create -f environment.yml
conda activate data_proact
```

**Python and key library versions:**

| Package | Version |
|---|---|
| Python | 3.11.9 |
| pandas | 3.0.0 |
| numpy | 2.4.2 |
| pyarrow | 23.0.1 |
| scikit-learn | 1.8.0 |
| scipy | 1.17.0 |
| matplotlib | 3.10.8 |
| openpyxl | 3.1.5 |
| tqdm | 4.67.3 |

---

## Running the pipeline

The complete PRO-ACT curation and machine-learning workflow is executed through a single entry point:

python Treatments/main.py

Before execution, update the path variables defined at the top of `main.py` so that they point to your local PRO-ACT installation and output directories.

The pipeline automatically executes all stages in the correct dependency order

Stages D and E can require substantial execution time depending on hardware configuration and the number of generated prediction datasets.

---

## Script reference

### Preprocessing — Stage A

Each `PROACT_<TABLE>_processing.py` script exposes a `run(data_path, proact_path)` function and implements the full cleaning chain for its table, including missing-value handling, unit harmonisation, duplicate resolution, and temporal consistency checks.  Key design decisions per table are documented in the script docstrings.

Tables that have no temporal Delta column (ALSHISTORY, DEATHDATA, DEMOGRAPHICS, ELESCORIAL, FAMILYHISTORY, RILUZOLE, TREATMENT) and tables whose Delta column was dropped during preprocessing (ADVERSEEVENTS, CONMEDS) do not produce interval feature files in Stage D.

### Merge — Stage B (`PROACT_MERGE_NODELTA.py`)

Performs a union join across all nine non-temporal tables, producing a single patient-level baseline feature matrix (`PROACT_MERGE_NODELTA_V2.csv`).

### Alignment — Stage C (`PROACT_FIRST_SYMPTOMS_alignment.py`)

Re-expresses all Delta columns relative to `HIS_Onset_Delta` (days from first symptom to study enrolment), producing first-symptom-aligned counterparts of every temporal table.

### Interval extraction — Stage D

| Script | Role |
|---|---|
| `PROACT_INTERVALS_COUNT.py` | Counts observations per 90-day interval per patient; produces distribution statistics |
| `INTERVALS_ALL.py` | Computes interval summary statistics (mean, median, slope, min, max, ...) for each clinical variable |
| `PROACT_INTERVALS_CUT.py` | Generates Fixed and Sliding prediction-ready CSV files; supports single-table, pairwise and full all-table merge configurations |

### Random Forest prediction — Stage E

| Script | Role |
|---|---|
| `RF_PREDICTION.py` | Trains and evaluates Random Forest regression models on every generated prediction dataset; computes MAE, RMSE and R² metrics using 10-fold cross-validation and exports feature importance rankings |

### Result parsing — Stage F

| Script | Role |
|---|---|
| `RESULT_TEXT_TO_EXCEL.py` | Parses raw text outputs generated during Stage E and produces formatted Excel workbooks summarising model performance across all experimental configurations |

---

## Citation

If you use this code in your work, please cite the associated manuscript:

```bibtex
@unpublished{bouclier_proact_2026,
  title  = {PRO-ACT database curation for Amyotrophic Lateral Sclerosis studies},
  author = {Bouclier, Lucas and Dartigues-Pallez, Christelle and Montagnat, Johan},
  note   = {Manuscript under peer review},
  year   = {2026}
}
```

The citation will be updated once the manuscript is formally published.

---

## Authors

- **Lucas Bouclier** — Université Côte d'Azur, I3S Laboratory
- **Christelle Dartigues-Pallez** — Université Côte d'Azur, I3S Laboratory
- **Johan Montagnat** — Université Côte d'Azur, I3S Laboratory

---

## Licence

This code is released for research reproducibility purposes.
See https://ncri1.partners.org/ProACT/Account/Register for access and licensing information.
