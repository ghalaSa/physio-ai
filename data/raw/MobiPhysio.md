# MobiPhysio Dataset (v1.1)

MobiPhysio is a **2D smartphone video dataset of physiotherapy exercises** designed to support research and development of **AI-driven exercise assessment and monitoring** using low-cost 2D camera devices (e.g., mobile phones). It includes multi-angle recordings and real-world variations (lighting, jitter, occlusion, and resolution), along with **expert physiotherapist evaluation scores**.

## Quick facts (from the Excel summary sheet)
- **Total videos:** 3,686
- **Participants:** 58 (**Male:** 39, **Female:** 19)
- **Exercises:** 9 Active Range of Motion (AROM) physiotherapy exercises (E01–E09)
- **Angles:** Front / Left / Right
- **Annotations:** Exercise Accuracy Assessment Questionnaire (EAAQ) scores by **three certified physiotherapists** (averaged and scaled to 0–100)

## Repository / DOI
Dataset DOI:
```text
doi:10.7910/DVN/XSI0QN
```

## Associated publication
Please refer to the accompanying **Data in Brief** article/manuscript (cited below) for the full dataset creation and annotation details.

## Dataset contents
Typical files included in the dataset package:
- **Video files** (`.mp4`): one file per exercise instance.
- **Annotation file** (CSV): `Physiotherapist Exercise Marking.csv`
- **Research paper / data descriptor:** See the associated publication for full methodology, protocol, and detailed dataset description.

## Exercises (9 classes)
Exercise code → name (as listed in the Excel file):
- **E01** Abduction
- **E02** Adduction
- **E03** Lateral Rotation
- **E04** Medial Rotation
- **E05** Circumduction
- **E06** Wrist Extension
- **E07** Hip Joint Flexion
- **E08** Both Leg Flexion
- **E09** Back Extension

## Dataset statistics (counts per exercise)
Total number of videos per exercise (sums to 3,686):

| Exercise | Videos |
|---|---:|
| E01 | 558 |
| E02 | 384 |
| E03 | 550 |
| E04 | 383 |
| E05 | 511 |
| E06 | 354 |
| E07 | 251 |
| E08 | 356 |
| E09 | 339 |

## Participant-wise raw counts
**Table 1. Participant-wise raw video counts per exercise and total videos (from the Excel sheet ‘Total Count of Dataset’).**

<details>
<summary>Click to expand Table 1 (participant-wise counts)</summary>

|   Person Number | Gender   |   E01 |   E02 |   E03 |   E04 |   E05 |   E06 |   E07 |   E08 |   E09 |   Number of Videos |
|----------------:|:---------|------:|------:|------:|------:|------:|------:|------:|------:|------:|-------------------:|
|               1 | Male     |    18 |    18 |    18 |    18 |    18 |     0 |     0 |     0 |     0 |                 90 |
|               2 | Male     |    18 |    18 |    18 |    18 |    18 |    18 |    18 |    18 |    18 |                162 |
|               3 | Male     |    18 |    18 |    18 |    18 |    15 |    18 |    18 |    18 |    18 |                159 |
|               4 | Male     |    18 |    18 |    18 |    18 |    17 |    18 |    18 |    18 |    18 |                161 |
|               5 | Male     |    18 |    18 |    18 |    18 |    18 |    18 |    18 |    18 |    18 |                162 |
|               6 | Male     |    18 |    18 |    18 |    18 |    18 |    17 |    18 |    18 |    18 |                161 |
|               7 | Male     |    18 |    18 |    18 |    18 |    18 |    18 |    18 |    18 |    18 |                162 |
|               8 | Male     |    18 |    18 |    18 |    18 |    17 |    18 |    18 |    18 |    18 |                161 |
|               9 | Female   |    12 |    12 |    12 |    12 |    12 |    12 |     0 |     0 |     0 |                 72 |
|              10 | Female   |    12 |    12 |    12 |    12 |    11 |    11 |     0 |     0 |     0 |                 70 |
|              11 | Female   |    11 |    12 |    12 |    12 |    12 |    11 |     0 |     0 |     0 |                 70 |
|              12 | Female   |    12 |    12 |    12 |    12 |    12 |    12 |     0 |     0 |     0 |                 72 |
|              13 | Female   |    18 |    18 |    18 |    18 |    18 |    18 |     0 |     0 |     0 |                108 |
|              14 | Female   |    12 |    12 |    12 |    12 |    12 |    12 |     0 |     0 |     0 |                 72 |
|              15 | Female   |    12 |    12 |    12 |    12 |    12 |    12 |     0 |     0 |     0 |                 72 |
|              16 | Female   |    18 |    18 |    18 |    18 |    18 |    17 |     0 |     0 |     0 |                107 |
|              17 | Male     |    18 |    18 |    17 |    17 |    17 |    17 |    18 |    18 |    17 |                157 |
|              18 | Male     |    18 |    18 |    18 |    18 |    18 |    18 |    18 |    18 |    18 |                162 |
|              19 | Male     |    18 |    18 |    18 |    18 |    18 |    17 |    17 |    18 |    18 |                160 |
|              20 | Male     |    17 |    18 |    18 |    18 |    18 |    18 |    18 |    18 |    18 |                161 |
|              21 | Male     |    18 |    18 |    18 |    18 |    18 |    18 |    18 |    18 |    18 |                162 |
|              22 | Male     |    18 |    18 |    18 |    18 |    18 |    18 |    18 |    18 |    18 |                162 |
|              23 | Male     |    18 |    18 |    18 |    18 |    18 |    18 |    18 |    18 |    18 |                162 |
|              24 | Male     |     6 |     6 |     6 |     6 |     0 |     0 |     0 |     0 |     0 |                 24 |
|              25 |          |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |                  0 |
|              26 | Female   |     7 |     0 |     9 |     0 |     8 |     0 |     0 |     0 |     0 |                 24 |
|              27 | Male     |     7 |     0 |     0 |     0 |     0 |     0 |     0 |     7 |     5 |                 19 |
|              28 | Male     |     9 |     0 |     0 |     0 |     7 |     0 |     0 |     8 |     9 |                 33 |
|              29 | Male     |     9 |     0 |     9 |     0 |     9 |     0 |     0 |     9 |     8 |                 44 |
|              30 | Female   |     8 |     0 |     7 |     0 |     8 |     0 |     0 |     0 |     0 |                 23 |
|              31 | Male     |     9 |     0 |     1 |     0 |     9 |     0 |     0 |     1 |     9 |                 29 |
|              32 | Male     |     9 |     0 |     9 |     0 |     8 |     0 |     0 |     8 |     8 |                 42 |
|              33 | Male     |     5 |     0 |     7 |     0 |     0 |     0 |     0 |     0 |     0 |                 12 |
|              34 | Female   |     8 |     0 |     9 |     0 |     9 |     0 |     0 |     0 |     0 |                 26 |
|              35 | Male     |     9 |     0 |     9 |     0 |     9 |     0 |     0 |     8 |     9 |                 44 |
|              36 | Male     |     9 |     0 |     1 |     0 |     9 |     0 |     0 |     0 |     0 |                 19 |
|              37 | Male     |     1 |     0 |     1 |     0 |     1 |     0 |     0 |     9 |     1 |                 13 |
|              38 | Male     |     0 |     0 |     0 |     0 |     1 |     0 |     0 |     9 |     9 |                 19 |
|              39 | Male     |     1 |     0 |     9 |     0 |     1 |     0 |     0 |     0 |     1 |                 12 |
|              40 | Male     |     8 |     0 |     8 |     0 |     8 |     0 |     0 |     0 |     0 |                 24 |
|              41 | Male     |     1 |     0 |     1 |     0 |     1 |     0 |     0 |     1 |     9 |                 13 |
|              42 | Male     |     1 |     0 |     1 |     0 |     1 |     0 |     0 |     9 |     8 |                 20 |
|              43 | Male     |     0 |     0 |     7 |     0 |     0 |     0 |     0 |     0 |     0 |                  7 |
|              44 |          |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |                  0 |
|              45 | Male     |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     6 |     6 |                 12 |
|              46 | Male     |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     6 |     6 |                 12 |
|              47 | Male     |     0 |     0 |     6 |     0 |     0 |     0 |     0 |     6 |     0 |                 12 |
|              48 | Male     |     4 |     0 |     5 |     0 |     0 |     0 |     0 |     6 |     0 |                 15 |
|              49 | Male     |     0 |     0 |     6 |     0 |     0 |     0 |     0 |     6 |     0 |                 12 |
|              50 | Male     |     0 |     0 |     6 |     0 |     0 |     0 |     0 |     5 |     0 |                 11 |
|              51 |          |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |                  0 |
|              52 | Male     |     9 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |                  9 |
|              53 | Female   |     9 |     0 |     9 |     0 |     7 |     0 |     0 |     0 |     0 |                 25 |
|              54 |          |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |                  0 |
|              55 | Male     |     8 |     0 |     0 |     0 |     7 |     0 |     0 |     0 |     0 |                 15 |
|              56 | Female   |     4 |     0 |     6 |     0 |     7 |     0 |     0 |     0 |     0 |                 17 |
|              57 | Female   |     6 |     0 |     7 |     0 |     7 |     0 |     0 |     0 |     0 |                 20 |
|              58 | Female   |     8 |     0 |     7 |     0 |     7 |     0 |     0 |     0 |     0 |                 22 |
|              59 | Female   |     7 |     0 |     6 |     0 |     7 |     0 |     0 |     0 |     0 |                 20 |
|              60 | Female   |     9 |     0 |     7 |     0 |     9 |     0 |     0 |     0 |     0 |                 25 |
|              61 | Female   |     3 |     0 |     8 |     0 |     0 |     0 |     0 |     0 |     0 |                 11 |
|              62 | Female   |     8 |     0 |     6 |     0 |     0 |     0 |     0 |     0 |     0 |                 14 |

</details>

**Note (excluded IDs):** Person numbers **25, 44, 51, and 54** appear as all-zero rows because the corresponding recordings were **removed during dataset curation** due to **technical/quality issues** (e.g., corrupted/missing video segments or clips containing non-exercise preparation footage that could not be reliably labeled). These IDs are kept only to preserve the original participant indexing, and they are **not counted as participants**.

## File naming convention
Each video filename follows a pattern that encodes exercise, participant, angle, variation, and gender:

```text
E{xx}_P{yy}_A{F|L|R}_V{FL|ML|LL|LJ|HJ|O|LR}_G{M|F}.mp4
```

Example:
```text
E01_P01_AF_VFL_GM.mp4
```

### Name format (decoded)
A filename is composed of **five parts**:

`E01` `_` `P01` `_` `AF` `_` `VFL` `_` `GM`

**Table 2. Video filename codebook (Name Format).**

| Part | Meaning | Codes |
|---|---|---|
| `E{xx}` | Exercise number | `E01`–`E09` |
| `P{yy}` | Person number | `P01`, `P02`, … |
| `A{F/L/R}` | Angle | `F`=Front, `L`=Left, `R`=Right |
| `V{...}` | Variation | `FL`=Full Light, `ML`=Medium Light, `LL`=Low Light, `LJ`=Low Jitter, `HJ`=High Jitter, `O`=Occlusion, `LR`=Low Resolution |
| `G{M/F}` | Gender | `M`=Male, `F`=Female |

## Annotations (physiotherapist scores)
The file `Physiotherapist Exercise Marking (1).xlsx` contains scoring information produced from exercise-specific **Exercise Accuracy Assessment Questionnaires (EAAQs)**.

- Use the **`Combine Dataset`** sheet as the master annotation table.
- The sheet stores scoring from **three judges** and also provides an **average score** scaled to **0–100**.

## Recommended evaluation protocol
To avoid leakage and better test generalization:
- **Split by participant** (e.g., 70% train / 10% validation / 20% test by participant IDs).
- Report results per **exercise**, per **variation**, and per **camera angle**.

## Ethics & consent
See the paper for the ethics statement and approval details.

## License
See the dataset repository page for the license/data use agreement.

## Citation
If you use this dataset, please cite:

```text
[1] M. T. B. Iqbal et al., “MobiPhysio,” Harvard Dataverse, 2025, doi:10.7910/DVN/XSI0QN.

[2] M. T. B. Iqbal et al., “MobiPhysio: A 2D Video Dataset of Physiotherapy Exercises for AI-Driven Assessment and Monitoring,” Data in Brief (manuscript), 2025.
```

## Contact
Corresponding author (from the paper):
- **Md. Tauhid Bin Iqbal** — tauhid.iqbal@ewubd.edu

---

**README last updated:** 2026-02-10
