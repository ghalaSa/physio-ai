# Sample clips for the live-upload demo

Real MobiPhysio recordings (CC0). Not versioned in git (see `.gitignore`); download
them with the file ids below from `https://dataverse.harvard.edu/api/access/datafile/<id>`.

| File | Exercise (truth) | Participant | Split | Physio score | Dataverse id | Size |
|---|---|---|---|---|---|---|
| E06_P19_AF_VFL_GM.mp4 | E06 Wrist Extension | P19 | test (held-out) | 100.0 | 11788097 | 13 MB |
| E07_P18_AF_VFL_GM.mp4 | E07 Hip Joint Flexion | P18 | test (held-out) | 92.9 | 11788620 | 23 MB |
| E08_P46_AF_VFL_GM.mp4 | E08 Both Leg Flexion | P46 | test (held-out) | not scored | 11789248 | 11 MB |
| E01_P01_AL_VFL_GM.mp4 | E01 Abduction | P01 | train | 78.1 | 11674927 | 25 MB |

Verified through the live upload pipeline on 2026-09-06 (local Mac, MediaPipe 0.10.35):

| File | Predicted | Confidence | Predicted score | Reps | Notes |
|---|---|---|---|---|---|
| E06_P19 | E06 Wrist Extension | 0.96 | 96.7 | 37 | 30 s clip; triggers range-of-motion, asymmetry, consistency and tempo observations |
| E07_P18 | E07 Hip Joint Flexion | 0.98 | 78.4 | 16 | 42 s clip; consistency and tempo observations |
| E08_P46 | E08 Both Leg Flexion | 0.97 | 71.1 | 1 | 5 s clip; shows the "few repetitions" information flag |
| E01_P01 | E01 Abduction | 0.92 | 79.9 | 13 | 30 s clip from a training participant |

Rejected candidates (not kept): E09_P31 (3 s, no repetitions detected), E03_P26 (5 s,
2 repetitions), E05_P26 (circumduction predicted as abduction, the known E05/E01 confusion).
