# Dataset

`ghs_modeling_dataset_14148.csv` — 14,148 single-substance industrial chemicals, each carrying
at least one valid GHS H-code. This is the modeling set used throughout the paper; 3 compounds
fail ETKDGv3 3D embedding, leaving 14,145 for model development.

## Provenance

Seeded from two Korean regulatory sources — the Korea Occupational Safety and Health Agency
(KOSHA) chemical database and the Dangjin City industrial chemical usage registry — and expanded
via the PubChem PUG REST API, which supplied CIDs, canonical SMILES and physicochemical
properties. CAS Registry Number is the join key. Entries that failed the PubChem lookup, or that
lacked a valid single-substance SMILES (mixtures such as gasoline), were excluded.

The full inventory before label filtering held 22,497 chemicals, of which 8,349 (37.1%) had no
recorded H-code, NFPA rating or flash point. Those unlabeled entries are the screening target of
the paper and are not part of this file.

## Columns

| Column | Description |
|---|---|
| `CAS` | CAS Registry Number (join key) |
| `CID` | PubChem Compound ID |
| `SMILES` | Canonical SMILES |
| `Formula` | Molecular formula |
| `MolWt` | Molecular weight |
| `LogP` | Octanol–water partition coefficient |
| `TPSA` | Topological polar surface area |
| `NumHDonors`, `NumHAcceptors` | Hydrogen-bond donor / acceptor counts |
| `NumRings` | Ring count |
| `FlashPoint_C` | Flash point (°C) |
| `LFL_percent`, `UFL_percent` | Lower / upper flammability limit (vol %) |
| `HeatOfComb_kJ_mol` | Heat of combustion (kJ/mol) |
| `BoilingPoint_C`, `MeltingPoint_C` | Boiling / melting point (°C) |
| `AutoignitionTemp_C` | Autoignition temperature (°C) |
| `VaporPressure_mmHg` | Vapor pressure (mmHg) |
| `Density_g_cm3` | Density (g/cm³) |
| `Y_NFPA_Health`, `Y_NFPA_Fire`, `Y_NFPA_React` | NFPA 704 ratings (0–4) |
| `Y_GHS_Signal` | GHS signal word (`Danger` / `Warning`) |
| `Y_GHS_H_Codes` | Comma-separated H-codes — **the prediction target** |

Physicochemical columns are sparsely populated (PubChem coverage varies by property) and are
not the model input on their own; the 2D feature vector is rebuilt from SMILES with RDKit at
runtime. Empty cells are genuine missing values, not zeros.

## Label construction

`Y_GHS_H_Codes` is parsed into a 22-dimensional binary vector by `build_label_vector()` in
`src/ghs_full_pipeline.py`. The 73 raw H-codes present in the dataset map onto 22 mechanistically
coherent classes; severity-graded codes within one endpoint are merged, while toxicologically
unrelated endpoints stay separate.

| # | Class | H-codes | Positives |
|---|---|---|---|
| 01 | Explosive / self-reactive | H200–H205, H240–H242 | 158 |
| 02 | Flammable gas | H220, H221, H230, H231 | 76 |
| 03 | Gas under pressure | H280, H281 | 147 |
| 04 | Flammable liquid | H224–H227 | 2,320 |
| 05 | Flammable solid | H228 | 220 |
| 06 | Spontaneous combustion | H250–H252, H260, H261 | 211 |
| 07 | Oxidizing | H270–H272 | 244 |
| 08 | Corrosive to metals | H290 | 249 |
| 09 | Acute toxicity (oral) | H300–H303 | 6,417 |
| 10 | Acute toxicity (dermal) | H310–H313 | 2,514 |
| 11 | Acute toxicity (inhalation) | H330–H333 | 3,320 |
| 12 | Aspiration hazard | H304, H305 | 456 |
| 13 | Skin corrosion / irritation | H314–H316 | 8,540 |
| 14 | Eye damage / irritation | H318–H320 | 9,220 |
| 15 | Sensitization | H317, H334 | 2,772 |
| 16 | Mutagenicity | H340, H341 | 864 |
| 17 | Carcinogenicity | H350, H351 | 1,350 |
| 18 | Reproductive toxicity | H360–H362 | 1,877 |
| 19 | STOT single exposure (merged) | H335, H336, H370, H371 | 6,675 |
| 20 | STOT repeated exposure | H372, H373 | 2,583 |
| 21 | Aquatic hazard | H400–H402, H410–H413 | 5,480 |
| 22 | Ozone layer hazard | H420 | 27 |

Class prevalence spans nearly three orders of magnitude — a 341-fold imbalance between eye
damage/irritation (65.2%) and ozone layer hazard (0.19%) — which is what the focal loss,
inverse-frequency weighted sampler and targeted conformer augmentation in `run_stage2.py` are
there to handle.

## License

CC BY 4.0 — see `../LICENSE-DATA`.
