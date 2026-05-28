# CL-bench_context_ge5 Dataset Distribution

## Overview

| Metric | Value |
|--------|-------|
| Total Samples | 884 |
| Total Contexts | 120 |
| Mean Samples / Context | 7.37 |
| Median Samples / Context | 7 |
| Range | 5 ~ 12 |

---

## Samples per Context Distribution

| Samples / Context | Context Count | Percentage |
|:-----------------:|:-------------:|:----------:|
| 5 | 13 | 10.8% |
| 6 | 16 | 13.3% |
| 7 | 62 | 51.7% |
| 8 | 6 | 5.0% |
| 9 | 7 | 5.8% |
| 10 | 4 | 3.3% |
| 11 | 6 | 5.0% |
| 12 | 6 | 5.0% |

---

## Top-Level Category Distribution

| Category | Samples | Percentage |
|----------|--------:|:----------:|
| Procedural Task Execution | 306 | 34.6% |
| Domain Knowledge Reasoning | 294 | 33.3% |
| Rule System Application | 257 | 29.1% |
| Empirical Discovery & Simulation | 27 | 3.1% |
| **Total** | **884** | **100%** |

---

## Subcategory Distribution

### Rule System Application (257 samples)

| Subcategory | Samples | % of Category |
|-------------|--------:|:-------------:|
| Technical Standards | 125 | 48.6% |
| Legal & Regulatory | 59 | 23.0% |
| Game Mechanics | 42 | 16.3% |
| Mathematical Formalism | 19 | 7.4% |
| Programming Syntax | 12 | 4.7% |

### Domain Knowledge Reasoning (294 samples)

| Subcategory | Samples | % of Category |
|-------------|--------:|:-------------:|
| Management | 92 | 31.3% |
| Healthcare | 46 | 15.6% |
| Humanities | 45 | 15.3% |
| Legal Advisory | 41 | 13.9% |
| Finance | 31 | 10.5% |
| Science | 26 | 8.8% |
| Lifestyle | 13 | 4.4% |

### Procedural Task Execution (306 samples)

| Subcategory | Samples | % of Category |
|-------------|--------:|:-------------:|
| Workflow Orchestration | 198 | 64.7% |
| Operational Procedures | 108 | 35.3% |

### Empirical Discovery & Simulation (27 samples)

| Subcategory | Samples | % of Category |
|-------------|--------:|:-------------:|
| Observational Data | 13 | 48.1% |
| Experimental Data | 9 | 33.3% |
| Simulation Environment | 5 | 18.5% |

---

## Notes

- This is the filtered subset (≥5 samples per context) of the full CL-bench dataset (1,899 samples total).
- The filter retains 120 out of 500 contexts (24%) and 884 out of 1,899 samples (46.6%).
- **Empirical Discovery & Simulation** is heavily underrepresented after filtering: its share drops from 10.5% (full dataset) to 3.1% (filtered), suggesting its contexts tend to have fewer samples.
- 7-sample contexts dominate at 51.7%, making 7 the strong mode.
