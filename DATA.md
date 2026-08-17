# Data Setup

The raw data files are not committed to Git because they are too large for a normal GitHub repository.

## Expected Local Files

Place these files in the project root before running the pipelines:

| File | Current local size | Used by |
| --- | ---: | --- |
| `EXTR_Parcel_processed.csv` | 215.8 MB | `pipeline_v2.py`, `pipeline_v3_fast.py`, `pipeline_v3_enhanced.py`, `analyze_fixed.py` |
| `EXTR_ResBldg.csv` | 154.4 MB | `pipeline_v2.py`, `pipeline_v3_fast.py`, `pipeline_v3_enhanced.py` |
| `EXTR_RPSale.csv` | 646.2 MB | `pipeline_v2.py`, `pipeline_v3_fast.py`, `pipeline_v3_enhanced.py` |
| `EXTR_Parcel.csv` | 247.5 MB | `analyze.py`, `compare_columns.py` |

The included scripts currently read these files from the project root.

## GitHub Strategy

For a clean public repository, keep the CSV extracts out of Git and document how to obtain them. The `.gitignore` file excludes `EXTR_*.csv` so a normal `git add .` will not stage the large data.

If you need to publish data with the project, use one of these options:

- Add a small sample CSV that is safe to commit, such as `sample_properties.csv`.
- Upload the full extracts to a release, cloud storage, or another data host.
- Use Git LFS if you specifically want Git-managed large files.
