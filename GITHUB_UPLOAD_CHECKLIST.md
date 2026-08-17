# GitHub Upload Checklist

Use this checklist after creating a new empty GitHub repository.

## 1. Verify Local Prep

```powershell
git status --short
git check-ignore -v EXTR_Parcel.csv EXTR_Parcel_processed.csv EXTR_ResBldg.csv EXTR_RPSale.csv
```

The `git check-ignore` command should show that each large CSV is ignored by `.gitignore`.

## 2. Commit Locally

```powershell
git add .
git status --short
git commit -m "Initial King County ML pipeline"
```

Before committing, confirm the large `EXTR_*.csv` files are not listed as staged files.

## 3. Connect GitHub Remote

Replace the URL with your new repository URL:

```powershell
git remote add origin https://github.com/<your-user>/<your-repo>.git
git branch -M main
```

## 4. Push

```powershell
git push -u origin main
```

## 5. Optional Polish Before Public Release

- Add a license file if this will be public.
- Add screenshots or final metrics after running the pipeline.
- Add a small sample dataset if reviewers should run a quick demo without downloading full extracts.
