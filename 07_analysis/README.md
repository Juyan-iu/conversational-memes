# Label Audit Workbook

Run this folder from the target repository:

```bash
cd /home/exouser/git/conversational-memes/07_analysis/label_audit
python -m pip install -r requirements.txt

python make_label_audit_workbook.py \
  --input ../../03_filter_and_label/labeled_final/labeled_memes.jsonl \
  --image-root ../../03_filter_and_label/labeled_final \
  --output ./meme_label_audit_100.xlsx \
  --sample-size 100 \
  --seed 42 \
  --stratify-by-month
```

If the image files are not already saved under `labeled_final/images/meme_reply/...`, add `--download-missing`. That uses the image URLs stored in the records and saves copies under `downloaded_audit_images/`.

The workbook contains:

- `Review_100`: one row per sampled meme, with context, existing labels, image thumbnail, and audit columns.
- `Codebook`: compact scoring rules embedded in the workbook.
- `Label_Definitions`: discourse and stance label definitions.
