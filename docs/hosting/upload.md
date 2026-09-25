# Uploading the viewer data

Checked against the command line tool installed in this project's environment:
`hf` from huggingface_hub 0.35.3 (`.venv/Scripts/hf.exe`). Older guides say `huggingface-cli`;
that is not the command here.

Every step below is run by hand. The token is pasted at the login prompt and is never written to a
file in the repository or passed on a command line.

Replace `NAME` with the dataset name you choose. The project name is still provisional, so
nothing here assumes one.

1. Log in, pasting a write token from huggingface.co/settings/tokens when asked:

```bash
.venv/Scripts/hf.exe auth login
```

2. Create the dataset repository:

```bash
.venv/Scripts/hf.exe repo create markolivaic/NAME --repo-type dataset
```

3. Upload the octrees, one folder per area, without the converter's temporary files:

```bash
.venv/Scripts/hf.exe upload markolivaic/NAME data/viewer . --repo-type dataset --exclude "*/chunks/*" "*/log.txt" --commit-message "Viewer octrees"
```

4. Upload the dataset card as the repository's README:

```bash
.venv/Scripts/hf.exe upload markolivaic/NAME docs/hosting/dataset_card.md README.md --repo-type dataset --commit-message "Dataset card"
```

5. Upload the licence for the predictions, which the card's `license_link` points to:

```bash
.venv/Scripts/hf.exe upload markolivaic/NAME LICENSE_MODEL.md LICENSE_MODEL.md --repo-type dataset --commit-message "Licence for predictions"
```

After a retrain the octrees change; step 3 again replaces them in a new commit. The repository's
history then holds old versions, which Hugging Face lets the owner squash; the code repository is
never touched by any of this.

Before step 3, check the manifests match the files:

```bash
.venv/Scripts/python scripts/pipeline/check_viewer_manifests.py
```
