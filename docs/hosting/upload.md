# Hosting the page and its data

The page is static, and the octrees it streams are too large for the code repository, so the two are
hosted apart: the octrees in a Hugging Face dataset, the built page on GitHub Pages.
`archive/investigations/hosting_check.py` measured, in a real
browser, that a page can read byte ranges from Hugging Face across origins
(`data/artifacts/hosting_check.json`).

Every step below is run by hand, in Git Bash from the repository root, logged in to your own
Hugging Face and GitHub accounts. A token is pasted at the login prompt and never written to a file
in the repository or passed on a command line.

Checked against the command line tool installed in this project's environment: `hf` from
huggingface_hub 0.35.3 (`.venv/Scripts/hf.exe`). Older guides say `huggingface-cli`; that is not the
command here. The dataset name below follows the page's own package name, `surrey-viewer`; any other
works if every command and `HOSTED_OCTREES` in `web/vite.config.js` use it.

## The octrees, to Hugging Face

0. Check the manifests match the files:

```bash
.venv/Scripts/python scripts/pipeline/check_viewer_manifests.py
```

1. Log in, pasting a write token from huggingface.co/settings/tokens when asked:

```bash
.venv/Scripts/hf.exe auth login
```

2. Create the dataset repository. It has to be public for a page to read it without a token:

```bash
.venv/Scripts/hf.exe repo create Maki101/surrey-viewer --repo-type dataset
```

3. Upload the octrees, one folder per area, without the converter's temporary files:

```bash
.venv/Scripts/hf.exe upload Maki101/surrey-viewer data/viewer . --repo-type dataset --exclude "*/chunks/*" "*/log.txt" --commit-message "Viewer octrees"
```

4. Upload the dataset card as the repository's README, and the licence it points to:

```bash
.venv/Scripts/hf.exe upload Maki101/surrey-viewer docs/hosting/dataset_card.md README.md --repo-type dataset --commit-message "Dataset card"
.venv/Scripts/hf.exe upload Maki101/surrey-viewer LICENSE_MODEL.md LICENSE_MODEL.md --repo-type dataset --commit-message "Licence for predictions"
```

5. Check one file answers a byte range, as the page will ask:

```bash
curl -s -L -r 0-15 -o /dev/null -w "%{http_code} %{size_download}\n" https://huggingface.co/datasets/Maki101/surrey-viewer/resolve/main/sunset/octree.bin
```

It prints `206 16`. After a retrain the octrees change; step 3 again replaces them in a new commit.

## The weights, to Hugging Face

A training checkpoint needs unpickling to read, which can run code, so what is published is the
weights alone, written by `export_weights.py` and read with `weights_only=True`.

1. Write the weights of the four models the README reports:

```bash
.venv/Scripts/python scripts/pipeline/export_weights.py data/checkpoints/m0_seedA.pt data/checkpoints/m0_seedB.pt data/checkpoints/m0_seedC.pt data/checkpoints/m0_big_seedA.pt
```

2. Create the model repository, logged in as in step 1 above:

```bash
.venv/Scripts/hf.exe repo create Maki101/surrey --repo-type model
```

3. Upload the weights, the model card as the repository's README, and the licence:

```bash
.venv/Scripts/hf.exe upload Maki101/surrey data/weights . --repo-type model --commit-message "Weights"
.venv/Scripts/hf.exe upload Maki101/surrey docs/hosting/model_card.md README.md --repo-type model --commit-message "Model card"
.venv/Scripts/hf.exe upload Maki101/surrey LICENSE_MODEL.md LICENSE_MODEL.md --repo-type model --commit-message "Licence for the weights"
```

4. Check one file reads back as weights only:

```bash
.venv/Scripts/hf.exe download Maki101/surrey m0_seedA.pt --local-dir data/weights_check
.venv/Scripts/python -c "from surrey.inference import load_checkpoint; load_checkpoint('data/weights_check/m0_seedA.pt', 'cpu'); print('reads with weights_only=True')"
```

## The page, to GitHub Pages

`.github/workflows/pages.yml` builds the page with `npm run build:pages` and deploys it on every push to
`main`. That build reads the octrees from the Hugging Face dataset above without being told, and the
build uses relative paths (`base: "./"` in `web/vite.config.js`), so it works from any folder. GitHub
Pages serves a private repository only on a paid plan, so this step waits until the repository is public.

1. In the repository's Settings, Pages, set the source to "GitHub Actions".

2. Run the workflow, or push to `main`:

```bash
gh workflow run pages.yml
```

3. Open https://markolivaic.github.io/surrey/ in a private browser window, logged in to nothing. The
   points should draw, and the network panel should show `octree.bin` answering 206 from Hugging Face.

Elsewhere, `?data=<url>` points the page at other octrees. Without the octrees it says so and shows the
stills. Locally, `scripts/pipeline/serve_page.py` serves `npm run build` with the octrees in `data/viewer`.
