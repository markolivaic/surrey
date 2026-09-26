# Hosting the page and its data

The page is static, and the octrees it streams are too large for the code repository, so the two are
hosted apart: the octrees in a Hugging Face dataset, the built page wherever static files are served,
told where the octrees are by `?data=`. `scripts/investigations/hosting_check.py` measured, in a real
browser, that a page can read byte ranges from Hugging Face across origins
(`data/artifacts/hosting_check.json`).

Every step below is run by hand, in Git Bash from the repository root. Accounts, tokens and settings are
the owner's steps. The token is pasted at the login prompt and is never written to a file in the
repository or passed on a command line.

Checked against the command line tool installed in this project's environment: `hf` from
huggingface_hub 0.35.3 (`.venv/Scripts/hf.exe`). Older guides say `huggingface-cli`; that is not the
command here. The dataset name below follows the page's own package name, `surrey-viewer`; any other
works if every command uses it.

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
.venv/Scripts/hf.exe repo create markolivaic/surrey-viewer --repo-type dataset
```

3. Upload the octrees, one folder per area, without the converter's temporary files:

```bash
.venv/Scripts/hf.exe upload markolivaic/surrey-viewer data/viewer . --repo-type dataset --exclude "*/chunks/*" "*/log.txt" --commit-message "Viewer octrees"
```

4. Upload the dataset card as the repository's README, and the licence it points to:

```bash
.venv/Scripts/hf.exe upload markolivaic/surrey-viewer docs/hosting/dataset_card.md README.md --repo-type dataset --commit-message "Dataset card"
.venv/Scripts/hf.exe upload markolivaic/surrey-viewer LICENSE_MODEL.md LICENSE_MODEL.md --repo-type dataset --commit-message "Licence for predictions"
```

5. Check one file answers a byte range, as the page will ask:

```bash
curl -s -L -r 0-15 -o /dev/null -w "%{http_code} %{size_download}\n" https://huggingface.co/datasets/markolivaic/surrey-viewer/resolve/main/sunset/octree.bin
```

It prints `206 16`. After a retrain the octrees change; step 3 again replaces them in a new commit.

## The page, to GitHub Pages

The build uses relative paths (`base: "./"` in `web/vite.config.js`), so `web/dist` works from any
folder. GitHub Pages serves a private repository only on a paid plan, so this step waits until the
repository is public or the plan allows it.

1. Build the page from the commit being published:

```bash
cd web && npm ci && npm run build && cd ..
```

2. Put the build on a `gh-pages` branch of its own, in a separate folder, and push it:

```bash
git worktree add --orphan -b gh-pages ../surrey-pages
cp -r web/dist/. ../surrey-pages/
touch ../surrey-pages/.nojekyll
git -C ../surrey-pages add -A
git -C ../surrey-pages commit -m "Page build"
git -C ../surrey-pages push origin gh-pages
git worktree remove ../surrey-pages
```

3. In the repository's Settings, Pages: source "Deploy from a branch", branch `gh-pages`, folder `/`.

4. The page, with its data:

```
https://markolivaic.github.io/surrey/?data=https://huggingface.co/datasets/markolivaic/surrey-viewer/resolve/main
```

Without `?data=` the page looks for `data/` next to itself, which only `scripts/pipeline/serve_page.py`
provides; the FIXTURE octree is served by it too and is not part of the build.

To publish a new build later, repeat step 1, then step 2 with its first line replaced by these two, so
the new build replaces the old one on the existing branch:

```bash
git worktree add ../surrey-pages gh-pages
git -C ../surrey-pages rm -rq .
```
