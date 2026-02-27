# Repository Setup and Sharing

This repository is set up to share code, configs, and documentation through GitHub while keeping datasets, checkpoints, and experiment outputs local to each machine.

## What should go to GitHub

Commit and push:

- `src/`
- `configs/`
- `README.md`
- `requirements.txt`
- `docs/`
- `scripts/`

Do not commit:

- `data/`
- `runs/`
- `wandb/`
- virtual environments such as `2dFM/` or `.venv/`
- model weights such as `*.pt`, `*.pth`, and `*.ckpt`

## Recommended workflow

Use one private GitHub repository as the source of truth.

On each machine:

1. Clone the repository.
2. Create a local virtual environment.
3. Install dependencies from `requirements.txt`.
4. Keep `data/` and `runs/` local on that machine.
5. Share code changes with `git add`, `git commit`, `git push`, and update other machines with `git pull`.

## Initial GitHub setup

From the repository root:

```bash
git init
git add .gitignore requirements.txt README.md src configs docs scripts
git commit -m "Initial commit"
git branch -M main
git remote add origin git@github.com:<your-github-id>/2dFM.git
git push -u origin main
```

## Local machine setup

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Data and run directories

Your configs already use relative paths:

- dataset root: `data`
- experiment output root: `runs`

That means each machine can keep its own local dataset cache and run outputs without changing the repository layout.

## PACE recommendation

On PACE, keep the cloned repository in home or project storage, but store large mutable outputs under scratch.

Suggested pattern:

- repository clone: `/home/$USER/2dFM` or project storage
- data cache: `/scratch/$USER/2dFM/data`
- experiment outputs: `/scratch/$USER/2dFM/runs`

The scripts in `scripts/` create symlinks so the code still sees `data/` and `runs/` inside the repo.

Basic PACE flow:

```bash
bash scripts/pace_setup.sh
sbatch scripts/pace_job.slurm
```

You can override the default config at submission time:

```bash
sbatch --export=REPO_ROOT=$HOME/2dFM,CONFIG_PATH=configs/mnist_imf_transformer.yaml scripts/pace_job.slurm
```

Check `scripts/pace_job.slurm` before the first submission and adjust `#SBATCH --partition` or loaded modules if your PACE account uses different names.

## SSH authentication

Set up an SSH key on each machine and add the public key to GitHub. This makes `git pull` and `git push` consistent across:

- local computer
- lab computer
- PACE login node

## Safe daily workflow

Before starting work:

```bash
git pull --rebase
```

After editing code:

```bash
git status
git add <files>
git commit -m "Describe the change"
git push
```

Before a commit, sanity check that no data or outputs are staged:

```bash
git status --short
```
