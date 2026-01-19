# Multimodal-Toy-task

A minimal **`multimodal toy task`** repository for controlled experiments and fast iteration.
Includes a baseline runner and an alternative entrypoint for **`JTT`**-style runs.

## Repository structure

- **`src/`** — core implementation (task definition, models, training, utilities).
- **`scripts/`** — helper scripts (batch runs, small utilities).
- **`main.py`** — baseline entrypoint.
- **`main_JTT.py`** — JTT-style entrypoint.
- **`run_main.sh`** — example baseline run script.
- **`run_main_JTT.sh`** — example JTT run script.
- **`TODO.md`** — next steps / planned improvements.
- **`observations.md`** — running notes / findings.

## Quickstart

1. **Clone**
   ```bash
   git clone https://github.com/BartekKrzepkowski/multimodal_toytask.git
   cd multimodal_toytask
