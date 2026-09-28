# Team demo script: the RYA baseline (10 minutes)

**Presenter:** Shlok · **Audience:** SIN-WEI, Shuyun, Pratik, Murat · **Goal:** everyone leaves able to run the baseline and knows what it means for their workstream.

## Before the meeting (15 min prep)

- Open `scrp-toolkit` in VS Code. Run `walkthrough.ipynb` fully (*Restart → Run All*, about 6 min) and **save it with outputs**, so you only scroll during the demo and nothing runs live.
- Have `docs/findings.md` open in a second tab.
- Open an Anaconda Prompt in the repo with `conda activate scrp` ready.
- Have `scrp-toolkit.bundle` ready to share.

---

## 0:00–1:00 · Why we did this

**[Screen: README.md, top]**

> "Before anyone improves the statistical core, we need to agree on exactly what RYA's core *is*. Their code in Box only ran on their own laptops, with hard-coded Mac paths, and Murat's notebook was a great start but differed from RYA's code in a few places.
>
> So I rebuilt the pipeline as one tested package and checked every part of it against RYA's own files and code. The result is a **baseline**: one command that proves your copy behaves exactly like RYA's. Every experiment we do from now on is measured against it."

**[Action: in the Anaconda Prompt, start it now so it finishes while you talk]**
```
python -m scrp_toolkit.cli baseline
```

## 1:00–2:30 · What's in the repo

**[Screen: README → Repository layout]**

> "One git repo, five personal branches plus `main`:
> - `scrp_toolkit/`: the package. Each file is one piece of RYA's pipeline: features, the model, scoring, data preparation.
> - `data/`: the customer's Box files in Box's own folder structure. All 9 inputs are byte-identical to Box; we check their fingerprints.
> - `walkthrough.ipynb`: nine steps, each comparing one file against RYA's version.
> - `docs/findings.md`: everything we found, plus open questions for RYA.
>
> There's one **SETTINGS** cell at the top of the notebook. Nothing needs editing if you use the repo as-is."

## 2:30–3:15 · The baseline result

**[Screen: the Anaconda Prompt, which should be finished by now]**

> "Seven sections, 359 checks, all passing: the files match Box, the model matches, and the outputs for a real RY25 school match.
>
> If you change anything in the pipeline, run this, and `baseline_report.csv` shows exactly which numbers moved. That report goes with your pull request."

## 3:15–7:00 · How we know it's identical to RYA (walkthrough highlights)

**[Screen: walkthrough.ipynb, scroll to each step's output. Spend about 30 s on each.]**

- **Step 2, the files.** "9 out of 9 fingerprints match what Box reports. One changed byte would fail."
- **Step 4, the features.** "The model never sees raw answers. It sees 40 numbers built from them. We ran RYA's own feature script from Box next to ours: difference 0.0. We also recomputed the features stored in all 63,000 training rows. They match, except one row, which I'll come back to."
- **Step 5, the model.** "Same inputs through RYA's scaling and the shipped model: identical output. For the ry9 example, shape 9.97, rate 20.93. We did fix one tiny precision difference in our code here."
- **Step 6, the data preparation.** "This one's the strongest proof. We ran RYA's actual training-data class, and it reproduces all 84 scaling numbers stored with the model. Ours produces the identical training set.
>   It also showed that Murat's notebook skipped a rate-rescaling step RYA used, which is why the correct row count is 61,155, not 60,810."
- **Step 7, the network.** "We loaded the model's trained weights into our network definition: outputs identical, 1.26 million parameters. The training loss is identical to RYA's too. So Murat can retrain or shrink from exactly RYA's starting point."
- **Step 8, a real school.** "One RY25 school, 656 before and 123 after, 40 items scored. The numbers match the baseline and Murat's run."

**Say this once, clearly:**
> "What the model outputs is **not** an item reliability score. It's the spread of differences you'd expect from students' answer noise alone, for that item and those group sizes. A real change between Pre and Post has to clearly beat it."

## 7:00–8:45 · What we found, and who it affects

**[Screen: docs/findings.md]**

| Finding | Why it matters | Owner |
|---|---|---|
| **6-option items (chs1–6)** can't be scored by EXP002; the old notebook scored them anyway | Drop them from any analysis until a 6-option model exists | Everyone |
| **One corrupt training row** (1 student in a group, but answer shares that add up to 3) sets 9 of the model's scaling limits | Keep it for the baseline; **drop it before any retraining** | Murat |
| Model was trained only on **groups of 5 to about 9,900**, and gives no warning outside that range | Direct rules for the input-safety detector | SIN-WEI |
| **Everyone giving the same answer, or one respondent** produces NaN | Detector must catch these; the backup method must handle them | SIN-WEI, Shuyun |
| Two copies of RYA's reliability matrices disagree in **2 cells** (sun1, ry17) | Matters for any sensitivity analysis on the matrices | Pratik |
| RYA never saved EXP002's train/test split | Our seed-42 error figures are a fixed reference, not a true test score | Murat, Pratik |

> "None of these change the baseline. The baseline is RYA's system exactly as it is, warts included. They're the starting points for our improvements."

## 8:45–9:45 · How we work from here

**[Screen: README → Team git workflow]**

> "Clone with `git clone -b main scrp-toolkit.bundle scrp-toolkit`, set up the `scrp` environment from the Quick start, and run the baseline once to confirm 359/359.
>
> Then three rules:
> 1. **Work on your own branch.** `main` only changes through a reviewed pull request.
> 2. **Never edit `data/` or `known_answers/`.**
> 3. **Attach `baseline_report.csv` to every pull request**, so we all see what your change moved.
>
> Please keep this repo private. The RY25 files are real student survey data."

## 9:45–10:00 · Close

> "By Friday 9 October we join everything up: flagged cases go to the backup method, and real survey data flows through end to end. I'm sending RYA six questions, in `findings.md`, including the portal's p-value code. Questions?"

---

## If asked

- **"Can we recompute the reliability matrices?"** Not from Box. RYA's re-test data isn't there; we've asked for it.
- **"Why not retrain the model?"** We don't need to. We have RYA's trained model and proved we use it exactly. Retraining is Murat's experiment, measured against this baseline.
- **"How long does it take to run?"** About 2 minutes for the baseline, 6 for the full notebook.
- **"Git says 'dubious ownership'."** Run the `git config --global --add safe.directory ...` command git prints.
