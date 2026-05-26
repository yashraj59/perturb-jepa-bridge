# Amendment 001: Scratch 100-Experiment Run

## 1. Direct instruction
User instructed: delete prior autoresearch artifacts, start from scratch, run 100 experiments again, and operate autonomously.

## 2. Active baseline
`PerturbJEPABridge` at commit `0436cccd99a733fec6b57dfdb1fe1e6f4982922f`.

## 3. Fixed Step 0 reference numbers
Recomputed in this clean run under `step0_baselines/`.

## 4. What changed
Hard experiment cap is exactly 100. Collapse, saturation, and family-exhaustion stops are disabled for this user-directed scratch run. Candidate-level collapse labels remain recorded.

## 5. Family status
Reset to zero before experiment 001.

## 6. Updated gates
Still stop for infrastructure failure, eval-broken negative controls, real-data violation, GPU-safety violation, or completion of 100 experiments.

## 7. Updated compute budget
CPU-only micro screens, no real data, no larger datasets, no larger image size, no model-width escalation.

## 8. Do-not-run list
No real datasets, no public AnnData/image manifests, no pretrained biological or image backbones, no GPU-heavy jobs.

## 9. Immediate next experiment
Experiment 001 starts after Step 0 recomputation.

## 10. Stop/user-review trigger
Stop after experiment 100 and write final report.
