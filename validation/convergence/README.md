# Convergence traces

The current release writes one numerical CSV trace and one PNG convergence plot per stochastic run. The archived manuscript validation predates that export feature, so its original convergence artifacts are the PNG plots preserved under `../results/<ALGORITHM>/`.

Run `../regenerate_validation.sh` from the repository root after building `segm` to populate this directory with current-release numerical traces. Regenerated stochastic traces are not expected to be identical to the archived plots unless deterministic seeding is supported and explicitly selected.
