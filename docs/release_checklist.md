# Release Checklist

- [ ] Add the final GitHub repository URL to `CITATION.cff` after the repo is created.
- [ ] Confirm final paper title, venue, and author list.
- [ ] Decide whether checkpoint files will be pushed with Git LFS or uploaded as Release assets.
- [ ] Verify that AudioSet audio paths in `configs/*.yaml` match the target machine.
- [ ] Run `bash scripts/eval_camf.sh 0` on a machine with prepared AudioSet-Strong audio.
- [ ] Tag the first public release as `v0.1.0`.
