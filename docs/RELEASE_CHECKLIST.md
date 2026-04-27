# Release Checklist

Use this before publishing a public checkpoint, dataset, or demo.

## Repository

- [ ] README reflects the current actual model state.
- [ ] No local paths in public docs.
- [ ] No checkpoints committed to GitHub.
- [ ] No reference voices committed to GitHub.
- [ ] No generated datasets committed to GitHub.
- [ ] License and provenance notes are accurate.
- [ ] Install or reproduction instructions have been tested in a clean clone.

## Model

- [ ] Model card exists.
- [ ] Parameter count is stated honestly.
- [ ] Pipeline size includes enhancer if enhancer is required.
- [ ] Inference hardware requirements are listed.
- [ ] Known limitations are documented.
- [ ] Safety and voice-cloning caveats are included.

## Samples

- [ ] Samples include raw and enhanced outputs where relevant.
- [ ] Samples include at least one long prompt.
- [ ] Samples include at least one difficult/failure case.
- [ ] No private reference voice audio is bundled accidentally.
- [ ] Demo claims match the actual files.

## Dataset

- [ ] Dataset card exists.
- [ ] Source/provenance is described.
- [ ] License is not incorrectly inherited from repo code.
- [ ] Generated audio is clearly labeled synthetic.
- [ ] Any upstream model terms are acknowledged.

## Final Sanity

- [ ] Fresh clone works.
- [ ] README links resolve.
- [ ] Sample links work.
- [ ] The stated roadmap matches the actual project direction.
