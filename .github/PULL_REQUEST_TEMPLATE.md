## What does this PR do?

<!-- One or two sentences. -->

Task: Closes #<!-- issue number of the task; the GitHub app posts it to Slack and closes it on merge -->

## Checklist

- [ ] Branch follows GitHub Flow (`feature/...`, `fix/...`) and targets `main`
- [ ] PR title is a Conventional Commit (`feat: ...`, `fix: ...`); it becomes the merge commit
- [ ] `make qa` passes locally (ruff, pylint, pytest + coverage, Pynblint)
- [ ] No data, models or credentials committed to Git (use DVC)
- [ ] If the pipeline changed: `params.yaml` / `dvc.yaml` updated, `dvc repro` run, `dvc.lock` committed and `dvc push` done
- [ ] If the model changed: `make test` passes **with** `models/model.pkl` present (release gates), model card numbers updated, `make promote` run after merge
- [ ] Docs updated if behaviour changed
