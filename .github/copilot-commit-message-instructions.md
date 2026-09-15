# Commit message instructions

This repository uses [Conventional Commits](https://www.conventionalcommits.org/).

## Subject line

Format: `<type>: <description>`

- 50 characters or fewer, including the type
- Imperative present tense: `add training stage`, never `added` or `adds`
- Lowercase description; no trailing period
- Describe the change, not the files touched:
  `fix: correct recall threshold check`, not `fix: update evaluate.py`

### Types

| Type | Use for |
|------|---------|
| `feat` | A new capability: a pipeline stage, an endpoint, a metric |
| `fix` | Corrects wrong behaviour |
| `docs` | README, AGENTS.md, model card, dataset card, code comments |
| `refactor` | Restructures code without changing behaviour |
| `test` | Adds or changes tests only |
| `chore` | Dependencies, config, tooling, project scaffolding |
| `ci` | GitHub Actions workflows |
| `data` | DVC-tracked data or pipeline definition changes |

Add a scope only when it genuinely narrows the change: `feat(api): ...`,
`fix(train): ...`. Omit it rather than inventing one.

## Body

- Separate from the subject with a blank line; wrap at 72 characters
- Include a body whenever the change is not self-evident from the subject
- Explain **why** the change was made and what it affects.
  The diff already shows what changed.
- Write the body in the present tense, describing the new state
- Omit the body entirely for trivial changes (typos, formatting)
- Note any change to `params.yaml` or `dvc.yaml`, since those invalidate
  pipeline stages for every teammate

## Rules

- One logical change per commit
- Never mention the AI assistant or the tool that produced the change
- Never add a `Co-Authored-By:` trailer naming an AI, and never add a
  `Generated with ...` footer
- Do not list changed filenames in a bullet list

## Examples

```
feat: add data validation stage

The training stage assumed the label column was present and binary. A bad
upstream file reached the model instead of failing the pipeline, so the
expectations now run before training.

Adds a stage to dvc.yaml, so teammates need `dvc repro` after pulling.
```

```
docs: clarify DagsHub token setup

DagsHub issues one token rather than a key/secret pair, which was not
obvious from the DVC commands. The template now states where the same
token goes for both MLflow and DVC.
```
