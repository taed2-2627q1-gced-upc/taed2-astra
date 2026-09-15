# Commit message instructions

Write commit messages for this repository in the following format.

## Subject line

- Imperative present tense: `add training stage`, never `added` or `adds`
- Lowercase after the first word; no trailing period
- 50 characters or fewer
- Describe the change, not the files touched:
  `fix recall threshold check`, not `update evaluate.py`

## Body

- Separate from the subject with a blank line; wrap at 72 characters
- Include a body whenever the change is not self-evident from the subject
- Explain **why** the change was made and what it affects.
  The diff already shows what changed.
- Omit the body entirely for trivial changes (typos, formatting)

## Rules

- One logical change per commit
- Never mention the AI assistant or the tool that produced the change
- Never add a `Co-Authored-By:` trailer naming an AI, and never add a
  `Generated with ...` footer
- Do not list changed filenames in a bullet list
- Do not invent a scope prefix (`feat:`, `chore:`); this repository does not
  use Conventional Commits

## Example

```
add data validation stage

The training stage assumed the label column was present and binary. A bad
upstream file reached the model instead of failing the pipeline, so the
expectations now run before training.
```
