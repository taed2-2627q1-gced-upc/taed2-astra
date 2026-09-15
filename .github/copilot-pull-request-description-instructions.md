# Pull request description instructions

Match the structure of `.github/PULL_REQUEST_TEMPLATE.md`.

- Open with one or two sentences saying what the pull request does and why
- Keep the checklist from the template, unticked
- Do not summarise the diff file by file
- Mention any change to `params.yaml` or `dvc.yaml`, since those invalidate
  pipeline stages for everyone
- Never mention the AI assistant or the tool that produced the change, and
  never add a `Generated with ...` footer
