"""Fail when a Pynblint JSON report contains any repository- or notebook-level lint.

Pynblint always exits 0, so on its own it can report smells but never block a
pull request. This turns its report into a pass/fail gate for CI and `make lint-nb`.

Usage: python .github/scripts/check_pynblint.py <report.json>
"""

import json
import sys


def main(path: str) -> int:
    """Print every lint in the report and return 1 if there was at least one."""
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)

    found = [("repository", lint) for lint in report.get("lints", [])]
    for notebook in report.get("notebook_level_lints", []):
        name = notebook["notebook_metadata"]["notebook_name"]
        found += [(name, lint) for lint in notebook.get("lints", [])]

    for where, lint in found:
        print(f"{where}: {lint['slug']} - {lint['description']}")
    notebooks = report.get("repository_stats", {}).get("number_of_notebooks", 0)
    print(f"Pynblint: {len(found)} lint(s) across {notebooks} notebook(s) and the repository.")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
