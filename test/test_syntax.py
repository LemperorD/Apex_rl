#!/usr/bin/env python
"""Syntax check script for ApexRL."""

import sys
from pathlib import Path


def check_syntax():
    """Check all Python files for syntax errors."""
    src_dir = Path("src/apexrl")
    errors = []

    if not src_dir.exists():
        print(f"Error: {src_dir} not found")
        return 1

    for py_file in src_dir.rglob("*.py"):
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                compile(f.read(), py_file, "exec")
            print(f"OK: {py_file}")
        except SyntaxError as e:
            errors.append(f"Syntax error in {py_file}: {e}")
            print(f"ERROR: {py_file}: {e}")

    if errors:
        print("\n" + "=" * 50)
        print(f"Found {len(errors)} error(s)")
        for err in errors:
            print(err)
        return 1

    print("\nAll files passed syntax check!")
    return 0


if __name__ == "__main__":
    sys.exit(check_syntax())
