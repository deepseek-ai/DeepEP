#!/usr/bin/env bash
# Usage:
#    # Lint Python files that differ from origin/main.
#    bash format.sh
#
#    # Lint all Python files.
#    bash format.sh --all
#
# You are encouraged to run this locally before pushing changes for review.

# Cause the script to exit if a single command fails
set -eo pipefail

# If Ruff is not installed, install according to the requirements
if ! ruff --version &>/dev/null; then
    pip install -r requirements-lint.txt
fi

echo 'ruff: Check Start'
# Lint specified files
lint() {
    ruff check "$@"
}

# Lint Python files that differ from the main branch
lint_changed() {
    # Skip Ruff when there are no added, copied, or modified Python files.
    if git show-ref --verify --quiet refs/remotes/origin/main; then
        BASE_BRANCH="origin/main"
    else
        BASE_BRANCH="main"
    fi

    MERGEBASE="$(git merge-base $BASE_BRANCH HEAD)"

    if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &>/dev/null; then
        git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs \
             ruff check
    fi
}

# If `--all` is passed, then any further arguments are ignored and all
# Python files are linted.
if [[ "$1" == '--all' ]]; then
    lint .
else
    lint_changed
fi

echo 'ruff: Done'
echo 'All checks passed'
