#!/usr/bin/env bash
# Usage:
#    # Do work and commit your work.

#    # Format files that differ from origin/main.
#    bash format.sh

#    # Commit changed files with message 'Run yapf and ruff'
#
#
# YAPF + Clang formatter (if installed). This script formats all changed files from the last mergebase.
# You are encouraged to run this locally before pushing changes for review.

# Cause the script to exit if a single command fails
set -eo pipefail

# If yapf/ruff is not installed, install according to the requirements
if ! (yapf --version &>/dev/null && ruff --version &>/dev/null); then
    pip install -r requirements-lint.txt
fi

SED_CMD="sed -i"
if [[ "$OSTYPE" == "darwin"* ]]; then
    SED_CMD="sed -i ''"
fi

YAPF_VERSION=$(yapf --version | awk '{print $2}')
RUFF_VERSION=$(ruff --version | awk '{print $2}')

echo 'yapf: Check Start'

YAPF_FLAGS=(
    '--recursive'
    '--parallel'
)

YAPF_EXCLUDES=(
    '--exclude' 'build/**'
)

# Format specified files
format() {
    yapf --in-place "${YAPF_FLAGS[@]}" "$@"
}

# Format all files
format_all() {
    yapf --in-place "${YAPF_FLAGS[@]}" "${YAPF_EXCLUDES[@]}" .
}

# Format files that differ from main branch
format_changed() {
    # The `if` guard ensures that the list of filenames is not empty, which
    # could cause ruff to receive 0 positional arguments, making it hang
    # waiting for STDIN.
    #
    # `diff-filter=ACM` and $MERGEBASE is to ensure we only lint files that
    # exist on both branches.
    if git show-ref --verify --quiet refs/remotes/origin/main; then
        BASE_BRANCH="origin/main"
    else
        BASE_BRANCH="main"
    fi

    MERGEBASE="$(git merge-base $BASE_BRANCH HEAD)"

    if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &>/dev/null; then
        git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs -P 5 \
             yapf --in-place "${YAPF_EXCLUDES[@]}" "${YAPF_FLAGS[@]}"
    fi
}

# If `--all` is passed, then any further arguments are ignored and the
# entire python directory is formatted.
if [[ "$1" == '--all' ]]; then
   format_all
else
   # Format only the files that changed in last commit.
   format_changed
fi
echo 'yapf: Done'

echo 'ruff: Check Start'
# Lint specified files
lint() {
    ruff check "$@"
}

# Lint files that differ from main branch. Ignores dirs that are not slated
# for autolint yet.
lint_changed() {
    # The `if` guard ensures that the list of filenames is not empty, which
    # could cause ruff to receive 0 positional arguments, making it hang
    # waiting for STDIN.
    #
    # `diff-filter=ACM` and $MERGEBASE is to ensure we only lint files that
    # exist on both branches.
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

# Run Ruff
# If `--all` is passed, then any further arguments are ignored and the
# entire python directory is linted.
if [[ "$1" == '--all' ]]; then
   lint . 
else
   # Check only the files that changed in last commit.
   lint_changed
fi

echo 'ruff: Done'

# # params: tool name, tool version, required version
tool_version_check() {
    if [[ $2 != $3 ]]; then
        echo "Wrong $1 version installed: $3 is required, not $2."
        pip install -r requirements-lint.txt
    fi
}

echo 'clang-format: Check Start'
# If clang-format is available, run it; otherwise, skip
if command -v clang-format &>/dev/null; then
    CLANG_FORMAT_VERSION=$(clang-format --version | awk '{print $3}')
    tool_version_check "clang-format" "$CLANG_FORMAT_VERSION" "$(grep clang-format requirements-lint.txt | cut -d'=' -f3)"

    CLANG_FORMAT_FLAGS=("-i")

    # Format all C/C++ files in the repo, excluding specified directories
    clang_format_all() {
        # Replace "#pragma unroll" by "// #pragma unroll"
        find . -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.h' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' \) \
            -not -path "./build/*" \
            -exec $SED_CMD 's/#pragma unroll/\/\/#pragma unroll/g' {} +
        find . -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.h' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' \) \
            -not -path "./build/*" \
            -exec clang-format -i {} +
        # Replace "// #pragma unroll" by "#pragma unroll"
        find . -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.h' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' \) \
            -not -path "./build/*" \
            -exec $SED_CMD 's/\/\/ *#pragma unroll/#pragma unroll/g' {} +
    }

    # Format changed C/C++ files relative to main
    clang_format_changed() {
        if git show-ref --verify --quiet refs/remotes/origin/main; then
            BASE_BRANCH="origin/main"
        else
            BASE_BRANCH="main"
        fi

        MERGEBASE="$(git merge-base $BASE_BRANCH HEAD)"

        if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.c' '*.cc' '*.cpp' '*.h' '*.hpp' '*.cu' '*.cuh' &>/dev/null; then
            git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.c' '*.cc' '*.cpp' '*.h' '*.hpp' '*.cu' '*.cuh' | xargs $SED_CMD 's/#pragma unroll/\/\/#pragma unroll/g'
            git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.c' '*.cc' '*.cpp' '*.h' '*.hpp' '*.cu' '*.cuh' | xargs clang-format -i
            git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.c' '*.cc' '*.cpp' '*.h' '*.hpp' '*.cu' '*.cuh' | xargs $SED_CMD 's/\/\/ *#pragma unroll/#pragma unroll/g'
        fi
    }

    if [[ "$1" == '--all' ]]; then
       # If --all is given, format all eligible C/C++ files
       clang_format_all
    else
       # Otherwise, format only changed C/C++ files
       clang_format_changed
    fi
else
    echo "clang-format not found. Skipping C/C++ formatting."
fi
echo 'clang-format: Done'

# Check if there are any uncommitted changes after all formatting steps.
# If there are, ask the user to review and stage them.
if ! git diff --quiet &>/dev/null; then
    echo 'Reformatted files. Please review and stage the changes.'
    echo 'Changes not staged for commit:'
    echo
    git --no-pager diff --name-only

    exit 1
fi

echo 'All checks passed'