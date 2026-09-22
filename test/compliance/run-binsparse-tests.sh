#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Finch Developers
# SPDX-License-Identifier: MIT
#
# Run the Binsparse compliance test suite against Finch.jl using a persistent Julia server process.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
build_dir="${BINSPARSE_BUILD_DIR:-${repo_root}/build-compliance}"
tests_dir="${BINSPARSE_TESTS_DIR:-${build_dir}/binsparse-tests}"
tests_ref="${BINSPARSE_TESTS_REF:-main}"

# check prerequisites
for command in git pixi julia python3; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "error: required command '${command}' was not found" >&2
    exit 1
  fi
done

# clone / update binsparse-tests
mkdir -p "${build_dir}"

if [[ -d "${tests_dir}/.git" ]]; then
  git -C "${tests_dir}" fetch --depth 1 origin "${tests_ref}"
  git -C "${tests_dir}" checkout --detach FETCH_HEAD
elif [[ -e "${tests_dir}" ]]; then
  echo "error: ${tests_dir} exists but is not a Git checkout" >&2
  exit 1
else
  git clone --depth 1 https://github.com/Binsparse/binsparse-tests.git \
    "${tests_dir}"
  if [[ "${tests_ref}" != "main" ]]; then
    git -C "${tests_dir}" fetch --depth 1 origin "${tests_ref}"
    git -C "${tests_dir}" checkout --detach FETCH_HEAD
  fi
fi

# install binsparse-tests Python environment
(
  cd "${tests_dir}"
  pixi install -e test-hdf5
)

# instantiate the Julia test project so the server works
julia --project="${repo_root}/test" -e '
  using Pkg
  Pkg.develop(PackageSpec(; path=ARGS[1]))
  Pkg.instantiate()
' "${repo_root}"

# Set up FIFO for persistent Finch server
FIFO_DIR=$(mktemp -d)
FIFO="${FIFO_DIR}/finch_fifo"
READY_FILE="${FIFO_DIR}/ready"
mkfifo "$FIFO"

export FINCH_SERVER_FIFO="$FIFO"

cleanup() {
  if [[ -p "$FIFO" ]]; then
    echo '{"cmd": "shutdown"}' > "$FIFO" 2>/dev/null || true
  fi
  rm -rf "$FIFO_DIR"
}
trap cleanup EXIT

echo "Starting persistent Finch server..."
julia --project="${repo_root}/test" "${script_dir}/finch_server.jl" "$FIFO" "$READY_FILE" &
SERVER_PID=$!

# Wait for server to signal ready
while [[ ! -f "$READY_FILE" ]]; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "error: Finch server failed to start" >&2
    exit 1
  fi
  sleep 0.2
done
echo "Finch server ready!"

# point the harness at the Finch CLI wrappers
export BINSPARSE_TO_NPY="${script_dir}/binsparse_to_npy"
export BINSPARSE_TO_BINSPARSE="${script_dir}/binsparse_to_binsparse"

if [[ -z "${NPY_TO_BINSPARSE:-}" ]]; then
  export NPY_TO_BINSPARSE="${script_dir}/npy_to_binsparse"
fi

# run pytest
(
  cd "${tests_dir}"
  if [[ -f "${script_dir}/skips.txt" ]]; then
    pixi run -e test-hdf5 pytest -m hdf5 \
      --skips-file "${script_dir}/skips.txt" \
      binsparse_tests/ "$@"
  else
    pixi run -e test-hdf5 pytest -m hdf5 \
      binsparse_tests/ "$@"
  fi
)
