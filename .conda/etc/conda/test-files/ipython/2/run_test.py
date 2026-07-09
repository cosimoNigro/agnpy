from __future__ import annotations

import subprocess
import platform
from pathlib import Path
import sys

WIN = platform.system() == "Windows"
LINUX = platform.system() == "Linux"

COV_THRESHOLD = 57
PYTEST_SKIPS = [
    "decorator_skip",
    "pprint_heap_allocated",
    #: bad interactions with "clean" enviroment in rattler-build 0.62
    #: https://github.com/conda-forge/ipython-feedstock/pull/263
    "test_dirops",
]
UNLINK: list[str] = []

if LINUX:
    PYTEST_SKIPS += ["system_interrupt"]

if WIN:
    PYTEST_SKIPS += [
        #: bad interactions with "clean" enviroment in rattler-build 0.62
        #: https://github.com/conda-forge/ipython-feedstock/pull/263
        "(test_oinspect and test_info)",
    ]

PYTEST_ARGS = ["pytest", "-vv", "--color=yes", "--tb=long", "tests"]

if len(PYTEST_SKIPS) == 1:
    PYTEST_ARGS += ["-k", f"not {PYTEST_SKIPS[0]}"]
elif PYTEST_SKIPS:
    PYTEST_ARGS += ["-k", f"""not ({" or ".join(PYTEST_SKIPS)})"""]

COV = [sys.executable, "-m", "coverage"]
COV_RUN = [*COV, "run", "--source", "IPython", "--branch", "--append", "-m"]
COV_REPORT = [
    *COV,
    "report",
    "--show-missing",
    "--skip-covered",
    f"--fail-under={COV_THRESHOLD}",
]
COV_RUNS = [
    [*COV_RUN, "IPython", "--version"],
    [*COV_RUN, "IPython", "--help"],
    [*COV_RUN, *PYTEST_ARGS],
]


def do(args: list[str]) -> int:
    print(">>>", *args, flush=True)
    return subprocess.call(args)


def main() -> int:
    print("Testing on Windows?      ", WIN)
    print("Testing on Linux?        ", LINUX)
    for stem in UNLINK:
        path = (Path("tests") / stem).unlink()
        print("... removing", path)

    return any(map(do, [*COV_RUNS, COV_REPORT]))


if __name__ == "__main__":
    sys.exit(main())
