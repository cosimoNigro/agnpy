import json
import os
import platform
import sys
import pytest
from pathlib import Path

test_skips = []

py_major = sys.version_info[0]
machine = platform.machine().lower()
system = platform.system().lower()

is_win = system == "windows"
is_linux = system == "linux"

prefix = Path(os.environ["PREFIX"])


def check_kernel() -> int:
    print(f"              Machine: {machine}")
    print(f"               System: {system}")

    specfile = prefix / f"share/jupyter/kernels/python{py_major}/kernel.json"

    print(f"Checking Kernelspec at:     {specfile}...")

    raw_spec = specfile.read_text(encoding="utf-8")

    print(raw_spec)

    spec = json.loads(raw_spec)

    print("""Checking python executable: {spec["argv"][0]}""")

    if spec["argv"][0].replace("\\", "/") != sys.executable.replace("\\", "/"):
        print(
            "The kernelspec seems to have the wrong prefix. \n"
            f"""    Specfile: {spec["argv"][0]}"""
            "\n"
            f"""    Expected: {sys.executable}"""
        )
        return 1

    return 0


def build_pytest_args() -> list[str]:
    pytest_args = [
        "--color=yes",
        "--tb=long",
        "-vv",
        "--timeout=300",
        "--asyncio-mode=auto",
    ]

    if is_win:
        # test_pickleutil fails on windows, `pickleutil` deprecated anyway,
        test_skips.append("pickleutil")

    if is_linux:
        # getting x11 from yum isn't worth it
        test_skips.append("matplotlib_gui")

    if len(test_skips) == 1:
        pytest_args += ["-k", f"not {test_skips[0]}"]
    elif test_skips:
        pytest_args += ["-k", f"""not ({" or ".join(test_skips)})"""]

    return pytest_args


def run_pytest():
    pytest_args = build_pytest_args()

    print("Final pytest args:", "\t".join(pytest_args), flush=True)
    return int(pytest.main(pytest_args))


def main() -> int:
    return check_kernel() or run_pytest()


if __name__ == "__main__":
    sys.exit(main())
