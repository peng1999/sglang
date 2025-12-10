import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from sglang.srt.utils.common import kill_process_tree


@dataclass
class TestFile:
    name: str
    estimated_time: float = 60


@dataclass
class TestFailureInfo:
    """Information about a test failure."""

    filename: str
    reason: str
    stderr: str = ""
    stdout: str = ""
    error_lines: List[str] = None

    def __post_init__(self):
        if self.error_lines is None:
            self.error_lines = []


def write_failure_summary_to_github(failure_details: List["TestFailureInfo"]) -> None:
    """Write test failure summary to GitHub Step Summary.

    Args:
        failure_details: List of TestFailureInfo objects containing error information
    """
    if not failure_details:
        return

    # Check if we're in GitHub Actions
    github_summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not github_summary_path:
        return

    # Build markdown summary
    summary_lines = []
    summary_lines.append("## ❌ Test Failures\n")
    summary_lines.append(f"**{len(failure_details)} test(s) failed**\n\n")

    for failure in failure_details:
        # Extract just the test filename (not full path)
        test_name = os.path.basename(failure.filename)

        summary_lines.append(f"### {test_name}\n")
        summary_lines.append(f"**Reason:** {failure.reason}\n\n")

        if failure.error_lines:
            # Show key error lines in collapsible section
            summary_lines.append(f"<details>\n")
            summary_lines.append(
                f"<summary>🔍 Key Error Lines (click to expand)</summary>\n\n"
            )
            summary_lines.append("```python\n")
            for line in failure.error_lines[:50]:  # Limit to first 50 lines
                summary_lines.append(line + "\n")
            if len(failure.error_lines) > 50:
                summary_lines.append(
                    f"... ({len(failure.error_lines) - 50} more lines omitted)\n"
                )
            summary_lines.append("```\n\n")
            summary_lines.append(f"</details>\n\n")

        summary_lines.append("---\n\n")

    # Write to GitHub Step Summary
    try:
        with open(github_summary_path, "a") as f:
            f.writelines(summary_lines)
    except Exception as e:
        print(f"Warning: Failed to write to GitHub Step Summary: {e}", flush=True)


def extract_main_error_message(stderr: str, stdout: str) -> str:
    """Extract the main error message from output.

    Returns a concise error message like "TypeError: expected str, bytes..."
    or "AssertionError: a.shape[1] >= 256" instead of just "exit code 1"
    """
    combined_output = stderr + "\n" + stdout
    lines = combined_output.split("\n")

    # Common Python error types to look for (in order of priority)
    error_types = [
        "AssertionError",
        "TypeError",
        "ValueError",
        "FileNotFoundError",
        "RuntimeError",
        "KeyError",
        "AttributeError",
        "ImportError",
        "IndexError",
        "MemoryError",
        "Exception",
    ]

    # Look for the last occurrence of these errors (usually the root cause)
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        for error_type in error_types:
            if line.startswith(error_type + ":") or line.startswith(error_type):
                # Extract the error message (limit to 150 chars)
                error_msg = line[:150]
                if len(line) > 150:
                    error_msg += "..."
                return error_msg

    # Fallback: look for "ERROR:" or "FAILED" messages
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if "ERROR:" in line or "FAILED" in line:
            return line[:150]

    return "exit code non-zero (see error lines below)"


def extract_error_lines(stderr: str, stdout: str, max_lines: int = 50) -> List[str]:
    """Extract key error lines from stderr/stdout.

    Looks for patterns like:
    - Traceback (most recent call last):
    - Exception/Error: messages
    - AssertionError, RuntimeError, etc.
    - Failed/ERROR log lines

    Returns up to max_lines of the most relevant error context.
    """
    error_lines = []
    combined_output = stderr + "\n" + stdout
    lines = combined_output.split("\n")

    # Pattern to identify error-related lines
    error_patterns = [
        r"Traceback \(most recent call last\):",
        r"\w*Error:",  # Catches AssertionError, RuntimeError, ValueError, etc.
        r"\w*Exception:",
        r"FAILED",
        r"ERROR",
        r"Failed to",
        r"raise \w+Error",
    ]

    # Find lines matching error patterns and collect context
    i = 0
    while i < len(lines) and len(error_lines) < max_lines:
        line = lines[i]

        # Check if this line matches an error pattern
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in error_patterns):
            # Found an error - collect this line and context around it

            # If it's a traceback, capture the full traceback
            if "Traceback (most recent call last):" in line:
                error_lines.append(line)
                i += 1
                # Continue collecting lines until we hit the final exception line
                while i < len(lines) and len(error_lines) < max_lines:
                    error_lines.append(lines[i])
                    # Stop after the final exception message
                    if re.match(r"\w+Error:", lines[i]) or re.match(
                        r"\w+Exception:", lines[i]
                    ):
                        # Collect one more line if it's the exception message
                        if i + 1 < len(lines) and not lines[i + 1].strip().startswith(
                            "File "
                        ):
                            i += 1
                            if i < len(lines):
                                error_lines.append(lines[i])
                        break
                    i += 1
            else:
                # For other errors, collect a few lines of context
                # Add 2 lines before (if available)
                start = max(0, i - 2)
                for j in range(start, min(i + 3, len(lines))):
                    if len(error_lines) < max_lines:
                        error_lines.append(lines[j])
                i += 3
        else:
            i += 1

    return error_lines


def run_with_timeout(
    func: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    timeout: float = None,
):
    """Run a function with timeout."""
    ret_value = []

    def _target_func():
        ret_value.append(func(*args, **(kwargs or {})))

    t = threading.Thread(target=_target_func)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise TimeoutError()

    if not ret_value:
        raise RuntimeError()

    return ret_value[0]


def run_unittest_files(
    files: List[TestFile], timeout_per_file: float, continue_on_error: bool = False
):
    """
    Run a list of test files.

    Args:
        files: List of TestFile objects to run
        timeout_per_file: Timeout in seconds for each test file
        continue_on_error: If True, continue running remaining tests even if one fails.
                          If False, stop at first failure (default behavior for PR tests).
    """
    tic = time.perf_counter()
    success = True
    passed_tests = []
    failed_tests = []
    failure_details = []  # List of TestFailureInfo objects

    for i, file in enumerate(files):
        filename, estimated_time = file.name, file.estimated_time
        process = None
        captured_stdout = ""
        captured_stderr = ""

        def run_one_file(filename):
            nonlocal process, captured_stdout, captured_stderr

            filename = os.path.join(os.getcwd(), filename)
            print(
                f".\n.\nBegin ({i}/{len(files) - 1}):\npython3 {filename}\n.\n.\n",
                flush=True,
            )
            tic = time.perf_counter()

            # Capture stdout and stderr while also displaying to console
            process = subprocess.Popen(
                ["python3", filename],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ,
                text=True,  # Return output as strings instead of bytes
            )

            # Wait for process to complete and get output (calls wait() internally)
            captured_stdout, captured_stderr = process.communicate()

            # Print to console (preserving original behavior)
            if captured_stdout:
                print(captured_stdout, end="", flush=True)
            if captured_stderr:
                print(captured_stderr, end="", flush=True)

            elapsed = time.perf_counter() - tic

            print(
                f".\n.\nEnd ({i}/{len(files) - 1}):\n{filename=}, {elapsed=:.0f}, {estimated_time=}\n.\n.\n",
                flush=True,
            )
            return process.returncode

        try:
            ret_code = run_with_timeout(
                run_one_file, args=(filename,), timeout=timeout_per_file
            )
            if ret_code != 0:
                print(
                    f"\n✗ FAILED: {filename} returned exit code {ret_code}\n",
                    flush=True,
                )
                success = False

                # Extract error lines from captured output
                error_lines = extract_error_lines(captured_stderr, captured_stdout)

                # Extract main error message for better visibility
                error_msg = extract_main_error_message(captured_stderr, captured_stdout)

                # Create failure info
                failure_info = TestFailureInfo(
                    filename=filename,
                    reason=error_msg,
                    stderr=captured_stderr,
                    stdout=captured_stdout,
                    error_lines=error_lines,
                )
                failure_details.append(failure_info)
                failed_tests.append((filename, error_msg))

                if not continue_on_error:
                    # Stop at first failure for PR tests
                    break
                # Otherwise continue to next test for nightly tests
            else:
                passed_tests.append(filename)
        except TimeoutError:
            kill_process_tree(process.pid)
            time.sleep(5)
            print(
                f"\n✗ TIMEOUT: {filename} after {timeout_per_file} seconds\n",
                flush=True,
            )
            success = False

            # Create failure info for timeout
            failure_info = TestFailureInfo(
                filename=filename,
                reason=f"timeout after {timeout_per_file}s",
                stderr=captured_stderr,
                stdout=captured_stdout,
                error_lines=[],
            )
            failure_details.append(failure_info)
            failed_tests.append((filename, f"timeout after {timeout_per_file}s"))

            if not continue_on_error:
                # Stop at first timeout for PR tests
                break
            # Otherwise continue to next test for nightly tests

    if success:
        print(f"Success. Time elapsed: {time.perf_counter() - tic:.2f}s", flush=True)
    else:
        print(f"Fail. Time elapsed: {time.perf_counter() - tic:.2f}s", flush=True)

    # Print summary
    print(f"\n{'='*60}", flush=True)
    print(f"Test Summary: {len(passed_tests)}/{len(files)} passed", flush=True)
    print(f"{'='*60}", flush=True)
    if passed_tests:
        print("✓ PASSED:", flush=True)
        for test in passed_tests:
            print(f"  {test}", flush=True)
    if failed_tests:
        print("\n✗ FAILED:", flush=True)
        for test, reason in failed_tests:
            print(f"  {test} ({reason})", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Write failure summary to GitHub Actions Step Summary
    if failure_details:
        write_failure_summary_to_github(failure_details)

    return 0 if success else -1
