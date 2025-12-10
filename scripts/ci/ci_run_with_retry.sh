#!/bin/bash
# CI script to run tests with smart retry logic
# Only retries on accuracy/performance assertion failures, not code errors

set -o pipefail

MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
RETRY_WAIT=${RETRY_WAIT:-60}
COMMAND="$@"

# Patterns that indicate retriable accuracy/performance failures
RETRIABLE_PATTERNS=(
    "AssertionError:.*not greater than"
    "AssertionError:.*not less than"
    "AssertionError:.*not equal to"
    "AssertionError:.*!=.*expected"
    "accuracy"
    "score"
    "latency"
    "throughput"
)

# Patterns that indicate non-retriable failures (real errors)
NON_RETRIABLE_PATTERNS=(
    "SyntaxError"
    "ImportError"
    "ModuleNotFoundError"
    "NameError"
    "TypeError"
    "AttributeError"
    "RuntimeError"
    "CUDA out of memory"
    "OOM"
    "Segmentation fault"
    "core dumped"
    "ConnectionRefusedError"
    "FileNotFoundError"
)

is_retriable_failure() {
    local output="$1"

    # Check for non-retriable patterns first
    for pattern in "${NON_RETRIABLE_PATTERNS[@]}"; do
        if echo "$output" | grep -qE "$pattern"; then
            echo "[CI Retry] Non-retriable error detected: $pattern"
            return 1
        fi
    done

    # Check for retriable patterns
    for pattern in "${RETRIABLE_PATTERNS[@]}"; do
        if echo "$output" | grep -qiE "$pattern"; then
            echo "[CI Retry] Retriable failure detected (pattern: $pattern)"
            return 0
        fi
    done

    # If we have an AssertionError but didn't match non-retriable, assume retriable
    if echo "$output" | grep -qE "AssertionError"; then
        echo "[CI Retry] AssertionError detected, assuming retriable"
        return 0
    fi

    # Default: not retriable
    echo "[CI Retry] Unknown failure type, not retrying"
    return 1
}

write_retry_summary() {
    local attempt=$1
    local max=$2
    local status=$3
    local was_retry=$4

    if [ -n "$GITHUB_STEP_SUMMARY" ]; then
        {
            echo "### CI Retry Summary"
            echo ""
            echo "| Metric | Value |"
            echo "|--------|-------|"
            echo "| Total Attempts | $attempt / $max |"
            echo "| Final Status | $status |"
            echo "| Retry Occurred | $was_retry |"
            echo "| Command | \`$COMMAND\` |"
            echo ""
        } >> "$GITHUB_STEP_SUMMARY"
    fi
}

run_with_retry() {
    local attempt=1
    local was_retry="No"

    while [ $attempt -le $MAX_ATTEMPTS ]; do
        echo ""
        echo "=========================================="
        echo "[CI Retry] Attempt $attempt of $MAX_ATTEMPTS"
        echo "[CI Retry] Command: $COMMAND"
        echo "=========================================="
        echo ""

        # Run command and capture output
        local output
        local exit_code
        output=$(eval "$COMMAND" 2>&1 | tee /dev/stderr)
        exit_code=${PIPESTATUS[0]}

        if [ $exit_code -eq 0 ]; then
            echo ""
            echo "[CI Retry] SUCCESS on attempt $attempt"
            write_retry_summary $attempt $MAX_ATTEMPTS "SUCCESS" "$was_retry"
            return 0
        fi

        echo ""
        echo "[CI Retry] FAILED on attempt $attempt (exit code: $exit_code)"

        # Check if we should retry
        if [ $attempt -lt $MAX_ATTEMPTS ]; then
            if is_retriable_failure "$output"; then
                echo "[CI Retry] Will retry in $RETRY_WAIT seconds..."
                was_retry="Yes"
                sleep $RETRY_WAIT
                attempt=$((attempt + 1))
            else
                echo "[CI Retry] Failure is not retriable, stopping"
                write_retry_summary $attempt $MAX_ATTEMPTS "FAILED (non-retriable)" "$was_retry"
                return $exit_code
            fi
        else
            echo "[CI Retry] Max attempts reached"
            write_retry_summary $attempt $MAX_ATTEMPTS "FAILED (max attempts)" "$was_retry"
            return $exit_code
        fi
    done
}

# Main
if [ -z "$COMMAND" ]; then
    echo "Usage: $0 <command>"
    echo "Environment variables:"
    echo "  MAX_ATTEMPTS - Maximum retry attempts (default: 3)"
    echo "  RETRY_WAIT - Seconds to wait between retries (default: 60)"
    exit 1
fi

run_with_retry
