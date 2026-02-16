#!/usr/bin/env python3
"""
Intelligent Retry Utilities for LLM API Calls
Handles rate limits, transient errors, and exponential backoff
"""

import re
import time
from typing import Optional, Callable, Any
from functools import wraps


class RetryError(Exception):
    """Base class for retry-related errors"""
    pass


class RateLimitError(RetryError):
    """Rate limit exceeded - need to wait or exit"""
    def __init__(self, wait_seconds: float, message: str):
        self.wait_seconds = wait_seconds
        self.message = message
        super().__init__(message)


class PermanentError(RetryError):
    """Permanent error - should not retry"""
    pass


def parse_rate_limit_wait_time(error_msg: str) -> Optional[float]:
    """
    Parse Groq/OpenAI rate limit error and extract wait time

    Examples:
        "Please try again in 12m8.352s" -> 728.352
        "Please try again in 1h23m45.6s" -> 5025.6
        "Please try again in 45.6s" -> 45.6

    Returns:
        Wait time in seconds, or None if not a rate limit error
    """
    # Common rate limit error patterns (case-insensitive)
    rate_limit_indicators = [
        "rate limit",
        "rate_limit",
        "too many requests",
        "quota exceeded",
        "requests per"
    ]

    error_msg_lower = error_msg.lower()
    if not any(indicator.lower() in error_msg_lower for indicator in rate_limit_indicators):
        return None

    # Parse wait time from various formats
    patterns = [
        r"(\d+)h(\d+)m(\d+(?:\.\d+)?)s",  # 1h23m45.6s (anywhere in message)
        r"(\d+)m(\d+(?:\.\d+)?)s",        # 12m8.352s
        r"(\d+(?:\.\d+)?)s",              # 45.6s (must have 's' suffix)
        r"after (\d+) seconds",            # after 60 seconds
        r"retry.*?(\d+)\s*seconds?",      # retry after 60 seconds (flexible)
    ]

    for pattern in patterns:
        match = re.search(pattern, error_msg, re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups) == 3:  # hours, minutes, seconds
                return int(groups[0]) * 3600 + int(groups[1]) * 60 + float(groups[2])
            elif len(groups) == 2:  # minutes, seconds
                return int(groups[0]) * 60 + float(groups[1])
            else:  # seconds only
                return float(groups[0])

    # Default: if we detect rate limit but can't parse time, assume 60 seconds
    return 60.0


def retry_with_intelligent_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    rate_limit_threshold: float = 3600.0,  # 1 hour in seconds
):
    """
    Decorator for intelligent retry with exponential backoff and rate limit handling

    Strategy:
    - Exponential backoff for transient errors (network, timeout, 500 errors)
    - Smart waiting for short rate limits (< threshold)
    - Raise RateLimitError for long rate limits (>= threshold) to exit gracefully

    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        backoff_factor: Multiplier for exponential backoff
        rate_limit_threshold: If rate limit wait > this, raise error to exit (default: 1 hour)

    Raises:
        RateLimitError: If rate limit wait time exceeds threshold (should exit and resume later)
        Exception: Original exception if all retries exhausted
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e
                    error_msg = str(e)

                    # Check if this is a rate limit error
                    wait_time = parse_rate_limit_wait_time(error_msg)

                    if wait_time is not None:
                        # Rate limit detected
                        if wait_time >= rate_limit_threshold:
                            # Long wait - should exit and resume later
                            print(f"\n⚠️  Rate limit exceeded")
                            print(f"    Wait time: {wait_time/60:.1f} minutes ({wait_time/3600:.2f} hours)")
                            print(f"    Threshold: {rate_limit_threshold/60:.1f} minutes")
                            print(f"💾 Exiting gracefully - resume with same command after rate limit resets")
                            raise RateLimitError(wait_time, error_msg)
                        else:
                            # Short wait - can handle it
                            buffer = 5  # Add 5 second buffer
                            total_wait = wait_time + buffer
                            print(f"\n⏳ Rate limit hit - waiting {total_wait:.0f}s (requested: {wait_time:.0f}s + {buffer}s buffer)")
                            time.sleep(total_wait)
                            continue  # Retry immediately after waiting

                    # Check for permanent errors (validation, authentication)
                    permanent_error_indicators = [
                        "validation error",
                        "invalid api key",
                        "authentication failed",
                        "not found",
                        "access denied"
                    ]

                    if any(indicator in error_msg.lower() for indicator in permanent_error_indicators):
                        print(f"❌ Permanent error detected - not retrying: {error_msg[:100]}")
                        raise  # Don't retry permanent errors

                    # Transient error - use exponential backoff
                    if attempt < max_attempts - 1:
                        print(f"⚠️  Attempt {attempt + 1}/{max_attempts} failed: {error_msg[:100]}")
                        print(f"   Retrying in {delay:.1f}s with exponential backoff...")
                        time.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        # Final attempt failed
                        print(f"❌ All {max_attempts} attempts exhausted")
                        raise

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception

            return None

        return wrapper
    return decorator


def is_rate_limit_error(error: Exception) -> bool:
    """Check if an exception is a rate limit error"""
    error_msg = str(error)
    return parse_rate_limit_wait_time(error_msg) is not None


def is_transient_error(error: Exception) -> bool:
    """
    Check if an exception is likely transient (worth retrying)

    Transient errors include:
    - Network errors (timeout, connection reset)
    - Server errors (500, 502, 503, 504)
    - Rate limits (handled separately)
    """
    error_msg = str(error).lower()

    transient_indicators = [
        "timeout",
        "timed out",
        "connection",
        "network",
        "temporary",
        "500",
        "502",
        "503",
        "504",
        "bad gateway",
        "service unavailable",
        "gateway timeout"
    ]

    return any(indicator in error_msg for indicator in transient_indicators)


# Example usage:
if __name__ == "__main__":
    # Test rate limit parsing
    test_cases = [
        "Rate limit reached. Please try again in 12m8.352s",
        "Rate limit: Too many requests. Try again in 1h23m45s",
        "Rate limit exceeded. Quota exceeded. Please wait 45.6s",
        "Error: rate_limit_exceeded, retry after 120 seconds",
        "Too many requests - try again in 30s",
        "Normal error message"
    ]

    print("Testing rate limit parsing:")
    for msg in test_cases:
        wait = parse_rate_limit_wait_time(msg)
        if wait:
            print(f"  ✅ '{msg[:50]}...' -> {wait:.1f}s ({wait/60:.1f}m)")
        else:
            print(f"  ❌ '{msg[:50]}...' -> Not a rate limit error")

    print("\n" + "="*60)
    print("Testing retry decorator:")

    # Test function that fails a few times then succeeds
    attempt_counter = [0]

    @retry_with_intelligent_backoff(max_attempts=3, initial_delay=0.1, rate_limit_threshold=10)
    def flaky_function():
        attempt_counter[0] += 1
        if attempt_counter[0] < 3:
            raise Exception("Temporary network timeout")
        return "Success!"

    result = flaky_function()
    print(f"  Result: {result} (after {attempt_counter[0]} attempts)")
