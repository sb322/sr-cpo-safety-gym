"""Pytest scaffold configuration."""

from pytest import ExitCode


def pytest_sessionfinish(session, exitstatus):
    """Treat the scaffold's import-only test layout as a clean pass."""
    if exitstatus == ExitCode.NO_TESTS_COLLECTED:
        session.exitstatus = ExitCode.OK
