"""Shared pytest configuration and fixtures."""


def pytest_addoption(parser):
    """Register --testdata option for conformance tests."""
    parser.addoption(
        "--testdata",
        action="store",
        default=None,
        help="Path to the ViSQOL testdata directory (for conformance tests)",
    )
