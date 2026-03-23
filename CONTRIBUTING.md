# Contributing to ViSQOL (Python)

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/visqol-python.git
   cd visqol-python
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/my-improvement
   ```

## Development Setup

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install in development mode with test dependencies
pip install -e ".[test]"
```

## Running Tests

```bash
# Quick smoke tests (no external data needed)
pytest tests/test_quick.py -v

# Full conformance tests (requires testdata directory)
pytest tests/test_conformance.py -v --testdata /path/to/visqol/testdata
```

## Code Style

- **Type hints**: All public functions and methods must include type annotations.
- **Docstrings**: Use Google-style docstrings for all public APIs.
- **Imports**: Use `from __future__ import annotations` at the top of every module.
- Keep line length ≤ 99 characters where practical.

## Making Changes

1. Write clean, well-documented code with type hints.
2. Add or update tests for any new functionality.
3. Ensure all existing tests still pass.
4. Update `CHANGELOG.md` under an `[Unreleased]` section.

## Pull Request Process

1. Update the `CHANGELOG.md` with details of your changes.
2. Ensure all tests pass locally.
3. Submit a pull request with a clear description of the changes.
4. Link any relevant issues.

## Reporting Bugs

Please open an [issue](https://github.com/talker93/visqol-python/issues) with:

- A clear, descriptive title
- Steps to reproduce the problem
- Expected vs. actual behavior
- Python version and OS
- Relevant audio file details (sample rate, duration, format)

## Versioning

This project follows [Semantic Versioning](https://semver.org/). The single source of truth for the version number is `visqol/__init__.py`.

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
