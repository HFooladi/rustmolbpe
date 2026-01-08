# Contributing to rustmolbpe

Thank you for your interest in contributing to rustmolbpe! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- [Rust](https://rustup.rs/) (stable toolchain)
- Python 3.9+
- [maturin](https://github.com/PyO3/maturin) for building Python bindings

### Setting Up the Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/HFooladi/rustmolbpe.git
   cd rustmolbpe
   ```

2. Create a Python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install maturin pytest
   ```

4. Build the package in development mode:
   ```bash
   maturin develop
   ```

   For optimized builds:
   ```bash
   maturin develop --release
   ```

## Running Tests

### Rust Tests

```bash
cargo test
```

### Python Tests

```bash
pytest tests/python/ -v
```

### Run All Tests

```bash
cargo test && pytest tests/python/ -v
```

## Code Style

### Rust

We use `rustfmt` for formatting and `clippy` for linting:

```bash
# Format code
cargo fmt

# Check formatting (CI will fail if not formatted)
cargo fmt --check

# Run linter
cargo clippy -- -D warnings
```

### Python

Follow PEP 8 guidelines for any Python code (tests, examples, scripts).

## Making Changes

### Workflow

1. Fork the repository
2. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes
4. Ensure tests pass and code is formatted:
   ```bash
   cargo fmt
   cargo clippy -- -D warnings
   cargo test
   maturin develop
   pytest tests/python/ -v
   ```
5. Commit your changes with a descriptive message
6. Push to your fork and open a Pull Request

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb in imperative mood (e.g., "Add", "Fix", "Update")
- Reference issues when applicable (e.g., "Fix #123")

### Pull Request Guidelines

- Provide a clear description of the changes
- Include any relevant issue numbers
- Ensure CI passes (tests, formatting, linting)
- Update documentation if needed
- Add tests for new functionality

## Adding New Features

When adding new features:

1. **Update Rust code** in `src/lib.rs`
2. **Add Python bindings** using PyO3 macros
3. **Update type stubs** in `rustmolbpe.pyi`
4. **Add tests** in both Rust (unit tests) and Python (`tests/python/`)
5. **Update documentation** in README.md and `docs/`
6. **Update CHANGELOG.md** with your changes

## Reporting Issues

When reporting bugs, please include:

- Python version
- Operating system
- Minimal code to reproduce the issue
- Expected vs actual behavior
- Full error message/traceback

## Questions?

Feel free to open an issue for questions or discussions about potential contributions.
