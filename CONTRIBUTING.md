# Contributing to ApexRL

Thank you for your interest in contributing!

## Development Setup

```bash
# Clone and install
git clone https://github.com/yourusername/apexrl.git
cd apexrl
pip install -e ".[dev,test]"

# Install pre-commit hooks
pre-commit install
```

## Code Style

This project uses Ruff for formatting and linting. Run before committing:

```bash
pre-commit run --all-files
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/apexrl

# Syntax check only
python3 test/test_syntax.py
```

## Pull Requests

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request
