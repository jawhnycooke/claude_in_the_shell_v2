# Code Standards

## Python Style

- Follow PEP 8 with Black formatting (88 char line length)
- Use type hints for all function signatures
- Write comprehensive docstrings (Google style)
- Minimum 80% test coverage
- Use `structlog` for logging

## Import Order

1. Standard library
2. Third-party packages
3. Local application imports

Separate each group with a blank line.

## Naming Conventions

- Modules: `snake_case.py`
- Classes: `PascalCase`
- Functions: `snake_case()`
- Constants: `UPPER_SNAKE_CASE`
- Private: `_leading_underscore`

## Error Handling

- Use specific exception types
- Include descriptive error messages
- Use context managers for resources
- Fail fast with clear errors

## Testing

- Test file names: `test_*.py`
- Test classes: `TestClassName`
- Test functions: `test_description`
- Use fixtures in `conftest.py`
- Mock external dependencies

## Documentation

- Every module needs docstring
- Every public function needs docstring
- Include examples in docstrings
- Keep CLAUDE.md updated
