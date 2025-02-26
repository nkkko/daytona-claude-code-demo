# Daytona SDK Project Guidelines

## Build & Test Commands
- **Python Setup**: `pip install daytona-sdk`
- **TypeScript Setup**: `npm install @daytonaio/sdk`
- **Run Python**: `python main.py`
- **Run TypeScript**: `npx tsx ./index.ts`
- **Debug Python**: `python -m debugpy --listen 5678 --wait-for-client main.py`
- **Python Test**: `pytest tests/`
- **Python Single Test**: `pytest tests/test_file.py::test_name -v`
- **TypeScript Test**: `npm test`
- **TypeScript Single Test**: `npm test -- -t "test name"`

## Code Style Guidelines
- **Naming**: camelCase for TypeScript/JavaScript, snake_case for Python
- **Imports**: Group by standard library, third-party packages, local modules
- **Python Types**: Use type hints for function parameters and return values
- **TypeScript Types**: Define interfaces/types for complex structures
- **Error Handling**: Use try/catch or try/except blocks with specific exceptions
- **Documentation**: Use docstrings (Python) or JSDoc (TypeScript)
- **Environment Variables**: Store sensitive data in .env files, not in code
- **API Access**: Use the DaytonaConfig class for configuration
- **File Structure**: Keep language-specific code in separate directories