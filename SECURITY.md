# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in rustmolbpe, please report it responsibly:

1. **Do not** open a public GitHub issue for security vulnerabilities
2. Email the maintainers directly at [hfooladi.work@gmail.com](mailto:hfooladi.work@gmail.com)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes (if applicable)

## Response Timeline

- **Initial response**: Within 48 hours
- **Status update**: Within 7 days
- **Fix timeline**: Depends on severity, typically within 30 days for critical issues

## Security Considerations

rustmolbpe processes SMILES strings which are user-provided input. The following security measures are in place:

- **Input validation**: The regex-based tokenizer only matches valid SMILES patterns
- **Memory safety**: Core implementation in Rust provides memory safety guarantees
- **No network access**: The library does not make any network requests
- **No file system access**: Except for explicit vocabulary load/save operations

## Scope

This security policy covers:
- The rustmolbpe Rust library
- The Python bindings (PyO3)
- Pre-trained vocabulary files in the `data/` directory

Out of scope:
- Third-party dependencies (report to respective projects)
- User applications built with rustmolbpe
