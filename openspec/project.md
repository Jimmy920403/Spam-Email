# Project Context

## Purpose
This repository holds the OpenSpec-driven project for [PROJECT NAME]. Its purpose is to maintain authoritative specifications in `openspec/specs/`, manage proposed changes in `openspec/changes/`, and coordinate design, implementation, and archiving of changes using the OpenSpec workflow.

## Tech Stack
- Primary languages: TypeScript (backend & tool scripts), Markdown for specs, and YAML for CI.
- Runtime / Platform: Node.js (LTS) for tools and automation.
 - Primary languages: TypeScript (backend & tool scripts), Python for data/ML workloads, Markdown for specs, and YAML for CI.
 - Runtime / Platform: Node.js (LTS) for tools and automation; Python 3.10+ for ML scripts and notebooks.
- Tooling: ESLint, Prettier, Jest for unit tests, Playwright/Cypress for optional E2E, and GitHub Actions for CI.
- CLI tooling: `openspec` (project-specific tool), ripgrep (`rg`) for search, and standard shell (PowerShell on Windows, bash on CI).

## Project Conventions

### Code Style
- Use Prettier for code formatting and common editor settings. Keep line width to 100 characters.
- Use ESLint with the recommended TypeScript rules. Fix linter issues before merging.
- Commit messages follow Conventional Commits (type(scope): short-description). Examples: `feat(auth): add 2fa endpoints` or `fix(api): handle nil body`.

### Architecture Patterns
- Organize code by capability (feature folders) rather than by technical layer when it improves clarity.
- Prefer small, single-purpose modules and explicit interfaces. Keep public APIs minimal and well-documented.
- Use hexagonal/clean architecture patterns for services that interact with multiple external systems.

### Testing Strategy
- Unit tests: Jest. Aim for high coverage on critical logic but avoid 100% as a strict requirement.
- Integration tests: Run against in-memory or ephemeral test instances where possible (e.g., SQLite, test containers).
- End-to-end: Use Playwright or Cypress for critical user flows; keep these small and fast.
- Every change that touches behavior MUST include at least one test demonstrating the expected scenario.

### Git Workflow
- Branching: `main` is protected. Work in feature branches named `feat/<short-name>` or `chore/<short-name>`.
- PRs should reference the OpenSpec change-id when implementing a proposal (e.g., `implements: add-openspec-proposal-generator`).
- Use PR templates that require linking proposal.md and tasks.md when a change-id exists.

## Domain Context
- This repository treats specs as the source of truth. Any behavioral change should be proposed as an OpenSpec change and validated before implementation.
- Capabilities are organized under `openspec/specs/<capability>/spec.md` and must use the required `#### Scenario:` format.

### ML & Data conventions
- Place dataset download and ingestion scripts in `scripts/` (e.g., `scripts/download_dataset.py`). Do not commit raw datasets unless licensing is explicit. Prefer a download script that fetches data at runtime.
- Place Jupyter notebooks under `notebooks/` and trained model artifacts under `models/`. Use `artifacts/` for CI-produced metrics and reports.
- Use a small `requirements.txt` or `pyproject.toml` to pin Python dependencies for ML tasks (e.g., scikit-learn, pandas, joblib).

## Important Constraints
- Keep tooling changes backwards compatible where possible.
- Security: secrets must be stored in the repository's secure secrets manager (e.g., GitHub encrypted secrets); never check credentials into the repo.
- Compliance: log / audit changes to security-related specs and require a review from the security owner.

## External Dependencies
- Authentication provider: (e.g., Auth0, Okta) — document the exact provider and tenant in a separate secure doc if used.
- Email/SMS providers for notifications (e.g., SendGrid, Twilio) — note usage and any sending quotas.
- CI/CD: GitHub Actions is the canonical runner for this project; runners are configured in `.github/workflows/`.

## Contacts
- Primary repository owner: add a name/email here.
- OpenSpec workflow owner: add a name/email here.

---

Notes:
- Keep this file updated when adding new capabilities or changing conventions. The assistant can help maintain this file as changes are proposed.
