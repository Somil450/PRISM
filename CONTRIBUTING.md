# Contributing to PRISM

Welcome to the PRISM (Provenance & Reasoning Intelligence System for Multimedia) project! We appreciate your interest in contributing. This document outlines the standards, workflows, and rules required to maintain the integrity and quality of our forensic platform.

## Table of Contents
1. [Git Workflow](#git-workflow)
2. [Conventional Commits](#conventional-commits)
3. [Code Standards & Linting](#code-standards--linting)
4. [Hardware Discipline Rules (VRAM)](#hardware-discipline-rules-vram)
5. [Peer Review Policy](#peer-review-policy)

---

## Git Workflow

We follow a structured branching model. **Never commit directly to the `main` or `dev` branches.**

1. **Ensure you are up to date:**
   ```bash
   git checkout dev
   git pull origin dev
   ```

2. **Create a feature branch:**
   Use the format `type/feature-name`.
   ```bash
   git checkout -b feature/your-feature-name
   # or bugfix/your-bugfix-name
   ```

3. **Develop and commit:**
   Follow the [Conventional Commits](#conventional-commits) specification for all commits.

4. **Push your branch:**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request:**
   Target your PR against the `dev` branch. `main` is reserved for stable releases only. Make sure to fill out the `.github/PULL_REQUEST_TEMPLATE.md` checklist when opening your PR.

---

## Conventional Commits

All commit messages must adhere to the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification. This allows for automated changelog generation and maintains a readable history.

**Format:**
```
<type>(<scope>): <description>
```

**Allowed Types:**
- `feat`: A new feature or agent implementation.
- `fix`: A bug fix.
- `docs`: Documentation only changes.
- `style`: Changes that do not affect the meaning of the code (white-space, formatting, etc.).
- `refactor`: A code change that neither fixes a bug nor adds a feature.
- `perf`: A code change that improves performance (critical for VRAM management).
- `test`: Adding missing tests or correcting existing tests.
- `chore`: Changes to the build process or auxiliary tools and libraries.

**Example:**
```
feat(fusion): implement adaptive quality discounting factor
perf(vision): optimize TruFor tensor transfer to CPU to save VRAM
```

---

## Code Standards & Linting

Before pushing your code or opening a PR, you **must** ensure it passes our styling and type-checking pipelines. We enforce strict PEP-8 compliance and static typing.

Run the following commands locally:

```bash
# 1. Formatting
black src/

# 2. Linting
flake8 src/

# 3. Static Type Checking
mypy src/
```

Any PR that fails the CI pipeline for `black`, `flake8`, or `mypy` will not be reviewed.

---

## Hardware Discipline Rules (VRAM)

PRISM is explicitly designed to operate on consumer-grade hardware (4GB-8GB VRAM). **Strict memory management is a core requirement, not an afterthought.**

If you are modifying or adding models, agents, or data pipelines, you MUST adhere to the following:

1. **Peak VRAM Cap:** Your code must not cause the total VRAM consumption to exceed **6.5 GB** during local execution.
2. **Dynamic Staging:** Ensure models are loaded into VRAM only when needed and immediately offloaded.
3. **Cleanup Protocol:** Always invoke agent cleanup methods and clear the CUDA cache after processing:
   ```python
   agent.cleanup()
   torch.cuda.empty_cache()
   # Hand off tensors to CPU before proceeding
   ```
4. **Validation:** Manually profile your changes using tools like `nvidia-smi` or `torch.cuda.memory_summary()` before submitting a PR.

---

## Peer Review Policy

To maintain the forensic integrity of the platform, all code changes require a thorough review.

1. **Required Approvers:** All PRs must be approved by at least **one Lead Engineer** associated with the domain you are modifying (e.g., Somil for Vision, Kulsum for Audio/UI, Tanush for Architecture/Fusion).
2. **Review Criteria:**
   - Adherence to the VRAM peak limit.
   - Proper use of `black`, `flake8`, and `mypy`.
   - Clear and documented mathematical logic (especially in fusion or modeling layers).
   - Identity leakage prevention in data splits.
3. **Response Time:** Reviewers aim to provide feedback within 48 hours.
