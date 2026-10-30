# ADR-001: Python Environment & Runtime Selection

## Context
The technical blueprint originally specified Python 3.11 for MedAgentX. Upon performing repository forensics on the host system, Python 3.12.4 was detected. We must determine whether to standardize on Python 3.12.4 or enforce Python 3.11.

## Decision
Standardize the host execution environment on **Python 3.12.4** while locking package dependencies in `requirements.txt` to ensure full cross-platform compatibility with FastAPI, Pydantic v2, LangGraph, Qdrant Client, and Neo4j Driver.

## Alternatives Considered
1. **Downgrade to Python 3.11:** Would require installing a secondary Python distribution on the host machine.
2. **Standardize on Python 3.12.4:** Leverages installed runtime; Python 3.12 offers improved asyncio performance and lower memory overhead.

## Advantages
- Zero setup friction on the host system.
- Improved async execution performance for FastAPI endpoints and concurrent vector/graph database calls.

## Disadvantages
- Minor risk of version mismatch with legacy PyTorch packages (mitigated by selecting PyTorch 2.2+ compiled for Python 3.12).

## Research & Engineering Implications
- All type annotations will use Python 3.12 syntax (`list[str]`, `dict[str, Any]`, `str | None`).
- Pydantic Settings v2 (`pydantic-settings`) will be used for environment validation.

## Status
**ACCEPTED** (Date: August 13, 2026)
