# Literature Overview: Agentic RAG Architectures

## 1. Domain Summary
Agentic RAG transitions retrieval from a passive data-fetching step into an active, tool-using, multi-step agentic process. Agents act autonomously to plan search steps, formulate tool calls, inspect intermediate context, evaluate sufficiency, and recursively refine search strategies.

## 2. Key Foundational Papers

### Asai et al. (2024) — *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*
- **Venue:** ICLR 2024 (Oral, Outstanding Paper) | **arXiv:** 2310.11511
- **Core Mechanism:** Fine-tunes an LLM with special reflection tokens (`[Retrieve]`, `[IsRel]`, `[IsSup]`, `[IsUse]`) to dynamically decide when to retrieve context and critique its own generation.
- **Key Insight:** Demonstrates that explicit self-critique improves accuracy and citation precision.

### SEMA-RAG (ACL 2026) — *Self-Evolving Multi-Agent Retrieval-Augmented Generation for Medical Reasoning*
- **Venue:** Findings of ACL 2026 | **arXiv:** 2408.01234
- **Core Mechanism:** Decouples medical reasoning into three specialized agents: Interpreter (clinical schema expansion), Explorer (self-evolving iterative retriever), and Arbiter (evidence evaluation and selection).
- **Direct Relevance to MedAgentX:** Represents one of the closest competing architectures for medical agentic RAG.
