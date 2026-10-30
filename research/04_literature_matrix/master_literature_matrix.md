# Master Literature Matrix

This matrix synthesizes 20 landmark research papers across RAG, Adaptive RAG, Corrective RAG, Agentic RAG, Multi-Agent Systems, Graph RAG, Medical LLMs, and Clinical AI from 2020 through 2026.

## Literature Summary Table

| Paper | Year | Venue | Architecture | Key Contribution | Relevance to MedAgentX | Novelty Threat |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Adaptive-RAG** | 2024 | NAACL | Classifier Routing | Query-complexity based RAG routing | High (Routing strategy) | Medium |
| **CRAG (Corrective RAG)** | 2024 | arXiv | Evaluator + Web Search | Evaluator score triggers web search correction | High (Verification concept) | Medium |
| **Self-RAG** | 2024 | ICLR | Special Critique Tokens | Model critiques retrieval and output tokens | High (Self-reflection) | High |
| **Microsoft GraphRAG** | 2024 | arXiv | Graph Community Summaries | Global entity-relation community summaries | High (Hybrid graph baseline) | Medium |
| **Med-PaLM 2** | 2025 | Nature Med | Parametric Med LLM | 86.5% MedQA accuracy (USMLE benchmark) | High (Medical LLM baseline) | Low |
| **AMIE** | 2025 | Nature | Self-Play Diagnostic AI | Physician-level diagnostic dialogue & history | High (Clinical reasoning) | Low |
| **SEMA-RAG** | 2026 | ACL Find | Decoupled Medical RAG | Interpreter / Explorer / Arbiter agents | **Very High (Direct Competitor)** | **High** |
| **MedCoAct** | 2025 | MDPI | Confidence Multi-Agent | Collaborative confidence loops for clinical QA | **Very High (Direct Competitor)** | **High** |
| **ArbGraph** | 2026 | arXiv | Graph Evidence Arbiter | Resolves evidence contradictions via graphs | High (Contradiction resolution) | Medium |
| **DeepRare** | 2025 | arXiv | Confidence Triage | Agent activation triage for rare diseases | High (Confidence triage) | Medium |
| **CONFRAG** | 2026 | OpenRev | Confidence Dynamic RAG | Entropy-based dynamic retrieval stopping | High (Confidence calculation) | Medium |
| **Reflexion** | 2023 | NeurIPS | Verbal RL Memory | Self-reflection memory buffer for agents | Medium (Reflection loop) | Low |
| **Self-Refine** | 2023 | NeurIPS | Multi-Prompt Feedback | Generator-Feedback-Refiner iterative loop | Medium (Critique module) | Low |
| **MetaGPT** | 2024 | ICLR | SOP Multi-Agent | Software engineering multi-agent SOPs | Medium (SOP roles) | Low |
| **AutoGen** | 2023 | arXiv | Conversational Agents | Flexible group chat multi-agent framework | Medium (Infrastructure) | Low |
| **DyLAN** | 2024 | ICLR | Dynamic Agent Selection | Importance scoring to prune inactive agents | High (Dynamic activation) | Medium |
| **BioMistral** | 2024 | arXiv | Open Medical LLM | 7B open-weights model fine-tuned on PubMed | High (Local model backbone) | Low |
| **OpenBioLLM-70B** | 2024 | HuggingFace | Open Medical 70B | State-of-the-art open medical instruction model | High (Local model backbone) | Low |
| **FLARE** | 2023 | EMNLP | Mid-Generation RAG | Active retrieval triggered by low token logprob | Medium (Token confidence) | Low |
| **Med-Gemini** | 2024 | arXiv | Multimodal Med AI | 91.1% MedQA accuracy; web search integration | High (SOTA benchmark) | High |

---

## Detailed Paper Summaries Directory

Granular, single-paper extraction files are located in:  
[`research/04_literature_matrix/paper_summaries/`](file:///c:/Users/Bhukya%20Shrikanth/OneDrive/Desktop/Major%20Project/research/04_literature_matrix/paper_summaries/)
