Paper Title: SEMA-RAG: Self-Evolving Multi-Agent Retrieval-Augmented Generation for Medical Reasoning
Authors: Chen et al.
Year: 2026
Venue: Findings of ACL 2026
DOI: 10.48550/arXiv.2408.01234
URL: https://arxiv.org/abs/2408.01234
Peer Reviewed: Yes
Citation Count: 18

Problem: Static single-round RAG fails to resolve complex medical questions requiring dynamic evidence retrieval and multi-step reasoning.
Research Question: Can a decoupled multi-agent framework improve medical RAG accuracy by separating schema expansion, retrieval, and evidence arbitration?

Architecture: Decoupled Multi-Agent RAG
Models: GPT-4o / LLaMA-3-8B
Retrieval: Iterative Dense Retrieval
RAG Type: Agentic Multi-Agent RAG
Agents: 3 Agents (Interpreter, Explorer, Arbiter)
Planner: No (Interpreter handles query transformation)
Retriever: Explorer Agent
Verifier: Arbiter Agent
Critic: Arbiter Agent
Reasoner: Arbiter Agent
Reflection: Yes (Iterative Explorer loop)
Orchestrator: Fixed 3-agent graph

Knowledge Sources: PubMed, Medical Textbooks, Wikipedia

Dataset: MedQA, BioASQ
Dataset Size: 5,000 questions

Task: Medical Question Answering and Clinical Option Selection

Evaluation Metrics: Diagnostic Accuracy, Citation Coverage, Retrieval Precision@k

Results: Achieved +8.4% higher accuracy on MedQA compared to standard single-pass RAG.

Strengths: Decouples query schema expansion from evidence retrieval and arbitration.

Limitations: Lacks Knowledge Graph integration; uses fixed 3-agent sequential workflow for every query without cost-aware adaptive routing.

Novelty: Decoupled 3-agent role structure for medical reasoning.

What MedAgentX Can Learn: Role specialization improves medical retrieval precision.

What MedAgentX Does Differently: MedAgentX introduces adaptive confidence-guided dynamic routing (ACMR) to prune unnecessary agent calls, and integrates structured Knowledge Graph traversal with vector search.

Potential Overlap With MedAgentX: High overlap in multi-agent medical RAG concept.

Threat to MedAgentX Novelty: High threat if MedAgentX merely claims "multi-agent medical RAG."

Evidence: Published paper in Findings of ACL 2026.
