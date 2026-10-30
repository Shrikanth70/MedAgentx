Paper Title: Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity
Authors: Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, Jong C. Park
Year: 2024
Venue: NAACL 2024
DOI: 10.48550/arXiv.2403.14403
URL: https://arxiv.org/abs/2403.14403
Peer Reviewed: Yes
Citation Count: 145

Problem: Standard RAG retrieves context for all queries indiscriminately, incurring unnecessary computational overhead for simple queries and failing on complex multi-step queries.
Research Question: Can a query complexity classifier dynamically determine whether to retrieve, and how deeply to retrieve?

Architecture: Classifier-Guided Adaptive Routing
Models: T5-base Classifier + LLaMA-2 / GPT-3.5
Retrieval: Dense (FAISS) + Sparse (BM25)
RAG Type: Adaptive RAG
Agents: Single Classifier Router
Planner: No
Retriever: Standard Dense Retriever
Verifier: No
Critic: No
Reasoner: No
Reflection: No
Orchestrator: Complexity Classifier

Knowledge Sources: Wikipedia, Open-domain text corpora

Dataset: MuSiQue, 2WikiMultihopQA, StrategyQA
Dataset Size: 15,000 samples

Task: Open-Domain Question Answering

Evaluation Metrics: Accuracy, Token Efficiency, Latency, Retrieval Count

Results: Reduced retrieval calls by 35% while maintaining or exceeding full multi-hop RAG accuracy.

Strengths: Simple, effective query complexity classifier trained on automatically generated labels.

Limitations: General open domain only; static query classification before retrieval (does not evaluate evidence state post-retrieval).

Novelty: Pre-retrieval question complexity classification for dynamic RAG routing.

What MedAgentX Can Learn: Pre-retrieval complexity routing saves substantial computation.

What MedAgentX Does Differently: MedAgentX evaluates evidence sufficiency *during* and *after* retrieval, dynamically selecting specialized agents based on real-time evidence confidence rather than just initial query text.

Potential Overlap With MedAgentX: Medium overlap in "adaptive retrieval routing."

Threat to MedAgentX Novelty: Medium. MedAgentX must differentiate post-retrieval evidence confidence from pre-retrieval query complexity.

Evidence: Published in Proceedings of NAACL 2024.
