# 14 Rejected Ideas Log

## Rejected Idea 1: Open Web Search Fallback for Low Confidence
- **Reason for Rejection:** Fallback to open web search engines (like CRAG 2024) introduces unvetted, inaccurate medical blog content, violating medical safety protocols.
- **Replacement:** Use structured Knowledge Graph traversal (Neo4j) and verified clinical guideline repositories (NICE/WHO) for fallback retrieval.

## Rejected Idea 2: Unconstrained Multi-Agent Chat Loops
- **Reason for Rejection:** Open-ended group chat conversations (like AutoGen 2023) lead to infinite agent loops, context drift, and high token costs.
- **Replacement:** Graph-based state machine (LangGraph) with strict `max_iterations=2` limits and explicit mathematical confidence stopping conditions ($C_{total} \ge 0.80$).
