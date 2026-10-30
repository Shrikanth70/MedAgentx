# Literature Overview: Multi-Agent LLM Systems

## 1. Domain Summary
Multi-agent LLM systems leverage multiple specialized agent roles (e.g., Planner, Executor, Critic, Verifier) coordinating through shared communication protocols or central orchestrators to tackle complex tasks beyond the capability of single LLM prompts.

## 2. Key Foundational Frameworks

### Hong et al. (2024) — *MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework*
- **Venue:** ICLR 2024
- **Key Insight:** Encodes Standard Operating Procedures (SOPs) into multi-agent workflows to streamline agent interaction and reduce trivial conversational loops.

### Wu et al. (2023) — *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation*
- **Venue:** arXiv:2308.08155
- **Key Insight:** Flexible conversation-driven multi-agent framework supporting customized agent roles and human-in-the-loop interactions.

### Liu et al. (2024) — *Dynamic LLM-Agent Network (DyLAN)*
- **Venue:** ICLR 2024
- **Key Insight:** Selectively activates agents based on intermediate contribution scores, demonstrating that fixed multi-agent networks incur unnecessary latency and token cost.
- **Direct Link to MedAgentX:** Supports MedAgentX's core premise that static multi-agent pipelines are inefficient and adaptive selection is superior.
