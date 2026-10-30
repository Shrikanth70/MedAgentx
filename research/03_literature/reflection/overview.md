# Literature Overview: Reflection & Self-Correction Mechanisms

## 1. Domain Summary
Reflection and self-correction paradigms enable language models and agents to inspect intermediate outputs, detect logical inconsistencies or missing evidence, generate explicit self-critiques, and refine solutions iteratively before presenting a final output.

## 2. Key Foundational Papers

### Shinn et al. (2023) — *Reflexion: Language Agents with Verbal Reinforcement Learning*
- **Venue:** NeurIPS 2023
- **Core Mechanism:** Equips agents with dynamic memory and self-reflection capabilities. Evaluates intermediate actions, writes verbal self-reflections into memory, and uses them to guide subsequent trials.

### Madaan et al. (2023) — *Self-Refine: Iterative Refinement with Self-Feedback*
- **Venue:** NeurIPS 2023
- **Core Mechanism:** Performs iterative output refinement through a single LLM acting as Generator, Feedback Provider, and Refiner.
