# 12 Reproducibility Specification

To ensure another researcher can independently reproduce every experimental finding:

## Hardware & Environment Lock
- **Python Version:** `3.11.8`
- **PyTorch Version:** `2.2.1+cu121`
- **LangChain / LangGraph:** `langgraph==0.0.26`
- **Vector DB:** Qdrant Docker `v1.8.0`
- **Graph DB:** Neo4j Docker `v5.17.0` (APOC plugin enabled)
- **Random Seeds:** Python `42`, NumPy `42`, PyTorch `42`
- **Model Parameters:** `temperature=0.0`, `top_p=1.0`, `max_tokens=1024`
- **Retrieval Parameters:** `vector_top_k=5`, `graph_max_depth=2`
- **Confidence Thresholds:** `High=0.80`, `Moderate=0.50`, `Low=0.50`
