"""Knowledge Graph Construction Pipeline for MedAgentX."""

import sys
import os
import json
from typing import Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import structlog
from config.settings import settings
from src.knowledge.graph_store import GraphStoreClient

logger = structlog.get_logger()

# Gold-standard seed clinical Knowledge Graph triples
SEED_GRAPH_TRIPLES = [
    {
        "subject": "Acute Coronary Syndrome",
        "predicate": "HAS_SUBTYPE",
        "object": "STEMI",
        "category": "Cardiology",
    },
    {
        "subject": "Acute Coronary Syndrome",
        "predicate": "HAS_SUBTYPE",
        "object": "NSTEMI",
        "category": "Cardiology",
    },
    {
        "subject": "Acute Coronary Syndrome",
        "predicate": "TREATED_BY",
        "object": "Dual Antiplatelet Therapy",
        "category": "Cardiology",
    },
    {
        "subject": "Dual Antiplatelet Therapy",
        "predicate": "INCLUDES",
        "object": "Aspirin",
        "category": "Pharmacology",
    },
    {
        "subject": "Dual Antiplatelet Therapy",
        "predicate": "INCLUDES",
        "object": "Ticagrelor",
        "category": "Pharmacology",
    },
    {
        "subject": "Nitroglycerin",
        "predicate": "CONTRAINDICATED_WITH",
        "object": "Right Ventricular Infarction",
        "category": "Safety",
    },
    {
        "subject": "Heart Failure with Reduced Ejection Fraction",
        "predicate": "TREATED_BY",
        "object": "ARNI (Sacubitril/Valsartan)",
        "category": "Cardiology",
    },
    {
        "subject": "Heart Failure with Reduced Ejection Fraction",
        "predicate": "TREATED_BY",
        "object": "SGLT2 Inhibitor",
        "category": "Cardiology",
    },
    {
        "subject": "Community Acquired Pneumonia",
        "predicate": "HAS_SYMPTOM",
        "object": "Cough with Sputum",
        "category": "Pulmonology",
    },
    {
        "subject": "Community Acquired Pneumonia",
        "predicate": "TREATED_BY",
        "object": "Amoxicillin",
        "category": "Infectious Disease",
    },
]


def build_and_index_graph(
    output_file: str = "data/processed/graph_triples.json",
) -> dict[str, Any]:
    """Save knowledge graph triples and ingest into Neo4j database."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(SEED_GRAPH_TRIPLES, f, indent=2)

    logger.info("Saved Knowledge Graph triples file", count=len(SEED_GRAPH_TRIPLES), path=output_file)

    graph_client = GraphStoreClient()
    health = graph_client.health_check()

    if health.get("status") == "healthy":
        for triple in SEED_GRAPH_TRIPLES:
            cypher = """
            MERGE (s:MedicalEntity {name: $subj})
            MERGE (o:MedicalEntity {name: $obj})
            MERGE (s)-[r:RELATION {type: $pred, category: $cat}]->(o)
            """
            graph_client.execute_cypher(
                cypher,
                {
                    "subj": triple["subject"],
                    "obj": triple["object"],
                    "pred": triple["predicate"],
                    "cat": triple["category"],
                },
            )
        logger.info("Successfully populated Neo4j Knowledge Graph nodes and relationships")
    else:
        logger.info("Neo4j database offline; local graph triples cache saved for deterministic offline retrieval")

    return {"status": "success", "count": len(SEED_GRAPH_TRIPLES)}


if __name__ == "__main__":
    build_and_index_graph()
