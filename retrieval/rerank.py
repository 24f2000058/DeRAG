import math

from blockchain.registry_client import verify_integrity

SCALE = 10
FLOOR_PENALTY = 50


def get_candidate_reputation(doc_hash: str) -> tuple[bool, int]:
    result = verify_integrity(doc_hash)
    return result["exists"], result["reputation"]


def normalize_reputation(candidates: list[dict]) -> list[dict]:
    existing_reputations = [c["reputation"] for c in candidates if c["exists"]]
    baseline = min(existing_reputations) if existing_reputations else 0
    floor = baseline - FLOOR_PENALTY

    result = []
    for candidate in candidates:
        effective_reputation = candidate["reputation"] if candidate["exists"] else floor
        normalized = 1 / (1 + math.exp(-effective_reputation / SCALE))
        result.append({**candidate, "normalized_reputation": normalized})
    return result


def hybrid_score(similarity: float, normalized_reputation: float, alpha: float = 0.7) -> float:
    return alpha * similarity + (1 - alpha) * normalized_reputation
