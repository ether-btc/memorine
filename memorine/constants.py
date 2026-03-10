"""
Memorine constants — tunable thresholds in one place.
"""

# ── Cortex ───────────────────────────────────────────────────────────

# Jaccard similarity: 0.5-0.95 = contradiction, >=0.95 = duplicate
CONTRADICTION_SIMILARITY_MIN = 0.5
DUPLICATE_SIMILARITY_MIN = 0.95
SUPERSEDE_CONFIDENCE_MIN = 0.8

WEIGHT_MIN = 0.1
WEIGHT_MAX = 10.0
WEIGHT_DEFAULT = 1.0

SEMANTIC_RELEVANCE_FLOOR = 0.55
# How much semantic score vs effective weight matters in blended ranking
SEMANTIC_BLEND_WEIGHT = 0.6
RECALL_REINFORCE_BOOST = 0.05

# ── Amygdala ─────────────────────────────────────────────────────────

SECONDS_PER_DAY = 86400
MAX_STABILITY_ACCESS_COUNT = 20
STABILITY_PER_ACCESS = 1.5
MIN_RETENTION = 0.01
ERROR_IMPORTANCE_MULTIPLIER = 2.5
REINFORCE_WEIGHT_CAP = 5.0
REINFORCE_BOOST = 0.1
WEAKEN_PENALTY = 0.2
CLEANUP_THRESHOLD = 0.05
CLEANUP_BATCH_SIZE = 500
PROFILE_WEIGHT_FLOOR = 0.1

# ── Cerebellum ───────────────────────────────────────────────────────

AUTO_OPTIMIZE_MIN_RUNS = 5
STEP_OPTIMIZE_MIN_RUNS = 3
STEP_SKIP_FAILURE_RATE = 0.7

# ── Embeddings ───────────────────────────────────────────────────────

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSIONS = 384
L2_DISTANCE_FALLBACK = 2.0
