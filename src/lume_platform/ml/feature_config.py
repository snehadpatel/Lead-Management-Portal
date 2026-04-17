"""Single source of truth for lead-scoring column names (training, API, Spark UDF)."""

# Richer but still pre-outcome signals (no post-conversion-only fields).
LEAD_NUMERIC_FEATURES: list[str] = [
    "totalvisits",
    "total_time_spent_on_website",
    "page_views_per_visit",
    "asymmetrique_activity_score",
    "asymmetrique_profile_score",
]

LEAD_CATEGORICAL_FEATURES: list[str] = [
    "lead_origin",
    "lead_source",
    "specialization",
    "what_is_your_current_occupation",
    "last_activity",
    "country",
    "lead_quality",
    "do_not_email",
    "do_not_call",
]
