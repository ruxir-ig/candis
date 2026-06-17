"""Location fit (soft). JD prefers India, especially Pune/Noida/Hyderabad/
Mumbai/Delhi NCR/Bangalore, but doesn't hard-reject elsewhere. We penalize
rather than eliminate, and credit willingness to relocate.
"""

PREFERRED_CITIES = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "new delhi", "gurgaon",
    "gurugram", "bangalore", "bengaluru", "ncr", "delhi ncr",
}


def location_score(candidate: dict) -> float:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    country = (profile.get("country", "") or "").lower()
    location = (profile.get("location", "") or "").lower()

    if "india" in country:
        if any(city in location for city in PREFERRED_CITIES):
            return 1.0
        return 0.92  # in India, but not a named hub

    # Outside India: soft penalty, eased by relocation willingness.
    return 0.88 if signals.get("willing_to_relocate") else 0.78
