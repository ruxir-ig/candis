import sys
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).parent.parent))

from src.loader import load_candidates
from src.honeypot_detector import is_honeypot

def test_honeypots():
    candidates_file = Path(__file__).parent.parent / "data/candidates.jsonl"
    print(f"Loading candidates from {candidates_file}...")
    
    candidates = list(load_candidates(candidates_file))
    print(f"Loaded {len(candidates)} candidates.")
    
    honeypots = [c for c in candidates if is_honeypot(c)]
    print(f"Found {len(honeypots)} honeypot candidates.")
    
    # Assert we found exactly 70 honeypots
    assert len(honeypots) == 70, f"Expected 70 honeypots, found {len(honeypots)}"
    print("Test passed! Exactly 70 honeypots identified.")
    
    # Print a few examples of honeypot candidates
    print("\nHoneypot Samples:")
    for h in honeypots[:5]:
        profile = h.get("profile", {})
        print(f"- {h['candidate_id']} ({profile.get('anonymized_name')}): YOE={profile.get('years_of_experience')}, Skills={len(h.get('skills', []))}")

if __name__ == "__main__":
    test_honeypots()
