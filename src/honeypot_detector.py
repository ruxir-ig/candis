from typing import Dict, Any, List

def is_honeypot(candidate: Dict[str, Any]) -> bool:
    """
    Identify honeypot candidates based on logical anomalies in their profiles.
    Returns True if the candidate is classified as a honeypot, False otherwise.
    
    Honeypot criteria:
    1. YOE Mismatch: Stated years of experience differs significantly from the sum 
       of durations of all career history entries (threshold: > 1.5 years).
    2. Skill Duration Paradox: Expert-level skills listed with 0 duration_months.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    
    # 1. Check YOE Mismatch
    yoe = profile.get("years_of_experience", 0)
    total_months = sum(job.get("duration_months", 0) for job in career)
    calc_yoe = total_months / 12.0
    
    if abs(yoe - calc_yoe) > 1.5:
        return True
        
    # 2. Check Skill Duration Paradox
    expert_zero_dur = sum(
        1 for s in skills 
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    )
    if expert_zero_dur >= 1:
        return True
        
    return False

def filter_honeypots(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filters out honeypots from a list of candidates.
    """
    return [c for c in candidates if not is_honeypot(c)]
