import sys
from pathlib import Path
from datetime import datetime
from collections import Counter

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.loader import load_candidates

def analyze_detailed():
    candidates_file = Path(__file__).parent.parent / "data/candidates.jsonl"
    
    company_starts = {}
    print("Collecting company start dates...")
    for c in load_candidates(candidates_file):
        for job in c.get('career_history', []):
            comp = job.get('company')
            start_s = job.get('start_date')
            if comp and start_s:
                try:
                    start_dt = datetime.strptime(start_s, "%Y-%m-%d")
                    if comp not in company_starts:
                        company_starts[comp] = []
                    company_starts[comp].append(start_dt)
                except Exception:
                    pass

    # Print company start date statistics
    print("\nCompany start date stats:")
    company_founded_est = {}
    for comp, dates in sorted(company_starts.items()):
        dates.sort()
        # Let's see if there's a huge gap or if we can find a clustering of start dates.
        # E.g., maybe most people start after a certain date, but a few start way before.
        min_date = dates[0]
        p1 = dates[int(len(dates)*0.01)] # 1st percentile
        p5 = dates[int(len(dates)*0.05)] # 5th percentile
        print(f"Company: {comp:<30} | Min: {min_date.strftime('%Y-%m-%d')} | 1%: {p1.strftime('%Y-%m-%d')} | 5%: {p5.strftime('%Y-%m-%d')} | Total: {len(dates)}")
        # Let's estimate foundation as the 1st percentile or 5th percentile to avoid being fooled by honeypots
        company_founded_est[comp] = p1

    print("\nScanning for specific honeypots...")
    
    honeypot_candidates = []
    
    # Let's track different potential honeypot flags
    flag_counts = Counter()
    
    for c in load_candidates(candidates_file):
        cid = c['candidate_id']
        profile = c.get('profile', {})
        career = c.get('career_history', [])
        skills = c.get('skills', [])
        signals = c.get('redrob_signals', {})
        
        c_flags = []
        
        # Flag 1: Expert proficiency in 10 skills with 0 years (0 duration_months) used
        # Let's count expert skills with duration_months == 0
        expert_zero_dur = sum(1 for s in skills if s.get('proficiency') == 'expert' and s.get('duration_months', 0) == 0)
        if expert_zero_dur >= 10:
            c_flags.append(f"expert_10_skills_0_dur (count={expert_zero_dur})")
            
        # Let's see if there are other skill thresholds
        expert_or_adv_zero_dur = sum(1 for s in skills if s.get('proficiency') in ['expert', 'advanced'] and s.get('duration_months', 0) == 0)
        if expert_or_adv_zero_dur >= 10:
            c_flags.append(f"expert_or_adv_10_skills_0_dur (count={expert_or_adv_zero_dur})")

        # Flag 2: Company founded paradox
        # "8 years of experience at a company founded 3 years ago"
        # Let's check if the candidate worked at a company before its estimated founding date (p1 or p5)
        for job in career:
            comp = job.get('company')
            start_s = job.get('start_date')
            dur_m = job.get('duration_months', 0)
            if comp and start_s and dur_m:
                try:
                    start_dt = datetime.strptime(start_s, "%Y-%m-%d")
                    # If start_dt is way before the 1st percentile of all start dates for this company, AND the duration is long
                    est_founded = company_founded_est[comp]
                    # If candidate started > 2 years before the 1st percentile (to be safe and avoid normal variance)
                    if (est_founded - start_dt).days > 365 * 2:
                        c_flags.append(f"worked_at_company_before_foundation ({comp}: started {start_s}, company est founded {est_founded.strftime('%Y-%m-%d')})")
                except Exception:
                    pass

        # Let's also check if there is an impossible duration in general:
        # e.g., duration_months is way longer than the difference between start_date and end_date
        # or duration_months is positive but start_date and end_date are the same, etc.
        for idx, job in enumerate(career):
            start_s = job.get('start_date')
            end_s = job.get('end_date')
            dur_m = job.get('duration_months')
            if start_s and end_s and dur_m:
                try:
                    start_dt = datetime.strptime(start_s, "%Y-%m-%d")
                    end_dt = datetime.strptime(end_s, "%Y-%m-%d")
                    calc_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
                    # A huge discrepancy (e.g. listed 96 months (8 years) but calculated 36 months (3 years))
                    if dur_m > calc_months + 12:
                        c_flags.append(f"job_{idx}_dur_longer_than_dates (listed={dur_m}, calculated={calc_months})")
                except Exception:
                    pass

        # Flag 3: Signup vs last active date?
        # But wait, why did 7561 candidates have signup_date > last_active_date?
        # Let's see if that's just a general noise pattern or something.
        
        if c_flags:
            for f in c_flags:
                flag_counts[f.split(' ')[0]] += 1
            honeypot_candidates.append({
                'candidate_id': cid,
                'name': profile.get('anonymized_name'),
                'flags': c_flags
            })

    print("\nHoneypot candidate flag counts:")
    for flag, cnt in flag_counts.items():
        print(f"  {flag}: {cnt}")
        
    print(f"\nTotal potential honeypots found: {len(honeypot_candidates)}")
    print("\nSamples:")
    for hc in honeypot_candidates[:20]:
        print(f"  - {hc['candidate_id']} ({hc['name']}): {hc['flags']}")

if __name__ == "__main__":
    analyze_detailed()
