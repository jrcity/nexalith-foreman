import sys
import time
import subprocess
from pathlib import Path

# Add agent path so we can import orchestrator
sys.path.insert(0, str(Path("/home/redemption/codebase/ai/nexalith-foreman/agent")))
from orchestrator import run_agent

QUERIES = [
    # Edge case 1: Requires two parallel reads, and complex reasoning across them
    "Find any new hire starting within 14 days and check if there is any CMS draft published under their ID. Just give me a yes or no.",
    
    # Edge case 2: Requires sequential reads and writes, testing hallucination guards
    "Please look up the customer 'Acme Corp', get their CRM ID, and then log a sales interaction saying 'Followed up about Q3 roadmap'. Do not log it if you can't find them.",
    
    # Edge case 3: Stressing context and instruction following (multi-turn hallucination trap)
    "Can you update the lead status for CRM ID 'L-9999' to 'closed_won' and onboard an employee named 'John Doe' (jdoe@example.com) as a 'DevOps Engineer' in 'Engineering'? I approve these actions."
]

def get_temps():
    try:
        output = subprocess.check_output(["sensors"], encoding="utf-8")
        # Extract a quick summary of core temps
        temps = []
        for line in output.split('\n'):
            if "Core" in line or "Package" in line or "Tctl" in line:
                temps.append(line.strip())
        return " | ".join(temps[:3]) if temps else "No temp data"
    except Exception:
        return "Sensors unavailable"

def main():
    print(f"--- STARTING EDGE TEST RUN ---")
    print(f"Initial Temps: {get_temps()}")
    
    start_total = time.time()
    
    for i, q in enumerate(QUERIES, 1):
        print(f"\n[Test {i}/3] Query: {q}")
        start_q = time.time()
        
        try:
            response = run_agent(q, max_tool_rounds=6)
            duration = time.time() - start_q
            print(f"Response: {response}")
            print(f"Time Taken: {duration:.2f}s")
            print(f"Current Temps: {get_temps()}")
        except Exception as e:
            print(f"Error during agent run: {e}")
            
    print(f"\n--- TEST RUN COMPLETE ---")
    print(f"Total Time: {time.time() - start_total:.2f}s")

if __name__ == "__main__":
    main()
