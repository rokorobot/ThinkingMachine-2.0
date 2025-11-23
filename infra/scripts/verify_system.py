import requests
import time
import json
import sys

BASE_URL = "http://localhost:8080"

def log(msg, color="white"):
    colors = {
        "green": "\033[92m",
        "red": "\033[91m",
        "blue": "\033[94m",
        "white": "\033[0m",
        "yellow": "\033[93m"
    }
    print(f"{colors.get(color, colors['white'])}{msg}{colors['white']}")

def check_health():
    log("[-] Checking System Health...", "blue")
    try:
        res = requests.get(f"{BASE_URL}/health")
        if res.status_code == 200:
            log("[+] System is Healthy!", "green")
            return True
        else:
            log(f"[!] Health Check Failed: {res.status_code}", "red")
            return False
    except Exception as e:
        log(f"[!] Connection Failed: {e}", "red")
        return False

def test_memory():
    log("\n[-] Testing Long-Term Memory...", "blue")
    user_id = "test_user_001"
    
    # 1. Store a memory
    log(f"    Sending request with memory note for user '{user_id}'...", "white")
    payload_store = {
        "input_text": "Hi, I am testing the memory system.",
        "user_external_id": user_id,
        "memory_note": "The secret code is BLUE_ORION."
    }
    try:
        res = requests.post(f"{BASE_URL}/task", json=payload_store)
        res.raise_for_status()
        log("    [+] Memory storage request successful.", "green")
        # print(f"    Response: {res.json()['output_text']}")
    except Exception as e:
        log(f"    [!] Failed to store memory: {e}", "red")
        return

    # Wait a bit for async processing if any (though currently synchronous in core_agent)
    time.sleep(1)

    # 2. Recall memory
    log(f"    Sending request to recall memory...", "white")
    payload_recall = {
        "input_text": "What is the secret code I told you?",
        "user_external_id": user_id
    }
    try:
        res = requests.post(f"{BASE_URL}/task", json=payload_recall)
        res.raise_for_status()
        response_text = res.json()['output_text']
        log(f"    Response: {response_text}", "yellow")
        
        if "BLUE_ORION" in response_text or "BLUE_ORION" in str(res.content):
            log("    [+] Memory Recall Successful! (Found 'BLUE_ORION')", "green")
        else:
            log("    [?] Memory Recall Uncertain (Did not find 'BLUE_ORION' explicitly). Check logs.", "yellow")
            
    except Exception as e:
        log(f"    [!] Failed to recall memory: {e}", "red")

def test_game_theory():
    log("\n[-] Testing Game Theory Admin API...", "blue")
    
    # 1. Preview
    log("    Fetching Game Theory Equilibrium Preview...", "white")
    try:
        res = requests.get(f"{BASE_URL}/admin/game-theory/preview?domain=medical&hours=24")
        res.raise_for_status()
        data = res.json()
        log(f"    [+] Preview Successful.", "green")
        log(f"        Strategy: {data['chosen_strategy']}", "yellow")
        log(f"        Metrics: {data['metrics']}", "white")
    except Exception as e:
        log(f"    [!] Failed to fetch preview: {e}", "red")

def main():
    log("=== Thinking Machine Verification Script ===\n", "blue")
    
    if not check_health():
        log("\n[!] Aborting. Please ensure docker-compose is running.", "red")
        sys.exit(1)
        
    test_memory()
    test_game_theory()
    
    log("\n=== Verification Complete ===", "blue")

if __name__ == "__main__":
    main()
