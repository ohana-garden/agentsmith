#!/usr/bin/env python3
"""
AgentSmith Startup Check
Runs before Agent Zero to verify environment
"""

import os
import sys

print("="*60)
print("AGENTSMITH STARTUP CHECK")
print("="*60)

# Python
print(f"\n[Python] {sys.version.split()[0]}")

# Required env vars
print("\n[Environment]")
checks = {
    "API_KEY_OPENROUTER": os.getenv("API_KEY_OPENROUTER"),
    "FALKORDB_HOST": os.getenv("FALKORDB_HOST", "localhost"),
    "FALKORDB_PORT": os.getenv("FALKORDB_PORT", "16379"),
    "FALKORDB_PASSWORD": os.getenv("FALKORDB_PASSWORD"),
    "GRAPH_NAME": os.getenv("GRAPH_NAME", "agentsmith"),
}

errors = []
for k, v in checks.items():
    if v:
        masked = v[:4] + "..." if k in ["API_KEY_OPENROUTER", "FALKORDB_PASSWORD"] and len(v) > 4 else v
        print(f"  ✓ {k}={masked}")
    else:
        print(f"  ✗ {k} NOT SET")
        if k == "API_KEY_OPENROUTER":
            errors.append(f"{k} is required")

# Test imports
print("\n[Imports]")
imports = [
    ("flask", "Flask"),
    ("falkordb", "FalkorDB"),
    ("litellm", "litellm"),
    ("openai", "openai"),
    ("tiktoken", "tiktoken"),
    ("sentence_transformers", "sentence-transformers"),
]

for module, name in imports:
    try:
        __import__(module)
        print(f"  ✓ {name}")
    except ImportError as e:
        print(f"  ✗ {name}: {e}")
        # Not fatal - log but continue

# Test FalkorDB connection
print("\n[FalkorDB Connection]")
try:
    from falkordb import FalkorDB
    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", "16379"))
    password = os.getenv("FALKORDB_PASSWORD", "CH@NG3M3N()W!")
    
    print(f"  Connecting to {host}:{port}...")
    db = FalkorDB(host=host, port=port, password=password)
    graph = db.select_graph(os.getenv("GRAPH_NAME", "agentsmith"))
    result = graph.query("RETURN 1 as test")
    print(f"  ✓ Connected to FalkorDB")
except Exception as e:
    print(f"  ✗ Connection failed: {e}")
    # Not fatal - might just be starting up

# Summary
print("\n" + "="*60)
if errors:
    print("STARTUP ERRORS:")
    for e in errors:
        print(f"  • {e}")
    print("="*60)
    print("\nSet API_KEY_OPENROUTER in Railway variables and redeploy.")
    sys.exit(1)
else:
    print("Startup checks passed - launching Agent Zero")
    print("="*60)
