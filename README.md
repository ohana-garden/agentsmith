# AgentSmith

Agent Zero + BMAD Personas + Temporal FalkorDB for Railway deployment.

## What's Different

| Feature | Standard Agent Zero | AgentSmith |
|---------|---------------------|------------|
| Memory | Flat files | Temporal graph with valid_from/to |
| Personas | None | 7 BMAD roles (graph-resident) |
| Conflicts | Manual | Auto-detected contradictions |
| History | Current state only | Full decision trail |

## Quick Deploy

1. Fork this repo to your GitHub
2. Railway → New Project → Deploy from GitHub → select your fork

All env vars pre-configured in `railway.toml`. FalkorDB already online.

## BMAD Personas

| Role | Name | Phase | Model | Purpose |
|------|------|-------|-------|---------|
| analyst | Mary | strategic | Sonnet 4.5 | Requirements, constraints |
| pm | John | strategic | Sonnet 4.5 | PRD, strategy |
| architect | Winston | strategic | Sonnet 4.5 | System design |
| po | Sarah | strategic | Sonnet 4.5 | Backlog, decisions |
| dev | Alex | tactical | Gemini Flash | Implementation |
| reviewer | Sam | tactical | Deepseek | Code review |
| qa | Quinn | tactical | Llama 3.3 | Testing, security |

## Usage

```python
# In AgentSmith chat:

# Load instruments
exec(open('/a0/instruments/graph_memory.py').read())
exec(open('/a0/instruments/bmad.py').read())

# Bootstrap (first time)
bootstrap()

# Run full ensemble
result = run_ensemble("Build a REST API for user auth")

# Or phase by phase
strategic = run_strategic("Build a REST API")
tactical = run_tactical(strategic["spec"])

# Check for contradictions
health_check()
```

## Graph Memory

```python
# Store
remember("User prefers JWT for auth")

# Recall
recall("auth")

# Temporal query (what was true at time X?)
g.temporal_query("auth_system", at_time=1704067200)

# Detect contradictions
g.detect_contradictions()
```

## Knowledge Graph

Full codebase awareness - files, tools, instruments, agents all tracked.

```python
exec(open('/a0/instruments/knowledge.py').read())

# Sync everything to graph
sync_all()

# Search codebase
search("auth")

# Project structure
structure()

# File dependencies
deps("/a0/python/helpers/graph.py")

# See what's registered
show_files()
show_instruments()
show_tools()
show_agents()
```

### What Gets Tracked
- Projects and their contents
- Agent instances and subagents
- BMAD personas controlling agents
- Tools and instruments
- Files with hashes and dependencies
- Prompts and settings
- Blueprints with version chains

## Environment Variables

All pre-configured in `railway.toml`:

| Variable | Value |
|----------|-------|
| `API_KEY_OPENROUTER` | Included |
| `FALKORDB_HOST` | `shinkansen.proxy.rlwy.net` |
| `FALKORDB_PORT` | `33564` |
| `FALKORDB_PASSWORD` | Included |
| `GRAPH_NAME` | `agentsmith` |
| `DEBUG` | `true` |

## Model Costs (Fully Autonomous)

| Role | Model | Cost |
|------|-------|------|
| Strategic | Sonnet 4.5 | $$ |
| Dev | Gemini Flash | $ |
| Reviewer | Deepseek | $ |
| QA | Llama 3.3 70B | Free |

~$0.05-0.15 per ensemble run.

## Using Graph Memory

In AgentSmith chat:

```python
# Load the graph instrument
exec(open('/a0/instruments/graph_memory.py').read())

# Check connection
health_check()

# Store memories
remember("User prefers Python for scripting")
remember("Project deadline is next Friday")

# Recall memories
recall("Python")
recall("deadline")

# Create entities and relationships
add_entity("Steve", "Person", {"role": "developer"})
add_entity("AgentSmith", "Project", {"status": "active"})
link("Steve", "AgentSmith", "WORKS_ON")

# Query directly
query("MATCH (p:Person)-[:WORKS_ON]->(proj:Project) RETURN p.name, proj.name")
```

## Architecture

```
MAX (claude.ai) ←→ MCP Server ←→ FalkorDB ←→ Agent Zero
     strategic        bridge      temporal      tactical
        ↓                          memory
  Analyst/Architect                  ↓
                             Episodes → Facts
                             Temporal validity
                             Contradiction detection
```

## Development

```bash
# Clone
git clone https://github.com/YOUR_ORG/agentsmith.git
cd agentsmith

# Install deps
pip install -r requirements.txt

# Run locally
python run_ui.py --port 5000
```

## License

MIT - Based on Agent Zero by agent0ai
