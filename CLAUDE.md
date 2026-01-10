# AgentSmith: BMAD + Temporal Graph + Agent Zero

Read this entire file before doing anything.

## Architecture

```
MAX (claude.ai) ←→ MCP Server ←→ FalkorDB ←→ Agent Zero
     strategic        bridge      temporal      tactical
        ↓                          memory
  Analyst/Architect                  ↓
                             Episodes + Facts
                             Temporal validity
                             Contradiction detection
```

## Critical Concepts

### Split Orchestration
- **MAX (Opus)**: Strategic thinking via MCP. Cannot be called via API.
- **Agent Zero**: Tactical execution via OpenRouter.
- **MCP Server**: Bridge between MAX and graph.

### Temporal Memory
Everything has time:
- **Episodes**: Immutable events (what happened)
- **Facts**: Extracted knowledge with valid_from/valid_to/invalid_at
- **Contradictions**: Auto-detected when facts conflict

### BMAD Personas
Not config files - graph entities with history:

| Role | Name | Phase | Model | Purpose |
|------|------|-------|-------|---------|
| analyst | Mary | strategic | opus | Requirements, constraints |
| pm | John | strategic | opus | PRD, strategy |
| architect | Winston | strategic | opus | System design |
| dev | Alex | tactical | sonnet | Implementation |
| reviewer | Sam | tactical | gemini-flash | Code review |
| qa | Quinn | tactical | deepseek | Testing, security |
| po | Sarah | strategic | opus | Backlog, decisions |

## Bootstrap Sequence

1. **Graph Connection**
   ```python
   exec(open('/a0/instruments/graph_memory.py').read())
   health_check()
   ```

2. **Sync Codebase to Graph**
   ```python
   exec(open('/a0/instruments/knowledge.py').read())
   sync_all()  # Syncs files, instruments, tools, prompts
   ```

3. **Bootstrap BMAD**
   ```python
   exec(open('/a0/instruments/bmad.py').read())
   bootstrap()  # Creates personas in graph
   show_personas()
   ```

4. **Run Tasks**
   ```python
   # Full pipeline
   result = run_ensemble("Build a REST API for user auth")
   
   # Or phase by phase
   strategic = run_strategic("Build a REST API for user auth")
   tactical = run_tactical(strategic["spec"])
   ```

## Knowledge Graph

Everything is in the graph. BMAD agents navigate it to understand the codebase.

### Node Types
- **Project** - Workspace container
- **Agent** - Agent instances (primary, subagents)
- **Persona** - BMAD roles (Mary, John, Winston, etc.)
- **Tool** - Available tools
- **Instrument** - Custom scripts
- **File** - All tracked files
- **Directory** - Folder structure
- **Prompt** - System/agent prompts
- **Setting** - Configuration
- **Blueprint** - Specs/designs
- **Episode** - Memory events
- **Fact** - Extracted knowledge
- **Decision** - Recorded decisions

### Key Relationships
```
(Project)-[:CONTAINS]->(Agent|File|Blueprint)
(Agent)-[:USES]->(Tool|Instrument)
(Persona)-[:CONTROLS]->(Agent)
(Persona)-[:DECIDED]->(Decision)
(Directory)-[:CONTAINS]->(File|Directory)
(File)-[:IMPLEMENTS]->(Blueprint)
(File)-[:DEPENDS_ON]->(File)
(Blueprint)-[:EVOLVED_FROM]->(Blueprint)
```

### Knowledge Queries
```python
exec(open('/a0/instruments/knowledge.py').read())

# Search codebase
search("auth")

# Get project structure
structure()

# Find files for a blueprint
files_for("user_api")

# File dependencies
deps("/a0/python/helpers/graph.py")

# Agent capabilities
agent_capabilities("agent_001")

# List everything
show_files()
show_instruments()
show_tools()
show_agents()
```

## Graph Operations

### Episodes (Raw Memory)
```python
from python.helpers.temporal_graph import get_graph
g = get_graph()

# Record event
g.add_episode("User requested auth system", source="user", episode_type="request")

# Search
g.recall("auth")
```

### Facts (Extracted Knowledge)
```python
# Extract fact from episode
g.extract_fact(episode_id, subject="auth_system", predicate="uses", obj="JWT")

# Query valid facts
g.get_valid_facts(subject="auth_system")

# Invalidate when things change
g.invalidate_fact(fact_id, reason="Switched to session-based auth")
```

### Temporal Queries
```python
# What was true at a specific time?
g.temporal_query("auth_system", at_time=1704067200)

# Detect contradictions
contradictions = g.detect_contradictions()
```

### Blueprints (Versioned Specs)
```python
# Create
g.create_blueprint("user_api", "REST API for user management", priority=1)

# Evolve (creates version chain)
g.evolve_blueprint("user_api", changes="Added OAuth support", changed_by="Winston")

# History
g.get_blueprint_history("user_api")
```

## Environment Variables

All pre-configured in `railway.toml`:

- `API_KEY_OPENROUTER` - Included
- `FALKORDB_HOST` - shinkansen.proxy.rlwy.net
- `FALKORDB_PORT` - 33564
- `FALKORDB_PASSWORD` - Included
- `GRAPH_NAME` - agentsmith
- `DEBUG` - true

## Model Cost Strategy

All models via OpenRouter. Fully autonomous.

| Role | Model | Cost |
|------|-------|------|
| Strategic (Mary, John, Winston, Sarah) | Sonnet 4.5 | $$ |
| Dev (Alex) | Gemini 2.5 Flash | $ |
| Reviewer (Sam) | Deepseek | $ |
| QA (Quinn) | Llama 3.3 70B | Free |

Total per ensemble run: ~$0.05-0.15 depending on task complexity.

## File Structure

```
/a0/
├── instruments/
│   ├── bmad.py           # BMAD ensemble instrument
│   ├── knowledge.py      # Codebase sync to graph
│   └── graph_memory.py   # Basic graph operations
├── python/helpers/
│   ├── temporal_graph.py # Temporal graph + BMAD personas
│   ├── knowledge_graph.py # Full hierarchy graph
│   ├── graph.py          # Simple graph wrapper
│   └── config.py         # TOML config loader
├── config/
│   └── agentsmith.toml   # Configuration
├── prompts/              # Agent Zero prompts
├── webui/                # Web interface
└── CLAUDE.md             # This file
```

## Rules

1. **Graph is Truth** - If it's not in the graph, it doesn't exist
2. **Episodes are Immutable** - Never modify, only add new
3. **Facts have Time** - Always track validity
4. **Personas Accumulate** - Decisions build history
5. **Contradictions Surface** - Don't hide conflicts

## Common Tasks

### Add to Memory
```python
g.remember("Important fact about the project")
```

### Search Memory
```python
g.recall("project")
```

### Run BMAD on Feature
```python
result = run_ensemble("Add password reset to auth API")
print(result["final_implementation"])
```

### Check for Problems
```python
health = g.health_check()
print(f"Contradictions: {health['contradiction_count']}")
```

### See Persona History
```python
show_decisions("architect", limit=5)
```
