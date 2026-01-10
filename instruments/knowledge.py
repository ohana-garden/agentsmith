"""
Knowledge Graph Instrument for AgentSmith
Full codebase awareness for BMAD agents

Usage in agent chat:
    exec(open('/a0/instruments/knowledge.py').read())
    
    # First time: sync everything to graph
    sync_all()
    
    # Search codebase
    search("auth")
    
    # Get project structure
    structure()
    
    # Find files for a blueprint
    files_for("user_api")
    
    # Check what an agent can do
    agent_capabilities("agent_001")
    
    # Full health check
    health()
    
    # Debug mode
    debug()
"""

import os
import traceback
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if os.getenv("DEBUG") else logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("knowledge")

try:
    from python.helpers.knowledge_graph import (
        KnowledgeGraph, get_knowledge_graph, sync_all as _sync_all
    )
    KG_AVAILABLE = True
    log.info("Knowledge graph module loaded")
except ImportError as e:
    KG_AVAILABLE = False
    log.error(f"Knowledge graph not available: {e}")


# Global instance
_kg = None

def _get_kg() -> Optional[KnowledgeGraph]:
    global _kg
    if not KG_AVAILABLE:
        log.error("Knowledge graph module not loaded")
        return None
    if _kg is None:
        log.debug("Initializing knowledge graph")
        _kg = get_knowledge_graph()
    return _kg


# === SYNC OPERATIONS ===

def sync_all(base_path: str = "/a0") -> Dict[str, Any]:
    """
    Sync entire AgentSmith deployment to graph.
    Run this first time or after major changes.
    """
    kg = _get_kg()
    if not kg:
        return {"error": "Knowledge graph not available"}
    
    result = kg.full_sync(base_path)
    print(f"\n[+] Sync complete:")
    print(f"    Directories: {result['directories']}")
    print(f"    Files: {result['files']}")
    print(f"    Instruments: {result['instruments']}")
    print(f"    Prompts: {result['prompts']}")
    print(f"    Tools: {result['tools']}")
    return result


def sync_dir(path: str) -> Dict[str, int]:
    """Sync a specific directory"""
    kg = _get_kg()
    if not kg:
        return {"error": "Knowledge graph not available"}
    return kg.sync_directory(path)


# === SEARCH & QUERY ===

def search(query: str, node_type: str = "File") -> List[Dict]:
    """
    Search the codebase.
    node_type: File, Instrument, Tool, Prompt, Blueprint
    """
    kg = _get_kg()
    if not kg:
        return []
    
    results = kg.search_codebase(query, node_type)
    print(f"\n[Search: {query}] Found {len(results)} results:")
    for r in results[:10]:
        name = r.get("name") or r.get("path", "?").split("/")[-1]
        desc = r.get("description", "")[:50]
        print(f"  • {name}: {desc}")
    return results


def files_for(blueprint_name: str) -> List[Dict]:
    """Find all files implementing a blueprint"""
    kg = _get_kg()
    if not kg:
        return []
    
    results = kg.find_files_for_blueprint(blueprint_name)
    print(f"\n[Files for '{blueprint_name}'] Found {len(results)}:")
    for r in results:
        print(f"  • {r['path']} ({r['type']})")
    return results


def deps(file_path: str, depth: int = 3) -> List[Dict]:
    """Get file dependencies"""
    kg = _get_kg()
    if not kg:
        return []
    
    results = kg.get_file_dependencies(file_path, depth)
    print(f"\n[Dependencies for {file_path}]")
    for r in results:
        indent = "  " * r["depth"]
        print(f"{indent}→ {r['path']}")
    return results


# === PROJECT & STRUCTURE ===

def structure(project_name: str = "agentsmith") -> Dict[str, Any]:
    """Get project structure overview"""
    kg = _get_kg()
    if not kg:
        return {}
    
    result = kg.get_project_structure(project_name)
    
    print(f"\n[Project: {project_name}]")
    print(f"  Directories: {len(result['directories'])}")
    print(f"  File types:")
    for ft, count in result['file_types'].items():
        print(f"    • {ft}: {count}")
    print(f"  Blueprints: {len(result['blueprints'])}")
    for b in result['blueprints']:
        print(f"    • {b['name']} ({b.get('status', 'unknown')})")
    
    return result


# === AGENTS & CAPABILITIES ===

def register_agent(agent_id: str, agent_type: str = "primary",
                   project: str = "agentsmith") -> bool:
    """Register an agent instance"""
    kg = _get_kg()
    if not kg:
        return False
    
    result = kg.register_agent(agent_id, agent_type, project=project)
    if result:
        print(f"[+] Registered agent: {agent_id} ({agent_type})")
    return result is not None


def spawn_agent(parent_id: str, child_id: str, purpose: str = "") -> bool:
    """Record subagent spawn"""
    kg = _get_kg()
    if not kg:
        return False
    
    result = kg.spawn_subagent(parent_id, child_id, purpose)
    if result:
        print(f"[+] Spawned: {parent_id} → {child_id}")
    return result


def agent_capabilities(agent_id: str) -> Dict[str, List]:
    """Get agent's available tools and instruments"""
    kg = _get_kg()
    if not kg:
        return {}
    
    result = kg.find_agent_capabilities(agent_id)
    
    print(f"\n[Agent {agent_id} Capabilities]")
    print(f"  Tools ({len(result['tools'])}):")
    for t in result['tools']:
        print(f"    • {t['name']}: {t.get('description', '')[:40]}")
    print(f"  Instruments ({len(result['instruments'])}):")
    for i in result['instruments']:
        print(f"    • {i['name']}: {i.get('description', '')[:40]}")
    
    return result


def link_persona(persona_name: str, agent_id: str) -> bool:
    """Link a BMAD persona to control an agent"""
    kg = _get_kg()
    if not kg:
        return False
    
    result = kg.link_persona_to_agent(persona_name, agent_id)
    if result:
        print(f"[+] {persona_name} now controls {agent_id}")
    return result


# === FILES & REGISTRATION ===

def register_file(path: str, file_type: str = "auto",
                  description: str = "", blueprint: str = None) -> bool:
    """Register a file in the knowledge graph"""
    kg = _get_kg()
    if not kg:
        return False
    
    if file_type == "auto":
        ext = path.split(".")[-1] if "." in path else "unknown"
        file_type = ext
    
    result = kg.register_file(path, file_type, description, blueprint)
    if result:
        print(f"[+] Registered: {path}")
    return result is not None


def add_dependency(from_path: str, to_path: str, dep_type: str = "imports") -> bool:
    """Record file dependency"""
    kg = _get_kg()
    if not kg:
        return False
    
    result = kg.add_file_dependency(from_path, to_path, dep_type)
    if result:
        print(f"[+] {from_path} → {to_path} ({dep_type})")
    return result


# === BLUEPRINTS ===

def create_blueprint(name: str, description: str, priority: int = 0) -> bool:
    """Create a new blueprint"""
    kg = _get_kg()
    if not kg:
        return False
    
    from python.helpers.temporal_graph import get_graph
    g = get_graph()
    result = g.create_blueprint(name, description, priority=priority)
    if result:
        print(f"[+] Blueprint created: {name}")
    return result is not None


def link_file_to_blueprint(file_path: str, blueprint_name: str) -> bool:
    """Link a file as implementing a blueprint"""
    kg = _get_kg()
    if not kg:
        return False
    
    # Just re-register with blueprint link
    return register_file(file_path, blueprint=blueprint_name)


# === SETTINGS ===

def set_config(key: str, value: Any, scope: str = "global") -> bool:
    """Set a configuration value"""
    kg = _get_kg()
    if not kg:
        return False
    
    result = kg.set_setting(key, value, scope)
    if result:
        print(f"[+] Set {scope}/{key}")
    return result is not None


def get_config(key: str, scope: str = "global") -> Any:
    """Get a configuration value"""
    kg = _get_kg()
    if not kg:
        return None
    return kg.get_setting(key, scope)


# === HEALTH ===

def health() -> Dict[str, Any]:
    """Full health check"""
    kg = _get_kg()
    if not kg:
        return {"error": "Knowledge graph not available"}
    
    result = kg.health_check()
    
    print("\n[Knowledge Graph Health]")
    print(f"  Status: {result['status']}")
    print(f"  Synced: {result['synced']}")
    print("  Nodes:")
    for ntype, count in result['nodes'].items():
        print(f"    • {ntype}: {count}")
    print("  Edges:")
    for etype, count in list(result['edges'].items())[:5]:
        print(f"    • {etype}: {count}")
    
    return result


# === SHOW COMMANDS ===

def show_files(path_pattern: str = "", limit: int = 20) -> None:
    """List registered files"""
    kg = _get_kg()
    if not kg:
        return
    
    results = kg.g.query("""
        MATCH (f:File)
        WHERE f.path CONTAINS $pattern
        RETURN f.path as path, f.file_type as type
        ORDER BY f.path
        LIMIT $limit
    """, {"pattern": path_pattern, "limit": limit})
    
    print(f"\n[Files matching '{path_pattern}']")
    for r in results:
        print(f"  • {r['path']} ({r['type']})")


def show_instruments() -> None:
    """List all instruments"""
    kg = _get_kg()
    if not kg:
        return
    
    results = kg.g.query("""
        MATCH (i:Instrument)
        RETURN i.name as name, i.path as path, i.description as desc
        ORDER BY i.name
    """)
    
    print("\n[Instruments]")
    for r in results:
        desc = r.get('desc', '')[:40] if r.get('desc') else ''
        print(f"  • {r['name']}: {desc}")


def show_tools() -> None:
    """List all tools"""
    kg = _get_kg()
    if not kg:
        return
    
    results = kg.g.query("""
        MATCH (t:Tool)
        RETURN t.name as name, t.tool_type as type, t.description as desc
        ORDER BY t.name
    """)
    
    print("\n[Tools]")
    for r in results:
        desc = r.get('desc', '')[:40] if r.get('desc') else ''
        print(f"  • {r['name']} ({r['type']}): {desc}")


def show_agents() -> None:
    """List all registered agents"""
    kg = _get_kg()
    if not kg:
        return
    
    results = kg.g.query("""
        MATCH (a:Agent)
        OPTIONAL MATCH (p:Persona)-[:CONTROLS]->(a)
        RETURN a.agent_id as id, a.agent_type as type, 
               a.status as status, p.name as persona
        ORDER BY a.registered_at DESC
    """)
    
    print("\n[Agents]")
    for r in results:
        persona = f" (controlled by {r['persona']})" if r.get('persona') else ""
        print(f"  • {r['id']} ({r['type']}) - {r['status']}{persona}")


def debug() -> Dict[str, Any]:
    """Full system debug info"""
    import sys
    
    info = {
        "python_version": sys.version,
        "env": {
            "DEBUG": os.getenv("DEBUG", "not set"),
            "API_KEY_OPENROUTER": "set" if os.getenv("API_KEY_OPENROUTER") else "NOT SET",
            "FALKORDB_HOST": os.getenv("FALKORDB_HOST", "not set"),
            "FALKORDB_PORT": os.getenv("FALKORDB_PORT", "not set"),
            "FALKORDB_PASSWORD": "set" if os.getenv("FALKORDB_PASSWORD") else "using default",
            "GRAPH_NAME": os.getenv("GRAPH_NAME", "not set"),
        },
        "modules": {
            "knowledge_graph": KG_AVAILABLE,
        },
        "graph_connected": False,
        "counts": {}
    }
    
    # Check graph connection
    kg = _get_kg()
    if kg and kg.g:
        try:
            # Test query
            result = kg.g.query("RETURN 1 as test")
            info["graph_connected"] = True
            
            # Get node counts
            counts = kg.g.query("""
                MATCH (n)
                RETURN labels(n)[0] as label, count(n) as count
            """)
            info["counts"] = {r["label"]: r["count"] for r in counts}
        except Exception as e:
            info["graph_error"] = str(e)
            log.error(f"Debug graph query failed: {e}")
            log.error(traceback.format_exc())
    
    # Print summary
    print("\n" + "="*60)
    print("AGENTSMITH DEBUG INFO")
    print("="*60)
    print(f"\nPython: {info['python_version'].split()[0]}")
    print(f"\n[Environment]")
    for k, v in info["env"].items():
        status = "✓" if v not in ["not set", "NOT SET"] else "✗"
        print(f"  {status} {k}: {v}")
    
    print(f"\n[Modules]")
    for k, v in info["modules"].items():
        status = "✓" if v else "✗"
        print(f"  {status} {k}")
    
    print(f"\n[Graph Connection]")
    if info["graph_connected"]:
        print("  ✓ Connected")
        if info["counts"]:
            print(f"\n[Node Counts]")
            for label, count in sorted(info["counts"].items()):
                print(f"  • {label}: {count}")
    else:
        print("  ✗ Not connected")
        if "graph_error" in info:
            print(f"  Error: {info['graph_error']}")
    
    print("="*60)
    return info


# Auto-announce
log.info("Knowledge Graph Instrument Loaded")
print("[Knowledge Graph Instrument Loaded]")
print("Commands: sync_all(), search(query), structure(), health(), debug()")
print("          show_files(), show_instruments(), show_tools(), show_agents()")
