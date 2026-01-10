"""
Graph Memory Instrument for AgentSmith
Allows agents to store and retrieve from FalkorDB graph

Usage in agent chat:
    exec(open('/a0/instruments/graph_memory.py').read())
    
    # Check connection
    health_check()
    
    # Store a memory
    remember("User prefers Python for scripting")
    
    # Recall memories
    recall("Python")
    
    # Direct Cypher query
    query("MATCH (n) RETURN n LIMIT 5")
    
    # Create nodes and relationships
    create_node("Person", {"name": "Steve", "role": "developer"})
    create_node("Project", {"name": "AgentSmith", "status": "active"})
"""

import os
import json
import time
import traceback
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if os.getenv("DEBUG") else logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("graph_memory")

# Try to import FalkorDB
try:
    from falkordb import FalkorDB
    FALKORDB_AVAILABLE = True
    log.info("FalkorDB module loaded")
except ImportError:
    FALKORDB_AVAILABLE = False
    log.error("FalkorDB not installed. Run: pip install falkordb")

# Global connection
_graph = None
_db = None


def connect() -> bool:
    """Connect to FalkorDB"""
    global _graph, _db
    
    if not FALKORDB_AVAILABLE:
        log.error("FalkorDB module not available")
        return False
    
    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", "16379"))
    password = os.getenv("FALKORDB_PASSWORD", "CH@NG3M3N()W!")
    graph_name = os.getenv("GRAPH_NAME", "agentsmith")
    
    log.info(f"Connecting to {host}:{port}/{graph_name}")
    
    try:
        _db = FalkorDB(
            host=host,
            port=port,
            password=password if password else None
        )
        _graph = _db.select_graph(graph_name)
        log.info(f"Connected to {host}:{port}/{graph_name}")
        return True
    except Exception as e:
        log.error(f"Connection failed: {e}")
        log.error(traceback.format_exc())
        return False


def health_check() -> Dict[str, Any]:
    """Check graph health and connection"""
    global _graph
    
    if _graph is None:
        if not connect():
            return {"status": "error", "message": "Not connected"}
    
    try:
        result = _graph.query("MATCH (n) RETURN count(n) as count")
        count = result.result_set[0][0] if result.result_set else 0
        return {
            "status": "ok",
            "connected": True,
            "node_count": count,
            "host": os.getenv("FALKORDB_HOST", "localhost"),
            "graph": os.getenv("GRAPH_NAME", "agentsmith")
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def query(cypher: str, params: Optional[Dict] = None) -> List[Dict]:
    """Execute a Cypher query"""
    global _graph
    
    if _graph is None:
        if not connect():
            return []
    
    try:
        result = _graph.query(cypher, params or {})
        return [dict(zip(result.header, row)) for row in result.result_set]
    except Exception as e:
        print(f"[!] Query error: {e}")
        return []


def create_node(label: str, properties: Dict[str, Any]) -> Optional[str]:
    """Create a node with given label and properties"""
    props_str = ", ".join(f"{k}: ${k}" for k in properties.keys())
    cypher = f"CREATE (n:{label} {{{props_str}}}) RETURN elementId(n) as id"
    result = query(cypher, properties)
    if result:
        print(f"[+] Created {label} node: {result[0]['id']}")
        return result[0]["id"]
    return None


def create_edge(from_id: str, to_id: str, rel_type: str, properties: Optional[Dict] = None) -> bool:
    """Create a relationship between two nodes"""
    props = properties or {}
    props_str = ", ".join(f"{k}: ${k}" for k in props.keys()) if props else ""
    rel_props = f" {{{props_str}}}" if props_str else ""
    
    cypher = f"""
    MATCH (a), (b) 
    WHERE elementId(a) = $from_id AND elementId(b) = $to_id
    CREATE (a)-[r:{rel_type}{rel_props}]->(b)
    RETURN type(r) as rel
    """
    params = {"from_id": from_id, "to_id": to_id, **props}
    result = query(cypher, params)
    if result:
        print(f"[+] Created {rel_type} relationship")
        return True
    return False


def remember(content: str, source: str = "agent", metadata: Optional[Dict] = None) -> Optional[str]:
    """Store a memory/episode in the graph"""
    props = {
        "content": content,
        "source": source,
        "timestamp": time.time(),
        "metadata": json.dumps(metadata or {})
    }
    node_id = create_node("Memory", props)
    if node_id:
        print(f"[+] Remembered: {content[:50]}...")
    return node_id


def recall(search_term: str, limit: int = 10) -> List[Dict]:
    """Recall memories matching a search term"""
    cypher = """
    MATCH (m:Memory)
    WHERE m.content CONTAINS $term
    RETURN m.content as content, m.timestamp as timestamp, m.source as source
    ORDER BY m.timestamp DESC
    LIMIT $limit
    """
    results = query(cypher, {"term": search_term, "limit": limit})
    if results:
        print(f"[+] Found {len(results)} memories")
        for r in results:
            print(f"  - {r['content'][:60]}...")
    else:
        print(f"[-] No memories found for '{search_term}'")
    return results


def add_entity(name: str, entity_type: str, properties: Optional[Dict] = None) -> Optional[str]:
    """Add or update an entity (person, project, concept, etc.)"""
    props = properties or {}
    props["name"] = name
    props["entity_type"] = entity_type
    props["updated_at"] = time.time()
    
    cypher = """
    MERGE (e:Entity {name: $name, entity_type: $entity_type})
    SET e += $props
    RETURN elementId(e) as id
    """
    result = query(cypher, {"name": name, "entity_type": entity_type, "props": props})
    if result:
        print(f"[+] Entity: {entity_type}/{name}")
        return result[0]["id"]
    return None


def link(from_name: str, to_name: str, relationship: str) -> bool:
    """Link two entities by name"""
    cypher = """
    MATCH (a:Entity {name: $from_name}), (b:Entity {name: $to_name})
    MERGE (a)-[r:""" + relationship + """]->(b)
    RETURN type(r) as rel
    """
    result = query(cypher, {"from_name": from_name, "to_name": to_name})
    if result:
        print(f"[+] Linked: {from_name} -[{relationship}]-> {to_name}")
        return True
    return False


def get_entity(name: str) -> Optional[Dict]:
    """Get an entity by name"""
    cypher = """
    MATCH (e:Entity {name: $name})
    RETURN e
    """
    result = query(cypher, {"name": name})
    return result[0] if result else None


def get_related(name: str, relationship: Optional[str] = None) -> List[Dict]:
    """Get entities related to a given entity"""
    rel_filter = f":{relationship}" if relationship else ""
    cypher = f"""
    MATCH (e:Entity {{name: $name}})-[r{rel_filter}]-(other:Entity)
    RETURN other.name as name, other.entity_type as type, type(r) as relationship
    """
    return query(cypher, {"name": name})


def show_graph(limit: int = 20) -> None:
    """Display current graph state"""
    nodes = query(f"MATCH (n) RETURN labels(n) as labels, count(*) as count")
    print("\n=== Graph State ===")
    print("Node counts:")
    for n in nodes:
        print(f"  {n['labels']}: {n['count']}")
    
    edges = query("MATCH ()-[r]->() RETURN type(r) as type, count(*) as count")
    print("Edge counts:")
    for e in edges:
        print(f"  {e['type']}: {e['count']}")


# Auto-connect on load
print("[AgentSmith Graph Memory]")
connect()
