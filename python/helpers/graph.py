"""
AgentSmith Graph Module
FalkorDB + Graphiti integration for persistent graph memory
"""

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import json

try:
    from falkordb import FalkorDB
    FALKORDB_AVAILABLE = True
except ImportError:
    FALKORDB_AVAILABLE = False

try:
    from graphiti_core import Graphiti
    from graphiti_core.nodes import EpisodeType
    GRAPHITI_AVAILABLE = True
except ImportError:
    GRAPHITI_AVAILABLE = False


@dataclass
class GraphConfig:
    """Graph database configuration"""
    host: str = "localhost"
    port: int = 16379
    password: str = ""
    graph_name: str = "agentsmith"
    graphiti_enabled: bool = True
    
    @classmethod
    def from_env(cls) -> "GraphConfig":
        """Load config from environment variables"""
        return cls(
            host=os.getenv("FALKORDB_HOST", "localhost"),
            port=int(os.getenv("FALKORDB_PORT", "16379")),
            password=os.getenv("FALKORDB_PASSWORD", "CH@NG3M3N()W!"),
            graph_name=os.getenv("GRAPH_NAME", "agentsmith"),
            graphiti_enabled=os.getenv("GRAPHITI_ENABLED", "true").lower() == "true"
        )


class AgentGraph:
    """
    Unified graph interface for AgentSmith.
    Combines FalkorDB for storage with Graphiti for temporal knowledge.
    """
    
    _instance: Optional["AgentGraph"] = None
    
    def __init__(self, config: Optional[GraphConfig] = None):
        self.config = config or GraphConfig.from_env()
        self._db: Optional[FalkorDB] = None
        self._graph = None
        self._graphiti: Optional[Graphiti] = None
        self._connected = False
    
    @classmethod
    def get_instance(cls, config: Optional[GraphConfig] = None) -> "AgentGraph":
        """Singleton pattern for graph access"""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance
    
    def connect(self) -> bool:
        """Establish connection to FalkorDB"""
        if not FALKORDB_AVAILABLE:
            print("[AgentGraph] FalkorDB not installed")
            return False
            
        try:
            self._db = FalkorDB(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password if self.config.password else None
            )
            self._graph = self._db.select_graph(self.config.graph_name)
            self._connected = True
            print(f"[AgentGraph] Connected to {self.config.host}:{self.config.port}/{self.config.graph_name}")
            return True
        except Exception as e:
            print(f"[AgentGraph] Connection failed: {e}")
            self._connected = False
            return False
    
    @property
    def is_connected(self) -> bool:
        return self._connected and self._graph is not None
    
    def query(self, cypher: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute a Cypher query"""
        if not self.is_connected:
            if not self.connect():
                return []
        
        try:
            result = self._graph.query(cypher, params or {})
            return [dict(zip(result.header, row)) for row in result.result_set]
        except Exception as e:
            print(f"[AgentGraph] Query error: {e}")
            return []
    
    def create_node(self, label: str, properties: Dict[str, Any]) -> Optional[str]:
        """Create a node and return its ID"""
        props_str = ", ".join(f"{k}: ${k}" for k in properties.keys())
        cypher = f"CREATE (n:{label} {{{props_str}}}) RETURN elementId(n) as id"
        result = self.query(cypher, properties)
        return result[0]["id"] if result else None
    
    def create_edge(self, from_id: str, to_id: str, rel_type: str, properties: Optional[Dict] = None) -> bool:
        """Create an edge between two nodes"""
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
        result = self.query(cypher, params)
        return len(result) > 0
    
    def find_nodes(self, label: str, filters: Optional[Dict] = None, limit: int = 100) -> List[Dict]:
        """Find nodes by label and optional filters"""
        where_clause = ""
        if filters:
            conditions = [f"n.{k} = ${k}" for k in filters.keys()]
            where_clause = "WHERE " + " AND ".join(conditions)
        
        cypher = f"MATCH (n:{label}) {where_clause} RETURN n LIMIT {limit}"
        return self.query(cypher, filters or {})
    
    def get_neighbors(self, node_id: str, rel_type: Optional[str] = None, direction: str = "both") -> List[Dict]:
        """Get neighboring nodes"""
        rel = f":{rel_type}" if rel_type else ""
        
        if direction == "out":
            pattern = f"(a)-[r{rel}]->(b)"
        elif direction == "in":
            pattern = f"(a)<-[r{rel}]-(b)"
        else:
            pattern = f"(a)-[r{rel}]-(b)"
        
        cypher = f"""
        MATCH {pattern}
        WHERE elementId(a) = $node_id
        RETURN b, type(r) as rel_type, r as rel_props
        """
        return self.query(cypher, {"node_id": node_id})
    
    # Graphiti-style temporal methods
    def add_episode(self, content: str, source: str = "user", metadata: Optional[Dict] = None) -> Optional[str]:
        """Add an episode (conversation turn, event, etc.) to the graph"""
        import time
        props = {
            "content": content,
            "source": source,
            "timestamp": time.time(),
            "metadata": json.dumps(metadata or {})
        }
        return self.create_node("Episode", props)
    
    def add_entity(self, name: str, entity_type: str, properties: Optional[Dict] = None) -> Optional[str]:
        """Add or update an entity in the graph"""
        props = properties or {}
        props["name"] = name
        props["entity_type"] = entity_type
        
        # Upsert pattern
        cypher = """
        MERGE (e:Entity {name: $name, entity_type: $entity_type})
        SET e += $props
        RETURN elementId(e) as id
        """
        result = self.query(cypher, {"name": name, "entity_type": entity_type, "props": props})
        return result[0]["id"] if result else None
    
    def link_episode_to_entity(self, episode_id: str, entity_id: str, relationship: str = "MENTIONS") -> bool:
        """Link an episode to an entity it references"""
        return self.create_edge(episode_id, entity_id, relationship)
    
    def get_entity_history(self, entity_name: str, limit: int = 50) -> List[Dict]:
        """Get temporal history of an entity"""
        cypher = """
        MATCH (ep:Episode)-[r]->(e:Entity {name: $name})
        RETURN ep.content as content, ep.timestamp as timestamp, type(r) as relationship
        ORDER BY ep.timestamp DESC
        LIMIT $limit
        """
        return self.query(cypher, {"name": entity_name, "limit": limit})
    
    def search_episodes(self, query: str, limit: int = 20) -> List[Dict]:
        """Search episodes by content (basic text match)"""
        cypher = """
        MATCH (ep:Episode)
        WHERE ep.content CONTAINS $query
        RETURN ep.content as content, ep.timestamp as timestamp, ep.source as source
        ORDER BY ep.timestamp DESC
        LIMIT $limit
        """
        return self.query(cypher, {"query": query, "limit": limit})
    
    def health_check(self) -> Dict[str, Any]:
        """Check graph connection health"""
        if not self.is_connected:
            connected = self.connect()
        else:
            connected = True
            
        result = {
            "connected": connected,
            "host": self.config.host,
            "port": self.config.port,
            "graph": self.config.graph_name
        }
        
        if connected:
            try:
                count = self.query("MATCH (n) RETURN count(n) as count")
                result["node_count"] = count[0]["count"] if count else 0
            except:
                result["node_count"] = "error"
        
        return result


# Convenience functions for instruments
def get_graph() -> AgentGraph:
    """Get the singleton graph instance"""
    return AgentGraph.get_instance()


def health_check() -> Dict[str, Any]:
    """Quick health check"""
    return get_graph().health_check()


def query(cypher: str, params: Optional[Dict] = None) -> List[Dict]:
    """Execute a Cypher query"""
    return get_graph().query(cypher, params)


def create_node(label: str, properties: Dict[str, Any]) -> Optional[str]:
    """Create a node"""
    return get_graph().create_node(label, properties)


def remember(content: str, source: str = "agent") -> Optional[str]:
    """Store something in graph memory"""
    return get_graph().add_episode(content, source)


def recall(query: str, limit: int = 10) -> List[Dict]:
    """Recall from graph memory"""
    return get_graph().search_episodes(query, limit)
