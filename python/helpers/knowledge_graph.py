"""
AgentSmith Knowledge Graph
Full hierarchy: Projects, Agents, Tools, Instruments, Files, Settings

Everything lives in the graph. If it's not here, agents don't know about it.
"""

import os
import hashlib
import json
import time
import traceback
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if os.getenv("DEBUG") else logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("knowledge_graph")

try:
    from python.helpers.temporal_graph import TemporalGraph, get_graph
    GRAPH_AVAILABLE = True
    log.info("Temporal graph available")
except ImportError:
    GRAPH_AVAILABLE = False
    log.error("Temporal graph not available")


# ============================================================================
# SCHEMA DEFINITIONS
# ============================================================================

SCHEMA = """
// Node Types
// ----------
// Project      - Workspace/project container
// Agent        - Agent instance (Agent Zero, subagents)
// Persona      - BMAD role (Mary, John, Winston, etc.)
// Tool         - Available tool/capability
// Instrument   - Custom script in /instruments
// File         - Any file in the system
// Directory    - Folder structure
// Prompt       - System/agent prompts
// Setting      - Configuration values
// Blueprint    - Spec/design document
// Episode      - Memory event
// Fact         - Extracted knowledge
// Decision     - Recorded decision

// Edge Types (all temporal: valid_from, valid_to, invalid_at)
// ----------
// (Project)-[:CONTAINS]->(Agent|File|Directory|Blueprint)
// (Agent)-[:USES]->(Tool|Instrument)
// (Agent)-[:HAS_PROMPT]->(Prompt)
// (Agent)-[:CONFIGURED_BY]->(Setting)
// (Agent)-[:SPAWNED]->(Agent)  // subagent relationship
// (Persona)-[:CONTROLS]->(Agent)
// (Persona)-[:DECIDED]->(Decision)
// (Directory)-[:CONTAINS]->(File|Directory)
// (File)-[:IMPLEMENTS]->(Blueprint)
// (File)-[:DEPENDS_ON]->(File)
// (Blueprint)-[:EVOLVED_FROM]->(Blueprint)
// (Episode)-[:EXTRACTED]->(Fact)
// (Fact)-[:ABOUT]->(any entity)
// (Decision)-[:AFFECTS]->(any entity)
"""


# ============================================================================
# HIERARCHY SYNC
# ============================================================================

class KnowledgeGraph:
    """
    Full knowledge graph for AgentSmith.
    Mirrors filesystem, tracks relationships, enables BMAD navigation.
    """
    
    def __init__(self, graph: Optional[TemporalGraph] = None):
        self.g = graph or get_graph()
        self._synced = False
    
    # === INITIALIZATION ===
    
    def init_schema(self) -> bool:
        """Create indexes and constraints"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (p:Project) ON (p.name)",
            "CREATE INDEX IF NOT EXISTS FOR (a:Agent) ON (a.agent_id)",
            "CREATE INDEX IF NOT EXISTS FOR (t:Tool) ON (t.name)",
            "CREATE INDEX IF NOT EXISTS FOR (i:Instrument) ON (i.name)",
            "CREATE INDEX IF NOT EXISTS FOR (f:File) ON (f.path)",
            "CREATE INDEX IF NOT EXISTS FOR (f:File) ON (f.hash)",
            "CREATE INDEX IF NOT EXISTS FOR (d:Directory) ON (d.path)",
            "CREATE INDEX IF NOT EXISTS FOR (pr:Prompt) ON (pr.name)",
            "CREATE INDEX IF NOT EXISTS FOR (s:Setting) ON (s.key)",
            "CREATE INDEX IF NOT EXISTS FOR (b:Blueprint) ON (b.name)",
            "CREATE INDEX IF NOT EXISTS FOR (pe:Persona) ON (pe.name)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Episode) ON (e.timestamp)",
            "CREATE INDEX IF NOT EXISTS FOR (fa:Fact) ON (fa.subject)",
        ]
        for idx in indexes:
            try:
                self.g.query(idx)
            except:
                pass
        return True
    
    # === PROJECT ===
    
    def create_project(self, name: str, path: str, description: str = "") -> Optional[str]:
        """Create or update a project"""
        ts = time.time()
        result = self.g.query("""
            MERGE (p:Project {name: $name})
            SET p.path = $path,
                p.description = $description,
                p.updated_at = $ts
            RETURN elementId(p) as id
        """, {"name": name, "path": path, "description": description, "ts": ts})
        return result[0]["id"] if result else None
    
    def get_project(self, name: str) -> Optional[Dict]:
        """Get project with all contents"""
        result = self.g.query("""
            MATCH (p:Project {name: $name})
            OPTIONAL MATCH (p)-[:CONTAINS]->(a:Agent)
            OPTIONAL MATCH (p)-[:CONTAINS]->(b:Blueprint)
            OPTIONAL MATCH (p)-[:CONTAINS]->(f:File)
            RETURN p, collect(DISTINCT a.agent_id) as agents,
                   collect(DISTINCT b.name) as blueprints,
                   count(DISTINCT f) as file_count
        """, {"name": name})
        return result[0] if result else None
    
    # === AGENTS ===
    
    def register_agent(self, agent_id: str, agent_type: str = "primary",
                       config: Optional[Dict] = None, project: Optional[str] = None) -> Optional[str]:
        """Register an agent instance"""
        ts = time.time()
        result = self.g.query("""
            MERGE (a:Agent {agent_id: $agent_id})
            SET a.agent_type = $agent_type,
                a.config = $config,
                a.registered_at = $ts,
                a.status = 'active'
            RETURN elementId(a) as id
        """, {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "config": json.dumps(config or {}),
            "ts": ts
        })
        
        if result and project:
            self.g.query("""
                MATCH (p:Project {name: $project}), (a:Agent {agent_id: $agent_id})
                MERGE (p)-[:CONTAINS {valid_from: $ts}]->(a)
            """, {"project": project, "agent_id": agent_id, "ts": ts})
        
        return result[0]["id"] if result else None
    
    def spawn_subagent(self, parent_id: str, child_id: str, purpose: str = "") -> bool:
        """Record subagent spawn relationship"""
        ts = time.time()
        result = self.g.query("""
            MATCH (parent:Agent {agent_id: $parent_id})
            MERGE (child:Agent {agent_id: $child_id})
            SET child.agent_type = 'subagent',
                child.purpose = $purpose,
                child.spawned_at = $ts
            MERGE (parent)-[:SPAWNED {timestamp: $ts, purpose: $purpose}]->(child)
            RETURN elementId(child) as id
        """, {"parent_id": parent_id, "child_id": child_id, "purpose": purpose, "ts": ts})
        return len(result) > 0
    
    def link_persona_to_agent(self, persona_name: str, agent_id: str) -> bool:
        """Link a BMAD persona to an agent instance"""
        ts = time.time()
        result = self.g.query("""
            MATCH (pe:Persona {name: $persona_name}), (a:Agent {agent_id: $agent_id})
            MERGE (pe)-[:CONTROLS {valid_from: $ts}]->(a)
            RETURN pe.name as persona
        """, {"persona_name": persona_name, "agent_id": agent_id, "ts": ts})
        return len(result) > 0
    
    # === TOOLS ===
    
    def register_tool(self, name: str, description: str, tool_type: str = "builtin",
                      parameters: Optional[Dict] = None) -> Optional[str]:
        """Register a tool"""
        ts = time.time()
        result = self.g.query("""
            MERGE (t:Tool {name: $name})
            SET t.description = $description,
                t.tool_type = $tool_type,
                t.parameters = $parameters,
                t.registered_at = $ts
            RETURN elementId(t) as id
        """, {
            "name": name,
            "description": description,
            "tool_type": tool_type,
            "parameters": json.dumps(parameters or {}),
            "ts": ts
        })
        return result[0]["id"] if result else None
    
    def agent_uses_tool(self, agent_id: str, tool_name: str) -> bool:
        """Record that an agent uses a tool"""
        ts = time.time()
        result = self.g.query("""
            MATCH (a:Agent {agent_id: $agent_id}), (t:Tool {name: $tool_name})
            MERGE (a)-[:USES {valid_from: $ts}]->(t)
            RETURN t.name as tool
        """, {"agent_id": agent_id, "tool_name": tool_name, "ts": ts})
        return len(result) > 0
    
    # === INSTRUMENTS ===
    
    def register_instrument(self, name: str, path: str, description: str = "",
                            capabilities: List[str] = None) -> Optional[str]:
        """Register an instrument (custom script)"""
        ts = time.time()
        
        # Get file hash if exists
        file_hash = ""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        
        result = self.g.query("""
            MERGE (i:Instrument {name: $name})
            SET i.path = $path,
                i.description = $description,
                i.capabilities = $capabilities,
                i.hash = $hash,
                i.updated_at = $ts
            RETURN elementId(i) as id
        """, {
            "name": name,
            "path": path,
            "description": description,
            "capabilities": json.dumps(capabilities or []),
            "hash": file_hash,
            "ts": ts
        })
        return result[0]["id"] if result else None
    
    # === FILES ===
    
    def register_file(self, path: str, file_type: str = "unknown",
                      description: str = "", blueprint: Optional[str] = None) -> Optional[str]:
        """Register a file in the knowledge graph"""
        ts = time.time()
        
        # Compute hash if file exists
        file_hash = ""
        size = 0
        if os.path.exists(path):
            with open(path, 'rb') as f:
                content = f.read()
                file_hash = hashlib.sha256(content).hexdigest()[:16]
                size = len(content)
        
        result = self.g.query("""
            MERGE (f:File {path: $path})
            SET f.file_type = $file_type,
                f.description = $description,
                f.hash = $hash,
                f.size = $size,
                f.updated_at = $ts
            RETURN elementId(f) as id
        """, {
            "path": path,
            "file_type": file_type,
            "description": description,
            "hash": file_hash,
            "size": size,
            "ts": ts
        })
        
        # Link to blueprint if specified
        if result and blueprint:
            self.g.query("""
                MATCH (f:File {path: $path}), (b:Blueprint {name: $blueprint})
                MERGE (f)-[:IMPLEMENTS {valid_from: $ts}]->(b)
            """, {"path": path, "blueprint": blueprint, "ts": ts})
        
        # Link to parent directory
        parent = str(Path(path).parent)
        self.g.query("""
            MERGE (d:Directory {path: $parent})
            WITH d
            MATCH (f:File {path: $path})
            MERGE (d)-[:CONTAINS]->(f)
        """, {"parent": parent, "path": path})
        
        return result[0]["id"] if result else None
    
    def register_directory(self, path: str, description: str = "") -> Optional[str]:
        """Register a directory"""
        ts = time.time()
        result = self.g.query("""
            MERGE (d:Directory {path: $path})
            SET d.description = $description,
                d.updated_at = $ts
            RETURN elementId(d) as id
        """, {"path": path, "description": description, "ts": ts})
        
        # Link to parent
        parent = str(Path(path).parent)
        if parent != path:
            self.g.query("""
                MERGE (p:Directory {path: $parent})
                WITH p
                MATCH (d:Directory {path: $path})
                MERGE (p)-[:CONTAINS]->(d)
            """, {"parent": parent, "path": path})
        
        return result[0]["id"] if result else None
    
    def add_file_dependency(self, from_path: str, to_path: str, dep_type: str = "imports") -> bool:
        """Record file dependency"""
        ts = time.time()
        result = self.g.query("""
            MATCH (f1:File {path: $from_path}), (f2:File {path: $to_path})
            MERGE (f1)-[:DEPENDS_ON {dep_type: $dep_type, valid_from: $ts}]->(f2)
            RETURN f1.path as from_file
        """, {"from_path": from_path, "to_path": to_path, "dep_type": dep_type, "ts": ts})
        return len(result) > 0
    
    # === PROMPTS ===
    
    def register_prompt(self, name: str, content: str, prompt_type: str = "system",
                        agent_id: Optional[str] = None) -> Optional[str]:
        """Register a prompt"""
        ts = time.time()
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        result = self.g.query("""
            MERGE (pr:Prompt {name: $name})
            SET pr.content = $content,
                pr.prompt_type = $prompt_type,
                pr.hash = $hash,
                pr.updated_at = $ts
            RETURN elementId(pr) as id
        """, {
            "name": name,
            "content": content,
            "prompt_type": prompt_type,
            "hash": content_hash,
            "ts": ts
        })
        
        if result and agent_id:
            self.g.query("""
                MATCH (a:Agent {agent_id: $agent_id}), (pr:Prompt {name: $name})
                MERGE (a)-[:HAS_PROMPT {valid_from: $ts}]->(pr)
            """, {"agent_id": agent_id, "name": name, "ts": ts})
        
        return result[0]["id"] if result else None
    
    # === SETTINGS ===
    
    def set_setting(self, key: str, value: Any, scope: str = "global",
                    agent_id: Optional[str] = None) -> Optional[str]:
        """Set a configuration value"""
        ts = time.time()
        result = self.g.query("""
            MERGE (s:Setting {key: $key, scope: $scope})
            SET s.value = $value,
                s.updated_at = $ts
            RETURN elementId(s) as id
        """, {
            "key": key,
            "value": json.dumps(value),
            "scope": scope,
            "ts": ts
        })
        
        if result and agent_id:
            self.g.query("""
                MATCH (a:Agent {agent_id: $agent_id}), (s:Setting {key: $key})
                MERGE (a)-[:CONFIGURED_BY {valid_from: $ts}]->(s)
            """, {"agent_id": agent_id, "key": key, "ts": ts})
        
        return result[0]["id"] if result else None
    
    def get_setting(self, key: str, scope: str = "global") -> Optional[Any]:
        """Get a configuration value"""
        result = self.g.query("""
            MATCH (s:Setting {key: $key, scope: $scope})
            RETURN s.value as value
        """, {"key": key, "scope": scope})
        if result:
            try:
                return json.loads(result[0]["value"])
            except:
                return result[0]["value"]
        return None
    
    # === SYNC OPERATIONS ===
    
    def sync_directory(self, path: str, recursive: bool = True, 
                       file_types: List[str] = None) -> Dict[str, int]:
        """Sync a directory to the graph"""
        if file_types is None:
            file_types = ['.py', '.md', '.txt', '.json', '.toml', '.yaml', '.yml']
        
        stats = {"dirs": 0, "files": 0, "skipped": 0}
        path = Path(path)
        
        if not path.exists():
            return stats
        
        # Register directory
        self.register_directory(str(path))
        stats["dirs"] += 1
        
        for item in path.iterdir():
            if item.name.startswith('.') or item.name == '__pycache__':
                stats["skipped"] += 1
                continue
            
            if item.is_dir():
                if recursive:
                    sub_stats = self.sync_directory(str(item), recursive, file_types)
                    stats["dirs"] += sub_stats["dirs"]
                    stats["files"] += sub_stats["files"]
                    stats["skipped"] += sub_stats["skipped"]
            elif item.is_file():
                if any(item.name.endswith(ft) for ft in file_types):
                    file_type = item.suffix[1:] if item.suffix else "unknown"
                    self.register_file(str(item), file_type=file_type)
                    stats["files"] += 1
                else:
                    stats["skipped"] += 1
        
        return stats
    
    def sync_instruments(self, instruments_path: str = "/a0/instruments") -> int:
        """Sync all instruments to graph"""
        count = 0
        path = Path(instruments_path)
        
        if not path.exists():
            return 0
        
        for item in path.rglob("*.py"):
            name = item.stem
            # Try to extract description from docstring
            description = ""
            try:
                content = item.read_text()
                if '"""' in content:
                    desc_start = content.index('"""') + 3
                    desc_end = content.index('"""', desc_start)
                    description = content[desc_start:desc_end].strip()[:200]
            except:
                pass
            
            self.register_instrument(name, str(item), description)
            count += 1
        
        return count
    
    def sync_prompts(self, prompts_path: str = "/a0/prompts") -> int:
        """Sync all prompts to graph"""
        count = 0
        path = Path(prompts_path)
        
        if not path.exists():
            return 0
        
        for item in path.rglob("*.md"):
            name = item.stem
            try:
                content = item.read_text()
                self.register_prompt(name, content, prompt_type="template")
                count += 1
            except:
                pass
        
        return count
    
    def sync_tools(self, tools_path: str = "/a0/python/tools") -> int:
        """Sync tool definitions to graph"""
        count = 0
        path = Path(tools_path)
        
        if not path.exists():
            return 0
        
        for item in path.rglob("*.py"):
            if item.name.startswith('_'):
                continue
            
            name = item.stem
            description = ""
            try:
                content = item.read_text()
                if '"""' in content:
                    desc_start = content.index('"""') + 3
                    desc_end = content.index('"""', desc_start)
                    description = content[desc_start:desc_end].strip()[:200]
            except:
                pass
            
            self.register_tool(name, description, tool_type="python")
            count += 1
        
        return count
    
    def full_sync(self, base_path: str = "/a0") -> Dict[str, Any]:
        """Full sync of AgentSmith to graph"""
        print("[KnowledgeGraph] Starting full sync...")
        
        # Initialize schema
        self.init_schema()
        
        # Create default project
        self.create_project("agentsmith", base_path, "AgentSmith deployment")
        
        # Sync directories
        dir_stats = self.sync_directory(base_path)
        print(f"  Directories: {dir_stats['dirs']}, Files: {dir_stats['files']}")
        
        # Sync instruments
        inst_count = self.sync_instruments(f"{base_path}/instruments")
        print(f"  Instruments: {inst_count}")
        
        # Sync prompts
        prompt_count = self.sync_prompts(f"{base_path}/prompts")
        print(f"  Prompts: {prompt_count}")
        
        # Sync tools
        tool_count = self.sync_tools(f"{base_path}/python/tools")
        print(f"  Tools: {tool_count}")
        
        self._synced = True
        
        return {
            "directories": dir_stats["dirs"],
            "files": dir_stats["files"],
            "instruments": inst_count,
            "prompts": prompt_count,
            "tools": tool_count
        }
    
    # === QUERIES ===
    
    def find_files_for_blueprint(self, blueprint_name: str) -> List[Dict]:
        """Find all files implementing a blueprint"""
        return self.g.query("""
            MATCH (f:File)-[:IMPLEMENTS]->(b:Blueprint {name: $name})
            RETURN f.path as path, f.file_type as type, f.hash as hash
        """, {"name": blueprint_name})
    
    def find_agent_capabilities(self, agent_id: str) -> Dict[str, List[str]]:
        """Find all tools and instruments an agent can use"""
        tools = self.g.query("""
            MATCH (a:Agent {agent_id: $agent_id})-[:USES]->(t:Tool)
            RETURN t.name as name, t.description as description
        """, {"agent_id": agent_id})
        
        instruments = self.g.query("""
            MATCH (a:Agent {agent_id: $agent_id})-[:USES]->(i:Instrument)
            RETURN i.name as name, i.description as description
        """, {"agent_id": agent_id})
        
        return {
            "tools": tools,
            "instruments": instruments
        }
    
    def get_file_dependencies(self, path: str, depth: int = 3) -> List[Dict]:
        """Get dependency tree for a file"""
        return self.g.query(f"""
            MATCH (f:File {{path: $path}})
            MATCH path = (f)-[:DEPENDS_ON*1..{depth}]->(dep:File)
            RETURN dep.path as path, length(path) as depth
            ORDER BY depth
        """, {"path": path})
    
    def search_codebase(self, query: str, node_type: str = "File") -> List[Dict]:
        """Search across the codebase"""
        return self.g.query(f"""
            MATCH (n:{node_type})
            WHERE n.path CONTAINS $query 
               OR n.name CONTAINS $query 
               OR n.description CONTAINS $query
            RETURN n.path as path, n.name as name, n.description as description
            LIMIT 20
        """, {"query": query})
    
    def get_project_structure(self, project_name: str) -> Dict[str, Any]:
        """Get complete project structure"""
        # Get directories
        dirs = self.g.query("""
            MATCH (p:Project {name: $name})-[:CONTAINS*]->(d:Directory)
            RETURN d.path as path
            ORDER BY d.path
        """, {"name": project_name})
        
        # Get file counts by type
        file_stats = self.g.query("""
            MATCH (p:Project {name: $name})-[:CONTAINS*]->(f:File)
            RETURN f.file_type as type, count(*) as count
        """, {"name": project_name})
        
        # Get blueprints
        blueprints = self.g.query("""
            MATCH (p:Project {name: $name})-[:CONTAINS]->(b:Blueprint)
            RETURN b.name as name, b.status as status
        """, {"name": project_name})
        
        return {
            "project": project_name,
            "directories": [d["path"] for d in dirs],
            "file_types": {s["type"]: s["count"] for s in file_stats},
            "blueprints": blueprints
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        counts = self.g.query("""
            MATCH (n)
            RETURN labels(n)[0] as type, count(*) as count
            ORDER BY count DESC
        """)
        
        edge_counts = self.g.query("""
            MATCH ()-[r]->()
            RETURN type(r) as type, count(*) as count
            ORDER BY count DESC
        """)
        
        return {
            "status": "ok",
            "synced": self._synced,
            "nodes": {c["type"]: c["count"] for c in counts},
            "edges": {e["type"]: e["count"] for e in edge_counts}
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_kg: Optional[KnowledgeGraph] = None

def get_knowledge_graph() -> KnowledgeGraph:
    """Get singleton knowledge graph instance"""
    global _kg
    if _kg is None:
        _kg = KnowledgeGraph()
    return _kg

def sync_all(base_path: str = "/a0") -> Dict[str, Any]:
    """Full sync helper"""
    return get_knowledge_graph().full_sync(base_path)

def search(query: str) -> List[Dict]:
    """Quick search"""
    return get_knowledge_graph().search_codebase(query)
