"""
AgentSmith Temporal Graph Layer
FalkorDB with temporal edges + BMAD persona support
"""

import os
import time
import json
import hashlib
import traceback
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if os.getenv("DEBUG") else logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("temporal_graph")

try:
    from falkordb import FalkorDB
    FALKORDB_AVAILABLE = True
    log.info("FalkorDB module loaded")
except ImportError:
    FALKORDB_AVAILABLE = False
    log.error("FalkorDB not available")


@dataclass
class TemporalEdge:
    """Edge with temporal validity"""
    from_id: str
    to_id: str
    rel_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    valid_from: float = field(default_factory=time.time)
    valid_to: Optional[float] = None
    invalid_at: Optional[float] = None


class TemporalGraph:
    """
    FalkorDB with temporal edge support.
    All edges carry valid_from, valid_to, invalid_at.
    Enables "what was true at time X?" queries.
    """
    
    _instance: Optional["TemporalGraph"] = None
    
    def __init__(self):
        self.host = os.getenv("FALKORDB_HOST", "localhost")
        self.port = int(os.getenv("FALKORDB_PORT", "16379"))
        self.password = os.getenv("FALKORDB_PASSWORD", "CH@NG3M3N()W!")
        self.graph_name = os.getenv("GRAPH_NAME", "agentsmith")
        self._db = None
        self._graph = None
        self._connected = False
    
    @classmethod
    def get_instance(cls) -> "TemporalGraph":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def connect(self) -> bool:
        if not FALKORDB_AVAILABLE:
            log.error("FalkorDB not installed")
            return False
        log.info(f"Connecting to {self.host}:{self.port}/{self.graph_name}")
        try:
            self._db = FalkorDB(
                host=self.host,
                port=self.port,
                password=self.password if self.password else None
            )
            self._graph = self._db.select_graph(self.graph_name)
            self._connected = True
            log.info(f"Connected: {self.host}:{self.port}/{self.graph_name}")
            return True
        except Exception as e:
            log.error(f"Connection failed: {e}")
            log.error(traceback.format_exc())
            return False
    
    def _ensure_connected(self) -> bool:
        if not self._connected:
            return self.connect()
        return True
    
    def query(self, cypher: str, params: Optional[Dict] = None) -> List[Dict]:
        if not self._ensure_connected():
            return []
        try:
            result = self._graph.query(cypher, params or {})
            return [dict(zip(result.header, row)) for row in result.result_set]
        except Exception as e:
            log.error(f"Query error: {e}")
            log.debug(f"Query was: {cypher}")
            log.debug(traceback.format_exc())
            return []
    
    # === TEMPORAL OPERATIONS ===
    
    def add_episode(self, content: str, source: str = "agent", 
                    episode_type: str = "action", metadata: Optional[Dict] = None) -> Optional[str]:
        """
        Record an episode (atomic unit of memory).
        Episodes are immutable - they record what happened.
        """
        ts = time.time()
        props = {
            "content": content,
            "source": source,
            "episode_type": episode_type,
            "timestamp": ts,
            "metadata": json.dumps(metadata or {})
        }
        cypher = """
        CREATE (e:Episode {
            content: $content,
            source: $source,
            episode_type: $episode_type,
            timestamp: $timestamp,
            metadata: $metadata
        })
        RETURN id(e) as id
        """
        result = self.query(cypher, props)
        return result[0]["id"] if result else None
    
    def extract_fact(self, episode_id: str, subject: str, predicate: str, 
                     obj: str, confidence: float = 1.0) -> Optional[str]:
        """
        Extract a fact from an episode.
        Facts have temporal validity - they can become invalid.
        """
        ts = time.time()
        fact_id = hashlib.md5(f"{subject}:{predicate}:{obj}".encode()).hexdigest()[:12]
        
        cypher = """
        MATCH (ep:Episode) WHERE id(ep) = $episode_id
        MERGE (f:Fact {fact_id: $fact_id})
        SET f.subject = $subject,
            f.predicate = $predicate,
            f.object = $obj,
            f.confidence = $confidence,
            f.valid_from = $ts,
            f.valid_to = null,
            f.invalid_at = null
        MERGE (ep)-[:EXTRACTED {timestamp: $ts}]->(f)
        RETURN id(f) as id
        """
        result = self.query(cypher, {
            "episode_id": episode_id,
            "fact_id": fact_id,
            "subject": subject,
            "predicate": predicate,
            "obj": obj,
            "confidence": confidence,
            "ts": ts
        })
        return result[0]["id"] if result else None
    
    def invalidate_fact(self, fact_id: str, reason: str = "") -> bool:
        """Mark a fact as no longer valid"""
        ts = time.time()
        cypher = """
        MATCH (f:Fact {fact_id: $fact_id})
        WHERE f.invalid_at IS NULL
        SET f.invalid_at = $ts,
            f.invalidation_reason = $reason
        RETURN f.fact_id as id
        """
        result = self.query(cypher, {"fact_id": fact_id, "ts": ts, "reason": reason})
        return len(result) > 0
    
    def get_valid_facts(self, subject: Optional[str] = None, 
                        at_time: Optional[float] = None) -> List[Dict]:
        """Get facts that were valid at a given time (default: now)"""
        ts = at_time or time.time()
        
        where_clause = "WHERE f.valid_from <= $ts AND (f.invalid_at IS NULL OR f.invalid_at > $ts)"
        if subject:
            where_clause += " AND f.subject = $subject"
        
        cypher = f"""
        MATCH (f:Fact)
        {where_clause}
        RETURN f.subject as subject, f.predicate as predicate, 
               f.object as object, f.confidence as confidence,
               f.valid_from as valid_from
        ORDER BY f.valid_from DESC
        """
        return self.query(cypher, {"ts": ts, "subject": subject})
    
    def detect_contradictions(self) -> List[Dict]:
        """Find facts that contradict each other"""
        cypher = """
        MATCH (f1:Fact), (f2:Fact)
        WHERE f1.subject = f2.subject 
          AND f1.predicate = f2.predicate
          AND f1.object <> f2.object
          AND f1.invalid_at IS NULL
          AND f2.invalid_at IS NULL
          AND id(f1) < id(f2)
        RETURN f1.subject as subject, f1.predicate as predicate,
               f1.object as value1, f2.object as value2,
               f1.valid_from as time1, f2.valid_from as time2
        """
        return self.query(cypher)
    
    def temporal_query(self, entity_name: str, at_time: Optional[float] = None) -> Dict[str, Any]:
        """Get everything known about an entity at a point in time"""
        ts = at_time or time.time()
        
        facts = self.query("""
        MATCH (f:Fact)
        WHERE f.subject = $name
          AND f.valid_from <= $ts
          AND (f.invalid_at IS NULL OR f.invalid_at > $ts)
        RETURN f.predicate as key, f.object as value, f.valid_from as since
        """, {"name": entity_name, "ts": ts})
        
        return {
            "entity": entity_name,
            "at_time": ts,
            "facts": {f["key"]: f["value"] for f in facts},
            "history": facts
        }
    
    # === BMAD PERSONA OPERATIONS ===
    
    def create_persona(self, name: str, role: str, phase: str, 
                       model: str, system_prompt: str,
                       capabilities: List[str] = None) -> Optional[str]:
        """
        Create a BMAD persona as a graph entity.
        Personas accumulate decision history over time.
        """
        ts = time.time()
        cypher = """
        MERGE (p:Persona {name: $name})
        SET p.role = $role,
            p.phase = $phase,
            p.model = $model,
            p.system_prompt = $system_prompt,
            p.capabilities = $capabilities,
            p.created_at = $ts,
            p.decision_count = 0
        RETURN id(p) as id
        """
        result = self.query(cypher, {
            "name": name,
            "role": role,
            "phase": phase,
            "model": model,
            "system_prompt": system_prompt,
            "capabilities": json.dumps(capabilities or []),
            "ts": ts
        })
        return result[0]["id"] if result else None
    
    def get_persona(self, name: str) -> Optional[Dict]:
        """Get persona with decision history"""
        cypher = """
        MATCH (p:Persona {name: $name})
        OPTIONAL MATCH (p)-[:DECIDED]->(d:Decision)
        RETURN p.name as name, p.role as role, p.phase as phase,
               p.model as model, p.system_prompt as system_prompt,
               p.capabilities as capabilities,
               count(d) as decision_count
        """
        result = self.query(cypher, {"name": name})
        return result[0] if result else None
    
    def record_decision(self, persona_name: str, decision: str, 
                        rationale: str, context: Optional[Dict] = None) -> Optional[str]:
        """Record a decision made by a persona"""
        ts = time.time()
        cypher = """
        MATCH (p:Persona {name: $persona_name})
        CREATE (d:Decision {
            decision: $decision,
            rationale: $rationale,
            context: $context,
            timestamp: $ts
        })
        CREATE (p)-[:DECIDED {timestamp: $ts}]->(d)
        SET p.decision_count = coalesce(p.decision_count, 0) + 1
        RETURN id(d) as id
        """
        result = self.query(cypher, {
            "persona_name": persona_name,
            "decision": decision,
            "rationale": rationale,
            "context": json.dumps(context or {}),
            "ts": ts
        })
        return result[0]["id"] if result else None
    
    def get_persona_decisions(self, persona_name: str, limit: int = 20) -> List[Dict]:
        """Get recent decisions by a persona"""
        cypher = """
        MATCH (p:Persona {name: $name})-[:DECIDED]->(d:Decision)
        RETURN d.decision as decision, d.rationale as rationale,
               d.timestamp as timestamp
        ORDER BY d.timestamp DESC
        LIMIT $limit
        """
        return self.query(cypher, {"name": persona_name, "limit": limit})
    
    # === BLUEPRINT OPERATIONS ===
    
    def create_blueprint(self, name: str, description: str, 
                         blueprint_type: str = "feature",
                         priority: int = 0) -> Optional[str]:
        """Create a blueprint with version tracking"""
        ts = time.time()
        cypher = """
        CREATE (b:Blueprint {
            name: $name,
            description: $description,
            blueprint_type: $blueprint_type,
            priority: $priority,
            version: 1,
            created_at: $ts,
            updated_at: $ts,
            status: 'draft'
        })
        RETURN id(b) as id
        """
        result = self.query(cypher, {
            "name": name,
            "description": description,
            "blueprint_type": blueprint_type,
            "priority": priority,
            "ts": ts
        })
        return result[0]["id"] if result else None
    
    def evolve_blueprint(self, name: str, changes: str, 
                         changed_by: str) -> Optional[str]:
        """
        Record blueprint evolution.
        Creates EVOLVED_FROM chain for history tracking.
        """
        ts = time.time()
        cypher = """
        MATCH (old:Blueprint {name: $name})
        WHERE NOT EXISTS((old)<-[:EVOLVED_FROM]-())
        CREATE (new:Blueprint {
            name: old.name,
            description: old.description,
            blueprint_type: old.blueprint_type,
            priority: old.priority,
            version: old.version + 1,
            created_at: $ts,
            updated_at: $ts,
            status: 'draft',
            changes: $changes,
            changed_by: $changed_by
        })
        CREATE (new)-[:EVOLVED_FROM {timestamp: $ts}]->(old)
        RETURN id(new) as id
        """
        result = self.query(cypher, {
            "name": name,
            "changes": changes,
            "changed_by": changed_by,
            "ts": ts
        })
        return result[0]["id"] if result else None
    
    def get_blueprint_history(self, name: str) -> List[Dict]:
        """Get full evolution history of a blueprint"""
        cypher = """
        MATCH (b:Blueprint {name: $name})
        MATCH path = (b)-[:EVOLVED_FROM*0..]->(ancestor:Blueprint)
        RETURN ancestor.version as version,
               ancestor.changes as changes,
               ancestor.changed_by as changed_by,
               ancestor.created_at as timestamp
        ORDER BY ancestor.version DESC
        """
        return self.query(cypher, {"name": name})
    
    # === CONVENIENCE FUNCTIONS ===
    
    def remember(self, content: str, source: str = "agent") -> Optional[str]:
        """Quick way to store a memory"""
        return self.add_episode(content, source, "memory")
    
    def recall(self, query: str, limit: int = 10) -> List[Dict]:
        """Search episodes by content"""
        cypher = """
        MATCH (e:Episode)
        WHERE e.content CONTAINS $query
        RETURN e.content as content, e.timestamp as timestamp,
               e.source as source, e.episode_type as type
        ORDER BY e.timestamp DESC
        LIMIT $limit
        """
        return self.query(cypher, {"query": query, "limit": limit})
    
    def health_check(self) -> Dict[str, Any]:
        """System health check"""
        if not self._ensure_connected():
            return {"status": "error", "connected": False}
        
        counts = self.query("""
        MATCH (n) 
        RETURN labels(n)[0] as label, count(*) as count
        """)
        
        contradictions = self.detect_contradictions()
        
        return {
            "status": "ok",
            "connected": True,
            "host": self.host,
            "graph": self.graph_name,
            "node_counts": {c["label"]: c["count"] for c in counts},
            "contradiction_count": len(contradictions)
        }


# === BMAD PERSONA DEFINITIONS ===

BMAD_PERSONAS = {
    "analyst": {
        "name": "Mary",
        "role": "Business Analyst",
        "phase": "strategic",
        "model": "openrouter/anthropic/claude-sonnet-4",  # Autonomous, capable, affordable
        "system_prompt": """You are Mary, the Business Analyst. Your role:
- Gather and clarify requirements
- Identify constraints and edge cases
- Create project briefs and market research
- Challenge assumptions constructively
Always think about what could go wrong and what's missing.""",
        "capabilities": ["requirements", "research", "constraints", "briefs"]
    },
    "pm": {
        "name": "John", 
        "role": "Product Manager",
        "phase": "strategic",
        "model": "openrouter/anthropic/claude-sonnet-4",
        "system_prompt": """You are John, the Product Manager. Your role:
- Create PRDs from requirements
- Define product strategy and roadmap
- Prioritize features by impact
- Communicate with stakeholders
Focus on user value and business outcomes.""",
        "capabilities": ["prd", "strategy", "prioritization", "roadmap"]
    },
    "architect": {
        "name": "Winston",
        "role": "Architect", 
        "phase": "strategic",
        "model": "openrouter/anthropic/claude-sonnet-4",
        "system_prompt": """You are Winston, the Architect. Your role:
- Design system architecture
- Select technologies and patterns
- Define APIs and data models
- Plan infrastructure and scaling
Think in systems. Consider failure modes. Design for change.""",
        "capabilities": ["architecture", "api_design", "infrastructure", "patterns"]
    },
    "dev": {
        "name": "Alex",
        "role": "Developer",
        "phase": "tactical",
        "model": "openrouter/google/gemini-2.0-flash-001",  # Fast, cheap
        "system_prompt": """You are Alex, the Developer. Your role:
- Implement features from specs
- Write clean, tested code
- Follow architectural decisions
- Document as you go
Write code that works, then make it better.""",
        "capabilities": ["implementation", "testing", "debugging", "documentation"]
    },
    "reviewer": {
        "name": "Sam",
        "role": "Code Reviewer",
        "phase": "tactical",
        "model": "openrouter/deepseek/deepseek-chat",  # Cheap, thorough
        "system_prompt": """You are Sam, the Code Reviewer. Your role:
- Review code for quality and correctness
- Check alignment with architecture
- Identify bugs and edge cases
- Suggest improvements
Be constructive. Catch what others miss.""",
        "capabilities": ["code_review", "quality", "edge_cases"]
    },
    "qa": {
        "name": "Quinn",
        "role": "QA Engineer",
        "phase": "tactical", 
        "model": "openrouter/meta-llama/llama-3.3-70b-instruct",  # Free on OpenRouter
        "system_prompt": """You are Quinn, the QA Engineer. Your role:
- Test for edge cases and failure modes
- Security review
- Performance considerations
- User experience issues
Break things before users do.""",
        "capabilities": ["testing", "security", "performance", "ux_testing"]
    },
    "po": {
        "name": "Sarah",
        "role": "Product Owner",
        "phase": "strategic",
        "model": "openrouter/anthropic/claude-sonnet-4",
        "system_prompt": """You are Sarah, the Product Owner. Your role:
- Manage and prioritize backlog
- Refine user stories
- Define acceptance criteria
- Sprint planning decisions
You own the what and why. Dev owns the how.""",
        "capabilities": ["backlog", "stories", "acceptance_criteria", "sprint_planning"]
    }
}


def bootstrap_bmad_personas(graph: Optional[TemporalGraph] = None) -> Dict[str, str]:
    """Create all BMAD personas in the graph"""
    g = graph or TemporalGraph.get_instance()
    results = {}
    
    for key, persona in BMAD_PERSONAS.items():
        node_id = g.create_persona(
            name=persona["name"],
            role=persona["role"],
            phase=persona["phase"],
            model=persona["model"],
            system_prompt=persona["system_prompt"],
            capabilities=persona["capabilities"]
        )
        results[key] = node_id
        print(f"[+] Created persona: {persona['name']} ({persona['role']})")
    
    # Record bootstrap as episode
    g.add_episode(
        content=f"BMAD personas bootstrapped: {', '.join(BMAD_PERSONAS.keys())}",
        source="system",
        episode_type="bootstrap"
    )
    
    return results


# Convenience singleton access
def get_graph() -> TemporalGraph:
    return TemporalGraph.get_instance()
