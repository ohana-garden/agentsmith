"""
Moral Agents Module

Implements the agent lifecycle and kuleana (responsibility) layer:
- Agent instantiation with 19D moral state
- Kuleana relationships (STEWARD_OF, RESPONSIBLE_FOR, ACCOUNTABLE_TO, DELEGATED_BY)
- User proxy agents
- Context agents
- SME and sensor agents
- Reputation trail
- Access derivation from kuleana

Key principle: Trust is NOT edges - it's the metric tensor in moral_geometry.
Kuleana is the operational layer for responsibility/accountability.
"""

import os
import time
import json
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from enum import Enum

from python.helpers.moral_geometry import (
    MoralState, MoralManifold, compute_kala, project_to_valid_manifold,
    get_default_state, update_state_from_action, VIRTUES
)

log = logging.getLogger("moral_agents")

# Kuleana relationship types
class KuleanaType(Enum):
    STEWARD_OF = "STEWARD_OF"           # Caretaking responsibility
    RESPONSIBLE_FOR = "RESPONSIBLE_FOR"  # Direct operational responsibility
    ACCOUNTABLE_TO = "ACCOUNTABLE_TO"    # Answerable to
    DELEGATED_BY = "DELEGATED_BY"        # Authority granted by


# Agent types
class AgentType(Enum):
    USER_PROXY = "user_proxy"    # Bound to human user
    CONTEXT = "context"          # Domain-specific work agent
    SME = "sme"                  # Subject matter expert
    SENSOR = "sensor"           # Data stream agent
    SECURITY = "security"        # Security audit/validation
    GENERAL = "general"          # General purpose


@dataclass
class KuleanaEdge:
    """Represents a kuleana relationship"""
    rel_type: KuleanaType
    target_id: str
    target_type: str  # "Human", "Agent", "Entity", "Domain", "Community"
    scope: str = "full"
    since: float = field(default_factory=time.time)
    revocable: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MoralAgent:
    """
    A graph-native, persistent agent with moral state.

    An agent IS:
    - A trajectory through the knowledge graph
    - Defined by its traversal pattern
    - Emergent from initial conditions + accumulated actions

    Properties stored in graph, computed properties derived from state.
    """
    id: str
    agent_type: AgentType
    state: MoralState
    autonomy_level: float = 0.5  # 0.0 = echo, 1.0 = full autonomy
    kala_current: float = 0.0
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_graph_properties(self) -> Dict[str, Any]:
        """Convert to flat dict for graph node properties"""
        state_dict = self.state.to_dict()
        return {
            "id": self.id,
            "agent_type": self.agent_type.value,
            "autonomy_level": self.autonomy_level,
            "kala_current": self.kala_current,
            "created_at": self.created_at,
            "metadata": json.dumps(self.metadata),
            # Flatten state into properties
            **state_dict
        }

    @classmethod
    def from_graph_properties(cls, props: Dict[str, Any]) -> "MoralAgent":
        """Create from graph node properties"""
        # Extract state properties
        state_dict = {
            "unity": props.get("unity", 0.5),
            "justice": props.get("justice", 0.5),
            "truthfulness": props.get("truthfulness", 0.5),
            "love": props.get("love", 0.5),
            "detachment": props.get("detachment", 0.5),
            "humility": props.get("humility", 0.5),
            "service": props.get("service", 0.5),
            "courage": props.get("courage", 0.5),
            "wisdom": props.get("wisdom", 0.5),
            "d_unity": props.get("d_unity", 0.0),
            "d_justice": props.get("d_justice", 0.0),
            "d_truthfulness": props.get("d_truthfulness", 0.0),
            "d_love": props.get("d_love", 0.0),
            "d_detachment": props.get("d_detachment", 0.0),
            "d_humility": props.get("d_humility", 0.0),
            "d_service": props.get("d_service", 0.0),
            "d_courage": props.get("d_courage", 0.0),
            "d_wisdom": props.get("d_wisdom", 0.0),
            "t": props.get("t", time.time())
        }

        metadata = props.get("metadata", "{}")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        return cls(
            id=props["id"],
            agent_type=AgentType(props.get("agent_type", "general")),
            state=MoralState.from_dict(state_dict),
            autonomy_level=props.get("autonomy_level", 0.5),
            kala_current=props.get("kala_current", 0.0),
            created_at=props.get("created_at", time.time()),
            metadata=metadata
        )


class MoralAgentGraph:
    """
    Graph operations for moral agents.

    Manages:
    - Agent CRUD
    - Kuleana edge management
    - Action recording
    - Reputation queries
    - Access derivation
    """

    _instance: Optional["MoralAgentGraph"] = None

    def __init__(self):
        # Lazy import to avoid circular deps
        self._graph = None

    @classmethod
    def get_instance(cls) -> "MoralAgentGraph":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_graph(self):
        """Lazy load graph connection"""
        if self._graph is None:
            from python.helpers.temporal_graph import TemporalGraph
            self._graph = TemporalGraph.get_instance()
            self._graph.connect()
        return self._graph

    def _generate_id(self, prefix: str = "agent") -> str:
        """Generate unique ID"""
        ts = str(time.time()).encode()
        return f"{prefix}_{hashlib.md5(ts).hexdigest()[:12]}"

    # === SCHEMA SETUP ===

    def setup_schema(self) -> bool:
        """Create indexes and constraints for moral agent nodes"""
        g = self._get_graph()

        indices = [
            "CREATE INDEX IF NOT EXISTS FOR (a:MoralAgent) ON (a.id)",
            "CREATE INDEX IF NOT EXISTS FOR (h:Human) ON (h.id)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.id)",
            "CREATE INDEX IF NOT EXISTS FOR (d:Domain) ON (d.id)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Community) ON (c.id)",
            "CREATE INDEX IF NOT EXISTS FOR (ev:Event) ON (ev.id)",
            "CREATE INDEX IF NOT EXISTS FOR (ac:Action) ON (ac.id)",
            "CREATE INDEX IF NOT EXISTS FOR (cv:Conversation) ON (cv.id)",
            "CREATE INDEX IF NOT EXISTS FOR (a:MoralAgent) ON (a.agent_type)",
        ]

        for cypher in indices:
            try:
                g.query(cypher)
            except Exception as e:
                log.warning(f"Index creation warning: {e}")

        log.info("Moral agent schema setup complete")
        return True

    # === AGENT LIFECYCLE ===

    def create_agent(
        self,
        agent_type: Union[AgentType, str],
        kuleana: Optional[List[KuleanaEdge]] = None,
        initial_state: Optional[MoralState] = None,
        autonomy_level: float = 0.5,
        metadata: Optional[Dict] = None
    ) -> MoralAgent:
        """
        Create a new moral agent node with initial conditions.

        The agent's identity will emerge from its traversal.
        Initial conditions just seed the trajectory.
        """
        g = self._get_graph()

        if isinstance(agent_type, str):
            agent_type = AgentType(agent_type)

        agent_id = self._generate_id("agent")

        # Get default state for agent type or use provided
        state = initial_state or get_default_state(agent_type.value)
        state = project_to_valid_manifold(state)  # Ensure valid

        # Compute initial Kala
        kala = compute_kala(state)

        agent = MoralAgent(
            id=agent_id,
            agent_type=agent_type,
            state=state,
            autonomy_level=autonomy_level,
            kala_current=kala,
            metadata=metadata or {}
        )

        # Create node
        props = agent.to_graph_properties()
        prop_list = ", ".join([f"{k}: ${k}" for k in props.keys()])

        cypher = f"""
        CREATE (a:MoralAgent {{{prop_list}}})
        RETURN a.id as id
        """
        result = g.query(cypher, props)

        if not result:
            raise RuntimeError(f"Failed to create agent {agent_id}")

        log.info(f"Created agent {agent_id} ({agent_type.value})")

        # Create kuleana edges
        if kuleana:
            for edge in kuleana:
                self.create_kuleana_edge(agent_id, edge)

        # Record creation as episode
        g.add_episode(
            content=f"Agent {agent_id} created (type: {agent_type.value})",
            source="system",
            episode_type="agent_creation",
            metadata={"agent_id": agent_id, "agent_type": agent_type.value}
        )

        return agent

    def get_agent(self, agent_id: str) -> Optional[MoralAgent]:
        """Retrieve agent by ID"""
        g = self._get_graph()

        cypher = """
        MATCH (a:MoralAgent {id: $id})
        RETURN a
        """
        result = g.query(cypher, {"id": agent_id})

        if not result:
            return None

        # FalkorDB returns node as dict of properties
        props = result[0].get("a", result[0])
        return MoralAgent.from_graph_properties(props)

    def update_agent_state(
        self,
        agent_id: str,
        action_impacts: Optional[Dict[str, float]] = None,
        new_state: Optional[MoralState] = None
    ) -> Optional[MoralAgent]:
        """
        Update agent's moral state.

        Either provide action_impacts (will update based on current state)
        or new_state (will replace entirely, after projection).
        """
        agent = self.get_agent(agent_id)
        if not agent:
            return None

        g = self._get_graph()
        manifold = MoralManifold.get_instance()

        if action_impacts:
            # Update based on action impacts
            agent.state = update_state_from_action(agent.state, action_impacts)
        elif new_state:
            # Replace with new state (projected)
            agent.state = project_to_valid_manifold(new_state)

        # Recompute Kala
        agent.kala_current = compute_kala(agent.state)
        agent.state.t = time.time()

        # Update in graph
        props = agent.to_graph_properties()
        set_clauses = ", ".join([f"a.{k} = ${k}" for k in props.keys() if k != "id"])

        cypher = f"""
        MATCH (a:MoralAgent {{id: $id}})
        SET {set_clauses}
        RETURN a.id as id
        """
        g.query(cypher, props)

        log.debug(f"Updated agent {agent_id} state, Kala={agent.kala_current:.3f}")
        return agent

    def delete_agent(self, agent_id: str) -> bool:
        """Delete agent and its kuleana edges (soft delete - keeps history)"""
        g = self._get_graph()

        # Mark as deleted rather than hard delete
        cypher = """
        MATCH (a:MoralAgent {id: $id})
        SET a.deleted_at = $ts,
            a.active = false
        RETURN a.id as id
        """
        result = g.query(cypher, {"id": agent_id, "ts": time.time()})
        return len(result) > 0

    # === KULEANA MANAGEMENT ===

    def create_kuleana_edge(self, agent_id: str, edge: KuleanaEdge) -> bool:
        """Create a kuleana relationship from agent to target"""
        g = self._get_graph()

        # Determine target label from type
        target_label = edge.target_type

        cypher = f"""
        MATCH (a:MoralAgent {{id: $agent_id}})
        MATCH (t:{target_label} {{id: $target_id}})
        CREATE (a)-[r:{edge.rel_type.value} {{
            scope: $scope,
            since: $since,
            revocable: $revocable,
            metadata: $metadata
        }}]->(t)
        RETURN id(r) as rel_id
        """

        result = g.query(cypher, {
            "agent_id": agent_id,
            "target_id": edge.target_id,
            "scope": edge.scope,
            "since": edge.since,
            "revocable": edge.revocable,
            "metadata": json.dumps(edge.metadata)
        })

        if result:
            log.info(f"Created {edge.rel_type.value} edge: {agent_id} -> {edge.target_id}")
            return True
        return False

    def remove_kuleana_edge(
        self,
        agent_id: str,
        rel_type: KuleanaType,
        target_id: str
    ) -> bool:
        """Remove a kuleana edge (if revocable)"""
        g = self._get_graph()

        cypher = f"""
        MATCH (a:MoralAgent {{id: $agent_id}})-[r:{rel_type.value} {{revocable: true}}]->(t {{id: $target_id}})
        DELETE r
        RETURN count(*) as deleted
        """
        result = g.query(cypher, {"agent_id": agent_id, "target_id": target_id})
        return result[0].get("deleted", 0) > 0 if result else False

    def get_agent_kuleana(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get all kuleana edges for an agent"""
        g = self._get_graph()

        cypher = """
        MATCH (a:MoralAgent {id: $agent_id})-[r:STEWARD_OF|RESPONSIBLE_FOR|ACCOUNTABLE_TO|DELEGATED_BY]->(t)
        RETURN type(r) as relationship, t.id as target_id, labels(t)[0] as target_type,
               r.scope as scope, r.since as since
        """
        return g.query(cypher, {"agent_id": agent_id})

    # === USER PROXY AGENTS ===

    def create_human(
        self,
        human_id: Optional[str] = None,
        name: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """Create a human node"""
        g = self._get_graph()

        hid = human_id or self._generate_id("human")

        cypher = """
        MERGE (h:Human {id: $id})
        SET h.name = $name,
            h.created_at = $ts,
            h.metadata = $metadata
        RETURN h.id as id
        """
        result = g.query(cypher, {
            "id": hid,
            "name": name or hid,
            "ts": time.time(),
            "metadata": json.dumps(metadata or {})
        })
        return hid

    def create_user_proxy(
        self,
        human_id: str,
        autonomy_level: float = 0.5
    ) -> MoralAgent:
        """
        Create a user proxy agent bound to a human.

        The proxy:
        - Is STEWARD_OF the human (caretaking)
        - Is ACCOUNTABLE_TO the human (answerable)
        - Has adjustable autonomy dial
        - Persists across sessions
        - Accumulates its own reputation and Kala
        """
        kuleana = [
            KuleanaEdge(
                rel_type=KuleanaType.STEWARD_OF,
                target_id=human_id,
                target_type="Human",
                scope="representation"
            ),
            KuleanaEdge(
                rel_type=KuleanaType.ACCOUNTABLE_TO,
                target_id=human_id,
                target_type="Human",
                scope="all_actions"
            )
        ]

        proxy = self.create_agent(
            agent_type=AgentType.USER_PROXY,
            kuleana=kuleana,
            autonomy_level=autonomy_level,
            metadata={"human_id": human_id}
        )

        log.info(f"Created user proxy {proxy.id} for human {human_id}")
        return proxy

    def get_user_proxy(self, human_id: str) -> Optional[MoralAgent]:
        """Get existing user proxy for a human"""
        g = self._get_graph()

        cypher = """
        MATCH (a:MoralAgent)-[:STEWARD_OF]->(h:Human {id: $human_id})
        WHERE a.agent_type = 'user_proxy' AND (a.deleted_at IS NULL OR a.active = true)
        RETURN a
        LIMIT 1
        """
        result = g.query(cypher, {"human_id": human_id})

        if not result:
            return None

        props = result[0].get("a", result[0])
        return MoralAgent.from_graph_properties(props)

    def get_or_create_user_proxy(
        self,
        human_id: str,
        autonomy_level: float = 0.5
    ) -> MoralAgent:
        """Get existing proxy or create new one"""
        proxy = self.get_user_proxy(human_id)
        if proxy:
            return proxy

        # Ensure human exists
        self.create_human(human_id)
        return self.create_user_proxy(human_id, autonomy_level)

    # === DOMAIN & ENTITY MANAGEMENT ===

    def create_domain(
        self,
        name: str,
        description: str = "",
        domain_id: Optional[str] = None
    ) -> str:
        """Create a domain node"""
        g = self._get_graph()
        did = domain_id or self._generate_id("domain")

        cypher = """
        MERGE (d:Domain {id: $id})
        SET d.name = $name,
            d.description = $description,
            d.created_at = $ts
        RETURN d.id as id
        """
        g.query(cypher, {
            "id": did,
            "name": name,
            "description": description,
            "ts": time.time()
        })
        return did

    def create_entity(
        self,
        name: str,
        entity_type: str,
        entity_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """Create an entity node"""
        g = self._get_graph()
        eid = entity_id or self._generate_id("entity")

        cypher = """
        MERGE (e:Entity {id: $id})
        SET e.name = $name,
            e.type = $entity_type,
            e.created_at = $ts,
            e.metadata = $metadata
        RETURN e.id as id
        """
        g.query(cypher, {
            "id": eid,
            "name": name,
            "entity_type": entity_type,
            "ts": time.time(),
            "metadata": json.dumps(metadata or {})
        })
        return eid

    def create_community(
        self,
        name: str,
        community_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """Create a community node"""
        g = self._get_graph()
        cid = community_id or self._generate_id("community")

        cypher = """
        MERGE (c:Community {id: $id})
        SET c.name = $name,
            c.created_at = $ts,
            c.metadata = $metadata
        RETURN c.id as id
        """
        g.query(cypher, {
            "id": cid,
            "name": name,
            "ts": time.time(),
            "metadata": json.dumps(metadata or {})
        })
        return cid

    # === CONTEXT & SME AGENTS ===

    def create_context_agent(
        self,
        domain_id: str,
        accountable_to_id: str,
        accountable_to_type: str = "Human"
    ) -> MoralAgent:
        """
        Create a context agent for a specific domain.

        Context agents:
        - Are STEWARD_OF a domain
        - Do the actual work in conversations
        - Can be instantiated fresh or recalled from graph
        """
        kuleana = [
            KuleanaEdge(
                rel_type=KuleanaType.STEWARD_OF,
                target_id=domain_id,
                target_type="Domain",
                scope="full"
            ),
            KuleanaEdge(
                rel_type=KuleanaType.ACCOUNTABLE_TO,
                target_id=accountable_to_id,
                target_type=accountable_to_type,
                scope="domain_actions"
            )
        ]

        return self.create_agent(
            agent_type=AgentType.CONTEXT,
            kuleana=kuleana,
            metadata={"domain_id": domain_id}
        )

    def create_sme_agent(self, domain_id: str) -> MoralAgent:
        """Create a Subject Matter Expert agent"""
        kuleana = [
            KuleanaEdge(
                rel_type=KuleanaType.STEWARD_OF,
                target_id=domain_id,
                target_type="Domain",
                scope="knowledge"
            )
        ]

        return self.create_agent(
            agent_type=AgentType.SME,
            kuleana=kuleana,
            metadata={"domain_id": domain_id}
        )

    def create_sensor_agent(self, entity_id: str) -> MoralAgent:
        """Create a sensor agent for data streams"""
        kuleana = [
            KuleanaEdge(
                rel_type=KuleanaType.STEWARD_OF,
                target_id=entity_id,
                target_type="Entity",
                scope="data_stream"
            )
        ]

        return self.create_agent(
            agent_type=AgentType.SENSOR,
            kuleana=kuleana,
            metadata={"entity_id": entity_id}
        )

    def create_security_agent(self, domain_id: str) -> MoralAgent:
        """Create a security-focused agent"""
        kuleana = [
            KuleanaEdge(
                rel_type=KuleanaType.STEWARD_OF,
                target_id=domain_id,
                target_type="Domain",
                scope="security_validation"
            )
        ]

        return self.create_agent(
            agent_type=AgentType.SECURITY,
            kuleana=kuleana,
            initial_state=get_default_state("security"),
            metadata={"domain_id": domain_id}
        )

    # === ACTION RECORDING & REPUTATION ===

    def record_action(
        self,
        agent_id: str,
        action_type: str,
        description: str = "",
        metadata: Optional[Dict] = None,
        virtue_impacts: Optional[Dict[str, float]] = None
    ) -> Optional[str]:
        """
        Record an action performed by an agent.

        Actions form the reputation trail - observable history.
        """
        g = self._get_graph()
        action_id = self._generate_id("action")
        ts = time.time()

        cypher = """
        MATCH (a:MoralAgent {id: $agent_id})
        CREATE (act:Action {
            id: $action_id,
            action_type: $action_type,
            description: $description,
            timestamp: $ts,
            metadata: $metadata
        })
        CREATE (a)-[:PERFORMED {timestamp: $ts}]->(act)
        RETURN act.id as id
        """

        result = g.query(cypher, {
            "agent_id": agent_id,
            "action_id": action_id,
            "action_type": action_type,
            "description": description,
            "ts": ts,
            "metadata": json.dumps(metadata or {})
        })

        if result and virtue_impacts:
            # Update agent's moral state based on action
            self.update_agent_state(agent_id, action_impacts=virtue_impacts)

        return action_id if result else None

    def get_reputation(self, agent_id: str) -> Dict[str, Any]:
        """
        Get agent's reputation - retrospective, auditable trail.

        Reputation is observable history, not a score.
        """
        g = self._get_graph()

        cypher = """
        MATCH (a:MoralAgent {id: $agent_id})
        OPTIONAL MATCH (a)-[:PERFORMED]->(act:Action)
        OPTIONAL MATCH (a)-[p:PARTICIPATED_IN]->(ev:Event)
        WITH a, collect(DISTINCT act) as actions, collect(DISTINCT {event: ev, participation: p}) as participations
        RETURN
            a.kala_current as kala_current,
            size(actions) as total_actions,
            [act IN actions | {
                id: act.id,
                type: act.action_type,
                timestamp: act.timestamp,
                description: act.description
            }] as action_history,
            [p IN participations WHERE p.event IS NOT NULL | {
                event_id: p.event.id,
                kala_earned: p.participation.kala_earned
            }] as participation_history
        """

        result = g.query(cypher, {"agent_id": agent_id})

        if not result:
            return {"error": "Agent not found"}

        r = result[0]
        total_kala_earned = sum(
            p.get("kala_earned", 0) or 0
            for p in (r.get("participation_history") or [])
        )

        return {
            "agent_id": agent_id,
            "kala_current": r.get("kala_current", 0),
            "total_actions": r.get("total_actions", 0),
            "total_kala_earned": total_kala_earned,
            "action_history": r.get("action_history", []),
            "participation_history": r.get("participation_history", [])
        }

    # === EVENT PARTICIPATION ===

    def create_event(
        self,
        name: str,
        kala_rate: float = 50.0,
        event_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """Create an event for participation tracking"""
        g = self._get_graph()
        eid = event_id or self._generate_id("event")

        cypher = """
        CREATE (e:Event {
            id: $id,
            name: $name,
            timestamp: $ts,
            kala_rate: $kala_rate,
            metadata: $metadata
        })
        RETURN e.id as id
        """
        g.query(cypher, {
            "id": eid,
            "name": name,
            "ts": time.time(),
            "kala_rate": kala_rate,
            "metadata": json.dumps(metadata or {})
        })
        return eid

    def record_participation(
        self,
        agent_id: str,
        event_id: str,
        duration_hours: float
    ) -> float:
        """
        Record agent participation in an event.

        Kala accrues at event's rate (e.g., 50 Kala/hour).
        Non-transferable. Cannot be spent. Recognition of presence.
        """
        g = self._get_graph()

        # Get event's kala rate
        event_query = """
        MATCH (e:Event {id: $event_id})
        RETURN e.kala_rate as kala_rate
        """
        event_result = g.query(event_query, {"event_id": event_id})

        if not event_result:
            return 0.0

        kala_rate = event_result[0].get("kala_rate", 50.0)
        kala_earned = kala_rate * duration_hours

        # Create participation edge
        cypher = """
        MATCH (a:MoralAgent {id: $agent_id})
        MATCH (e:Event {id: $event_id})
        CREATE (a)-[:PARTICIPATED_IN {
            duration_hours: $duration,
            kala_earned: $kala_earned,
            timestamp: $ts
        }]->(e)
        RETURN $kala_earned as earned
        """
        g.query(cypher, {
            "agent_id": agent_id,
            "event_id": event_id,
            "duration": duration_hours,
            "kala_earned": kala_earned,
            "ts": time.time()
        })

        # Update agent's Kala (Kala never decreases)
        agent = self.get_agent(agent_id)
        if agent:
            # Recalculate total Kala from state + participations
            state_kala = compute_kala(agent.state)
            g.query("""
                MATCH (a:MoralAgent {id: $agent_id})
                SET a.kala_current = $kala
                """, {"agent_id": agent_id, "kala": state_kala + kala_earned}
            )

        log.info(f"Agent {agent_id} earned {kala_earned:.1f} Kala from event {event_id}")
        return kala_earned

    # === ACCESS CONTROL VIA KULEANA ===

    def can_access(self, agent_id: str, resource_id: str) -> bool:
        """
        Derive access from kuleana relationships.

        NO separate ACL edges. Access = f(kuleana).
        """
        g = self._get_graph()

        # Direct kuleana
        cypher = """
        MATCH (a:MoralAgent {id: $agent_id})-[:STEWARD_OF|RESPONSIBLE_FOR]->(r {id: $resource_id})
        RETURN count(*) > 0 as has_access
        """
        result = g.query(cypher, {"agent_id": agent_id, "resource_id": resource_id})

        if result and result[0].get("has_access"):
            return True

        # Delegated access
        cypher = """
        MATCH (a:MoralAgent {id: $agent_id})<-[:DELEGATED_BY]-(delegator)-[:STEWARD_OF|RESPONSIBLE_FOR]->(r {id: $resource_id})
        RETURN count(*) > 0 as has_delegated_access
        """
        result = g.query(cypher, {"agent_id": agent_id, "resource_id": resource_id})

        return result[0].get("has_delegated_access", False) if result else False

    def validate_action(
        self,
        agent_id: str,
        action_type: str,
        target_id: str
    ) -> bool:
        """
        Validate that an action falls within agent's kuleana scope.

        Actions outside scope are rejected - not punished, just not possible.
        """
        kuleana = self.get_agent_kuleana(agent_id)

        for k in kuleana:
            if k.get("target_id") == target_id:
                scope = k.get("scope", "")
                # Check if action_type is within scope
                # "full" scope allows everything
                if scope == "full" or action_type in scope:
                    return True

        return False


# Singleton accessor
def get_moral_agent_graph() -> MoralAgentGraph:
    return MoralAgentGraph.get_instance()
