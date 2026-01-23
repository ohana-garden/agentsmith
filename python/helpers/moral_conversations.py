"""
Moral Conversation Architecture

Implements the three-layer conversation structure:
1. User Proxy Agent (bound to human)
2. Context Agent (bound to domain)
3. Supporting Agents (SMEs, sensors, etc.)

Message routing:
Human → User Proxy → Context Agent → SMEs → Context Agent → User Proxy → Human

The proxy's autonomy_level determines how much it interprets vs echoes.
"""

import time
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum

from python.helpers.moral_agents import (
    MoralAgentGraph, MoralAgent, AgentType, KuleanaEdge, KuleanaType,
    get_moral_agent_graph
)
from python.helpers.moral_geometry import compute_kala, update_state_from_action

log = logging.getLogger("moral_conversations")


class MessageRole(Enum):
    HUMAN = "human"
    USER_PROXY = "user_proxy"
    CONTEXT = "context"
    SME = "sme"
    SYSTEM = "system"


@dataclass
class Message:
    """A message in a conversation"""
    content: str
    role: MessageRole
    agent_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationState:
    """Runtime state for a conversation"""
    conversation_id: str
    human_id: str
    proxy_agent: MoralAgent
    context_agent: MoralAgent
    supporting_agents: List[MoralAgent] = field(default_factory=list)
    messages: List[Message] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationManager:
    """
    Manages conversations with three-layer architecture.

    Responsibilities:
    - Initialize conversations with proper agent structure
    - Route messages through agent layers
    - Record actions for reputation
    - Manage agent state updates
    """

    _instance: Optional["ConversationManager"] = None

    def __init__(self):
        self._graph = None
        self._agent_graph = None
        self._active_conversations: Dict[str, ConversationState] = {}

    @classmethod
    def get_instance(cls) -> "ConversationManager":
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

    def _get_agent_graph(self) -> MoralAgentGraph:
        if self._agent_graph is None:
            self._agent_graph = get_moral_agent_graph()
        return self._agent_graph

    def _generate_id(self, prefix: str = "conv") -> str:
        import hashlib
        ts = str(time.time()).encode()
        return f"{prefix}_{hashlib.md5(ts).hexdigest()[:12]}"

    # === CONVERSATION LIFECYCLE ===

    def start_conversation(
        self,
        human_id: str,
        domain_id: str,
        additional_agent_ids: Optional[List[str]] = None
    ) -> ConversationState:
        """
        Initialize a conversation with the standard three-layer structure:
        1. User Proxy Agent (bound to human)
        2. Context Agent (bound to domain)
        3. Supporting Agents (SMEs, sensors, etc.)
        """
        g = self._get_graph()
        ag = self._get_agent_graph()

        conversation_id = self._generate_id("conv")

        # Get or create user proxy
        proxy = ag.get_or_create_user_proxy(human_id)

        # Get or create context agent for domain
        context_agent = self._get_or_create_context_agent(domain_id, human_id)

        # Create conversation node in graph
        cypher = """
        CREATE (c:Conversation {
            id: $id,
            started_at: $ts,
            human_id: $human_id,
            domain_id: $domain_id,
            metadata: $metadata
        })
        RETURN c.id as id
        """
        g.query(cypher, {
            "id": conversation_id,
            "ts": time.time(),
            "human_id": human_id,
            "domain_id": domain_id,
            "metadata": json.dumps({})
        })

        # Link participants to conversation
        self._link_to_conversation(conversation_id, proxy.id, "user_proxy")
        self._link_to_conversation(conversation_id, context_agent.id, "context")

        # Add supporting agents
        supporting_agents = []
        for agent_id in (additional_agent_ids or []):
            agent = ag.get_agent(agent_id)
            if agent:
                self._link_to_conversation(conversation_id, agent_id, agent.agent_type.value)
                supporting_agents.append(agent)

        # Link human
        g.query("""
            MATCH (h:Human {id: $human_id}), (c:Conversation {id: $conv_id})
            CREATE (h)-[:INITIATED {timestamp: $ts}]->(c)
        """, {"human_id": human_id, "conv_id": conversation_id, "ts": time.time()})

        # Create state object
        state = ConversationState(
            conversation_id=conversation_id,
            human_id=human_id,
            proxy_agent=proxy,
            context_agent=context_agent,
            supporting_agents=supporting_agents
        )

        # Cache active conversation
        self._active_conversations[conversation_id] = state

        log.info(f"Started conversation {conversation_id}: human={human_id}, domain={domain_id}")

        # Record as episode
        g.add_episode(
            content=f"Conversation {conversation_id} started",
            source="system",
            episode_type="conversation_start",
            metadata={
                "conversation_id": conversation_id,
                "human_id": human_id,
                "domain_id": domain_id,
                "proxy_id": proxy.id,
                "context_agent_id": context_agent.id
            }
        )

        return state

    def _get_or_create_context_agent(
        self,
        domain_id: str,
        human_id: str
    ) -> MoralAgent:
        """Get existing context agent for domain or create new one"""
        g = self._get_graph()
        ag = self._get_agent_graph()

        # Check for existing context agent for this domain
        cypher = """
        MATCH (a:MoralAgent)-[:STEWARD_OF]->(d:Domain {id: $domain_id})
        WHERE a.agent_type = 'context' AND (a.deleted_at IS NULL OR a.active = true)
        RETURN a
        LIMIT 1
        """
        result = g.query(cypher, {"domain_id": domain_id})

        if result:
            props = result[0].get("a", result[0])
            return MoralAgent.from_graph_properties(props)

        # Create new context agent
        return ag.create_context_agent(domain_id, human_id, "Human")

    def _link_to_conversation(
        self,
        conversation_id: str,
        agent_id: str,
        role: str
    ):
        """Link an agent to a conversation"""
        g = self._get_graph()
        g.query("""
            MATCH (a:MoralAgent {id: $agent_id}), (c:Conversation {id: $conv_id})
            CREATE (a)-[:PARTICIPATED_IN_CONVERSATION {role: $role, joined_at: $ts}]->(c)
        """, {
            "agent_id": agent_id,
            "conv_id": conversation_id,
            "role": role,
            "ts": time.time()
        })

    def get_conversation(self, conversation_id: str) -> Optional[ConversationState]:
        """Get conversation state"""
        # Check cache first
        if conversation_id in self._active_conversations:
            return self._active_conversations[conversation_id]

        # Load from graph
        return self._load_conversation(conversation_id)

    def _load_conversation(self, conversation_id: str) -> Optional[ConversationState]:
        """Load conversation from graph"""
        g = self._get_graph()
        ag = self._get_agent_graph()

        cypher = """
        MATCH (c:Conversation {id: $conv_id})
        OPTIONAL MATCH (h:Human)-[:INITIATED]->(c)
        OPTIONAL MATCH (proxy:MoralAgent)-[:PARTICIPATED_IN_CONVERSATION {role: 'user_proxy'}]->(c)
        OPTIONAL MATCH (ctx:MoralAgent)-[:PARTICIPATED_IN_CONVERSATION {role: 'context'}]->(c)
        RETURN c, h.id as human_id, proxy, ctx
        """
        result = g.query(cypher, {"conv_id": conversation_id})

        if not result:
            return None

        r = result[0]
        conv = r.get("c", {})
        human_id = r.get("human_id", "")

        proxy_props = r.get("proxy")
        ctx_props = r.get("ctx")

        if not proxy_props or not ctx_props:
            return None

        proxy = MoralAgent.from_graph_properties(proxy_props)
        context_agent = MoralAgent.from_graph_properties(ctx_props)

        # Load supporting agents
        support_query = """
        MATCH (a:MoralAgent)-[p:PARTICIPATED_IN_CONVERSATION]->(c:Conversation {id: $conv_id})
        WHERE p.role NOT IN ['user_proxy', 'context']
        RETURN a
        """
        support_result = g.query(support_query, {"conv_id": conversation_id})
        supporting = [
            MoralAgent.from_graph_properties(r.get("a", r))
            for r in support_result
        ]

        state = ConversationState(
            conversation_id=conversation_id,
            human_id=human_id,
            proxy_agent=proxy,
            context_agent=context_agent,
            supporting_agents=supporting,
            started_at=conv.get("started_at", time.time())
        )

        self._active_conversations[conversation_id] = state
        return state

    def end_conversation(self, conversation_id: str):
        """End a conversation"""
        g = self._get_graph()

        g.query("""
            MATCH (c:Conversation {id: $conv_id})
            SET c.ended_at = $ts
        """, {"conv_id": conversation_id, "ts": time.time()})

        # Record as episode
        g.add_episode(
            content=f"Conversation {conversation_id} ended",
            source="system",
            episode_type="conversation_end",
            metadata={"conversation_id": conversation_id}
        )

        # Remove from cache
        if conversation_id in self._active_conversations:
            del self._active_conversations[conversation_id]

        log.info(f"Ended conversation {conversation_id}")

    # === MESSAGE ROUTING ===

    def route_message(
        self,
        conversation_id: str,
        message: str,
        from_human: bool = True
    ) -> Dict[str, Any]:
        """
        Route a message through the agent architecture.

        Human messages:
        1. Received by user proxy
        2. Proxy transmits to context agent (based on autonomy level)
        3. Context agent may consult SMEs/sensors
        4. Response flows back through proxy to human

        The proxy's autonomy_level determines how much it interprets vs echoes.
        """
        state = self.get_conversation(conversation_id)
        if not state:
            return {"error": f"Conversation {conversation_id} not found"}

        ag = self._get_agent_graph()
        proxy = state.proxy_agent
        context_agent = state.context_agent

        # Record incoming message
        incoming = Message(
            content=message,
            role=MessageRole.HUMAN if from_human else MessageRole.SYSTEM,
            timestamp=time.time()
        )
        state.messages.append(incoming)

        if from_human:
            # Process through proxy based on autonomy level
            processed_message = self._proxy_process(proxy, message)

            # Context agent handles the work
            response_content = self._context_process(
                context_agent,
                processed_message,
                state
            )

            # Proxy may modify response based on autonomy
            final_response = self._proxy_respond(proxy, response_content)

            # Record action for reputation
            ag.record_action(
                context_agent.id,
                "processed_message",
                description=f"Processed message in conversation {conversation_id}",
                virtue_impacts={"service": 0.01, "truthfulness": 0.005}
            )

            # Update Kala for both agents
            proxy = ag.update_agent_state(proxy.id)
            context_agent = ag.update_agent_state(context_agent.id)

            # Record response message
            response_msg = Message(
                content=final_response,
                role=MessageRole.CONTEXT,
                agent_id=context_agent.id,
                timestamp=time.time()
            )
            state.messages.append(response_msg)

            return {
                "response": final_response,
                "conversation_id": conversation_id,
                "proxy_id": proxy.id if proxy else None,
                "context_agent_id": context_agent.id if context_agent else None,
                "message_count": len(state.messages)
            }

        return {"error": "Non-human messages not yet implemented"}

    def _proxy_process(self, proxy: MoralAgent, message: str) -> str:
        """
        Process incoming message through user proxy.

        autonomy_level determines behavior:
        - < 0.2: Echo mode, transmit verbatim
        - 0.2-0.8: Semi-autonomous, may add context
        - > 0.8: Full autonomous, may interpret/transform
        """
        autonomy = proxy.autonomy_level

        if autonomy < 0.2:
            # Echo mode - verbatim transmission
            return message

        # Semi-autonomous or higher - would add context/interpretation
        # For now, just add metadata about autonomy level
        # In production, this would involve LLM processing
        context_prefix = f"[autonomy={autonomy:.1f}] "
        return context_prefix + message

    def _context_process(
        self,
        context_agent: MoralAgent,
        message: str,
        state: ConversationState
    ) -> str:
        """
        Process message in context agent.

        In production, this would:
        1. Check kuleana scope
        2. Possibly consult SMEs
        3. Generate response via LLM
        4. Validate response against moral geometry

        For now, returns acknowledgment placeholder.
        """
        # Check if we need SME consultation
        sme_responses = []
        for sme in state.supporting_agents:
            # In production, would route to appropriate SME based on content
            pass

        # Placeholder response - in production would use LLM
        response = f"Context agent {context_agent.id} received: {message[:100]}..."

        # In production: generate via LLM, validate against moral geometry
        return response

    def _proxy_respond(self, proxy: MoralAgent, response: str) -> str:
        """
        Process outgoing response through proxy.

        May modify based on autonomy level and human preferences.
        """
        autonomy = proxy.autonomy_level

        if autonomy < 0.2:
            # Echo mode - pass through
            return response

        # Higher autonomy could filter/modify
        return response

    # === SME CONSULTATION ===

    def add_supporting_agent(
        self,
        conversation_id: str,
        agent_id: str
    ) -> bool:
        """Add a supporting agent to an active conversation"""
        state = self.get_conversation(conversation_id)
        if not state:
            return False

        ag = self._get_agent_graph()
        agent = ag.get_agent(agent_id)
        if not agent:
            return False

        self._link_to_conversation(conversation_id, agent_id, agent.agent_type.value)
        state.supporting_agents.append(agent)

        log.info(f"Added agent {agent_id} to conversation {conversation_id}")
        return True

    def consult_sme(
        self,
        conversation_id: str,
        sme_agent_id: str,
        query: str
    ) -> Dict[str, Any]:
        """
        Consult an SME agent within a conversation.

        Records the consultation as an action for both agents.
        """
        state = self.get_conversation(conversation_id)
        if not state:
            return {"error": "Conversation not found"}

        ag = self._get_agent_graph()
        sme = ag.get_agent(sme_agent_id)
        if not sme:
            return {"error": "SME agent not found"}

        # Record consultation
        g = self._get_graph()
        g.query("""
            MATCH (ctx:MoralAgent {id: $ctx_id}), (sme:MoralAgent {id: $sme_id})
            CREATE (ctx)-[:CONSULTED {
                timestamp: $ts,
                topic: $topic,
                conversation_id: $conv_id
            }]->(sme)
        """, {
            "ctx_id": state.context_agent.id,
            "sme_id": sme_agent_id,
            "ts": time.time(),
            "topic": query[:200],
            "conv_id": conversation_id
        })

        # Record action
        ag.record_action(
            state.context_agent.id,
            "consulted_sme",
            description=f"Consulted SME {sme_agent_id}",
            virtue_impacts={"wisdom": 0.01, "humility": 0.01}
        )

        # In production, would route query to SME and get response
        return {
            "consultation_id": f"consult_{int(time.time())}",
            "sme_id": sme_agent_id,
            "query": query,
            "response": f"SME {sme_agent_id} response placeholder"
        }

    # === AUTONOMY MANAGEMENT ===

    def set_proxy_autonomy(
        self,
        human_id: str,
        autonomy_level: float
    ) -> Optional[MoralAgent]:
        """
        Set the autonomy level for a user's proxy agent.

        0.0 = echo (proxy transmits verbatim)
        0.5 = semi-autonomous (proxy may add context)
        1.0 = full autonomy (proxy may interpret/transform)
        """
        ag = self._get_agent_graph()
        proxy = ag.get_user_proxy(human_id)

        if not proxy:
            return None

        # Clamp to valid range
        autonomy_level = max(0.0, min(1.0, autonomy_level))

        g = self._get_graph()
        g.query("""
            MATCH (a:MoralAgent {id: $agent_id})
            SET a.autonomy_level = $autonomy
        """, {"agent_id": proxy.id, "autonomy": autonomy_level})

        proxy.autonomy_level = autonomy_level
        log.info(f"Set proxy {proxy.id} autonomy to {autonomy_level}")

        return proxy

    # === CONVERSATION QUERIES ===

    def get_conversation_history(
        self,
        conversation_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get message history for a conversation"""
        state = self.get_conversation(conversation_id)
        if not state:
            return []

        return [
            {
                "content": m.content,
                "role": m.role.value,
                "agent_id": m.agent_id,
                "timestamp": m.timestamp
            }
            for m in state.messages[-limit:]
        ]

    def get_human_conversations(
        self,
        human_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get all conversations for a human"""
        g = self._get_graph()

        cypher = """
        MATCH (h:Human {id: $human_id})-[:INITIATED]->(c:Conversation)
        RETURN c.id as id, c.started_at as started_at, c.ended_at as ended_at
        ORDER BY c.started_at DESC
        LIMIT $limit
        """
        return g.query(cypher, {"human_id": human_id, "limit": limit})


# Singleton accessor
def get_conversation_manager() -> ConversationManager:
    return ConversationManager.get_instance()
