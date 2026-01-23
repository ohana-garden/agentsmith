"""
Moral Geometry Instrument for AgentSmith
Graph-native agents with 19D moral manifold + Kala flow

Usage in agent chat:
    exec(open('/a0/instruments/moral.py').read())

    # Setup schema (first time only)
    setup()

    # Create a user proxy for a human
    proxy = create_user_proxy("human_123")

    # Create a context agent for a domain
    ctx = create_context_agent("software_development", "human_123")

    # Start a conversation
    conv = start_conversation("human_123", "software_development")

    # Route a message
    response = route_message(conv["conversation_id"], "Build a REST API")

    # Check Kala (alignment/flow)
    kala = compute_kala("agent_xyz")

    # View reputation trail
    rep = get_reputation("agent_xyz")
"""

import os
import json
import traceback
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if os.getenv("DEBUG") else logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("moral")

# Import modules
MODULES_AVAILABLE = True
try:
    from python.helpers.moral_geometry import (
        MoralState, MoralManifold, VIRTUES,
        compute_kala as _compute_kala,
        compute_kala_gradient as _compute_kala_gradient,
        project_to_valid_manifold,
        update_state_from_action,
        get_default_state
    )
    log.info("Moral geometry module loaded")
except ImportError as e:
    MODULES_AVAILABLE = False
    log.error(f"Moral geometry import failed: {e}")

try:
    from python.helpers.moral_agents import (
        MoralAgentGraph, MoralAgent, AgentType, KuleanaType, KuleanaEdge,
        get_moral_agent_graph
    )
    log.info("Moral agents module loaded")
except ImportError as e:
    MODULES_AVAILABLE = False
    log.error(f"Moral agents import failed: {e}")

try:
    from python.helpers.moral_conversations import (
        ConversationManager, ConversationState,
        get_conversation_manager
    )
    log.info("Moral conversations module loaded")
except ImportError as e:
    MODULES_AVAILABLE = False
    log.error(f"Moral conversations import failed: {e}")


# === SETUP ===

def setup() -> bool:
    """Initialize moral geometry schema in the graph"""
    if not MODULES_AVAILABLE:
        print("[!] Moral geometry modules not available")
        return False

    ag = get_moral_agent_graph()
    result = ag.setup_schema()

    if result:
        print("[+] Moral geometry schema initialized")
    return result


def health_check() -> Dict[str, Any]:
    """Check moral geometry system health"""
    if not MODULES_AVAILABLE:
        return {"status": "error", "message": "Modules not available"}

    from python.helpers.temporal_graph import get_graph
    g = get_graph()
    g.connect()

    # Base health
    health = g.health_check()

    # Count moral agents
    agents = g.query("""
        MATCH (a:MoralAgent)
        WHERE a.deleted_at IS NULL
        RETURN count(a) as count
    """)

    # Count conversations
    convs = g.query("""
        MATCH (c:Conversation)
        RETURN count(c) as count
    """)

    health["moral_agents"] = agents[0]["count"] if agents else 0
    health["conversations"] = convs[0]["count"] if convs else 0

    print(f"[+] Health: {health['status']}")
    print(f"    Moral agents: {health['moral_agents']}")
    print(f"    Conversations: {health['conversations']}")

    return health


# === AGENT CREATION ===

def create_agent(
    agent_type: str = "general",
    autonomy_level: float = 0.5,
    metadata: Optional[Dict] = None
) -> MoralAgent:
    """
    Create a new moral agent.

    Types: user_proxy, context, sme, sensor, security, general
    Autonomy: 0.0 (echo) to 1.0 (full autonomy)
    """
    if not MODULES_AVAILABLE:
        raise RuntimeError("Modules not available")

    ag = get_moral_agent_graph()
    agent = ag.create_agent(
        agent_type=agent_type,
        autonomy_level=autonomy_level,
        metadata=metadata or {}
    )

    print(f"[+] Created agent {agent.id} ({agent_type})")
    print(f"    Autonomy: {autonomy_level}")
    print(f"    Kala: {agent.kala_current:.4f}")

    return agent


def create_user_proxy(
    human_id: str,
    autonomy_level: float = 0.5
) -> MoralAgent:
    """
    Create a user proxy agent bound to a human.

    The proxy:
    - Is STEWARD_OF the human (caretaking)
    - Is ACCOUNTABLE_TO the human (answerable)
    - Has adjustable autonomy dial
    """
    if not MODULES_AVAILABLE:
        raise RuntimeError("Modules not available")

    ag = get_moral_agent_graph()

    # Ensure human exists
    ag.create_human(human_id)

    proxy = ag.create_user_proxy(human_id, autonomy_level)

    print(f"[+] Created user proxy {proxy.id}")
    print(f"    Human: {human_id}")
    print(f"    Autonomy: {autonomy_level}")

    return proxy


def create_context_agent(
    domain_name: str,
    accountable_to_id: str,
    accountable_to_type: str = "Human"
) -> MoralAgent:
    """
    Create a context agent for a specific domain.

    Context agents do the actual work in conversations.
    """
    if not MODULES_AVAILABLE:
        raise RuntimeError("Modules not available")

    ag = get_moral_agent_graph()

    # Ensure domain exists
    domain_id = ag.create_domain(domain_name, f"Domain: {domain_name}")

    agent = ag.create_context_agent(domain_id, accountable_to_id, accountable_to_type)

    print(f"[+] Created context agent {agent.id}")
    print(f"    Domain: {domain_name}")
    print(f"    Accountable to: {accountable_to_id}")

    return agent


def create_sme_agent(domain_name: str) -> MoralAgent:
    """Create a Subject Matter Expert agent"""
    if not MODULES_AVAILABLE:
        raise RuntimeError("Modules not available")

    ag = get_moral_agent_graph()
    domain_id = ag.create_domain(domain_name, f"SME Domain: {domain_name}")

    agent = ag.create_sme_agent(domain_id)

    print(f"[+] Created SME agent {agent.id}")
    print(f"    Domain: {domain_name}")

    return agent


def create_security_agent(domain_name: str = "security") -> MoralAgent:
    """Create a security-focused agent"""
    if not MODULES_AVAILABLE:
        raise RuntimeError("Modules not available")

    ag = get_moral_agent_graph()
    domain_id = ag.create_domain(domain_name, f"Security Domain: {domain_name}")

    agent = ag.create_security_agent(domain_id)

    print(f"[+] Created security agent {agent.id}")
    print(f"    Initial Kala: {agent.kala_current:.4f}")
    print(f"    Truthfulness: {agent.state.truthfulness:.2f}")
    print(f"    Justice: {agent.state.justice:.2f}")
    print(f"    Courage: {agent.state.courage:.2f}")

    return agent


# === AGENT OPERATIONS ===

def get_agent(agent_id: str) -> Optional[MoralAgent]:
    """Get an agent by ID"""
    if not MODULES_AVAILABLE:
        return None

    ag = get_moral_agent_graph()
    return ag.get_agent(agent_id)


def update_agent(
    agent_id: str,
    action_impacts: Dict[str, float]
) -> Optional[MoralAgent]:
    """
    Update agent's moral state based on action impacts.

    action_impacts: Dict mapping virtue names to impact values
    Example: {"truthfulness": 0.1, "service": -0.05}
    """
    if not MODULES_AVAILABLE:
        return None

    ag = get_moral_agent_graph()
    agent = ag.update_agent_state(agent_id, action_impacts=action_impacts)

    if agent:
        print(f"[+] Updated agent {agent_id}")
        print(f"    New Kala: {agent.kala_current:.4f}")

    return agent


def show_agent(agent_id: str) -> None:
    """Display detailed agent information"""
    agent = get_agent(agent_id)
    if not agent:
        print(f"[!] Agent {agent_id} not found")
        return

    print(f"\n=== Agent {agent.id} ===")
    print(f"Type: {agent.agent_type.value}")
    print(f"Autonomy: {agent.autonomy_level:.2f}")
    print(f"Kala: {agent.kala_current:.4f}")
    print(f"\nMoral State (Position):")
    for virtue in VIRTUES:
        val = getattr(agent.state, virtue)
        bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
        print(f"  {virtue:15s} [{bar}] {val:.3f}")
    print(f"\nMomentum:")
    for virtue in VIRTUES:
        dval = getattr(agent.state, f"d_{virtue}")
        if abs(dval) > 0.001:
            direction = "↑" if dval > 0 else "↓"
            print(f"  {virtue:15s} {direction} {abs(dval):.4f}")


# === MORAL GEOMETRY ===

def compute_kala(agent_id: str) -> float:
    """
    Compute Kala (alignment/flow) for an agent.

    Higher Kala = more in-flow with community-positive outcomes.
    """
    if not MODULES_AVAILABLE:
        return 0.0

    ag = get_moral_agent_graph()
    agent = ag.get_agent(agent_id)

    if not agent:
        print(f"[!] Agent {agent_id} not found")
        return 0.0

    kala = _compute_kala(agent.state)

    print(f"[+] Kala for {agent_id}: {kala:.4f}")
    return kala


def compute_gradient(agent_id: str) -> Dict[str, float]:
    """
    Compute gradient of Kala field.

    Returns direction of increasing alignment - which virtues to develop.
    """
    if not MODULES_AVAILABLE:
        return {}

    ag = get_moral_agent_graph()
    agent = ag.get_agent(agent_id)

    if not agent:
        print(f"[!] Agent {agent_id} not found")
        return {}

    gradient = _compute_kala_gradient(agent.state)
    gradient_dict = {VIRTUES[i]: gradient[i] for i in range(len(VIRTUES))}

    # Find strongest pull
    max_virtue = max(gradient_dict, key=gradient_dict.get)

    print(f"[+] Kala gradient for {agent_id}")
    print(f"    Strongest pull: {max_virtue} (+{gradient_dict[max_virtue]:.4f})")

    return gradient_dict


def project_state(agent_id: str) -> MoralState:
    """
    Project agent's state to valid manifold surface.

    Invalid states are auto-corrected (mercy, not punishment).
    """
    if not MODULES_AVAILABLE:
        raise RuntimeError("Modules not available")

    ag = get_moral_agent_graph()
    agent = ag.get_agent(agent_id)

    if not agent:
        raise ValueError(f"Agent {agent_id} not found")

    manifold = MoralManifold.get_instance()
    was_valid = manifold.is_valid(agent.state)

    projected = project_to_valid_manifold(agent.state)

    if not was_valid:
        ag.update_agent_state(agent_id, new_state=projected)
        print(f"[+] State projected to valid surface")
    else:
        print(f"[+] State was already valid")

    return projected


# === KULEANA MANAGEMENT ===

def add_kuleana(
    agent_id: str,
    rel_type: str,
    target_id: str,
    target_type: str,
    scope: str = "full"
) -> bool:
    """
    Add a kuleana (responsibility) relationship.

    rel_type: STEWARD_OF, RESPONSIBLE_FOR, ACCOUNTABLE_TO, DELEGATED_BY
    target_type: Human, Agent, Entity, Domain, Community
    """
    if not MODULES_AVAILABLE:
        return False

    ag = get_moral_agent_graph()

    edge = KuleanaEdge(
        rel_type=KuleanaType(rel_type),
        target_id=target_id,
        target_type=target_type,
        scope=scope
    )

    success = ag.create_kuleana_edge(agent_id, edge)

    if success:
        print(f"[+] Created {rel_type}: {agent_id} -> {target_id}")

    return success


def get_kuleana(agent_id: str) -> List[Dict]:
    """Get all kuleana relationships for an agent"""
    if not MODULES_AVAILABLE:
        return []

    ag = get_moral_agent_graph()
    kuleana = ag.get_agent_kuleana(agent_id)

    print(f"\n=== Kuleana for {agent_id} ===")
    for k in kuleana:
        print(f"  {k.get('relationship')}: -> {k.get('target_id')}")
        print(f"    Scope: {k.get('scope')}")

    return kuleana


def can_access(agent_id: str, resource_id: str) -> bool:
    """
    Check if agent can access a resource.

    Access is derived from kuleana - NO separate ACL edges.
    """
    if not MODULES_AVAILABLE:
        return False

    ag = get_moral_agent_graph()
    has_access = ag.can_access(agent_id, resource_id)

    print(f"[{'+'if has_access else '!'}] {agent_id} {'CAN' if has_access else 'CANNOT'} access {resource_id}")

    return has_access


# === ACTIONS & REPUTATION ===

def record_action(
    agent_id: str,
    action_type: str,
    description: str = "",
    virtue_impacts: Optional[Dict[str, float]] = None
) -> Optional[str]:
    """
    Record an action performed by an agent.

    Actions form the reputation trail - observable history.
    """
    if not MODULES_AVAILABLE:
        return None

    ag = get_moral_agent_graph()
    action_id = ag.record_action(
        agent_id,
        action_type,
        description=description,
        virtue_impacts=virtue_impacts
    )

    if action_id:
        print(f"[+] Recorded action {action_id}")

    return action_id


def get_reputation(agent_id: str) -> Dict[str, Any]:
    """
    Get agent's reputation trail.

    Reputation is retrospective, auditable - what you've done.
    """
    if not MODULES_AVAILABLE:
        return {}

    ag = get_moral_agent_graph()
    rep = ag.get_reputation(agent_id)

    print(f"\n=== Reputation for {agent_id} ===")
    print(f"Kala (current): {rep.get('kala_current', 0):.4f}")
    print(f"Total actions: {rep.get('total_actions', 0)}")
    print(f"Total Kala earned: {rep.get('total_kala_earned', 0):.2f}")

    if rep.get("action_history"):
        print(f"\nRecent actions:")
        for a in rep["action_history"][:5]:
            print(f"  - [{a.get('type')}] {a.get('description', '')[:50]}")

    return rep


# === CONVERSATIONS ===

def start_conversation(
    human_id: str,
    domain_name: str,
    additional_agent_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Start a conversation with three-layer architecture:
    1. User Proxy (bound to human)
    2. Context Agent (bound to domain)
    3. Supporting Agents (SMEs, etc.)
    """
    if not MODULES_AVAILABLE:
        return {"error": "Modules not available"}

    cm = get_conversation_manager()
    ag = get_moral_agent_graph()

    # Ensure domain exists
    domain_id = ag.create_domain(domain_name, f"Conversation domain: {domain_name}")

    state = cm.start_conversation(human_id, domain_id, additional_agent_ids)

    print(f"\n=== Started Conversation ===")
    print(f"ID: {state.conversation_id}")
    print(f"Human: {human_id}")
    print(f"Domain: {domain_name}")
    print(f"Proxy: {state.proxy_agent.id}")
    print(f"Context Agent: {state.context_agent.id}")
    if state.supporting_agents:
        print(f"Supporting: {[a.id for a in state.supporting_agents]}")

    return {
        "conversation_id": state.conversation_id,
        "proxy_id": state.proxy_agent.id,
        "context_agent_id": state.context_agent.id,
        "supporting_agents": [a.id for a in state.supporting_agents]
    }


def route_message(
    conversation_id: str,
    message: str,
    from_human: bool = True
) -> Dict[str, Any]:
    """
    Route a message through the conversation architecture.

    Human → Proxy → Context Agent → SMEs → back to Human
    """
    if not MODULES_AVAILABLE:
        return {"error": "Modules not available"}

    cm = get_conversation_manager()
    result = cm.route_message(conversation_id, message, from_human)

    if "error" in result:
        print(f"[!] {result['error']}")
    else:
        print(f"[+] Message routed in {conversation_id}")
        print(f"    Response: {result.get('response', '')[:100]}...")

    return result


def end_conversation(conversation_id: str) -> None:
    """End a conversation"""
    if not MODULES_AVAILABLE:
        return

    cm = get_conversation_manager()
    cm.end_conversation(conversation_id)

    print(f"[+] Ended conversation {conversation_id}")


def set_autonomy(human_id: str, level: float) -> Optional[MoralAgent]:
    """
    Set autonomy level for a user's proxy.

    0.0 = echo (verbatim transmission)
    0.5 = semi-autonomous (may add context)
    1.0 = full autonomy (may interpret/transform)
    """
    if not MODULES_AVAILABLE:
        return None

    cm = get_conversation_manager()
    proxy = cm.set_proxy_autonomy(human_id, level)

    if proxy:
        print(f"[+] Set proxy autonomy to {level:.2f}")

    return proxy


# === EVENTS & PARTICIPATION ===

def create_event(
    name: str,
    kala_rate: float = 50.0
) -> str:
    """
    Create an event for participation tracking.

    Kala accrues at the specified rate per hour of participation.
    """
    if not MODULES_AVAILABLE:
        return ""

    ag = get_moral_agent_graph()
    event_id = ag.create_event(name, kala_rate)

    print(f"[+] Created event {event_id}")
    print(f"    Name: {name}")
    print(f"    Kala rate: {kala_rate}/hour")

    return event_id


def record_participation(
    agent_id: str,
    event_id: str,
    hours: float
) -> float:
    """
    Record agent participation in an event.

    Returns Kala earned (non-transferable, cannot decrease).
    """
    if not MODULES_AVAILABLE:
        return 0.0

    ag = get_moral_agent_graph()
    kala_earned = ag.record_participation(agent_id, event_id, hours)

    print(f"[+] Recorded {hours}h participation")
    print(f"    Kala earned: {kala_earned:.2f}")

    return kala_earned


# === DISPLAY FUNCTIONS ===

def show_virtues() -> None:
    """Display the nine virtues"""
    print("\n=== The Nine Virtues ===")
    descriptions = {
        "unity": "Oneness, interconnection with all",
        "justice": "Fairness, equity in actions",
        "truthfulness": "Honesty, authenticity (LOAD-BEARING)",
        "love": "Compassion, genuine care",
        "detachment": "Objectivity, non-attachment",
        "humility": "Modesty, openness to correction",
        "service": "Contribution to community",
        "courage": "Bravery, willingness to act",
        "wisdom": "Discernment, deep understanding"
    }
    for v in VIRTUES:
        print(f"  {v:15s} - {descriptions.get(v, '')}")


def show_kuleana_types() -> None:
    """Display kuleana relationship types"""
    print("\n=== Kuleana Types ===")
    print("  STEWARD_OF      - Caretaking responsibility")
    print("  RESPONSIBLE_FOR - Direct operational responsibility")
    print("  ACCOUNTABLE_TO  - Answerable to")
    print("  DELEGATED_BY    - Authority granted by another")


def show_agent_types() -> None:
    """Display agent types"""
    print("\n=== Agent Types ===")
    print("  user_proxy - Bound to human, adjustable autonomy")
    print("  context    - Domain-specific work agent")
    print("  sme        - Subject matter expert")
    print("  sensor     - Data stream agent")
    print("  security   - Security audit/validation")
    print("  general    - General purpose")


# Auto-announce on load
print("[Moral Geometry Loaded]")
print("Commands: setup(), health_check(), show_virtues()")
print("Agents: create_user_proxy(human_id), create_context_agent(domain, human)")
print("Conversations: start_conversation(human, domain), route_message(conv_id, msg)")
print("Geometry: compute_kala(agent_id), compute_gradient(agent_id)")
