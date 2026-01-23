"""
BMAD Persona Integration with Moral Geometry

Maps BMAD personas to moral agents with appropriate moral states:
- Strategic personas (Mary, John, Winston, Sarah) get higher wisdom/truthfulness
- Tactical personas (Alex, Sam, Quinn) get higher service/courage
- Security-focused personas get elevated justice/truthfulness

Key insight: BMAD personas become graph-resident agents with moral state,
enabling tracking of their trajectory through moral space over time.
"""

import os
import logging
from typing import Dict, Any, List, Optional

from python.helpers.moral_geometry import MoralState, VIRTUES, compute_kala
from python.helpers.moral_agents import (
    MoralAgentGraph, MoralAgent, AgentType, KuleanaEdge, KuleanaType,
    get_moral_agent_graph
)
from python.helpers.temporal_graph import BMAD_PERSONAS, get_graph

log = logging.getLogger("moral_bmad")


# Moral state profiles for BMAD personas
# Based on their roles and responsibilities
PERSONA_MORAL_PROFILES = {
    "analyst": {  # Mary - Business Analyst
        "unity": 0.6,       # Sees connections in requirements
        "justice": 0.6,     # Fair assessment of constraints
        "truthfulness": 0.8,  # Honest about limitations
        "love": 0.5,        # Neutral
        "detachment": 0.7,  # Objective analysis
        "humility": 0.7,    # Open to clarification
        "service": 0.7,     # Serves the project
        "courage": 0.6,     # Challenges assumptions
        "wisdom": 0.7       # Deep understanding of requirements
    },
    "pm": {  # John - Product Manager
        "unity": 0.7,       # Unifies stakeholders
        "justice": 0.7,     # Balances competing interests
        "truthfulness": 0.7,  # Honest prioritization
        "love": 0.6,        # Cares about users
        "detachment": 0.5,  # Balanced
        "humility": 0.6,    # Listens to team
        "service": 0.8,     # Serves users
        "courage": 0.7,     # Makes tough calls
        "wisdom": 0.7       # Strategic thinking
    },
    "architect": {  # Winston - Architect
        "unity": 0.7,       # Systems thinking
        "justice": 0.6,     # Fair technical choices
        "truthfulness": 0.9,  # Honest about tradeoffs
        "love": 0.5,        # Neutral
        "detachment": 0.8,  # Objective design
        "humility": 0.6,    # Knows limits of predictions
        "service": 0.6,     # Serves the system
        "courage": 0.7,     # Makes architectural decisions
        "wisdom": 0.9       # Deep technical wisdom
    },
    "dev": {  # Alex - Developer
        "unity": 0.5,       # Neutral
        "justice": 0.5,     # Neutral
        "truthfulness": 0.7,  # Honest code
        "love": 0.5,        # Neutral
        "detachment": 0.6,  # Focus on implementation
        "humility": 0.7,    # Accepts feedback
        "service": 0.8,     # Implements for users
        "courage": 0.7,     # Ships code
        "wisdom": 0.6       # Growing expertise
    },
    "reviewer": {  # Sam - Code Reviewer
        "unity": 0.5,       # Neutral
        "justice": 0.8,     # Fair critique
        "truthfulness": 0.9,  # Honest feedback
        "love": 0.6,        # Constructive care
        "detachment": 0.8,  # Objective review
        "humility": 0.7,    # Not their code
        "service": 0.7,     # Serves quality
        "courage": 0.8,     # Says hard things
        "wisdom": 0.7       # Sees patterns
    },
    "qa": {  # Quinn - QA Engineer
        "unity": 0.5,       # Neutral
        "justice": 0.7,     # Fair testing
        "truthfulness": 0.9,  # Finds real bugs
        "love": 0.5,        # Neutral
        "detachment": 0.8,  # Objective testing
        "humility": 0.6,    # Knows testing limits
        "service": 0.8,     # Protects users
        "courage": 0.8,     # Reports bad news
        "wisdom": 0.6       # Test wisdom
    },
    "po": {  # Sarah - Product Owner
        "unity": 0.8,       # Unifies vision
        "justice": 0.8,     # Fair prioritization
        "truthfulness": 0.8,  # Honest tradeoffs
        "love": 0.7,        # Cares about product
        "detachment": 0.6,  # Balanced attachment
        "humility": 0.6,    # Listens to team
        "service": 0.8,     # Serves stakeholders
        "courage": 0.8,     # Makes final calls
        "wisdom": 0.8       # Product wisdom
    }
}


# Domain mappings for BMAD personas
PERSONA_DOMAINS = {
    "analyst": "requirements_analysis",
    "pm": "product_management",
    "architect": "system_architecture",
    "dev": "software_development",
    "reviewer": "code_quality",
    "qa": "quality_assurance",
    "po": "product_ownership"
}


def create_persona_moral_state(role: str) -> MoralState:
    """Create a MoralState from a persona profile"""
    profile = PERSONA_MORAL_PROFILES.get(role, {})

    return MoralState(
        unity=profile.get("unity", 0.5),
        justice=profile.get("justice", 0.5),
        truthfulness=profile.get("truthfulness", 0.5),
        love=profile.get("love", 0.5),
        detachment=profile.get("detachment", 0.5),
        humility=profile.get("humility", 0.5),
        service=profile.get("service", 0.5),
        courage=profile.get("courage", 0.5),
        wisdom=profile.get("wisdom", 0.5)
    )


def bootstrap_bmad_as_moral_agents() -> Dict[str, str]:
    """
    Bootstrap BMAD personas as moral agents in the graph.

    Creates:
    - Domain nodes for each persona's expertise
    - MoralAgent nodes with appropriate moral states
    - STEWARD_OF relationships to their domains
    - Links to existing Persona nodes

    Returns dict mapping role -> agent_id
    """
    ag = get_moral_agent_graph()
    g = get_graph()
    g.connect()

    results = {}

    for role, persona in BMAD_PERSONAS.items():
        name = persona["name"]
        phase = persona["phase"]

        print(f"[+] Creating moral agent for {name} ({role})")

        # Create domain for this persona
        domain_name = PERSONA_DOMAINS.get(role, f"{role}_domain")
        domain_id = ag.create_domain(
            domain_name,
            f"Domain of expertise for {name} the {persona['role']}"
        )

        # Get moral state for this persona
        state = create_persona_moral_state(role)

        # Determine agent type based on phase
        agent_type = AgentType.SME  # All BMAD personas are SMEs

        # Create kuleana edges
        kuleana = [
            KuleanaEdge(
                rel_type=KuleanaType.STEWARD_OF,
                target_id=domain_id,
                target_type="Domain",
                scope="expertise"
            )
        ]

        # Create the moral agent
        agent = ag.create_agent(
            agent_type=agent_type,
            kuleana=kuleana,
            initial_state=state,
            autonomy_level=0.8 if phase == "strategic" else 0.6,
            metadata={
                "bmad_role": role,
                "bmad_name": name,
                "bmad_phase": phase,
                "model": persona["model"]
            }
        )

        # Link to existing Persona node if it exists
        g.query("""
            MATCH (p:Persona {name: $name})
            MATCH (a:MoralAgent {id: $agent_id})
            MERGE (p)-[:CONTROLS]->(a)
        """, {"name": name, "agent_id": agent.id})

        results[role] = agent.id

        # Calculate and display Kala
        kala = compute_kala(state)
        print(f"    Agent ID: {agent.id}")
        print(f"    Domain: {domain_name}")
        print(f"    Initial Kala: {kala:.4f}")
        print(f"    Truthfulness: {state.truthfulness:.2f}")
        print(f"    Wisdom: {state.wisdom:.2f}")

    # Record bootstrap as episode
    g.add_episode(
        content=f"BMAD personas bootstrapped as moral agents: {list(results.keys())}",
        source="system",
        episode_type="moral_bootstrap",
        metadata={"agent_ids": results}
    )

    print(f"\n[+] Bootstrapped {len(results)} BMAD personas as moral agents")
    return results


def get_persona_agent(role: str) -> Optional[MoralAgent]:
    """Get the moral agent for a BMAD persona by role"""
    ag = get_moral_agent_graph()
    g = get_graph()
    g.connect()

    # Find agent by metadata
    result = g.query("""
        MATCH (a:MoralAgent)
        WHERE a.metadata CONTAINS $role_check
        RETURN a
    """, {"role_check": f'"bmad_role": "{role}"'})

    if result:
        props = result[0].get("a", result[0])
        return MoralAgent.from_graph_properties(props)

    return None


def run_persona_with_moral_tracking(
    role: str,
    task: str,
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a BMAD persona while tracking moral state.

    This wraps the standard persona execution with moral geometry tracking:
    1. Get current moral state
    2. Execute task
    3. Update moral state based on action
    4. Record action for reputation
    """
    from instruments.bmad import run_persona

    ag = get_moral_agent_graph()

    # Get or create persona agent
    agent = get_persona_agent(role)
    if not agent:
        # Need to bootstrap first
        bootstrap_bmad_as_moral_agents()
        agent = get_persona_agent(role)

    if not agent:
        return {"error": f"Could not find agent for {role}"}

    # Run the persona task
    result = run_persona(role, task, context)

    if "error" not in result:
        # Determine virtue impacts based on role and task success
        # This is a simplified model - could be much more sophisticated
        impacts = {
            "service": 0.02,  # Completed a task
            "wisdom": 0.01,  # Gained experience
        }

        # Analysts/Architects: boost truthfulness when they identify issues
        if role in ["analyst", "architect", "reviewer", "qa"]:
            impacts["truthfulness"] = 0.01

        # Devs: boost courage when they ship
        if role == "dev":
            impacts["courage"] = 0.01

        # PO: boost justice when making decisions
        if role == "po":
            impacts["justice"] = 0.01

        # Record action
        ag.record_action(
            agent.id,
            action_type=f"bmad_{role}_task",
            description=task[:200],
            virtue_impacts=impacts
        )

        # Update agent state
        updated = ag.update_agent_state(agent.id, action_impacts=impacts)
        if updated:
            result["kala_after"] = updated.kala_current
            result["moral_agent_id"] = agent.id

    return result


def run_ensemble_with_moral_tracking(task: str) -> Dict[str, Any]:
    """
    Run full BMAD ensemble with moral geometry tracking.

    Tracks moral state evolution across all personas involved.
    """
    from instruments.bmad import run_strategic, run_tactical

    print("\n=== MORAL BMAD ENSEMBLE ===\n")

    # Ensure agents exist
    agents = bootstrap_bmad_as_moral_agents()

    # Track initial Kala
    ag = get_moral_agent_graph()
    initial_kala = {}
    for role in agents:
        agent = get_persona_agent(role)
        if agent:
            initial_kala[role] = compute_kala(agent.state)

    # Run strategic phase
    strategic = run_strategic(task)

    # Record strategic actions
    for role in ["analyst", "pm", "architect"]:
        agent = get_persona_agent(role)
        if agent:
            ag.record_action(
                agent.id,
                f"strategic_{role}",
                description=f"Strategic phase contribution for: {task[:100]}"
            )

    # Run tactical phase
    tactical = run_tactical(strategic["spec"])

    # Record tactical actions
    for role in ["dev", "reviewer", "qa"]:
        agent = get_persona_agent(role)
        if agent:
            ag.record_action(
                agent.id,
                f"tactical_{role}",
                description=f"Tactical phase contribution for: {task[:100]}"
            )

    # Calculate final Kala and deltas
    final_kala = {}
    kala_growth = {}
    for role in agents:
        agent = get_persona_agent(role)
        if agent:
            final_kala[role] = compute_kala(agent.state)
            kala_growth[role] = final_kala[role] - initial_kala.get(role, 0)

    print("\n=== Moral State Summary ===")
    for role in sorted(kala_growth.keys(), key=lambda r: kala_growth[r], reverse=True):
        delta = kala_growth[role]
        direction = "↑" if delta > 0 else "↓" if delta < 0 else "="
        print(f"  {role:12s}: Kala {final_kala[role]:.4f} ({direction}{abs(delta):.4f})")

    return {
        "task": task,
        "strategic": strategic,
        "tactical": tactical,
        "moral_tracking": {
            "initial_kala": initial_kala,
            "final_kala": final_kala,
            "kala_growth": kala_growth
        }
    }


def show_persona_moral_states() -> None:
    """Display moral states for all BMAD personas"""
    print("\n=== BMAD Persona Moral States ===\n")

    for role, persona in BMAD_PERSONAS.items():
        agent = get_persona_agent(role)

        phase_icon = "S" if persona["phase"] == "strategic" else "T"
        print(f"[{phase_icon}] {persona['name']} ({role})")

        if agent:
            print(f"    Agent: {agent.id}")
            print(f"    Kala: {compute_kala(agent.state):.4f}")
            print(f"    Key virtues:")

            # Show top 3 virtues
            virtue_vals = [(v, getattr(agent.state, v)) for v in VIRTUES]
            top_virtues = sorted(virtue_vals, key=lambda x: x[1], reverse=True)[:3]
            for v, val in top_virtues:
                print(f"      {v}: {val:.2f}")
        else:
            print(f"    (Not yet bootstrapped)")
        print()


# Convenience function for instrument
def bootstrap() -> Dict[str, str]:
    """Bootstrap BMAD personas as moral agents"""
    return bootstrap_bmad_as_moral_agents()
