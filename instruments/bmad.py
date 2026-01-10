"""
BMAD Ensemble Instrument for AgentSmith
Multi-model collaboration with graph-resident personas

Usage in agent chat:
    exec(open('/a0/instruments/bmad.py').read())
    
    # Bootstrap personas (first time only)
    bootstrap()
    
    # Get a persona
    analyst = get_persona("analyst")
    
    # Run strategic phase (Analyst + Architect)
    strategic_output = run_strategic("Build a REST API for user management")
    
    # Run tactical phase (Dev + Reviewer + QA)
    tactical_output = run_tactical(strategic_output)
    
    # Full ensemble
    result = run_ensemble("Build a REST API for user management")
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
log = logging.getLogger("bmad")

# Import temporal graph
try:
    from python.helpers.temporal_graph import (
        TemporalGraph, get_graph, 
        BMAD_PERSONAS, bootstrap_bmad_personas
    )
    GRAPH_AVAILABLE = True
    log.info("Temporal graph loaded")
except ImportError as e:
    GRAPH_AVAILABLE = False
    log.error(f"Temporal graph failed: {e}")

# Import LiteLLM for multi-model calls
try:
    import litellm
    LITELLM_AVAILABLE = True
    log.info("LiteLLM loaded")
except ImportError:
    LITELLM_AVAILABLE = False
    log.error("LiteLLM not available. Run: pip install litellm")


def bootstrap() -> Dict[str, str]:
    """Bootstrap all BMAD personas into the graph"""
    if not GRAPH_AVAILABLE:
        print("[!] Graph not available")
        return {}
    return bootstrap_bmad_personas()


def get_persona(role: str) -> Optional[Dict]:
    """Get a persona by role (analyst, pm, architect, dev, reviewer, qa, po)"""
    if not GRAPH_AVAILABLE:
        # Fallback to static definition
        return BMAD_PERSONAS.get(role)
    
    g = get_graph()
    persona_def = BMAD_PERSONAS.get(role)
    if not persona_def:
        print(f"[!] Unknown role: {role}")
        return None
    
    # Get from graph (includes decision history)
    persona = g.get_persona(persona_def["name"])
    if persona:
        return {**persona_def, **persona}
    return persona_def


def call_model(model: str, system_prompt: str, user_message: str,
               temperature: float = 0.7) -> str:
    """Call a model via OpenRouter using LiteLLM"""
    if not LITELLM_AVAILABLE:
        log.error("LiteLLM not available")
        return f"[LiteLLM not available - would call {model}]"
    
    log.debug(f"Calling {model}")
    
    # Set OpenRouter API key for LiteLLM
    api_key = os.getenv("API_KEY_OPENROUTER")
    if not api_key:
        return "[Error: API_KEY_OPENROUTER not set]"
    
    try:
        # LiteLLM handles openrouter/ prefix automatically
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=temperature,
            api_key=api_key
        )
        log.debug(f"Got response from {model}")
        return response.choices[0].message.content
    except Exception as e:
        log.error(f"Model call failed: {model}")
        log.error(traceback.format_exc())
        return f"[Error calling {model}: {e}]"


def run_persona(role: str, task: str, context: Optional[str] = None) -> Dict[str, Any]:
    """Run a single persona on a task"""
    persona = get_persona(role)
    if not persona:
        return {"error": f"Unknown role: {role}"}
    
    # Build prompt
    prompt = task
    if context:
        prompt = f"Context:\n{context}\n\nTask:\n{task}"
    
    # Get decision history if available
    if GRAPH_AVAILABLE:
        g = get_graph()
        history = g.get_persona_decisions(persona["name"], limit=5)
        if history:
            history_str = "\n".join([f"- {d['decision']}" for d in history])
            prompt = f"Your recent decisions:\n{history_str}\n\n{prompt}"
    
    # Call model
    print(f"[BMAD] Running {persona['name']} ({persona['role']})...")
    response = call_model(
        model=persona["model"],
        system_prompt=persona["system_prompt"],
        user_message=prompt
    )
    
    # Record decision if graph available
    if GRAPH_AVAILABLE:
        g = get_graph()
        g.record_decision(
            persona_name=persona["name"],
            decision=response[:500],  # Truncate for storage
            rationale=f"Task: {task[:200]}",
            context={"full_task": task, "context": context}
        )
        g.add_episode(
            content=f"{persona['name']} completed task: {task[:100]}",
            source=persona["name"],
            episode_type="decision"
        )
    
    return {
        "role": role,
        "persona": persona["name"],
        "model": persona["model"],
        "response": response
    }


def run_strategic(task: str) -> Dict[str, Any]:
    """
    Run strategic phase: Analyst → PM → Architect
    Returns comprehensive spec for tactical execution.
    """
    print("\n=== STRATEGIC PHASE ===\n")
    
    # Analyst: Requirements & constraints
    analyst_result = run_persona("analyst", f"""
Analyze this request and produce:
1. Clear requirements (functional and non-functional)
2. Constraints and assumptions
3. Edge cases to consider
4. Questions that need answers

Request: {task}
""")
    
    # PM: Product spec
    pm_result = run_persona("pm", f"""
Based on the analyst's findings, create:
1. User stories with acceptance criteria
2. Priority ranking (must-have vs nice-to-have)
3. Success metrics
4. Scope boundaries

Analyst findings:
{analyst_result['response']}
""")
    
    # Architect: Technical design
    architect_result = run_persona("architect", f"""
Based on the requirements and product spec, design:
1. System architecture (components, data flow)
2. Technology choices with rationale
3. API design (endpoints, data models)
4. Implementation phases

Product spec:
{pm_result['response']}
""")
    
    return {
        "phase": "strategic",
        "analyst": analyst_result,
        "pm": pm_result,
        "architect": architect_result,
        "spec": architect_result["response"]
    }


def run_tactical(spec: str, context: Optional[str] = None) -> Dict[str, Any]:
    """
    Run tactical phase: Dev → Reviewer → QA
    Takes strategic spec and produces reviewed implementation.
    """
    print("\n=== TACTICAL PHASE ===\n")
    
    full_context = spec
    if context:
        full_context = f"{context}\n\nSpec:\n{spec}"
    
    # Dev: Implementation
    dev_result = run_persona("dev", f"""
Implement according to this specification:

{full_context}

Produce:
1. Working code with comments
2. Unit tests
3. Usage examples
""")
    
    # Reviewer: Code review
    reviewer_result = run_persona("reviewer", f"""
Review this implementation:

Specification:
{spec}

Implementation:
{dev_result['response']}

Check for:
1. Correctness - does it meet the spec?
2. Code quality - is it clean and maintainable?
3. Edge cases - are they handled?
4. Improvements - what could be better?
""")
    
    # QA: Testing & security
    qa_result = run_persona("qa", f"""
QA review of this implementation:

Specification:
{spec}

Implementation:
{dev_result['response']}

Code Review:
{reviewer_result['response']}

Check for:
1. Security vulnerabilities
2. Performance issues
3. Missing error handling
4. UX problems
5. Test coverage gaps
""")
    
    return {
        "phase": "tactical",
        "dev": dev_result,
        "reviewer": reviewer_result,
        "qa": qa_result,
        "implementation": dev_result["response"],
        "review_notes": reviewer_result["response"],
        "qa_notes": qa_result["response"]
    }


def run_ensemble(task: str) -> Dict[str, Any]:
    """
    Full BMAD ensemble: Strategic → Tactical
    Complete pipeline from request to reviewed implementation.
    """
    print(f"\n{'='*60}")
    print(f"BMAD ENSEMBLE: {task[:50]}...")
    print(f"{'='*60}\n")
    
    # Strategic phase
    strategic = run_strategic(task)
    
    # Tactical phase
    tactical = run_tactical(strategic["spec"])
    
    # Detect contradictions
    contradictions = []
    if GRAPH_AVAILABLE:
        g = get_graph()
        contradictions = g.detect_contradictions()
        if contradictions:
            print(f"\n[!] Found {len(contradictions)} contradictions in knowledge base")
    
    return {
        "task": task,
        "strategic": strategic,
        "tactical": tactical,
        "contradictions": contradictions,
        "final_implementation": tactical["implementation"]
    }


def reconcile(disagreement: str) -> str:
    """
    When personas disagree, run reconciliation.
    Uses PO as tiebreaker.
    """
    po_result = run_persona("po", f"""
The team has a disagreement that needs resolution:

{disagreement}

As Product Owner, make the call:
1. What's the decision?
2. Why?
3. What are the tradeoffs?
""")
    
    return po_result["response"]


def show_personas() -> None:
    """Display all available personas"""
    print("\n=== BMAD PERSONAS ===\n")
    for key, p in BMAD_PERSONAS.items():
        phase_icon = "🎯" if p["phase"] == "strategic" else "⚡"
        print(f"{phase_icon} {key}: {p['name']} - {p['role']}")
        print(f"   Model: {p['model']}")
        print(f"   Capabilities: {', '.join(p['capabilities'])}")
        print()


def show_decisions(role: str, limit: int = 10) -> None:
    """Show recent decisions by a persona"""
    if not GRAPH_AVAILABLE:
        print("[!] Graph not available")
        return
    
    persona = get_persona(role)
    if not persona:
        return
    
    g = get_graph()
    decisions = g.get_persona_decisions(persona["name"], limit)
    
    print(f"\n=== {persona['name']}'s Recent Decisions ===\n")
    for d in decisions:
        print(f"- {d['decision'][:100]}...")
        print(f"  Rationale: {d['rationale'][:50]}...")
        print()


# Auto-announce on load
print("[BMAD Ensemble Loaded]")
print("Commands: bootstrap(), show_personas(), run_ensemble(task)")
print("Personas: analyst, pm, architect, dev, reviewer, qa, po")
