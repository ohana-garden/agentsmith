"""
MCP Tools for Moral Geometry Operations

Exposes moral geometry operations as MCP tools for claude.ai integration:
- Agent lifecycle (create, get, update, delete)
- Kuleana management (create/remove edges)
- Moral geometry (compute Kala, gradients, projections)
- Conversation management (start, route, end)
- Reputation queries

These tools integrate with the existing FastMCP server.
"""

from typing import Annotated, Optional, List, Dict, Any, Union
from pydantic import Field
from openai import BaseModel

# Tool response models
class MoralToolResponse(BaseModel):
    """Standard response for moral tools"""
    status: str = Field(description="success or error")
    data: Dict[str, Any] = Field(description="Response data")
    message: str = Field(description="Human-readable message", default="")


class MoralToolError(BaseModel):
    """Error response for moral tools"""
    status: str = Field(default="error")
    error: str = Field(description="Error message")


def register_moral_mcp_tools(mcp_server):
    """
    Register moral geometry MCP tools with the server.

    Call this from mcp_server.py to add moral tools:
        from python.helpers.moral_mcp_tools import register_moral_mcp_tools
        register_moral_mcp_tools(mcp_server)
    """

    # === AGENT LIFECYCLE TOOLS ===

    @mcp_server.tool(
        name="moral_create_agent",
        description="Create a new moral agent with initial state and kuleana relationships",
        tags={"moral", "agent", "create"}
    )
    async def moral_create_agent(
        agent_type: Annotated[str, Field(
            description="Agent type: user_proxy, context, sme, sensor, security, general"
        )],
        autonomy_level: Annotated[float, Field(
            description="Autonomy level 0.0-1.0 (0=echo, 1=full autonomy)",
            default=0.5
        )] = 0.5,
        metadata: Annotated[Optional[Dict[str, Any]], Field(
            description="Optional metadata for the agent",
            default=None
        )] = None
    ) -> MoralToolResponse:
        """Create a new moral agent"""
        try:
            from python.helpers.moral_agents import get_moral_agent_graph, AgentType
            ag = get_moral_agent_graph()

            agent = ag.create_agent(
                agent_type=agent_type,
                autonomy_level=autonomy_level,
                metadata=metadata or {}
            )

            return MoralToolResponse(
                status="success",
                data={
                    "agent_id": agent.id,
                    "agent_type": agent.agent_type.value,
                    "autonomy_level": agent.autonomy_level,
                    "kala_current": agent.kala_current
                },
                message=f"Created agent {agent.id}"
            )
        except Exception as e:
            return MoralToolError(error=str(e))

    @mcp_server.tool(
        name="moral_get_agent",
        description="Get a moral agent by ID",
        tags={"moral", "agent", "read"}
    )
    async def moral_get_agent(
        agent_id: Annotated[str, Field(description="Agent ID")]
    ) -> MoralToolResponse:
        """Get agent details"""
        try:
            from python.helpers.moral_agents import get_moral_agent_graph
            ag = get_moral_agent_graph()

            agent = ag.get_agent(agent_id)
            if not agent:
                return MoralToolError(error=f"Agent {agent_id} not found")

            return MoralToolResponse(
                status="success",
                data={
                    "agent_id": agent.id,
                    "agent_type": agent.agent_type.value,
                    "autonomy_level": agent.autonomy_level,
                    "kala_current": agent.kala_current,
                    "state": agent.state.to_dict(),
                    "created_at": agent.created_at
                },
                message=f"Agent {agent_id}"
            )
        except Exception as e:
            return MoralToolError(error=str(e))

    @mcp_server.tool(
        name="moral_update_agent_state",
        description="Update a moral agent's state based on action impacts",
        tags={"moral", "agent", "update"}
    )
    async def moral_update_agent_state(
        agent_id: Annotated[str, Field(description="Agent ID")],
        action_impacts: Annotated[Dict[str, float], Field(
            description="Dict mapping virtue names to impact values (e.g., {'truthfulness': 0.1, 'service': -0.05})"
        )]
    ) -> MoralToolResponse:
        """Update agent's moral state"""
        try:
            from python.helpers.moral_agents import get_moral_agent_graph
            ag = get_moral_agent_graph()

            agent = ag.update_agent_state(agent_id, action_impacts=action_impacts)
            if not agent:
                return MoralToolError(error=f"Agent {agent_id} not found")

            return MoralToolResponse(
                status="success",
                data={
                    "agent_id": agent.id,
                    "kala_current": agent.kala_current,
                    "state": agent.state.to_dict()
                },
                message=f"Updated agent {agent_id}, Kala={agent.kala_current:.3f}"
            )
        except Exception as e:
            return MoralToolError(error=str(e))

    # === KULEANA MANAGEMENT TOOLS ===

    @mcp_server.tool(
        name="moral_create_kuleana_edge",
        description="Create a kuleana (responsibility) relationship from agent to target",
        tags={"moral", "kuleana", "create"}
    )
    async def moral_create_kuleana_edge(
        agent_id: Annotated[str, Field(description="Source agent ID")],
        rel_type: Annotated[str, Field(
            description="Relationship type: STEWARD_OF, RESPONSIBLE_FOR, ACCOUNTABLE_TO, DELEGATED_BY"
        )],
        target_id: Annotated[str, Field(description="Target node ID")],
        target_type: Annotated[str, Field(
            description="Target node type: Human, Agent, Entity, Domain, Community"
        )],
        scope: Annotated[str, Field(
            description="Scope of responsibility",
            default="full"
        )] = "full"
    ) -> MoralToolResponse:
        """Create a kuleana edge"""
        try:
            from python.helpers.moral_agents import (
                get_moral_agent_graph, KuleanaEdge, KuleanaType
            )
            ag = get_moral_agent_graph()

            edge = KuleanaEdge(
                rel_type=KuleanaType(rel_type),
                target_id=target_id,
                target_type=target_type,
                scope=scope
            )

            success = ag.create_kuleana_edge(agent_id, edge)

            if success:
                return MoralToolResponse(
                    status="success",
                    data={
                        "agent_id": agent_id,
                        "relationship": rel_type,
                        "target_id": target_id
                    },
                    message=f"Created {rel_type} edge"
                )
            return MoralToolError(error="Failed to create edge")
        except Exception as e:
            return MoralToolError(error=str(e))

    @mcp_server.tool(
        name="moral_remove_kuleana_edge",
        description="Remove a kuleana edge (if revocable)",
        tags={"moral", "kuleana", "delete"}
    )
    async def moral_remove_kuleana_edge(
        agent_id: Annotated[str, Field(description="Source agent ID")],
        rel_type: Annotated[str, Field(description="Relationship type")],
        target_id: Annotated[str, Field(description="Target node ID")]
    ) -> MoralToolResponse:
        """Remove a kuleana edge"""
        try:
            from python.helpers.moral_agents import get_moral_agent_graph, KuleanaType
            ag = get_moral_agent_graph()

            success = ag.remove_kuleana_edge(
                agent_id,
                KuleanaType(rel_type),
                target_id
            )

            if success:
                return MoralToolResponse(
                    status="success",
                    data={"removed": True},
                    message=f"Removed {rel_type} edge"
                )
            return MoralToolError(error="Edge not found or not revocable")
        except Exception as e:
            return MoralToolError(error=str(e))

    @mcp_server.tool(
        name="moral_get_agent_kuleana",
        description="Get all kuleana relationships for an agent",
        tags={"moral", "kuleana", "read"}
    )
    async def moral_get_agent_kuleana(
        agent_id: Annotated[str, Field(description="Agent ID")]
    ) -> MoralToolResponse:
        """Get agent's kuleana edges"""
        try:
            from python.helpers.moral_agents import get_moral_agent_graph
            ag = get_moral_agent_graph()

            kuleana = ag.get_agent_kuleana(agent_id)

            return MoralToolResponse(
                status="success",
                data={"kuleana": kuleana},
                message=f"Found {len(kuleana)} kuleana relationships"
            )
        except Exception as e:
            return MoralToolError(error=str(e))

    # === MORAL GEOMETRY TOOLS ===

    @mcp_server.tool(
        name="moral_compute_kala",
        description="Compute the Kala (alignment/flow) value for an agent",
        tags={"moral", "geometry", "kala"}
    )
    async def moral_compute_kala(
        agent_id: Annotated[str, Field(description="Agent ID")]
    ) -> MoralToolResponse:
        """Compute Kala scalar field value"""
        try:
            from python.helpers.moral_agents import get_moral_agent_graph
            from python.helpers.moral_geometry import compute_kala
            ag = get_moral_agent_graph()

            agent = ag.get_agent(agent_id)
            if not agent:
                return MoralToolError(error=f"Agent {agent_id} not found")

            kala = compute_kala(agent.state)

            return MoralToolResponse(
                status="success",
                data={
                    "agent_id": agent_id,
                    "kala": kala,
                    "truthfulness": agent.state.truthfulness
                },
                message=f"Kala = {kala:.4f}"
            )
        except Exception as e:
            return MoralToolError(error=str(e))

    @mcp_server.tool(
        name="moral_compute_kala_gradient",
        description="Compute gradient of Kala field - direction of increasing alignment",
        tags={"moral", "geometry", "gradient"}
    )
    async def moral_compute_kala_gradient(
        agent_id: Annotated[str, Field(description="Agent ID")]
    ) -> MoralToolResponse:
        """Compute Kala gradient"""
        try:
            from python.helpers.moral_agents import get_moral_agent_graph
            from python.helpers.moral_geometry import (
                compute_kala_gradient, VIRTUES
            )
            ag = get_moral_agent_graph()

            agent = ag.get_agent(agent_id)
            if not agent:
                return MoralToolError(error=f"Agent {agent_id} not found")

            gradient = compute_kala_gradient(agent.state)

            # Create readable gradient dict
            gradient_dict = {
                VIRTUES[i]: gradient[i]
                for i in range(len(VIRTUES))
            }

            # Find strongest pull direction
            max_idx = gradient.index(max(gradient))
            pull_direction = VIRTUES[max_idx]

            return MoralToolResponse(
                status="success",
                data={
                    "agent_id": agent_id,
                    "gradient": gradient_dict,
                    "strongest_pull": pull_direction
                },
                message=f"Strongest pull toward: {pull_direction}"
            )
        except Exception as e:
            return MoralToolError(error=str(e))

    @mcp_server.tool(
        name="moral_project_state",
        description="Project agent state to valid manifold surface",
        tags={"moral", "geometry", "projection"}
    )
    async def moral_project_state(
        agent_id: Annotated[str, Field(description="Agent ID")]
    ) -> MoralToolResponse:
        """Project state to valid manifold"""
        try:
            from python.helpers.moral_agents import get_moral_agent_graph
            from python.helpers.moral_geometry import (
                project_to_valid_manifold, MoralManifold
            )
            ag = get_moral_agent_graph()

            agent = ag.get_agent(agent_id)
            if not agent:
                return MoralToolError(error=f"Agent {agent_id} not found")

            manifold = MoralManifold.get_instance()
            was_valid = manifold.is_valid(agent.state)

            projected = project_to_valid_manifold(agent.state)

            # Update if projection changed state
            if not was_valid:
                ag.update_agent_state(agent_id, new_state=projected)

            return MoralToolResponse(
                status="success",
                data={
                    "agent_id": agent_id,
                    "was_valid": was_valid,
                    "projected_state": projected.to_dict()
                },
                message="State was valid" if was_valid else "State projected to valid surface"
            )
        except Exception as e:
            return MoralToolError(error=str(e))

    # === ACTIONS AND REPUTATION TOOLS ===

    @mcp_server.tool(
        name="moral_record_action",
        description="Record an action performed by an agent (builds reputation trail)",
        tags={"moral", "action", "reputation"}
    )
    async def moral_record_action(
        agent_id: Annotated[str, Field(description="Agent ID")],
        action_type: Annotated[str, Field(description="Type of action")],
        description: Annotated[str, Field(
            description="Description of the action",
            default=""
        )] = "",
        virtue_impacts: Annotated[Optional[Dict[str, float]], Field(
            description="Optional virtue impact dict",
            default=None
        )] = None
    ) -> MoralToolResponse:
        """Record an action"""
        try:
            from python.helpers.moral_agents import get_moral_agent_graph
            ag = get_moral_agent_graph()

            action_id = ag.record_action(
                agent_id,
                action_type,
                description=description,
                virtue_impacts=virtue_impacts
            )

            if action_id:
                return MoralToolResponse(
                    status="success",
                    data={"action_id": action_id},
                    message=f"Recorded action {action_id}"
                )
            return MoralToolError(error="Failed to record action")
        except Exception as e:
            return MoralToolError(error=str(e))

    @mcp_server.tool(
        name="moral_get_reputation",
        description="Get an agent's reputation trail (observable history)",
        tags={"moral", "reputation", "read"}
    )
    async def moral_get_reputation(
        agent_id: Annotated[str, Field(description="Agent ID")]
    ) -> MoralToolResponse:
        """Get agent reputation"""
        try:
            from python.helpers.moral_agents import get_moral_agent_graph
            ag = get_moral_agent_graph()

            reputation = ag.get_reputation(agent_id)

            return MoralToolResponse(
                status="success",
                data=reputation,
                message=f"Reputation: {reputation.get('total_actions', 0)} actions, Kala={reputation.get('kala_current', 0):.3f}"
            )
        except Exception as e:
            return MoralToolError(error=str(e))

    # === CONVERSATION TOOLS ===

    @mcp_server.tool(
        name="moral_start_conversation",
        description="Start a new moral conversation with three-layer agent architecture",
        tags={"moral", "conversation", "create"}
    )
    async def moral_start_conversation(
        human_id: Annotated[str, Field(description="Human participant ID")],
        domain_id: Annotated[str, Field(description="Domain for the conversation")],
        additional_agent_ids: Annotated[Optional[List[str]], Field(
            description="Optional list of supporting agent IDs",
            default=None
        )] = None
    ) -> MoralToolResponse:
        """Start a conversation"""
        try:
            from python.helpers.moral_conversations import get_conversation_manager
            cm = get_conversation_manager()

            # Ensure domain exists
            from python.helpers.moral_agents import get_moral_agent_graph
            ag = get_moral_agent_graph()
            ag.create_domain(domain_id, domain_id)  # Create if not exists

            state = cm.start_conversation(
                human_id,
                domain_id,
                additional_agent_ids
            )

            return MoralToolResponse(
                status="success",
                data={
                    "conversation_id": state.conversation_id,
                    "proxy_id": state.proxy_agent.id,
                    "context_agent_id": state.context_agent.id,
                    "supporting_agents": [a.id for a in state.supporting_agents]
                },
                message=f"Started conversation {state.conversation_id}"
            )
        except Exception as e:
            return MoralToolError(error=str(e))

    @mcp_server.tool(
        name="moral_route_message",
        description="Route a message through the conversation's agent architecture",
        tags={"moral", "conversation", "message"}
    )
    async def moral_route_message(
        conversation_id: Annotated[str, Field(description="Conversation ID")],
        message: Annotated[str, Field(description="Message content")],
        from_human: Annotated[bool, Field(
            description="Whether message is from human",
            default=True
        )] = True
    ) -> MoralToolResponse:
        """Route a message"""
        try:
            from python.helpers.moral_conversations import get_conversation_manager
            cm = get_conversation_manager()

            result = cm.route_message(conversation_id, message, from_human)

            if "error" in result:
                return MoralToolError(error=result["error"])

            return MoralToolResponse(
                status="success",
                data=result,
                message="Message routed"
            )
        except Exception as e:
            return MoralToolError(error=str(e))

    @mcp_server.tool(
        name="moral_end_conversation",
        description="End a conversation",
        tags={"moral", "conversation", "end"}
    )
    async def moral_end_conversation(
        conversation_id: Annotated[str, Field(description="Conversation ID")]
    ) -> MoralToolResponse:
        """End a conversation"""
        try:
            from python.helpers.moral_conversations import get_conversation_manager
            cm = get_conversation_manager()

            cm.end_conversation(conversation_id)

            return MoralToolResponse(
                status="success",
                data={"ended": True},
                message=f"Ended conversation {conversation_id}"
            )
        except Exception as e:
            return MoralToolError(error=str(e))

    # === USER PROXY TOOLS ===

    @mcp_server.tool(
        name="moral_create_user_proxy",
        description="Create a user proxy agent bound to a human",
        tags={"moral", "agent", "proxy"}
    )
    async def moral_create_user_proxy(
        human_id: Annotated[str, Field(description="Human ID to bind to")],
        autonomy_level: Annotated[float, Field(
            description="Autonomy level 0.0-1.0",
            default=0.5
        )] = 0.5
    ) -> MoralToolResponse:
        """Create user proxy"""
        try:
            from python.helpers.moral_agents import get_moral_agent_graph
            ag = get_moral_agent_graph()

            # Ensure human exists
            ag.create_human(human_id)

            proxy = ag.create_user_proxy(human_id, autonomy_level)

            return MoralToolResponse(
                status="success",
                data={
                    "proxy_id": proxy.id,
                    "human_id": human_id,
                    "autonomy_level": proxy.autonomy_level
                },
                message=f"Created proxy {proxy.id} for human {human_id}"
            )
        except Exception as e:
            return MoralToolError(error=str(e))

    @mcp_server.tool(
        name="moral_set_proxy_autonomy",
        description="Set the autonomy level for a user's proxy agent",
        tags={"moral", "agent", "autonomy"}
    )
    async def moral_set_proxy_autonomy(
        human_id: Annotated[str, Field(description="Human ID")],
        autonomy_level: Annotated[float, Field(
            description="Autonomy level 0.0 (echo) to 1.0 (full autonomy)"
        )]
    ) -> MoralToolResponse:
        """Set proxy autonomy"""
        try:
            from python.helpers.moral_conversations import get_conversation_manager
            cm = get_conversation_manager()

            proxy = cm.set_proxy_autonomy(human_id, autonomy_level)

            if proxy:
                return MoralToolResponse(
                    status="success",
                    data={
                        "proxy_id": proxy.id,
                        "autonomy_level": proxy.autonomy_level
                    },
                    message=f"Set autonomy to {autonomy_level}"
                )
            return MoralToolError(error="Proxy not found")
        except Exception as e:
            return MoralToolError(error=str(e))

    # === GRAPH QUERY TOOL ===

    @mcp_server.tool(
        name="moral_query_graph",
        description="Execute a raw Cypher query on the moral agent graph",
        tags={"moral", "graph", "query"}
    )
    async def moral_query_graph(
        cypher: Annotated[str, Field(description="Cypher query to execute")],
        params: Annotated[Optional[Dict[str, Any]], Field(
            description="Query parameters",
            default=None
        )] = None
    ) -> MoralToolResponse:
        """Execute graph query"""
        try:
            from python.helpers.temporal_graph import TemporalGraph
            g = TemporalGraph.get_instance()
            g.connect()

            result = g.query(cypher, params or {})

            return MoralToolResponse(
                status="success",
                data={"result": result},
                message=f"Returned {len(result)} rows"
            )
        except Exception as e:
            return MoralToolError(error=str(e))

    # === HEALTH CHECK ===

    @mcp_server.tool(
        name="moral_health_check",
        description="Check moral geometry system health",
        tags={"moral", "health"}
    )
    async def moral_health_check() -> MoralToolResponse:
        """System health check"""
        try:
            from python.helpers.temporal_graph import TemporalGraph
            from python.helpers.moral_agents import get_moral_agent_graph

            g = TemporalGraph.get_instance()
            g.connect()

            # Count moral agents
            agent_count = g.query("""
                MATCH (a:MoralAgent)
                WHERE a.deleted_at IS NULL
                RETURN count(a) as count
            """)

            # Count conversations
            conv_count = g.query("""
                MATCH (c:Conversation)
                RETURN count(c) as count
            """)

            # Base health from temporal graph
            base_health = g.health_check()

            return MoralToolResponse(
                status="success",
                data={
                    **base_health,
                    "moral_agents": agent_count[0]["count"] if agent_count else 0,
                    "conversations": conv_count[0]["count"] if conv_count else 0
                },
                message="Moral geometry system healthy"
            )
        except Exception as e:
            return MoralToolError(error=str(e))

    print("[+] Registered moral geometry MCP tools")
    return mcp_server
