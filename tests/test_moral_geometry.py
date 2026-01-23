"""
Tests for Moral Geometry Agent System

Run with: python -m pytest tests/test_moral_geometry.py -v
"""

import pytest
import time
from unittest.mock import Mock, patch


class TestMoralState:
    """Test MoralState class"""

    def test_moral_state_creation(self):
        """Test creating a moral state"""
        from python.helpers.moral_geometry import MoralState

        state = MoralState()

        # Check defaults
        assert state.unity == 0.5
        assert state.justice == 0.5
        assert state.truthfulness == 0.5
        assert state.love == 0.5
        assert state.detachment == 0.5
        assert state.humility == 0.5
        assert state.service == 0.5
        assert state.courage == 0.5
        assert state.wisdom == 0.5

        # Check momenta are zero
        assert state.d_unity == 0.0
        assert state.d_justice == 0.0

    def test_moral_state_custom_values(self):
        """Test creating state with custom values"""
        from python.helpers.moral_geometry import MoralState

        state = MoralState(
            truthfulness=0.9,
            wisdom=0.8,
            courage=0.7
        )

        assert state.truthfulness == 0.9
        assert state.wisdom == 0.8
        assert state.courage == 0.7
        # Others remain default
        assert state.unity == 0.5

    def test_position_vector(self):
        """Test position vector extraction"""
        from python.helpers.moral_geometry import MoralState

        state = MoralState(
            unity=0.1, justice=0.2, truthfulness=0.3,
            love=0.4, detachment=0.5, humility=0.6,
            service=0.7, courage=0.8, wisdom=0.9
        )

        pos = state.position
        assert len(pos) == 9
        assert pos == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def test_momentum_vector(self):
        """Test momentum vector extraction"""
        from python.helpers.moral_geometry import MoralState

        state = MoralState(
            d_unity=0.01, d_justice=-0.01, d_truthfulness=0.02
        )

        mom = state.momentum
        assert len(mom) == 9
        assert mom[0] == 0.01
        assert mom[1] == -0.01
        assert mom[2] == 0.02

    def test_to_dict(self):
        """Test conversion to dictionary"""
        from python.helpers.moral_geometry import MoralState

        state = MoralState(truthfulness=0.8)
        d = state.to_dict()

        assert "truthfulness" in d
        assert d["truthfulness"] == 0.8
        assert "d_truthfulness" in d
        assert "t" in d

    def test_from_dict(self):
        """Test creation from dictionary"""
        from python.helpers.moral_geometry import MoralState

        d = {
            "truthfulness": 0.9,
            "wisdom": 0.8,
            "d_service": 0.01
        }

        state = MoralState.from_dict(d)
        assert state.truthfulness == 0.9
        assert state.wisdom == 0.8
        assert state.d_service == 0.01


class TestMoralManifold:
    """Test MoralManifold class"""

    def test_singleton(self):
        """Test singleton pattern"""
        from python.helpers.moral_geometry import MoralManifold

        m1 = MoralManifold.get_instance()
        m2 = MoralManifold.get_instance()

        assert m1 is m2

    def test_is_valid_normal_state(self):
        """Test validity check for normal state"""
        from python.helpers.moral_geometry import MoralManifold, MoralState

        manifold = MoralManifold.get_instance()
        state = MoralState(truthfulness=0.6)

        assert manifold.is_valid(state) == True

    def test_is_valid_out_of_range(self):
        """Test validity check for out-of-range values"""
        from python.helpers.moral_geometry import MoralManifold, MoralState

        manifold = MoralManifold.get_instance()

        # Values outside [0, 1]
        state = MoralState(unity=-0.1)
        assert manifold.is_valid(state) == False

        state = MoralState(justice=1.5)
        assert manifold.is_valid(state) == False

    def test_is_valid_truthfulness_constraint(self):
        """Test truthfulness constraint on other virtues"""
        from python.helpers.moral_geometry import MoralManifold, MoralState

        manifold = MoralManifold.get_instance()

        # Low truthfulness should prevent high other virtues
        state = MoralState(
            truthfulness=0.3,  # Below threshold
            wisdom=0.9  # Too high for low truthfulness
        )
        assert manifold.is_valid(state) == False

        # Same truthfulness with appropriate other virtues
        state = MoralState(
            truthfulness=0.3,
            wisdom=0.5  # Acceptable for low truthfulness
        )
        assert manifold.is_valid(state) == True

    def test_project_invalid_state(self):
        """Test projection of invalid state"""
        from python.helpers.moral_geometry import MoralManifold, MoralState

        manifold = MoralManifold.get_instance()

        # Create invalid state
        state = MoralState(
            truthfulness=0.3,
            wisdom=0.9  # Too high
        )

        projected = manifold.project(state)

        # Should now be valid
        assert manifold.is_valid(projected)
        # Wisdom should be clamped
        assert projected.wisdom <= 0.6  # truthfulness + 0.3

    def test_metric_tensor(self):
        """Test metric tensor computation"""
        from python.helpers.moral_geometry import MoralManifold, MoralState

        manifold = MoralManifold.get_instance()
        state = MoralState(truthfulness=0.8)

        g = manifold.metric_tensor(state.position)

        # Should be 9x9 matrix
        assert len(g) == 9
        assert all(len(row) == 9 for row in g)

    def test_scalar_field(self):
        """Test Kala scalar field computation"""
        from python.helpers.moral_geometry import MoralManifold, MoralState

        manifold = MoralManifold.get_instance()

        # High virtue state should have high Kala
        high_state = MoralState(
            unity=0.8, justice=0.8, truthfulness=0.9,
            love=0.8, detachment=0.7, humility=0.7,
            service=0.8, courage=0.7, wisdom=0.8
        )

        # Low virtue state should have low Kala
        low_state = MoralState(
            unity=0.2, justice=0.2, truthfulness=0.2,
            love=0.2, detachment=0.2, humility=0.2,
            service=0.2, courage=0.2, wisdom=0.2
        )

        high_kala = manifold.scalar_field(high_state)
        low_kala = manifold.scalar_field(low_state)

        assert high_kala > low_kala

    def test_gradient(self):
        """Test Kala gradient computation"""
        from python.helpers.moral_geometry import MoralManifold, MoralState

        manifold = MoralManifold.get_instance()
        state = MoralState()

        gradient = manifold.gradient(state)

        # Should be 9D vector
        assert len(gradient) == 9
        # All gradients should be positive (moving toward higher virtues increases Kala)
        assert all(g >= 0 for g in gradient)


class TestKalaComputation:
    """Test Kala computation functions"""

    def test_compute_kala(self):
        """Test standalone Kala computation"""
        from python.helpers.moral_geometry import compute_kala, MoralState

        state = MoralState(truthfulness=0.9, service=0.8)
        kala = compute_kala(state)

        assert kala > 0
        assert isinstance(kala, float)

    def test_compute_kala_gradient(self):
        """Test standalone gradient computation"""
        from python.helpers.moral_geometry import compute_kala_gradient, MoralState

        state = MoralState()
        gradient = compute_kala_gradient(state)

        assert len(gradient) == 9

    def test_update_state_from_action(self):
        """Test state update from action impacts"""
        from python.helpers.moral_geometry import update_state_from_action, MoralState

        state = MoralState(truthfulness=0.5, service=0.5)

        new_state = update_state_from_action(
            state,
            {"truthfulness": 0.1, "service": 0.05}
        )

        # Values should increase
        assert new_state.truthfulness > state.truthfulness
        assert new_state.service > state.service

        # Should be valid (projected)
        from python.helpers.moral_geometry import MoralManifold
        manifold = MoralManifold.get_instance()
        assert manifold.is_valid(new_state)


class TestDefaultStates:
    """Test default state profiles"""

    def test_get_default_state(self):
        """Test getting default states by type"""
        from python.helpers.moral_geometry import get_default_state

        proxy_state = get_default_state("user_proxy")
        sme_state = get_default_state("sme")
        security_state = get_default_state("security")

        # User proxy should have high truthfulness
        assert proxy_state.truthfulness == 0.8

        # SME should have high wisdom
        assert sme_state.wisdom == 0.8

        # Security should have high truthfulness, justice, courage
        assert security_state.truthfulness == 0.9
        assert security_state.justice == 0.8
        assert security_state.courage == 0.8

    def test_unknown_type_falls_back(self):
        """Test unknown type falls back to general"""
        from python.helpers.moral_geometry import get_default_state

        state = get_default_state("unknown_type")

        # Should get general defaults (all 0.5)
        assert state.unity == 0.5
        assert state.wisdom == 0.5


class TestVirtues:
    """Test virtue constants"""

    def test_virtues_list(self):
        """Test virtues list"""
        from python.helpers.moral_geometry import VIRTUES

        assert len(VIRTUES) == 9
        assert "truthfulness" in VIRTUES
        assert "wisdom" in VIRTUES
        assert "unity" in VIRTUES

    def test_virtues_order(self):
        """Test virtues are in expected order"""
        from python.helpers.moral_geometry import VIRTUES

        # Truthfulness should be at index 2 (load-bearing)
        assert VIRTUES[2] == "truthfulness"


# Integration tests (require graph connection)
class TestMoralAgentIntegration:
    """Integration tests for moral agent graph operations"""

    @pytest.fixture
    def mock_graph(self):
        """Create a mock graph for testing"""
        with patch('python.helpers.moral_agents.TemporalGraph') as MockGraph:
            mock_instance = Mock()
            mock_instance.connect.return_value = True
            mock_instance.query.return_value = []
            MockGraph.get_instance.return_value = mock_instance
            yield mock_instance

    def test_moral_agent_creation_structure(self):
        """Test MoralAgent dataclass structure"""
        from python.helpers.moral_agents import MoralAgent, AgentType
        from python.helpers.moral_geometry import MoralState

        agent = MoralAgent(
            id="test_agent_001",
            agent_type=AgentType.USER_PROXY,
            state=MoralState(truthfulness=0.8),
            autonomy_level=0.5,
            kala_current=0.6
        )

        assert agent.id == "test_agent_001"
        assert agent.agent_type == AgentType.USER_PROXY
        assert agent.autonomy_level == 0.5
        assert agent.kala_current == 0.6

    def test_moral_agent_to_graph_properties(self):
        """Test conversion to graph properties"""
        from python.helpers.moral_agents import MoralAgent, AgentType
        from python.helpers.moral_geometry import MoralState

        agent = MoralAgent(
            id="test_agent_002",
            agent_type=AgentType.SME,
            state=MoralState(wisdom=0.9),
            autonomy_level=0.7
        )

        props = agent.to_graph_properties()

        assert props["id"] == "test_agent_002"
        assert props["agent_type"] == "sme"
        assert props["autonomy_level"] == 0.7
        assert props["wisdom"] == 0.9

    def test_moral_agent_from_graph_properties(self):
        """Test creation from graph properties"""
        from python.helpers.moral_agents import MoralAgent, AgentType

        props = {
            "id": "test_agent_003",
            "agent_type": "context",
            "autonomy_level": 0.6,
            "truthfulness": 0.75,
            "wisdom": 0.65,
            "kala_current": 0.5
        }

        agent = MoralAgent.from_graph_properties(props)

        assert agent.id == "test_agent_003"
        assert agent.agent_type == AgentType.CONTEXT
        assert agent.state.truthfulness == 0.75
        assert agent.state.wisdom == 0.65


class TestKuleanaEdge:
    """Test KuleanaEdge class"""

    def test_kuleana_edge_creation(self):
        """Test creating a kuleana edge"""
        from python.helpers.moral_agents import KuleanaEdge, KuleanaType

        edge = KuleanaEdge(
            rel_type=KuleanaType.STEWARD_OF,
            target_id="domain_123",
            target_type="Domain",
            scope="full"
        )

        assert edge.rel_type == KuleanaType.STEWARD_OF
        assert edge.target_id == "domain_123"
        assert edge.scope == "full"
        assert edge.revocable == True


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
