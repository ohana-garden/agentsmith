"""
Moral Geometry Module

Implements the 19D moral manifold where:
- Position (9D): virtue space coordinates
- Momentum (9D): rate of change (direction agent is heading)
- Time (1D): temporal context

Key concepts:
- Trustworthiness is the metric tensor (defines geometry, not edges)
- Kala is the scalar field / reward function
- Invalid states are projected to valid manifold surface (mercy, not punishment)
"""

import math
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import json

log = logging.getLogger("moral_geometry")

# The nine virtues that define position in moral space
VIRTUES = [
    "unity",        # 0: Oneness, interconnection
    "justice",      # 1: Fairness, equity
    "truthfulness", # 2: Honesty, authenticity (LOAD-BEARING)
    "love",         # 3: Compassion, care
    "detachment",   # 4: Objectivity, non-attachment
    "humility",     # 5: Modesty, openness to correction
    "service",      # 6: Contribution, helpfulness
    "courage",      # 7: Bravery, willingness to act
    "wisdom",       # 8: Discernment, understanding
]

# Truthfulness threshold - below this, other virtues cannot exist at high levels
TRUTHFULNESS_THRESHOLD = 0.5

# Kala field parameters
KALA_BASE_RATE = 1.0  # Base Kala accrual per time unit


@dataclass
class MoralState:
    """
    19D state vector for an agent in moral phase space.

    position: 9D virtue coordinates (where the agent is)
    momentum: 9D virtue derivatives (where agent is heading)
    t: temporal context
    """
    # Position (9 dimensions) - current virtue levels
    unity: float = 0.5
    justice: float = 0.5
    truthfulness: float = 0.5
    love: float = 0.5
    detachment: float = 0.5
    humility: float = 0.5
    service: float = 0.5
    courage: float = 0.5
    wisdom: float = 0.5

    # Momentum (9 dimensions) - rate of change
    d_unity: float = 0.0
    d_justice: float = 0.0
    d_truthfulness: float = 0.0
    d_love: float = 0.0
    d_detachment: float = 0.0
    d_humility: float = 0.0
    d_service: float = 0.0
    d_courage: float = 0.0
    d_wisdom: float = 0.0

    # Time
    t: float = field(default_factory=time.time)

    @property
    def position(self) -> List[float]:
        """Get 9D position vector"""
        return [
            self.unity, self.justice, self.truthfulness,
            self.love, self.detachment, self.humility,
            self.service, self.courage, self.wisdom
        ]

    @property
    def momentum(self) -> List[float]:
        """Get 9D momentum vector"""
        return [
            self.d_unity, self.d_justice, self.d_truthfulness,
            self.d_love, self.d_detachment, self.d_humility,
            self.d_service, self.d_courage, self.d_wisdom
        ]

    @property
    def full_state(self) -> List[float]:
        """Get full 19D state vector (position + momentum + time)"""
        return self.position + self.momentum + [self.t]

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for storage"""
        return {
            # Position
            "unity": self.unity,
            "justice": self.justice,
            "truthfulness": self.truthfulness,
            "love": self.love,
            "detachment": self.detachment,
            "humility": self.humility,
            "service": self.service,
            "courage": self.courage,
            "wisdom": self.wisdom,
            # Momentum
            "d_unity": self.d_unity,
            "d_justice": self.d_justice,
            "d_truthfulness": self.d_truthfulness,
            "d_love": self.d_love,
            "d_detachment": self.d_detachment,
            "d_humility": self.d_humility,
            "d_service": self.d_service,
            "d_courage": self.d_courage,
            "d_wisdom": self.d_wisdom,
            # Time
            "t": self.t
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "MoralState":
        """Create from dictionary"""
        return cls(
            unity=d.get("unity", 0.5),
            justice=d.get("justice", 0.5),
            truthfulness=d.get("truthfulness", 0.5),
            love=d.get("love", 0.5),
            detachment=d.get("detachment", 0.5),
            humility=d.get("humility", 0.5),
            service=d.get("service", 0.5),
            courage=d.get("courage", 0.5),
            wisdom=d.get("wisdom", 0.5),
            d_unity=d.get("d_unity", 0.0),
            d_justice=d.get("d_justice", 0.0),
            d_truthfulness=d.get("d_truthfulness", 0.0),
            d_love=d.get("d_love", 0.0),
            d_detachment=d.get("d_detachment", 0.0),
            d_humility=d.get("d_humility", 0.0),
            d_service=d.get("d_service", 0.0),
            d_courage=d.get("d_courage", 0.0),
            d_wisdom=d.get("d_wisdom", 0.0),
            t=d.get("t", time.time())
        )


class MoralManifold:
    """
    The moral geometry manifold.

    Trustworthiness is the metric tensor - it defines:
    - Distances between states
    - Curvature of the space
    - What paths are possible

    Trust is NOT a relationship. Trust is the medium.
    """

    _instance: Optional["MoralManifold"] = None

    def __init__(self):
        # Coupling strengths between virtues (symmetric matrix)
        # These define how virtues interact in the metric
        self.coupling = self._default_coupling()

    @classmethod
    def get_instance(cls) -> "MoralManifold":
        """Singleton access"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _default_coupling(self) -> List[List[float]]:
        """
        Default coupling matrix between virtues.

        Key coupling relationships:
        - Truthfulness couples strongly to all virtues (load-bearing)
        - Unity and love are coupled
        - Justice and courage are coupled
        - Wisdom couples to all others moderately
        """
        # Initialize identity (self-coupling = 1.0)
        n = len(VIRTUES)
        c = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

        # Truthfulness (index 2) couples to everything
        for i in range(n):
            if i != 2:
                c[2][i] = 0.8
                c[i][2] = 0.8

        # Unity (0) - Love (3) coupling
        c[0][3] = 0.6
        c[3][0] = 0.6

        # Justice (1) - Courage (7) coupling
        c[1][7] = 0.5
        c[7][1] = 0.5

        # Wisdom (8) - moderate coupling to all
        for i in range(n):
            if i != 8 and c[8][i] == 0.0:
                c[8][i] = 0.3
                c[i][8] = 0.3

        # Humility (5) - Detachment (4) coupling
        c[5][4] = 0.4
        c[4][5] = 0.4

        return c

    def metric_tensor(self, position: List[float]) -> List[List[float]]:
        """
        Return the metric tensor g_ij at a given position.

        Trustworthiness determines the geometry:
        - High trustworthiness = smooth, navigable space
        - Low trustworthiness = warped geometry, limited paths
        """
        n = len(VIRTUES)

        # Start with coupling matrix as base
        g = [[self.coupling[i][j] for j in range(n)] for i in range(n)]

        # Truthfulness (index 2) warps the space
        truthfulness = position[2]
        warp = self._compute_warp(truthfulness)

        # Apply warp to metric
        for i in range(n):
            for j in range(n):
                g[i][j] *= warp

        return g

    def _compute_warp(self, truthfulness: float) -> float:
        """
        Truthfulness below threshold warps geometry such that
        other virtues cannot exist above their thresholds.

        This is not punishment - it's topology. Low truthfulness
        mathematically prevents high virtue states from existing.
        """
        if truthfulness < TRUTHFULNESS_THRESHOLD:
            # Severe warping - space becomes "curved away" from high-virtue states
            deficit = TRUTHFULNESS_THRESHOLD - truthfulness
            return 1.0 + deficit * 10.0  # Up to 6x warp at truthfulness=0
        return 1.0  # Flat space when truthfulness is sufficient

    def is_valid(self, state: MoralState) -> bool:
        """
        Check if a state is valid on the manifold.

        Invalid states:
        - Any virtue < 0 or > 1
        - High virtues with low truthfulness (geometrically impossible)
        """
        pos = state.position

        # Range check
        for v in pos:
            if v < 0.0 or v > 1.0:
                return False

        # Truthfulness constraint: if truthfulness is low,
        # other virtues cannot be high
        truthfulness = pos[2]
        if truthfulness < TRUTHFULNESS_THRESHOLD:
            max_allowed = truthfulness + 0.3  # Virtues capped near truthfulness
            for i, v in enumerate(pos):
                if i != 2 and v > max_allowed:
                    return False

        return True

    def project(self, state: MoralState) -> MoralState:
        """
        Project invalid state to nearest valid manifold surface point.

        This is mercy baked into the math:
        - Invalid states auto-correct, not punish
        - Agent is gently moved to valid configuration
        """
        if self.is_valid(state):
            return state

        new_state = MoralState.from_dict(state.to_dict())

        # Clamp all values to [0, 1]
        for virtue in VIRTUES:
            val = getattr(new_state, virtue)
            setattr(new_state, virtue, max(0.0, min(1.0, val)))

        # If truthfulness is low, clamp other virtues
        truthfulness = new_state.truthfulness
        if truthfulness < TRUTHFULNESS_THRESHOLD:
            max_allowed = truthfulness + 0.3
            for virtue in VIRTUES:
                if virtue != "truthfulness":
                    val = getattr(new_state, virtue)
                    if val > max_allowed:
                        setattr(new_state, virtue, max_allowed)
                        log.debug(f"Projected {virtue} from {val:.2f} to {max_allowed:.2f} (truthfulness constraint)")

        new_state.t = time.time()  # Update timestamp
        return new_state

    def distance(self, state1: MoralState, state2: MoralState) -> float:
        """
        Compute geodesic distance between two states.

        Uses metric tensor at midpoint for approximation.
        """
        pos1 = state1.position
        pos2 = state2.position

        # Midpoint for metric evaluation
        midpoint = [(p1 + p2) / 2 for p1, p2 in zip(pos1, pos2)]
        g = self.metric_tensor(midpoint)

        # Compute ds^2 = g_ij * dx^i * dx^j
        delta = [p2 - p1 for p1, p2 in zip(pos1, pos2)]

        ds_squared = 0.0
        n = len(VIRTUES)
        for i in range(n):
            for j in range(n):
                ds_squared += g[i][j] * delta[i] * delta[j]

        return math.sqrt(max(0, ds_squared))

    def scalar_field(self, state: MoralState) -> float:
        """
        Compute Kala scalar field value at a state.

        Kala measures alignment/flow:
        - Higher Kala = more in-flow with community-positive outcomes
        - Non-transferable, cannot decrease (mercy)
        """
        pos = state.position
        mom = state.momentum

        # Base Kala: geometric mean of all virtues (balanced development)
        base = 1.0
        for v in pos:
            base *= max(0.01, v)  # Avoid zero
        base = math.pow(base, 1.0 / len(VIRTUES))

        # Momentum bonus: moving toward higher virtues adds flow
        momentum_bonus = 0.0
        for i, dv in enumerate(mom):
            if dv > 0:  # Positive direction
                momentum_bonus += dv * 0.1

        # Truthfulness multiplier (load-bearing virtue)
        truthfulness = pos[2]
        truth_multiplier = 0.5 + truthfulness  # Range [0.5, 1.5]

        # Combined Kala
        kala = base * truth_multiplier * (1.0 + momentum_bonus)

        return kala

    def gradient(self, state: MoralState) -> List[float]:
        """
        Compute gradient of Kala field at current state.

        Returns 9D vector pointing toward increasing Kala.
        This is the "pull" toward alignment - attractive, not punitive.
        """
        epsilon = 0.01
        pos = state.position
        base_kala = self.scalar_field(state)

        gradient = []
        for i, virtue in enumerate(VIRTUES):
            # Perturb in positive direction
            perturbed = state.to_dict()
            perturbed[virtue] = min(1.0, pos[i] + epsilon)
            perturbed_state = MoralState.from_dict(perturbed)
            perturbed_kala = self.scalar_field(perturbed_state)

            # Numerical derivative
            grad_i = (perturbed_kala - base_kala) / epsilon
            gradient.append(grad_i)

        return gradient


def compute_kala(state: MoralState, manifold: Optional[MoralManifold] = None) -> float:
    """
    Compute Kala scalar field value for a moral state.

    Kala measures alignment/flow at current position in 19D phase space.
    Higher Kala = more in-flow with community-positive outcomes.

    Properties:
    - Non-transferable
    - Cannot decrease (mercy)
    - Only accrues through genuine participation
    """
    m = manifold or MoralManifold.get_instance()
    return m.scalar_field(state)


def compute_kala_gradient(state: MoralState, manifold: Optional[MoralManifold] = None) -> List[float]:
    """
    Compute gradient of Kala - direction of increasing alignment.

    Returns 9D vector indicating which virtues to develop.
    """
    m = manifold or MoralManifold.get_instance()
    return m.gradient(state)


def project_to_valid_manifold(state: MoralState, manifold: Optional[MoralManifold] = None) -> MoralState:
    """
    Project invalid state to nearest valid manifold surface.

    Mercy baked into the math - invalid states auto-correct.
    """
    m = manifold or MoralManifold.get_instance()
    return m.project(state)


def update_state_from_action(
    state: MoralState,
    action_impacts: Dict[str, float],
    decay_rate: float = 0.01
) -> MoralState:
    """
    Update moral state based on action impacts.

    action_impacts: Dict mapping virtue names to impact values
                   (positive = increases virtue, negative = decreases)
    decay_rate: How fast momentum decays toward zero

    Returns new projected state (always valid).
    """
    manifold = MoralManifold.get_instance()
    new_state = MoralState.from_dict(state.to_dict())

    # Apply impacts to position and update momentum
    for virtue, impact in action_impacts.items():
        if virtue in VIRTUES:
            # Update position
            current = getattr(new_state, virtue)
            new_val = current + impact
            setattr(new_state, virtue, new_val)

            # Update momentum (smoothed)
            d_virtue = f"d_{virtue}"
            current_d = getattr(new_state, d_virtue)
            # Exponential moving average of momentum
            new_d = 0.7 * current_d + 0.3 * impact
            setattr(new_state, d_virtue, new_d)

    # Decay all momenta toward zero (equilibrium)
    for virtue in VIRTUES:
        d_virtue = f"d_{virtue}"
        current_d = getattr(new_state, d_virtue)
        setattr(new_state, d_virtue, current_d * (1.0 - decay_rate))

    # Update timestamp
    new_state.t = time.time()

    # Project to valid manifold
    return manifold.project(new_state)


# Default starting states for different agent types
DEFAULT_STATES = {
    "user_proxy": MoralState(
        # Proxy starts with high truthfulness (faithful representation)
        truthfulness=0.8,
        # Other virtues neutral
        unity=0.5, justice=0.5, love=0.5,
        detachment=0.5, humility=0.5, service=0.5,
        courage=0.5, wisdom=0.5
    ),
    "context": MoralState(
        # Context agents: balanced, slightly elevated service
        unity=0.5, justice=0.5, truthfulness=0.5,
        love=0.5, detachment=0.5, humility=0.5,
        service=0.6, courage=0.5, wisdom=0.5
    ),
    "sme": MoralState(
        # SME agents: high wisdom and truthfulness in their domain
        unity=0.5, justice=0.5, truthfulness=0.8,
        love=0.5, detachment=0.5, humility=0.5,
        service=0.5, courage=0.5, wisdom=0.8
    ),
    "sensor": MoralState(
        # Sensor agents: high truthfulness, detachment (objective data)
        unity=0.5, justice=0.5, truthfulness=0.9,
        love=0.5, detachment=0.8, humility=0.5,
        service=0.5, courage=0.5, wisdom=0.5
    ),
    "security": MoralState(
        # Security agents: high truthfulness, justice, courage, detachment
        unity=0.5, justice=0.8, truthfulness=0.9,
        love=0.5, detachment=0.7, humility=0.5,
        service=0.5, courage=0.8, wisdom=0.6
    ),
    "general": MoralState(
        # General purpose: all virtues neutral
        unity=0.5, justice=0.5, truthfulness=0.5,
        love=0.5, detachment=0.5, humility=0.5,
        service=0.5, courage=0.5, wisdom=0.5
    )
}


def get_default_state(agent_type: str) -> MoralState:
    """Get default starting state for an agent type"""
    return MoralState.from_dict(
        DEFAULT_STATES.get(agent_type, DEFAULT_STATES["general"]).to_dict()
    )
