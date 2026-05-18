"""PyTorch modules for the Perturb-JEPA Bridge model."""

from perturb_jepa.models.adversary import BatchAdversary, GradientReversalLayer, gradient_reversal_ramp
from perturb_jepa.models.bag_aggregator import MultiPrototypeBagAggregator, MultiPrototypeBagAggregatorOutput
from perturb_jepa.models.bridge import PerturbJEPABridge, PerturbJEPABridgeConfig
from perturb_jepa.models.counterfactual import (
    CounterfactualResponseOutput,
    CounterfactualResponsePredictor,
    PerturbationConditionEncoder,
)
from perturb_jepa.models.projection_heads import ImageProjectionHead, RNAProjectionHead

__all__ = [
    "BatchAdversary",
    "CounterfactualResponseOutput",
    "CounterfactualResponsePredictor",
    "GradientReversalLayer",
    "ImageProjectionHead",
    "MultiPrototypeBagAggregator",
    "MultiPrototypeBagAggregatorOutput",
    "PerturbJEPABridge",
    "PerturbJEPABridgeConfig",
    "PerturbationConditionEncoder",
    "RNAProjectionHead",
    "gradient_reversal_ramp",
]
