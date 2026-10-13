"""Expose the minimal derived-done orchestrator public surface."""

from __future__ import annotations

from . import subjects
from .config import (
    OrchestratorConfig,
    StackResolutionError,
    TargetResolutionError,
    resolve_bridged_agents,
    resolve_stack_id,
    resolve_target_agent_id,
)
from .fleet_signals import FleetBusDoneSource, FleetSignalSource, evidence_from_envelope
from .graph import GraphStore, InMemoryGraphStore, TaskGraph
from .orchestrator import Orchestrator
from .supervised_start import (
    ExternalStarter,
    ExternalStartSupervisor,
    KnownAgent,
    LivenessProbe,
    PrincipalResolver,
    StartAuthorizer,
    StartGuard,
)
from .transport import FrameBus, FrameHandler

__version__ = "0.0.1"

__all__ = [
    "FleetBusDoneSource",
    "FleetSignalSource",
    "FrameBus",
    "FrameHandler",
    "GraphStore",
    "InMemoryGraphStore",
    "ExternalStarter",
    "ExternalStartSupervisor",
    "KnownAgent",
    "LivenessProbe",
    "Orchestrator",
    "PrincipalResolver",
    "StartAuthorizer",
    "StartGuard",
    "OrchestratorConfig",
    "StackResolutionError",
    "TargetResolutionError",
    "TaskGraph",
    "evidence_from_envelope",
    "resolve_bridged_agents",
    "resolve_stack_id",
    "resolve_target_agent_id",
    "subjects",
]
