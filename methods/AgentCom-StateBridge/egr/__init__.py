"""Evidence-Grounded Revision method development on frozen ICR artifacts."""

from .evidence import build_evidence_packet
from .scoring import egr_gate

__all__ = ["build_evidence_packet", "egr_gate"]
