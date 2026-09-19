"""Independent -> Communicate -> Revise evaluation framework."""

PROTOCOL = "ICR-MEDQA300-V2"  # legacy V2 runs
PROTOCOL_V3 = "ICR-V3"
CONDITIONS = (
    "none",
    "true_text",
    "self_text",
    "other_text",
    "true_evidence",
    "self_evidence",
    "other_evidence",
    "true_statebridge",
    "self_statebridge",
    "other_statebridge",
    "true_latentmas",
    "self_latentmas",
    "other_latentmas",
)
DIRECTIONS = ("A_to_B", "B_to_A")
