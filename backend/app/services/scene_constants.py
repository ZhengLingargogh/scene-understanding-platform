"""Shared scene identifiers (no heavy CV dependencies)."""

DEFAULT_SCENE_ID = "nature"

SCENE_HEAD_WEIGHTS: dict[str, str] = {
    "nature": "nature_base.pt",
    "nature_base": "nature_base.pt",
    "urban": "urban_base.pt",
    "urban_base": "urban_base.pt",
    "natureoop": "natureoop_base.pt",
    "natureoop_base": "natureoop_base.pt",
    "urbanoop": "urbanoop_base.pt",
    "urbanoop_base": "urbanoop_base.pt",
    "inTraj": "inTraj_base.pt",
    "inTraj_base": "inTraj_base.pt",
    "outTraj": "outTraj_base.pt",
    "outTraj_base": "outTraj_base.pt",
}
