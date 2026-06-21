"""Standalone PointPillars deployment package."""
from .pointpillars_standalone import PointPillarsStandalone
from .weight_mapper import load_pcdet_checkpoint_into_standalone

__all__ = ['PointPillarsStandalone', 'load_pcdet_checkpoint_into_standalone']
