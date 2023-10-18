# Copyright (c) OpenMMLab. All rights reserved.

from custom_mmpkg.custom_mmcv.utils import Registry

from .temporal_encoder import TemporalGRUEncoder

NECKS = Registry('necks')

NECKS.register_module(name='TemporalGRUEncoder', module=TemporalGRUEncoder)


def build_neck(cfg):
    """Build neck."""
    if cfg is None:
        return None
    return NECKS.build(cfg)
