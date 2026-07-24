"""PepMem-AI — biblioteca central de predição peptídeo–membrana.

Exporta ``PepMemPredictor`` como API principal para dashboard, CLI e Spaces.
"""

from pepmem.predictor import PepMemPredictor

__all__ = ["PepMemPredictor"]
