"""
Face Enhancers Package for MuseTalk
===================================

This package contains face enhancement models and utilities for improving
the quality of AI-generated faces in the MuseTalk pipeline.

Available enhancers:
- GPENBFREnhancer: ONNX-based GPEN-BFR 256 (recommended)
- GFPGANEnhancer: PyTorch-based GFPGAN (legacy, has import issues)
"""

__version__ = "1.0.0"
__author__ = "MuseTalk Enhancement Team"

# Import main enhancers for easy access
try:
    from .gpen_bfr_enhancer import GPENBFREnhancer
    __all__ = ["GPENBFREnhancer"]
except ImportError:
    # GPEN-BFR not available
    __all__ = []

try:
    from .gfpgan_enhancer import GFPGANEnhancer
    __all__.append("GFPGANEnhancer")
except ImportError:
    # GFPGAN not available
    pass
