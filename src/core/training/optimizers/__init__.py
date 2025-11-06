"""
Custom optimizers for neural operator training.

Available optimizers:
- SOAP: Shampoo with Adam in the Preconditioner
"""

from .soap import SOAP

__all__ = ['SOAP']
