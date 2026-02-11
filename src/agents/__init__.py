"""
Multi-Agent Evaluation System

This module contains specialized agents for hackathon presentation evaluation.
Each agent has a distinct focus area and evaluation bias.
"""

from .base_agent import BaseAgent
from .technical_judge import TechnicalJudge
from .product_judge import ProductJudge
from .execution_judge import ExecutionJudge
from .aggregator import Aggregator

__all__ = [
    'BaseAgent',
    'TechnicalJudge',
    'ProductJudge',
    'ExecutionJudge',
    'Aggregator',
]
