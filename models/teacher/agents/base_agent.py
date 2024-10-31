from abc import ABC, abstractmethod
# -*- coding: utf-8 -*-
"""Abstract base model"""

class BaseAgent(ABC):
    """Abstract Model class that is inherited to all models"""

    def __init__(self):
        pass

    @abstractmethod
    def act_train(self, state):
        """Given a state, choose an epsilon-greedy action"""
        pass

    @abstractmethod
    def act_test(self, state):
        """Given a state, choose an epsilon-greedy action"""
        pass