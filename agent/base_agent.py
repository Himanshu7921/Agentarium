"""
base_agent.py

This module defines the abstract BaseAgent class, which serves as the 
foundation for all specialized agents in the Agentarium framework. 

Every agent (e.g., ResearchAgent, SummarizerAgent, CriticAgent) 
inherits from BaseAgent and implements its own tools and behavior.
"""

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    Abstract base class for all agents in Agentarium.

    Each agent is expected to:
    - Wrap around an LLM or other reasoning backend.
    - Define its role (`agent_type`) within the multi-agent system.
    - Implement specialized tools or behaviors required for its task.

    Attributes:
        llm: The language model or reasoning engine the agent uses.
        agent_type (str): A string identifier describing the agent's role.
        verbose (bool): Whether to print/log detailed agent activity.
    """

    def __init__(self, llm, agent_type: str, verbose: bool = False):
        """
        Initialize a BaseAgent.

        Args:
            llm: The language model instance used by the agent.
            agent_type (str): The role/type of the agent (e.g., "researcher").
            verbose (bool): If True, enables detailed logging.
        """
        self.llm = llm
        self.agent_type = agent_type
        self.verbose = verbose

    @abstractmethod
    def get_agent(self):
        """
        Retrieve the agent instance.

        Returns:
            The concrete implementation of the agent (e.g., LangChain agent).
        """
        pass

    @abstractmethod
    def define_agent_tools(self):
        """
        Define and return the tools available to this agent.

        Returns:
            list: A list of tools/functions that the agent can use.
        """
        pass