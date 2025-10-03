"""
critic_agent.py

This module defines the CriticAgent, a specialized agent in the 
Agentarium framework responsible for evaluating and providing 
constructive feedback on generated content.

The agent leverages an LLM and a predefined prompt template to 
analyze clarity, structure, engagement, factual accuracy, flow, 
and overall quality of content.
"""

from langchain.agents import Tool, initialize_agent
from agent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class CriticAgent(BaseAgent):
    """
    A specialized agent that critically evaluates generated content 
    and provides structured, actionable feedback.

    Inherits from:
        BaseAgent

    Attributes:
        llm: The language model instance used for critique.
        agent_type (str): Identifier describing this agent's type.
        verbose (bool): Flag for enabling detailed logs.
    """

    def __init__(self, llm, agent_type: str, verbose: bool = False):
        """
        Initialize the CriticAgent.

        Args:
            llm: The language model instance.
            agent_type (str): The agent's role/type (e.g., "critic").
            verbose (bool): If True, enables detailed logging.
        """
        super().__init__(llm, agent_type, verbose)
        self.llm = llm
        self.agent_type = agent_type
        self.verbose = verbose

    def get_agent(self):
        """
        Initialize and return the critic agent with its tools.

        Returns:
            A LangChain agent instance configured with critique tools.
        """
        self.tools = self.define_agent_tools()
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=self.agent_type,
            verbose=self.verbose
        )
        return self.agent

    def define_agent_tools(self):
        """
        Define the tools available to the CriticAgent.

        Currently includes:
            - Critic Tool: Evaluates written content and 
              provides structured feedback on strengths, 
              weaknesses, and areas for improvement.

        Returns:
            list: A list containing the critic tool.
        """
        critic_tool = Tool.from_function(
            func=self.evaluate_content,
            name="Critic Tool",
            description=(
                "Use this tool to critically evaluate blog posts or written content. "
                "The tool analyzes clarity, structure, engagement, factual accuracy, "
                "flow, and overall quality, and provides actionable suggestions for improvement."
            )
        )
        return [critic_tool]

    def evaluate_content(self, content: str) -> str:
        """
        Critically review a given piece of content and provide structured feedback.

        Args:
            content (str): The content to evaluate (e.g., blog post, summary).

        Returns:
            str: Detailed feedback including strengths, weaknesses, and 
                 suggestions for improvement.
        """
        with open("prompts/critic_agent_prompt.txt", "r", encoding="utf-8") as f:
            critic_agent_template = f.read()

        # Build a chain: system prompt → user content → LLM → output parser
        return (
            ChatPromptTemplate.from_messages([
                ("system", critic_agent_template),
                ("user", "Critically evaluate the following content: {content}")
            ])
            | self.llm
            | StrOutputParser()
        )