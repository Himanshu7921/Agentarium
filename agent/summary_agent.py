"""
summary_agent.py

This module defines the SummaryAgent, a specialized agent in the Agentarium 
framework designed to generate concise, structured summaries from detailed text.

The SummaryAgent uses a LangChain-compatible LLM and prompt templates to:
1. Condense research outputs into clear, readable summaries.
2. Provide structured information for downstream agents such as BlogWriterAgent 
   and CriticAgent.
"""

from langchain.agents import Tool, initialize_agent
from agent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class SummaryAgent(BaseAgent):
    """
    A specialized agent that generates clear and concise summaries from text.

    Attributes:
        llm: The language model instance used for summarization.
        agent_type (str): Identifier describing this agent's type/role.
        verbose (bool): Flag to enable detailed logging.
    """

    def __init__(self, llm, agent_type: str, verbose: bool = False):
        """
        Initialize the SummaryAgent.

        Args:
            llm: A LangChain-compatible language model instance.
            agent_type (str): The LangChain agent type (e.g., ZERO_SHOT_REACT_DESCRIPTION).
            verbose (bool): If True, enables detailed logs.
        """
        super().__init__(llm, agent_type, verbose)
        self.llm = llm
        self.agent_type = agent_type
        self.verbose = verbose

    def get_agent(self):
        """
        Initialize and return the LangChain agent with configured summarization tools.

        Returns:
            agent: A LangChain agent instance ready to summarize input text.
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
        Define the tools available to the SummaryAgent.

        Currently includes:
            - LLM Summarizer Tool: Generates concise, structured summaries from text.

        Returns:
            list: A list containing the summarizer tool.
        """
        summarizer_tool = Tool.from_function(
            func=self.summarize_using_llm,
            name="LLM Summarizer",
            description=(
                "Use this tool to generate clear and concise summaries from detailed text, "
                "research findings, or documents. The tool identifies key points, extracts "
                "essential information, and produces structured output suitable for analysis, "
                "reporting, or multi-agent workflows."
            )
        )
        return [summarizer_tool]

    def summarize_using_llm(self, content_to_summarize: str) -> str:
        """
        Generate a concise summary of a given input using an LLM.

        This function uses a predefined prompt template to instruct the LLM to
        extract key points and produce a structured summary.

        Args:
            content_to_summarize (str): Text, research output, or document to summarize.

        Returns:
            str: A short, structured summary of the input content.
        """
        with open("prompts/summary_agent_prompt.txt", "r", encoding="utf-8") as f:
            summary_agent_template = f.read()

        # Build a chain: system prompt → user content → LLM → output parser
        return (
            ChatPromptTemplate.from_messages([
                ("system", summary_agent_template),
                ("user", "Generate a concise summary of the following paragraph produced by the Research Agent: {content_to_summarize}")
            ])
            | self.llm
            | StrOutputParser()
        )