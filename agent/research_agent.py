"""
research_agent.py

This module defines the ResearchAgent, a specialized agent in the Agentarium 
framework designed to perform research tasks using Wikipedia and a LangChain 
compatible LLM (e.g., Google Gemini).

The ResearchAgent can:
1. Fetch concise, structured summaries from Wikipedia for a given query.
2. Integrate with multi-agent workflows using LangChain.
3. Be initialized with any LangChain-compatible LLM.

Requirements:
- Python 3.10+
- langchain-core
- langchain-google-genai
- wikipedia
- A valid Google Gemini API key set as environment variable 'GOOGLE_API_KEY'

Usage Example:
--------------
from agent import ResearchAgent
llm = build_llm()
agent = ResearchAgent(llm, AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True).get_agent()
result = agent.invoke("Research the latest trends in quantum computing 2025")
print(result)
"""

from langchain.agents import Tool, initialize_agent
from agent import BaseAgent
import wikipedia


class ResearchAgent(BaseAgent):
    """
    A specialized agent for performing research tasks using Wikipedia.

    This agent is designed to fetch concise, structured summaries from 
    Wikipedia, suitable for downstream multi-agent workflows such as 
    summarization, blog writing, and critique.

    Attributes:
        llm: The language model instance powering reasoning and agent actions.
        agent_type (str): Identifier for the type/role of the agent.
        verbose (bool): If True, enables detailed logging for debugging.
    """

    def __init__(self, llm, agent_type: str, verbose: bool = False):
        """
        Initialize the ResearchAgent.

        Args:
            llm: A LangChain-compatible language model instance.
            agent_type (str): The LangChain agent type (e.g., ZERO_SHOT_REACT_DESCRIPTION).
            verbose (bool): If True, prints detailed execution logs.
        """
        super().__init__(llm, agent_type, verbose)
        self.llm = llm
        self.agent_type = agent_type
        self.verbose = verbose

    def define_agent_tools(self) -> list:
        """
        Define the tools this agent will use.

        Currently includes:
            - Wikipedia Research Tool: Fetches concise, structured summaries
              from Wikipedia for a given query.

        Returns:
            list: A list containing LangChain Tool instances.
        """
        wikipedia_research_tool = Tool.from_function(
            func=self.wiki_research,
            name="Wikipedia Research",
            description=(
                "Use this tool to perform research on a topic by fetching a concise, "
                "structured summary from Wikipedia. Returns key facts, trends, or insights "
                "suitable for multi-agent workflows."
            ),
            return_direct=True
        )
        return [wikipedia_research_tool]

    def get_agent(self):
        """
        Initialize and return a LangChain agent configured with tools and LLM.

        Returns:
            agent: LangChain agent instance ready to process queries.
        """
        self.tools = self.define_agent_tools()
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=self.agent_type,
            verbose=self.verbose
        )
        return self.agent

    def wiki_research(self, query: str) -> str:
        """
        Perform a research lookup on Wikipedia for a given topic.

        Searches Wikipedia, retrieves the most relevant page, and returns a
        summary of up to 5 sentences. Handles errors gracefully.

        Args:
            query (str): Topic or search term for research.

        Returns:
            str: Concise research summary or informative error message.
        """
        try:
            search_results = wikipedia.search(query)
            if not search_results:
                return "No research results found on Wikipedia."
            research_summary = wikipedia.summary(search_results[0], sentences=5)
            return research_summary
        except Exception as e:
            return f"Error during Wikipedia research: {e}"