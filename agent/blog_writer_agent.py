"""
blog_writer_agent.py

This module defines the BlogWriterAgent, a specialized agent in the 
Agentarium framework responsible for converting summarized content 
into a detailed, blog-ready article.

The agent uses a predefined prompt template and a language model (LLM) 
to generate structured, engaging, and publication-ready blog posts.
"""

from langchain.agents import Tool, initialize_agent
from agent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class BlogWriterAgent(BaseAgent):
    """
    A specialized agent that transforms summarized content into 
    a structured, detailed blog article.

    Inherits from:
        BaseAgent

    Attributes:
        llm: The language model instance used for content generation.
        agent_type (str): Identifier describing this agent's type.
        verbose (bool): Flag for enabling detailed logs.
    """

    def __init__(self, llm, agent_type: str, verbose: bool = False):
        """
        Initialize the BlogWriterAgent.

        Args:
            llm: The language model instance.
            agent_type (str): The agent's role/type (e.g., "blog_writer").
            verbose (bool): If True, enables detailed logging.
        """
        super().__init__(llm, agent_type, verbose)
        self.llm = llm
        self.agent_type = agent_type
        self.verbose = verbose

    def get_agent(self):
        """
        Initialize and return the blog writer agent with its tools.

        Returns:
            A LangChain agent instance configured with blog writing tools.
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
        Define the tools available to the BlogWriterAgent.

        Currently includes:
            - Blog Writer Tool: Expands summarized content into a detailed, 
              engaging, and structured blog post.

        Returns:
            list: A list containing the blog writer tool.
        """
        blog_writer_tool = Tool.from_function(
            func=self.write_blog_post,
            name="Blog Writer Tool",
            description=(
                "Use this tool to convert summarized content into a full-length, engaging blog post. "
                "The tool expands the content with an introduction, detailed explanation of key points, "
                "subheadings, examples, and a conclusion. The output should be ready for publishing."
            )
        )
        return [blog_writer_tool]

    def write_blog_post(self, summary_content: str) -> str:
        """
        Generate a detailed blog post from summarized content.

        Args:
            summary_content (str): A concise summary produced by the SummaryAgent.

        Returns:
            str: A structured blog post with introduction, body sections, 
                 examples, and a conclusion.
        """
        with open("prompts/blog_writer_agent_prompt.txt", "r", encoding="utf-8") as f:
            blog_prompt_template = f.read()

        # Build a chain: system prompt → user summary → LLM → output parser
        return (
            ChatPromptTemplate.from_messages([
                ("system", blog_prompt_template),
                ("user", "Write a blog based on this summary: {summary_content}")
            ])
            | self.llm
            | StrOutputParser()
        )