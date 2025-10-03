"""
orchestrator_agent.py

This module defines the MainAgent (orchestrator) in the Agentarium framework. 
The orchestrator coordinates the workflow of multiple specialized agents 
(ResearchAgent → SummaryAgent → BlogWriterAgent → CriticAgent) to form a 
complete content generation and refinement pipeline.
"""

from agent import BaseAgent


class MainAgent:
    """
    Orchestrates the workflow of specialized agents in the content creation pipeline.

    The orchestrator is responsible for:
        1. Delegating research tasks to the ResearchAgent.
        2. Passing research output to the SummaryAgent.
        3. Converting summaries into a blog article via the BlogWriterAgent.
        4. Running final critique and feedback using the CriticAgent.

    Each specialized agent is expected to:
        - Inherit from BaseAgent.
        - Expose a `get_agent()` method that returns a LangChain-compatible agent.
    """

    def __init__(
        self,
        research_agent: BaseAgent,
        summary_agent: BaseAgent,
        blog_writer_agent: BaseAgent,
        critic_agent: BaseAgent
    ):
        """
        Initialize the orchestrator with specialized agents.

        Args:
            research_agent (BaseAgent): The agent responsible for research.
            summary_agent (BaseAgent): The agent responsible for summarization.
            blog_writer_agent (BaseAgent): The agent responsible for blog generation.
            critic_agent (BaseAgent): The agent responsible for critique/review.
        """
        self.research_agent = research_agent
        self.summary_agent = summary_agent
        self.blog_writer_agent = blog_writer_agent
        self.critic_agent = critic_agent

    def run(self, topic: str) -> dict:
        """
        Execute the full content creation and review pipeline.

        Args:
            topic (str): The main topic to research, summarize, expand into a blog, 
                         and critique.

        Returns:
            dict: A dictionary containing outputs from each stage:
                - "raw_research": The unprocessed research output.
                - "summary": Condensed version of research.
                - "blog": A detailed, structured blog post.
                - "review": Critique and feedback on the blog.
        """

        # Step 1: Research phase
        raw_content = self.research_agent.get_agent().invoke(topic)

        # Step 2: Summarization phase
        summary = self.summary_agent.get_agent().invoke(raw_content)

        # Step 3: Blog writing phase
        blog_content = self.blog_writer_agent.get_agent().invoke(summary)

        # Step 4: Critique phase
        review = self.critic_agent.get_agent().invoke(blog_content)

        return {
            "raw_research": raw_content,
            "summary": summary,
            "blog": blog_content,
            "review": review
        }