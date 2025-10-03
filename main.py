from agent.research_agent import ResearchAgent
from agent.summary_agent import SummaryAgent
from agent.blog_writer_agent import BlogWriterAgent
from agent.critic_agent import CriticAgent
from agent.orchestrator_agent import MainAgent
import os


from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import AgentType

llm = GoogleGenerativeAI(model="gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))

research_agent = ResearchAgent(llm=llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
summary_agent  = SummaryAgent(llm=llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
blog_agent     = BlogWriterAgent(llm=llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
critic_agent   = CriticAgent(llm=llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent = MainAgent(research_agent, summary_agent, blog_agent, critic_agent)
print(agent.run("Conduct research and provide a detailed explanation on Quantum Computer."))