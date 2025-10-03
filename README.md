# Agentarium

**Agentarium** is an experimental playground for building and orchestrating **multi-agent systems** using the **Agentic design pattern**.
It’s designed as a learning project but built with real-world inspiration — where multiple specialized agents (Researcher, Summarizer, Writer, Critic, etc.) collaborate to solve complex problems in a structured and autonomous way.

Think of it as a **digital think tank**, where different agents have unique roles, coordinate through an **orchestrator**, and work together to generate high-quality outputs.

---

## ✨ Features

* 🔹 **Agent-Oriented Design** – Each agent encapsulates its role (research, summarization, writing, critique).
* 🔹 **Central Orchestration** – A dedicated Orchestrator manages task flow between agents.
* 🔹 **Extendable Framework** – Easy to add new agents or swap out logic.
* 🔹 **Prompt-Driven** – Agents operate using modular prompt templates.
* 🔹 **Practical Use-Cases** – Automates tasks like research, summarization, and blog/article writing.

---

## 🏗 Project Structure

```
Agentarium/
├── README.md
├── agent
│   ├── __init__.py
│   ├── base_agent.py                   # Base class for all agents
│   ├── blog_writer_agent.py            # Generates blog-style content
│   ├── critic_agent.py                 # Provides constructive feedback
│   ├── orchestrator_agent.py           # Coordinates tasks across agents
│   ├── research_agent.py               # Conducts research and gathers information
│   └── summary_agent.py                # Summarizes research into concise form
|
├── main.py                             # Entry point of the project
├── prompts
│   ├── blog_writer_agent_prompt.txt    # Prompt template for blog writer
│   ├── critic_agent_prompt.txt         # Prompt template for critic
│   └── summary_agent_prompt.txt        # Prompt template for summarizer
```

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Himanshu7921/Agentarium
cd Agentarium
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the System

```bash
python main.py
```

---

## 🔧 How It Works

1. **Problem Definition** – Define the task/problem inside `problem.md`.
2. **Research Agent** – Gathers information and drafts a knowledge base.
3. **Summarizer Agent** – Condenses research into structured summaries.
4. **Blog Writer Agent** – Converts summaries into human-friendly articles.
5. **Critic Agent** – Reviews, critiques, and improves the final draft.
6. **Orchestrator Agent** – Controls the pipeline and manages agent collaboration.

---

## 📜 License

MIT License © 2025 — Created with curiosity while exploring **multi-agent design**.