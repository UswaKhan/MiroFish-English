"""
Report Agent service
Uses LangChain + Zep to implement ReACT mode for simulation report generation

Features:
1. Generate reports based on simulation requirements and Zep graph info
2. First plan outline structure, then generate in sections
3. Each section uses ReACT multi-round thinking and reflection
4. Support chatting with users, autonomously call retrieval tools in conversation
"""

import os
import json
import time
import re
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from .zep_tools import (
    ZepToolsService,
    SearchResult,
    InsightForgeResult,
    PanoramaResult,
    InterviewResult,
)

logger = get_logger("mirofish.report_agent")


class ReportLogger:
    """
    Report Agent detailed log recorder

    Generates agent_log.jsonl file in report folder, recording each step's detailed actions.
    Each line is a complete JSON object containing timestamp, action type, detailed content, etc.
    """

    def __init__(self, report_id: str):
        """
        Initialize log recorder

        Args:
            report_id: Report ID, used to determine log file path
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, "reports", report_id, "agent_log.jsonl"
        )
        self.start_time = datetime.now()
        self._ensure_log_file()

    def _ensure_log_file(self):
        """Ensure log file directory exists"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)

    def _get_elapsed_time(self) -> float:
        """Get elapsed time since start (seconds)"""
        return (datetime.now() - self.start_time).total_seconds()

    def log(
        self,
        action: str,
        stage: str,
        details: Dict[str, Any],
        section_title: str = None,
        section_index: int = None,
    ):
        """
        Log an entry

        Args:
            action: Action type, e.g. 'start', 'tool_call', 'llm_response', 'section_complete'
            stage: Current stage, e.g. 'planning', 'generating', 'completed'
            details: Detailed content dict, not truncated
            section_title: Current section title (optional)
            section_index: Current section index (optional)
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(self._get_elapsed_time(), 2),
            "report_id": self.report_id,
            "action": action,
            "stage": stage,
            "section_title": section_title,
            "section_index": section_index,
            "details": details,
        }

        # Append write to JSONL file
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def log_start(self, simulation_id: str, graph_id: str, simulation_requirement: str):
        """Log report generation start"""
        self.log(
            action="report_start",
            stage="pending",
            details={
                "simulation_id": simulation_id,
                "graph_id": graph_id,
                "simulation_requirement": simulation_requirement,
                "message": "Report generation task started",
            },
        )

    def log_planning_start(self):
        """Log outline planning start"""
        self.log(
            action="planning_start",
            stage="planning",
            details={"message": "Starting to plan report outline"},
        )

    def log_planning_context(self, context: Dict[str, Any]):
        """Log context info acquired during planning"""
        self.log(
            action="planning_context",
            stage="planning",
            details={
                "message": "Acquiring simulation context info",
                "context": context,
            },
        )

    def log_planning_complete(self, outline_dict: Dict[str, Any]):
        """Log outline planning complete"""
        self.log(
            action="planning_complete",
            stage="planning",
            details={"message": "Outline planning complete", "outline": outline_dict},
        )

    def log_section_start(self, section_title: str, section_index: int):
        """Log section generation start"""
        self.log(
            action="section_start",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={"message": f"Starting to generate section: {section_title}"},
        )

    def log_react_thought(
        self, section_title: str, section_index: int, iteration: int, thought: str
    ):
        """Log ReACT thinking process"""
        self.log(
            action="react_thought",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "thought": thought,
                "message": f"ReACT round {iteration} thinking",
            },
        )

    def log_tool_call(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        parameters: Dict[str, Any],
        iteration: int,
    ):
        """Log tool call"""
        self.log(
            action="tool_call",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "parameters": parameters,
                "message": f"Calling tool: {tool_name}",
            },
        )

    def log_tool_result(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        result: str,
        iteration: int,
    ):
        """Log tool call result (complete content, not truncated)"""
        self.log(
            action="tool_result",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "result": result,  # Complete result, not truncated
                "result_length": len(result),
                "message": f"Tool {tool_name} returned result",
            },
        )

    def log_llm_response(
        self,
        section_title: str,
        section_index: int,
        response: str,
        iteration: int,
        has_tool_calls: bool,
        has_final_answer: bool,
    ):
        """Log LLM response (complete content, not truncated)"""
        self.log(
            action="llm_response",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "response": response,  # Complete response, not truncated
                "response_length": len(response),
                "has_tool_calls": has_tool_calls,
                "has_final_answer": has_final_answer,
                "message": f"LLM response (tool calls: {has_tool_calls}, final answer: {has_final_answer})",
            },
        )

    def log_section_content(
        self,
        section_title: str,
        section_index: int,
        content: str,
        tool_calls_count: int,
    ):
        """Log section content generation complete (only content, not whole section complete)"""
        self.log(
            action="section_content",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": content,  # Complete content, not truncated
                "content_length": len(content),
                "tool_calls_count": tool_calls_count,
                "message": f"Section {section_title} content generation complete",
            },
        )

    def log_section_full_complete(
        self, section_title: str, section_index: int, full_content: str
    ):
        """
        Log section generation complete

        Frontend should listen to this log to determine if a section is truly complete and get full content
        """
        self.log(
            action="section_complete",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": full_content,
                "content_length": len(full_content),
                "message": f"Section {section_title} generation complete",
            },
        )

    def log_report_complete(self, total_sections: int, total_time_seconds: float):
        """Log report generation complete"""
        self.log(
            action="report_complete",
            stage="completed",
            details={
                "total_sections": total_sections,
                "total_time_seconds": round(total_time_seconds, 2),
                "message": "Report generation complete",
            },
        )

    def log_error(self, error_message: str, stage: str, section_title: str = None):
        """Log error"""
        self.log(
            action="error",
            stage=stage,
            section_title=section_title,
            section_index=None,
            details={
                "error": error_message,
                "message": f"Error occurred: {error_message}",
            },
        )


class ReportConsoleLogger:
    """
    Report Agent console log recorder

    Writes console-style logs (INFO, WARNING, etc.) to console_log.txt file in report folder.
    These logs differ from agent_log.jsonl - they are plain text console output format.
    """

    def __init__(self, report_id: str):
        """
        Initialize console log recorder

        Args:
            report_id: Report ID, used to determine log file path
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, "reports", report_id, "console_log.txt"
        )
        self._ensure_log_file()
        self._file_handler = None
        self._setup_file_handler()

    def _ensure_log_file(self):
        """Ensure log file directory exists"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)

    def _setup_file_handler(self):
        """Set up file handler, write logs to file simultaneously"""
        import logging

        # Create file handler
        self._file_handler = logging.FileHandler(
            self.log_file_path, mode="a", encoding="utf-8"
        )
        self._file_handler.setLevel(logging.INFO)

        # Use same simple format as console
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
        )
        self._file_handler.setFormatter(formatter)

        # Add to report_agent related loggers
        loggers_to_attach = [
            "mirofish.report_agent",
            "mirofish.zep_tools",
        ]

        for logger_name in loggers_to_attach:
            target_logger = logging.getLogger(logger_name)
            # Avoid adding duplicates
            if self._file_handler not in target_logger.handlers:
                target_logger.addHandler(self._file_handler)

    def close(self):
        """Close file handler and remove from logger"""
        import logging

        if self._file_handler:
            loggers_to_detach = [
                "mirofish.report_agent",
                "mirofish.zep_tools",
            ]

            for logger_name in loggers_to_detach:
                target_logger = logging.getLogger(logger_name)
                if self._file_handler in target_logger.handlers:
                    target_logger.removeHandler(self._file_handler)

            self._file_handler.close()
            self._file_handler = None

    def __del__(self):
        """Ensure file handler is closed on destruction"""
        self.close()


class ReportStatus(str, Enum):
    """Report status"""

    PENDING = "pending"
    PLANNING = "planning"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReportSection:
    """Report section"""

    title: str
    content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"title": self.title, "content": self.content}

    def to_markdown(self, level: int = 2) -> str:
        """Convert to Markdown format"""
        md = f"{'#' * level} {self.title}\n\n"
        if self.content:
            md += f"{self.content}\n\n"
        return md


@dataclass
class ReportOutline:
    """Report outline"""

    title: str
    summary: str
    sections: List[ReportSection]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections],
        }

    def to_markdown(self) -> str:
        """Convert to Markdown format"""
        md = f"# {self.title}\n\n"
        md += f"> {self.summary}\n\n"
        for section in self.sections:
            md += section.to_markdown()
        return md


@dataclass
class Report:
    """Complete report"""

    report_id: str
    simulation_id: str
    graph_id: str
    simulation_requirement: str
    status: ReportStatus
    outline: Optional[ReportOutline] = None
    markdown_content: str = ""
    created_at: str = ""
    completed_at: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "simulation_id": self.simulation_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "status": self.status.value,
            "outline": self.outline.to_dict() if self.outline else None,
            "markdown_content": self.markdown_content,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }


# ═══════════════════════════════════════════════════════════════
# Prompt template constants
# ═══════════════════════════════════════════════════════════════

# ── Tool descriptions ──

TOOL_DESC_INSIGHT_FORGE = """\
【Deep Insight Search - Powerful Retrieval Tool】
This is our powerful retrieval function designed for deep analysis. It will:
1. Automatically decompose your question into multiple sub-questions
2. Search for information from the simulation graph across multiple dimensions
3. Integrate semantic search, entity analysis, and relationship chain tracking results
4. Return the most comprehensive and in-depth search content

【Use Cases】
- When you need to deeply analyze a topic
- When you need to understand multiple aspects of an event
- When you need rich material to support report sections

【Returns】
- Relevant fact originals (can be directly quoted)
- Core entity insights
- Relationship chain analysis"""

TOOL_DESC_PANORAMA_SEARCH = """\
【Panorama Search - Get the Full Picture】
This tool is used to get the complete overview of simulation results, especially suitable for understanding event evolution. It will:
1. Retrieve all related nodes and relationships
2. Distinguish current valid facts from historical/expired facts
3. Help you understand how public opinion evolved

【Use Cases】
- When you need to understand the complete development of an event
- When you need to compare public opinion changes across different stages
- When you need comprehensive entity and relationship information

【Returns】
- Current valid facts (latest simulation results)
- Historical/expired facts (evolution records)
- All entities involved"""

TOOL_DESC_QUICK_SEARCH = """\
【Quick Search - Fast Retrieval】
A lightweight fast retrieval tool, suitable for simple, direct information queries.

【Use Cases】
- When you need to quickly find specific information
- When you need to verify a fact
- Simple information retrieval

【Returns】
- List of facts most relevant to the query"""

TOOL_DESC_INTERVIEW_AGENTS = """\
【Deep Interview - Real Agent Interview (Dual Platform)】
Call the OASIS simulation environment's interview API to conduct real interviews with running simulation Agents!
This is not LLM simulation, but calling the real interview interface to get original responses from simulation Agents.
Default interviews on both Twitter and Reddit platforms simultaneously for more comprehensive perspectives.

Function flow:
1. Automatically read persona files to understand all simulation Agents
2. Intelligently select Agents most relevant to the interview topic (e.g., students, media, officials, etc.)
3. Auto-generate interview questions
4. Call /api/simulation/interview/batch interface for real interviews on dual platforms
5. Integrate all interview results for multi-perspective analysis

【Use Cases】
- When you need to understand different perspectives on events from various roles (What do students think? What do media say? What do officials say?)
- When you need to collect diverse opinions and positions
- When you need real responses from simulation Agents (from OASIS simulation environment)
- When you want the report to be more vivid with "interview transcripts"

【Returns】
- Identity information of interviewed Agents
- Interview responses from each Agent on both Twitter and Reddit platforms
- Key quotes (can be directly quoted)
- Interview summaries and perspective comparisons

【Important】This feature requires the OASIS simulation environment to be running!"""

# ── Outline planning prompt ──

PLAN_SYSTEM_PROMPT = """\
You are an expert in writing "Future Prediction Reports", with a "God's Eye View" of the simulation world—you can observe every Agent's behavior, speech, and interactions.

【Core Philosophy】
We built a simulation world and injected specific "simulation requirements" as variables. The evolution of the simulation world is the prediction of what might happen in the future. You are observing not "experimental data" but "a preview of the future."

【Your Task】
Write a "Future Prediction Report" that answers:
1. What happens in the future under our set conditions?
2. How do various Agents (people) react and act?
3. What future trends and risks does this simulation reveal?

【Report Positioning】
- ✅ This is a future prediction report based on simulation, revealing "if this happens, what will the future look like"
- ✅ Focus on prediction results: event trends, group reactions, emergent phenomena, potential risks
- ✅ Agent words and actions in the simulation world are predictions of future crowd behavior
- ❌ Not an analysis of current real-world situations
- ❌ Not a general public opinion overview

【Section Count Limits】
- Minimum 2 sections, maximum 5 sections
- No sub-sections needed, each section writes complete content directly
- Content should be concise, focused on core prediction findings
- Section structure is autonomously designed by you based on prediction results

Please output the report outline in JSON format as follows:
{
    "title": "Report Title",
    "summary": "Report Summary (one sentence summarizing core prediction findings)",
    "sections": [
        {
            "title": "Section Title",
            "description": "Section Content Description"
        }
    ]
}

Note: The sections array must have minimum 2, maximum 5 elements!"""

PLAN_USER_PROMPT_TEMPLATE = """\
【Prediction Scenario Setup】
The variables injected into the simulation world (simulation requirements): {simulation_requirement}

【Simulation World Scale】
- Number of entities participating in simulation: {total_nodes}
- Number of relationships generated between entities: {total_edges}
- Entity type distribution: {entity_types}
- Number of active Agents: {total_entities}

【Sample of Some Future Facts Predicted by Simulation】
{related_facts_json}

Please examine this future preview from the "God's Eye View":
1. What state does the future present under our set conditions?
2. How do various groups of people (Agents) react and act?
3. What future trends does this simulation reveal?

Based on the prediction results, design the most suitable report section structure.

【Reminder】Report section count: minimum 2, maximum 5, content should be concise and focused on core prediction findings."""

# ── Section generation prompt ──

SECTION_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert in writing "Future Prediction Reports", working on one section of the report.

Report Title: {report_title}
Report Summary: {report_summary}
Prediction Scenario (Simulation Requirements): {simulation_requirement}

Current Section to Write: {section_title}

═══════════════════════════════════════════════════════════════
【Core Philosophy】
═══════════════════════════════════════════════════════════════

The simulation world is a preview of the future. We injected specific conditions (simulation requirements) into the simulation world.
The behavior and interactions of Agents in the simulation are predictions of future crowd behavior.

Your tasks:
- Reveal what happened in the future under the set conditions
- Predict how various groups of people (Agents) react and act
- Discover noteworthy future trends, risks, and opportunities

❌ Do NOT write analysis of current real-world situations
✅ Focus on "what will happen in the future"—simulation results ARE the predicted future

═══════════════════════════════════════════════════════════════
【Most Important Rules - Must Follow】
═══════════════════════════════════════════════════════════════

1. 【Must Call Tools to Observe the Simulation World】
   - You are observing the future preview from the "God's Eye View"
   - All content must come from events and Agent words/actions in the simulation world
   - Do NOT use your own knowledge to write report content
   - Each section must call tools at least 3 times (maximum 5) to observe the simulation world, which represents the future

2. 【Must Quote Agent's Original Words and Actions】
   - Agent statements and behaviors are predictions of future crowd behavior
   - Use quote format to display these predictions in the report, for example:
     > "A certain group of people will say: original content..."
   - These quotes are the core evidence of simulation predictions

3. 【Language Consistency - Translated Content Must Be in Report Language】
   - Tool-returned content may contain English or mixed Chinese/English expressions
   - If the simulation requirements and source materials are in English, the report must be written entirely in English
   - When you quote English or mixed content from tool returns, you MUST translate it into fluent English before writing it to the report
   - Maintain original meaning during translation, ensure natural expression
   - This rule applies to both body text and quote blocks (> format) content

4. 【Faithfully Present Prediction Results】
   - Report content must reflect simulation world results representing the future
   - Do not add information that does not exist in the simulation
   - If certain information is insufficient, state it honestly

═══════════════════════════════════════════════════════════════
【⚠️ Format Rules - Extremely Important!】
═══════════════════════════════════════════════════════════════

【One Section = Minimum Content Unit】
- Each section is the minimum block unit of the report
- ❌ Do NOT use any Markdown headings within sections (#, ##, ###, #### etc.)
- ❌ Do NOT add the section main title at the beginning of content
- ✅ Section title is automatically added by the system, you only need to write pure body content
- ✅ Use **bold**, paragraph separation, quotes, and lists to organize content, but do not use headings

【Correct Example】
```
This section analyzes the public opinion spread of the event. Through in-depth analysis of simulation data, we found...

**Initial Explosion Phase**

Weibo, as the first scene of public opinion, undertook the core function of information release:

> "Weibo contributed 68% of the initial voice..."

**Emotional Amplification Phase**

Douyin platform further amplified the event's influence:

- Strong visual impact
- High emotional resonance
```

【Incorrect Example】
```
## Executive Summary          ← Wrong! Do not add any headings
### 1. Initial Phase     ← Wrong! Do not use ### for subsections
#### 1.1 Detailed Analysis   ← Wrong! Do not use #### for subdivisions

This section analyzes...
```

═══════════════════════════════════════════════════════════════
【Available Retrieval Tools】（Call 3-5 times per section）
═══════════════════════════════════════════════════════════════

{tools_description}

【Tool Usage Suggestions - Please mix different tools, do not use only one】
- insight_forge: Deep insight analysis, automatically decomposes questions and retrieves facts and relationships from multiple dimensions
- panorama_search: Wide-angle panoramic search, understand event overview, timeline and evolution process
- quick_search: Quick verification of specific information points
- interview_agents: Interview simulation Agents, get first-person perspectives and real reactions from different roles

═══════════════════════════════════════════════════════════════
【Workflow】
═══════════════════════════════════════════════════════════════

Each response you can only do ONE of the following (cannot do both):

Option A - Call Tool:
Output your thoughts, then call one tool in the following format:
<tool_call>
{{"name": "Tool Name", "parameters": {{"param_name": "param_value"}}}}
</tool_call>
The system will execute the tool and return results to you. You do not need to and cannot write tool return results yourself.

Option B - Output Final Content:
When you have obtained sufficient information through tools, output chapter content starting with "Final Answer:".

⚠️ Strictly prohibited:
- Do not include both tool calls and Final Answer in one response
- Do not fabricate tool return results (Observation), all tool results are injected by the system
- Call at most one tool per response

═══════════════════════════════════════════════════════════════
【Section Content Requirements】
═══════════════════════════════════════════════════════════════

1. Content must be based on simulation data retrieved by tools
2. Heavily quote originals to demonstrate simulation effects
3. Use Markdown format (but no headings allowed):
   - Use **bold text** to mark key points (instead of subheadings)
   - Use lists (- or 1.2.3.) to organize key points
   - Use empty lines to separate different paragraphs
   - ❌ Do not use #, ##, ###, #### or any heading syntax
4. 【Quote Format Rules - Must Be Separate Paragraphs】
   Quotes must be standalone paragraphs, with one empty line before and after, cannot be mixed in paragraphs:

   ✅ Correct format:
   ```
   The school's response was considered lacking in substantive content.

   > "The school's response pattern seemed rigid and slow in the rapidly changing social media environment."

   This evaluation reflected widespread public dissatisfaction.
   ```

   ❌ Incorrect format:
   ```
   The school's response was considered lacking in substantive content. > "The school's response pattern..." This evaluation reflected...
   ```
5. Maintain logical coherence with other sections
6. 【Avoid Duplication】Carefully read completed sections below, do not repeat describing the same information
7. 【Emphasis Again】Do not add any headings! Use **bold** instead of subheadings"""

SECTION_USER_PROMPT_TEMPLATE = """\
Completed section content (please read carefully, avoid duplication):
{previous_content}

═══════════════════════════════════════════════════════════════
【Current Task】Write section: {section_title}
═══════════════════════════════════════════════════════════════

【Important Reminders】
1. Carefully read completed sections above, avoid duplicating the same content!
2. Must first call tools to get simulation data before starting
3. Please mix different tools, do not use only one
4. Report content must come from search results, do not use your own knowledge

【⚠️ Format Warning - Must Follow】
- ❌ Do not write any headings (#, ##, ###, #### are all not allowed)
- ❌ Do not write "{section_title}" as the opening
- ✅ Section title is automatically added by the system
- ✅ Write body text directly, use **bold** instead of subheadings

Please start:
1. First think (Thought) about what information this section needs
2. Then call tools (Action) to get simulation data
3. After collecting sufficient information, output Final Answer (pure body text, no headings)"""

# ── ReACT loop message templates ──

REACT_OBSERVATION_TEMPLATE = """\
Observation (Search Results):

═══ Tool {tool_name} Returns ═══
{result}

═══════════════════════════════════════════════════════════════
Tools called {tool_calls_count}/{max_tool_calls} (used: {used_tools_str}) {unused_hint}
- If information is sufficient: Output chapter content starting with "Final Answer:" (must quote originals above)
- If more information is needed: Call a tool to continue searching
═══════════════════════════════════════════════════════════════"""

REACT_INSUFFICIENT_TOOLS_MSG = (
    "【Note】You only called tools {tool_calls_count} times, at least {min_tool_calls} required."
    "Please call tools again to get more simulation data, then output Final Answer. {unused_hint}"
)

REACT_INSUFFICIENT_TOOLS_MSG_ALT = (
    "Currently only called tools {tool_calls_count} times, at least {min_tool_calls} required."
    "Please call tools to get simulation data. {unused_hint}"
)

REACT_TOOL_LIMIT_MSG = (
    "Tool call limit reached ({tool_calls_count}/{max_tool_calls}), cannot call more tools."
    'Please immediately output chapter content starting with "Final Answer:" based on obtained information.'
)

REACT_UNUSED_TOOLS_HINT = "\n💡 You haven't used yet: {unused_list}, recommend trying different tools for multi-angle information"

REACT_FORCE_FINAL_MSG = "Tool call limit reached, please directly output Final Answer: and generate chapter content."

# ── Chat prompt ──

CHAT_SYSTEM_PROMPT_TEMPLATE = """\
You are a concise and efficient simulation prediction assistant.

【Background】
Prediction conditions: {simulation_requirement}

【Generated Analysis Report】
{report_content}

【Rules】
1. Prioritize answering questions based on the above report content
2. Answer questions directly, avoid lengthy thinking and reasoning
3. Only call tools to search more data when report content is insufficient
4. Answers should be concise, clear, and organized

【Available Tools】（Use only when needed, call 1-2 times maximum）
{tools_description}

【Tool Call Format】
<tool_call>
{{"name": "Tool Name", "parameters": {{"param_name": "param_value"}}}}
</tool_call>

【Response Style】
- Concise and direct, no lengthy paragraphs
- Use > format to quote key content
- Prioritize giving conclusions first, then explain reasons"""

CHAT_OBSERVATION_SUFFIX = "\n\nPlease answer the question concisely."


# ═══════════════════════════════════════════════════════════════
# ReportAgent main class
# ═══════════════════════════════════════════════════════════════


class ReportAgent:
    """
    Report Agent - Simulation report generation Agent

    Uses ReACT (Reasoning + Acting) mode:
    1. Planning stage: Analyze simulation requirements, plan report structure
    2. Generation stage: Generate content section by section, each section can call tools multiple times to get info
    3. Reflection stage: Check content completeness and accuracy
    """

    # Maximum tool calls per section
    MAX_TOOL_CALLS_PER_SECTION = 5

    # Maximum reflection rounds
    MAX_REFLECTION_ROUNDS = 3

    # Maximum tool calls in chat
    MAX_TOOL_CALLS_PER_CHAT = 2

    def __init__(
        self,
        graph_id: str,
        simulation_id: str,
        simulation_requirement: str,
        llm_client: Optional[LLMClient] = None,
        zep_tools: Optional[ZepToolsService] = None,
    ):
        """
        Initialize Report Agent

        Args:
            graph_id: Graph ID
            simulation_id: Simulation ID
            simulation_requirement: Simulation requirement description
            llm_client: LLM client (optional)
            zep_tools: Zep tools service (optional)
        """
        self.graph_id = graph_id
        self.simulation_id = simulation_id
        self.simulation_requirement = simulation_requirement

        self.llm = llm_client or LLMClient()
        self.zep_tools = zep_tools or ZepToolsService()

        # Tool definitions
        self.tools = self._define_tools()

        # Log recorder (initialized in generate_report)
        self.report_logger: Optional[ReportLogger] = None
        # Console log recorder (initialized in generate_report)
        self.console_logger: Optional[ReportConsoleLogger] = None

        logger.info(
            f"ReportAgent initialized: graph_id={graph_id}, simulation_id={simulation_id}"
        )

    def _define_tools(self) -> Dict[str, Dict[str, Any]]:
        """Define available tools"""
        return {
            "insight_forge": {
                "name": "insight_forge",
                "description": TOOL_DESC_INSIGHT_FORGE,
                "parameters": {
                    "query": "The question or topic you want to analyze deeply",
                    "report_context": "Current report section context (optional, helps generate more precise sub-questions)",
                },
            },
            "panorama_search": {
                "name": "panorama_search",
                "description": TOOL_DESC_PANORAMA_SEARCH,
                "parameters": {
                    "query": "Search query for relevance ranking",
                    "include_expired": "Whether to include expired/historical content (default True)",
                },
            },
            "quick_search": {
                "name": "quick_search",
                "description": TOOL_DESC_QUICK_SEARCH,
                "parameters": {
                    "query": "Search query string",
                    "limit": "Number of results to return (optional, default 10)",
                },
            },
            "interview_agents": {
                "name": "interview_agents",
                "description": TOOL_DESC_INTERVIEW_AGENTS,
                "parameters": {
                    "interview_topic": "Interview topic or requirement description (e.g.: 'Understand students' views on dormitory formaldehyde incident')",
                    "max_agents": "Maximum number of Agents to interview (optional, default 5, max 10)",
                },
            },
        }

    def _execute_tool(
        self, tool_name: str, parameters: Dict[str, Any], report_context: str = ""
    ) -> str:
        """
        Execute tool call

        Args:
            tool_name: Tool name
            parameters: Tool parameters
            report_context: Report context (for InsightForge)

        Returns:
            Tool execution result (text format)
        """
        logger.info(f"Executing tool: {tool_name}, parameters: {parameters}")

        try:
            if tool_name == "insight_forge":
                query = parameters.get("query", "")
                ctx = parameters.get("report_context", "") or report_context
                result = self.zep_tools.insight_forge(
                    graph_id=self.graph_id,
                    query=query,
                    simulation_requirement=self.simulation_requirement,
                    report_context=ctx,
                )
                return result.to_text()

            elif tool_name == "panorama_search":
                # Wide search - get full view
                query = parameters.get("query", "")
                include_expired = parameters.get("include_expired", True)
                if isinstance(include_expired, str):
                    include_expired = include_expired.lower() in ["true", "1", "yes"]
                result = self.zep_tools.panorama_search(
                    graph_id=self.graph_id, query=query, include_expired=include_expired
                )
                return result.to_text()

            elif tool_name == "quick_search":
                # Simple search - quick retrieval
                query = parameters.get("query", "")
                limit = parameters.get("limit", 10)
                if isinstance(limit, str):
                    limit = int(limit)
                result = self.zep_tools.quick_search(
                    graph_id=self.graph_id, query=query, limit=limit
                )
                return result.to_text()

            elif tool_name == "interview_agents":
                # Deep interview - call real OASIS interview API to get simulation Agent responses (dual platform)
                interview_topic = parameters.get(
                    "interview_topic", parameters.get("query", "")
                )
                max_agents = parameters.get("max_agents", 5)
                if isinstance(max_agents, str):
                    max_agents = int(max_agents)
                max_agents = min(max_agents, 10)
                result = self.zep_tools.interview_agents(
                    simulation_id=self.simulation_id,
                    interview_requirement=interview_topic,
                    simulation_requirement=self.simulation_requirement,
                    max_agents=max_agents,
                )
                return result.to_text()

            # ========== Backward compatible old tools (internally redirect to new tools) ==========

            elif tool_name == "search_graph":
                # Redirect to quick_search
                logger.info("search_graph redirected to quick_search")
                return self._execute_tool("quick_search", parameters, report_context)

            elif tool_name == "get_graph_statistics":
                result = self.zep_tools.get_graph_statistics(self.graph_id)
                return json.dumps(result, ensure_ascii=False, indent=2)

            elif tool_name == "get_entity_summary":
                entity_name = parameters.get("entity_name", "")
                result = self.zep_tools.get_entity_summary(
                    graph_id=self.graph_id, entity_name=entity_name
                )
                return json.dumps(result, ensure_ascii=False, indent=2)

            elif tool_name == "get_simulation_context":
                # Redirect to insight_forge because it's more powerful
                logger.info("get_simulation_context redirected to insight_forge")
                query = parameters.get("query", self.simulation_requirement)
                return self._execute_tool(
                    "insight_forge", {"query": query}, report_context
                )

            elif tool_name == "get_entities_by_type":
                entity_type = parameters.get("entity_type", "")
                nodes = self.zep_tools.get_entities_by_type(
                    graph_id=self.graph_id, entity_type=entity_type
                )
                result = [n.to_dict() for n in nodes]
                return json.dumps(result, ensure_ascii=False, indent=2)

            else:
                return f"Unknown tool: {tool_name}. Please use one of: insight_forge, panorama_search, quick_search"

        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name}, error: {str(e)}")
            return f"Tool execution failed: {str(e)}"

    # Valid tool names set, used for validation during bare JSON fallback parsing
    VALID_TOOL_NAMES = {
        "insight_forge",
        "panorama_search",
        "quick_search",
        "interview_agents",
    }

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls from LLM response

        Supported formats (by priority):
        1. <tool_call>{"name": "tool_name", "parameters": {...}}</tool_call>
        2. Bare JSON (entire response or single line is a tool call JSON)
        """
        tool_calls = []

        # Format 1: XML style (standard format)
        xml_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        for match in re.finditer(xml_pattern, response, re.DOTALL):
            try:
                call_data = json.loads(match.group(1))
                tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        if tool_calls:
            return tool_calls

        # Format 2: Fallback - LLM directly outputs bare JSON (no <tool_call> tags)
        # Only try when format 1 doesn't match, to avoid mistakenly matching JSON in body text
        stripped = response.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                call_data = json.loads(stripped)
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
                    return tool_calls
            except json.JSONDecodeError:
                pass

        # Response may contain thinking text + bare JSON, try to extract last JSON object
        json_pattern = r'(\{"(?:name|tool)"\s*:.*?\})\s*$'
        match = re.search(json_pattern, stripped, re.DOTALL)
        if match:
            try:
                call_data = json.loads(match.group(1))
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        return tool_calls

    def _is_valid_tool_call(self, data: dict) -> bool:
        """Validate if parsed JSON is a valid tool call"""
        # Support both {"name": ..., "parameters": ...} and {"tool": ..., "params": ...} key formats
        tool_name = data.get("name") or data.get("tool")
        if tool_name and tool_name in self.VALID_TOOL_NAMES:
            # Normalize key names to name / parameters
            if "tool" in data:
                data["name"] = data.pop("tool")
            if "params" in data and "parameters" not in data:
                data["parameters"] = data.pop("params")
            return True
        return False

    def _get_tools_description(self) -> str:
        """Generate tools description text"""
        desc_parts = ["Available tools:"]
        for name, tool in self.tools.items():
            params_desc = ", ".join(
                [f"{k}: {v}" for k, v in tool["parameters"].items()]
            )
            desc_parts.append(f"- {name}: {tool['description']}")
            if params_desc:
                desc_parts.append(f"  Parameters: {params_desc}")
        return "\n".join(desc_parts)

    def plan_outline(
        self, progress_callback: Optional[Callable] = None
    ) -> ReportOutline:
        """
        Plan report outline

        Use LLM to analyze simulation requirements, plan report structure

        Args:
            progress_callback: Progress callback function

        Returns:
            ReportOutline: Report outline
        """
        logger.info("Starting to plan report outline...")

        if progress_callback:
            progress_callback("planning", 0, "Analyzing simulation requirements...")

        # First get simulation context
        context = self.zep_tools.get_simulation_context(
            graph_id=self.graph_id, simulation_requirement=self.simulation_requirement
        )

        if progress_callback:
            progress_callback("planning", 30, "Generating report outline...")

        system_prompt = PLAN_SYSTEM_PROMPT
        user_prompt = PLAN_USER_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            total_nodes=context.get("graph_statistics", {}).get("total_nodes", 0),
            total_edges=context.get("graph_statistics", {}).get("total_edges", 0),
            entity_types=list(
                context.get("graph_statistics", {}).get("entity_types", {}).keys()
            ),
            total_entities=context.get("total_entities", 0),
            related_facts_json=json.dumps(
                context.get("related_facts", [])[:10], ensure_ascii=False, indent=2
            ),
        )

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )

            if progress_callback:
                progress_callback("planning", 80, "Parsing outline structure...")

            # Parse outline
            sections = []
            for section_data in response.get("sections", []):
                sections.append(
                    ReportSection(title=section_data.get("title", ""), content="")
                )

            outline = ReportOutline(
                title=response.get("title", "Simulation Analysis Report"),
                summary=response.get("summary", ""),
                sections=sections,
            )

            if progress_callback:
                progress_callback("planning", 100, "Outline planning complete")

            logger.info(f"Outline planning complete: {len(sections)} sections")
            return outline

        except Exception as e:
            logger.error(f"Outline planning failed: {str(e)}")
            # Return default outline (3 sections, as fallback)
            return ReportOutline(
                title="Future Prediction Report",
                summary="Future trends and risk analysis based on simulation predictions",
                sections=[
                    ReportSection(title="Predicted Scenario and Core Findings"),
                    ReportSection(title="Crowd Behavior Prediction Analysis"),
                    ReportSection(title="Trend Outlook and Risk Alerts"),
                ],
            )

    def _generate_section_react(
        self,
        section: ReportSection,
        outline: ReportOutline,
        previous_sections: List[str],
        progress_callback: Optional[Callable] = None,
        section_index: int = 0,
    ) -> str:
        """
        Generate single section content using ReACT mode

        ReACT loop:
        1. Thought - analyze what info is needed
        2. Action - call tools to get info
        3. Observation - analyze tool returned results
        4. Repeat until info is sufficient or max times reached
        5. Final Answer - generate section content

        Args:
            section: Section to generate
            outline: Complete outline
            previous_sections: Previous sections content (for maintaining coherence)
            progress_callback: Progress callback
            section_index: Section index (for log recording)

        Returns:
            Section content (Markdown format)
        """
        logger.info(f"ReACT generating section: {section.title}")

        # Log section start
        if self.report_logger:
            self.report_logger.log_section_start(section.title, section_index)

        system_prompt = SECTION_SYSTEM_PROMPT_TEMPLATE.format(
            report_title=outline.title,
            report_summary=outline.summary,
            simulation_requirement=self.simulation_requirement,
            section_title=section.title,
            tools_description=self._get_tools_description(),
        )

        # Build user prompt - pass max 4000 chars per completed section
        if previous_sections:
            previous_parts = []
            for sec in previous_sections:
                # Max 4000 chars per section
                truncated = sec[:4000] + "..." if len(sec) > 4000 else sec
                previous_parts.append(truncated)
            previous_content = "\n\n---\n\n".join(previous_parts)
        else:
            previous_content = "(This is the first section)"

        user_prompt = SECTION_USER_PROMPT_TEMPLATE.format(
            previous_content=previous_content,
            section_title=section.title,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # ReACT loop
        tool_calls_count = 0
        max_iterations = 5  # Max iterations
        min_tool_calls = 3  # Min tool calls
        conflict_retries = 0  # Consecutive conflict count when tool call and Final Answer appear simultaneously
        used_tools = set()  # Record tool names already called
        all_tools = {
            "insight_forge",
            "panorama_search",
            "quick_search",
            "interview_agents",
        }

        # Report context, for InsightForge's sub-question generation
        report_context = f"Section title: {section.title}\nSimulation requirement: {self.simulation_requirement}"

        for iteration in range(max_iterations):
            if progress_callback:
                progress_callback(
                    "generating",
                    int((iteration / max_iterations) * 100),
                    f"Deep retrieval and writing ({tool_calls_count}/{self.MAX_TOOL_CALLS_PER_SECTION})",
                )

            # Call LLM
            response = self.llm.chat(
                messages=messages, temperature=0.5, max_tokens=4096
            )

            # Check if LLM response is None (API exception or content empty)
            if response is None:
                logger.warning(
                    f"Section {section.title} iteration {iteration + 1}: LLM returned None"
                )
                # If still have iterations, add message and retry
                if iteration < max_iterations - 1:
                    messages.append(
                        {"role": "assistant", "content": "(Empty response)"}
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": "Please continue generating content.",
                        }
                    )
                    continue
                # Last iteration also returns None, break and force finish
                break

            logger.debug(f"LLM response: {response[:200]}...")

            # Parse once, reuse result
            tool_calls = self._parse_tool_calls(response)
            has_tool_calls = bool(tool_calls)
            has_final_answer = "Final Answer:" in response

            # ── Conflict handling: LLM outputs both tool call and Final Answer ──
            if has_tool_calls and has_final_answer:
                conflict_retries += 1
                logger.warning(
                    f"Section {section.title} round {iteration + 1}: "
                    f"LLM outputs both tool call and Final Answer (conflict #{conflict_retries})"
                )

                if conflict_retries <= 2:
                    # First two times: discard this response, ask LLM to reply again
                    messages.append({"role": "assistant", "content": response})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "【Format Error】You included both tool call and Final Answer in one response, which is not allowed.\n"
                                "Each response can only do ONE of the following:\n"
                                "- Call a tool (output one <tool_call> block, do not write Final Answer)\n"
                                "- Output final content (start with 'Final Answer:', do not include <tool_call>)\n"
                                "Please reply again, only do one thing."
                            ),
                        }
                    )
                    continue
                else:
                    # Third time: degraded processing, truncate to first tool call, force execute
                    logger.warning(
                        f"Section {section.title}: {conflict_retries} consecutive conflicts, "
                        "degraded to truncate and execute first tool call"
                    )
                    first_tool_end = response.find("</tool_call>")
                    if first_tool_end != -1:
                        response = response[: first_tool_end + len("</tool_call>")]
                        tool_calls = self._parse_tool_calls(response)
                        has_tool_calls = bool(tool_calls)
                    has_final_answer = False
                    conflict_retries = 0

            # Log LLM response
            if self.report_logger:
                self.report_logger.log_llm_response(
                    section_title=section.title,
                    section_index=section_index,
                    response=response,
                    iteration=iteration + 1,
                    has_tool_calls=has_tool_calls,
                    has_final_answer=has_final_answer,
                )

            # ── Case 1: LLM outputs Final Answer ──
            if has_final_answer:
                # Tool calls insufficient, reject and ask to continue calling tools
                if tool_calls_count < min_tool_calls:
                    messages.append({"role": "assistant", "content": response})
                    unused_tools = all_tools - used_tools
                    unused_hint = (
                        f" (These tools haven't been used yet, recommend using them: {', '.join(unused_tools)})"
                        if unused_tools
                        else ""
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": REACT_INSUFFICIENT_TOOLS_MSG.format(
                                tool_calls_count=tool_calls_count,
                                min_tool_calls=min_tool_calls,
                                unused_hint=unused_hint,
                            ),
                        }
                    )
                    continue

                # Normal completion
                final_answer = response.split("Final Answer:")[-1].strip()
                logger.info(
                    f"Section {section.title} generation complete (tool calls: {tool_calls_count})"
                )

                if self.report_logger:
                    self.report_logger.log_section_content(
                        section_title=section.title,
                        section_index=section_index,
                        content=final_answer,
                        tool_calls_count=tool_calls_count,
                    )
                return final_answer

            # ── Case 2: LLM tries to call tool ──
            if has_tool_calls:
                # Tool quota exhausted → explicitly tell, ask to output Final Answer
                if tool_calls_count >= self.MAX_TOOL_CALLS_PER_SECTION:
                    messages.append({"role": "assistant", "content": response})
                    messages.append(
                        {
                            "role": "user",
                            "content": REACT_TOOL_LIMIT_MSG.format(
                                tool_calls_count=tool_calls_count,
                                max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                            ),
                        }
                    )
                    continue

                # Only execute first tool call
                call = tool_calls[0]
                if len(tool_calls) > 1:
                    logger.info(
                        f"LLM tries to call {len(tool_calls)} tools, only executing first: {call['name']}"
                    )

                if self.report_logger:
                    self.report_logger.log_tool_call(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        parameters=call.get("parameters", {}),
                        iteration=iteration + 1,
                    )

                result = self._execute_tool(
                    call["name"],
                    call.get("parameters", {}),
                    report_context=report_context,
                )

                if self.report_logger:
                    self.report_logger.log_tool_result(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        result=result,
                        iteration=iteration + 1,
                    )

                tool_calls_count += 1
                used_tools.add(call["name"])

                # Build unused tools hint
                unused_tools = all_tools - used_tools
                unused_hint = ""
                if unused_tools and tool_calls_count < self.MAX_TOOL_CALLS_PER_SECTION:
                    unused_hint = REACT_UNUSED_TOOLS_HINT.format(
                        unused_list="、".join(unused_tools)
                    )

                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {
                        "role": "user",
                        "content": REACT_OBSERVATION_TEMPLATE.format(
                            tool_name=call["name"],
                            result=result,
                            tool_calls_count=tool_calls_count,
                            max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                            used_tools_str=", ".join(used_tools),
                            unused_hint=unused_hint,
                        ),
                    }
                )
                continue

            # ── Case 3: Neither tool call nor Final Answer ──
            messages.append({"role": "assistant", "content": response})

            if tool_calls_count < min_tool_calls:
                # Tool calls insufficient, recommend unused tools
                unused_tools = all_tools - used_tools
                unused_hint = (
                    f" (These tools haven't been used yet, recommend using them: {', '.join(unused_tools)})"
                    if unused_tools
                    else ""
                )

                messages.append(
                    {
                        "role": "user",
                        "content": REACT_INSUFFICIENT_TOOLS_MSG_ALT.format(
                            tool_calls_count=tool_calls_count,
                            min_tool_calls=min_tool_calls,
                            unused_hint=unused_hint,
                        ),
                    }
                )
                continue

            # Tool calls sufficient, LLM outputs content but without "Final Answer:" prefix
            # Directly use this content as final answer, no more cycling
            logger.info(
                f"Section {section.title} no 'Final Answer:' prefix detected, directly adopting LLM output as final content (tool calls: {tool_calls_count})"
            )
            final_answer = response.strip()

            if self.report_logger:
                self.report_logger.log_section_content(
                    section_title=section.title,
                    section_index=section_index,
                    content=final_answer,
                    tool_calls_count=tool_calls_count,
                )
            return final_answer

        # Max iterations reached, force generate content
        logger.warning(
            f"Section {section.title} reached max iterations, forcing generation"
        )
        messages.append({"role": "user", "content": REACT_FORCE_FINAL_MSG})

        response = self.llm.chat(messages=messages, temperature=0.5, max_tokens=4096)

        # Check if force finish LLM returns None
        if response is None:
            logger.error(
                f"Section {section.title} force finish LLM returned None, using default error message"
            )
            final_answer = f"(Section generation failed: LLM returned empty response, please retry later)"
        elif "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
        else:
            final_answer = response

        # Log section content generation complete
        if self.report_logger:
            self.report_logger.log_section_content(
                section_title=section.title,
                section_index=section_index,
                content=final_answer,
                tool_calls_count=tool_calls_count,
            )

        return final_answer

    def generate_report(
        self,
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        report_id: Optional[str] = None,
    ) -> Report:
        """
        Generate complete report (section by section, real-time output)

        Each section is saved immediately after generation, no need to wait for entire report.
        File structure:
        reports/{report_id}/
            meta.json       - Report metadata
            outline.json    - Report outline
            progress.json   - Generation progress
            section_01.md   - Section 1
            section_02.md   - Section 2
            ...
            full_report.md  - Complete report

        Args:
            progress_callback: Progress callback function (stage, progress, message)
            report_id: Report ID (optional, auto-generated if not provided)

        Returns:
            Report: Complete report
        """
        import uuid

        # If no report_id provided, auto-generate
        if not report_id:
            report_id = f"report_{uuid.uuid4().hex[:12]}"
        start_time = datetime.now()

        report = Report(
            report_id=report_id,
            simulation_id=self.simulation_id,
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement,
            status=ReportStatus.PENDING,
            created_at=datetime.now().isoformat(),
        )

        # List of completed section titles (for progress tracking)
        completed_section_titles = []

        try:
            # Initialize: create report folder and save initial state
            ReportManager._ensure_report_folder(report_id)

            # Initialize log recorder (structured log agent_log.jsonl)
            self.report_logger = ReportLogger(report_id)
            self.report_logger.log_start(
                simulation_id=self.simulation_id,
                graph_id=self.graph_id,
                simulation_requirement=self.simulation_requirement,
            )

            # Initialize console log recorder (console_log.txt)
            self.console_logger = ReportConsoleLogger(report_id)

            ReportManager.update_progress(
                report_id, "pending", 0, "Initializing report...", completed_sections=[]
            )
            ReportManager.save_report(report)

            # Stage 1: Plan outline
            report.status = ReportStatus.PLANNING
            ReportManager.update_progress(
                report_id,
                "planning",
                5,
                "Starting to plan report outline...",
                completed_sections=[],
            )

            # Log planning start
            self.report_logger.log_planning_start()

            if progress_callback:
                progress_callback("planning", 0, "Starting to plan report outline...")

            outline = self.plan_outline(
                progress_callback=lambda stage, prog, msg: (
                    progress_callback(stage, prog // 5, msg)
                    if progress_callback
                    else None
                )
            )
            report.outline = outline

            # Log planning complete
            self.report_logger.log_planning_complete(outline.to_dict())

            # Save outline to file
            ReportManager.save_outline(report_id, outline)
            ReportManager.update_progress(
                report_id,
                "planning",
                15,
                f"Outline planning complete, {len(outline.sections)} sections",
                completed_sections=[],
            )
            ReportManager.save_report(report)

            logger.info(f"Outline saved to file: {report_id}/outline.json")

            # Stage 2: Generate section by section (save by section)
            report.status = ReportStatus.GENERATING

            total_sections = len(outline.sections)
            generated_sections = []  # Save content for context

            for i, section in enumerate(outline.sections):
                section_num = i + 1
                base_progress = 20 + int((i / total_sections) * 70)

                # Update progress
                ReportManager.update_progress(
                    report_id,
                    "generating",
                    base_progress,
                    f"Generating section: {section.title} ({section_num}/{total_sections})",
                    current_section=section.title,
                    completed_sections=completed_section_titles,
                )

                if progress_callback:
                    progress_callback(
                        "generating",
                        base_progress,
                        f"Generating section: {section.title} ({section_num}/{total_sections})",
                    )

                # Generate main section content
                section_content = self._generate_section_react(
                    section=section,
                    outline=outline,
                    previous_sections=generated_sections,
                    progress_callback=lambda stage, prog, msg: (
                        progress_callback(
                            stage, base_progress + int(prog * 0.7 / total_sections), msg
                        )
                        if progress_callback
                        else None
                    ),
                    section_index=section_num,
                )

                section.content = section_content
                generated_sections.append(f"## {section.title}\n\n{section_content}")

                # Save section
                ReportManager.save_section(report_id, section_num, section)
                completed_section_titles.append(section.title)

                # Log section complete
                full_section_content = f"## {section.title}\n\n{section_content}"

                if self.report_logger:
                    self.report_logger.log_section_full_complete(
                        section_title=section.title,
                        section_index=section_num,
                        full_content=full_section_content.strip(),
                    )

                logger.info(f"Section saved: {report_id}/section_{section_num:02d}.md")

                # Update progress
                ReportManager.update_progress(
                    report_id,
                    "generating",
                    base_progress + int(70 / total_sections),
                    f"Section {section.title} complete",
                    current_section=None,
                    completed_sections=completed_section_titles,
                )

            # Stage 3: Assemble complete report
            if progress_callback:
                progress_callback("generating", 95, "Assembling complete report...")

            ReportManager.update_progress(
                report_id,
                "generating",
                95,
                "Assembling complete report...",
                completed_sections=completed_section_titles,
            )

            # Use ReportManager to assemble complete report
            report.markdown_content = ReportManager.assemble_full_report(
                report_id, outline
            )
            report.status = ReportStatus.COMPLETED
            report.completed_at = datetime.now().isoformat()

            # Calculate total time
            total_time_seconds = (datetime.now() - start_time).total_seconds()

            # Log report complete
            if self.report_logger:
                self.report_logger.log_report_complete(
                    total_sections=total_sections, total_time_seconds=total_time_seconds
                )

            # Save final report
            ReportManager.save_report(report)
            ReportManager.update_progress(
                report_id,
                "completed",
                100,
                "Report generation complete",
                completed_sections=completed_section_titles,
            )

            if progress_callback:
                progress_callback("completed", 100, "Report generation complete")

            logger.info(f"Report generation complete: {report_id}")

            # Close console log recorder
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None

            return report

        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            report.status = ReportStatus.FAILED
            report.error = str(e)

            # Log error
            if self.report_logger:
                self.report_logger.log_error(str(e), "failed")

            # Save failed state
            try:
                ReportManager.save_report(report)
                ReportManager.update_progress(
                    report_id,
                    "failed",
                    -1,
                    f"Report generation failed: {str(e)}",
                    completed_sections=completed_section_titles,
                )
            except Exception:
                pass  # Ignore save failure errors

            # Close console log recorder
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None

            return report

    def chat(
        self, message: str, chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Chat with Report Agent

        In conversation, Agent can autonomously call retrieval tools to answer questions

        Args:
            message: User message
            chat_history: Chat history

        Returns:
            {
                "response": "Agent reply",
                "tool_calls": [list of called tools],
                "sources": [information sources]
            }
        """
        logger.info(f"Report Agent chat: {message[:50]}...")

        chat_history = chat_history or []

        # Get already generated report content
        report_content = ""
        try:
            report = ReportManager.get_report_by_simulation(self.simulation_id)
            if report and report.markdown_content:
                # Limit report length to avoid context too long
                report_content = report.markdown_content[:15000]
                if len(report.markdown_content) > 15000:
                    report_content += "\n\n... [Report content truncated] ..."
        except Exception as e:
            logger.warning(f"Failed to get report content: {e}")

        system_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            report_content=report_content if report_content else "(No report yet)",
            tools_description=self._get_tools_description(),
        )

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]

        # Add history
        for h in chat_history[-10:]:  # Limit history length
            messages.append(h)

        # Add user message
        messages.append({"role": "user", "content": message})

        # ReACT loop (simplified)
        tool_calls_made = []
        max_iterations = 2  # Reduced iterations

        for iteration in range(max_iterations):
            response = self.llm.chat(messages=messages, temperature=0.5)

            # Parse tool calls
            tool_calls = self._parse_tool_calls(response)

            if not tool_calls:
                # No tool calls, return response directly
                clean_response = re.sub(
                    r"<tool_call>.*?</tool_call>", "", response, flags=re.DOTALL
                )
                clean_response = re.sub(r"\[TOOL_CALL\].*?\)", "", clean_response)

                return {
                    "response": clean_response.strip(),
                    "tool_calls": tool_calls_made,
                    "sources": [
                        tc.get("parameters", {}).get("query", "")
                        for tc in tool_calls_made
                    ],
                }

            # Execute tool calls (limit count)
            tool_results = []
            for call in tool_calls[:1]:  # Max 1 tool call per round
                if len(tool_calls_made) >= self.MAX_TOOL_CALLS_PER_CHAT:
                    break
                result = self._execute_tool(call["name"], call.get("parameters", {}))
                tool_results.append(
                    {
                        "tool": call["name"],
                        "result": result[:1500],  # Limit result length
                    }
                )
                tool_calls_made.append(call)

            # Add result to messages
            messages.append({"role": "assistant", "content": response})
            observation = "\n".join(
                [f"[{r['tool']} result]\n{r['result']}" for r in tool_results]
            )
            messages.append(
                {"role": "user", "content": observation + CHAT_OBSERVATION_SUFFIX}
            )

        # Max iterations reached, get final response
        final_response = self.llm.chat(messages=messages, temperature=0.5)

        # Clean response
        clean_response = re.sub(
            r"<tool_call>.*?</tool_call>", "", final_response, flags=re.DOTALL
        )
        clean_response = re.sub(r"\[TOOL_CALL\].*?\)", "", clean_response)

        return {
            "response": clean_response.strip(),
            "tool_calls": tool_calls_made,
            "sources": [
                tc.get("parameters", {}).get("query", "") for tc in tool_calls_made
            ],
        }


class ReportManager:
    """
    Report manager

    Responsible for report persistent storage and retrieval

    File structure (sectioned output):
    reports/
      {report_id}/
        meta.json          - Report metadata and status
        outline.json       - Report outline
        progress.json      - Generation progress
        section_01.md      - Section 1
        section_02.md      - Section 2
        ...
        full_report.md     - Complete report
    """

    # Report storage directory
    REPORTS_DIR = os.path.join(Config.UPLOAD_FOLDER, "reports")

    @classmethod
    def _ensure_reports_dir(cls):
        """Ensure report root directory exists"""
        os.makedirs(cls.REPORTS_DIR, exist_ok=True)

    @classmethod
    def _get_report_folder(cls, report_id: str) -> str:
        """Get report folder path"""
        return os.path.join(cls.REPORTS_DIR, report_id)

    @classmethod
    def _ensure_report_folder(cls, report_id: str) -> str:
        """Ensure report folder exists and return path"""
        folder = cls._get_report_folder(report_id)
        os.makedirs(folder, exist_ok=True)
        return folder

    @classmethod
    def _get_report_path(cls, report_id: str) -> str:
        """Get report metadata file path"""
        return os.path.join(cls._get_report_folder(report_id), "meta.json")

    @classmethod
    def _get_report_markdown_path(cls, report_id: str) -> str:
        """Get complete report Markdown file path"""
        return os.path.join(cls._get_report_folder(report_id), "full_report.md")

    @classmethod
    def _get_outline_path(cls, report_id: str) -> str:
        """Get outline file path"""
        return os.path.join(cls._get_report_folder(report_id), "outline.json")

    @classmethod
    def _get_progress_path(cls, report_id: str) -> str:
        """Get progress file path"""
        return os.path.join(cls._get_report_folder(report_id), "progress.json")

    @classmethod
    def _get_section_path(cls, report_id: str, section_index: int) -> str:
        """Get section Markdown file path"""
        return os.path.join(
            cls._get_report_folder(report_id), f"section_{section_index:02d}.md"
        )

    @classmethod
    def _get_agent_log_path(cls, report_id: str) -> str:
        """Get Agent log file path"""
        return os.path.join(cls._get_report_folder(report_id), "agent_log.jsonl")

    @classmethod
    def _get_console_log_path(cls, report_id: str) -> str:
        """Get console log file path"""
        return os.path.join(cls._get_report_folder(report_id), "console_log.txt")

    @classmethod
    def get_console_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Get console log content

        This is console output log during report generation (INFO, WARNING, etc.),
        differs from agent_log.jsonl structured log.

        Args:
            report_id: Report ID
            from_line: Start reading from line (for incremental fetch, 0 means from beginning)

        Returns:
            {
                "logs": [list of log lines],
                "total_lines": total line count,
                "from_line": start line number,
                "has_more": whether there are more logs
            }
        """
        log_path = cls._get_console_log_path(report_id)

        if not os.path.exists(log_path):
            return {"logs": [], "total_lines": 0, "from_line": 0, "has_more": False}

        logs = []
        total_lines = 0

        with open(log_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    # Keep original log line, remove trailing newline
                    logs.append(line.rstrip("\n\r"))

        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False,  # Read to end
        }

    @classmethod
    def get_console_log_stream(cls, report_id: str) -> List[str]:
        """
        Get complete console log (get all at once)

        Args:
            report_id: Report ID

        Returns:
            List of log lines
        """
        result = cls.get_console_log(report_id, from_line=0)
        return result["logs"]

    @classmethod
    def get_agent_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Get Agent log content

        Args:
            report_id: Report ID
            from_line: Start reading from line (for incremental fetch, 0 means from beginning)

        Returns:
            {
                "logs": [list of log entries],
                "total_lines": total line count,
                "from_line": start line number,
                "has_more": whether there are more logs
            }
        """
        log_path = cls._get_agent_log_path(report_id)

        if not os.path.exists(log_path):
            return {"logs": [], "total_lines": 0, "from_line": 0, "has_more": False}

        logs = []
        total_lines = 0

        with open(log_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    try:
                        log_entry = json.loads(line.strip())
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        # Skip lines that fail to parse
                        continue

        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False,  # Read to end
        }

    @classmethod
    def get_agent_log_stream(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Get complete Agent log (get all at once)

        Args:
            report_id: Report ID

        Returns:
            List of log entries
        """
        result = cls.get_agent_log(report_id, from_line=0)
        return result["logs"]

    @classmethod
    def save_outline(cls, report_id: str, outline: ReportOutline) -> None:
        """
        Save report outline

        Called immediately after planning stage completes
        """
        cls._ensure_report_folder(report_id)

        with open(cls._get_outline_path(report_id), "w", encoding="utf-8") as f:
            json.dump(outline.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(f"Outline saved: {report_id}")

    @classmethod
    def save_section(
        cls, report_id: str, section_index: int, section: ReportSection
    ) -> str:
        """
        Save single section

        Called immediately after each section generation completes, enabling section-by-section output

        Args:
            report_id: Report ID
            section_index: Section index (starting from 1)
            section: Section object

        Returns:
            Saved file path
        """
        cls._ensure_report_folder(report_id)

        # Build section Markdown content - clean possible duplicate titles
        cleaned_content = cls._clean_section_content(section.content, section.title)
        md_content = f"## {section.title}\n\n"
        if cleaned_content:
            md_content += f"{cleaned_content}\n\n"

        # Save file
        file_suffix = f"section_{section_index:02d}.md"
        file_path = os.path.join(cls._get_report_folder(report_id), file_suffix)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"Section saved: {report_id}/{file_suffix}")
        return file_path

    @classmethod
    def _clean_section_content(cls, content: str, section_title: str) -> str:
        """
        Clean section content

        1. Remove duplicate Markdown title lines at content beginning
        2. Convert all ### and below level titles to bold text

        Args:
            content: Raw content
            section_title: Section title

        Returns:
            Cleaned content
        """
        import re

        if not content:
            return content

        content = content.strip()
        lines = content.split("\n")
        cleaned_lines = []
        skip_next_empty = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Check if it's a Markdown title line
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)

            if heading_match:
                level = len(heading_match.group(1))
                title_text = heading_match.group(2).strip()

                # Check if it's a duplicate title with section title (skip within first 5 lines)
                if i < 5:
                    if title_text == section_title or title_text.replace(
                        " ", ""
                    ) == section_title.replace(" ", ""):
                        skip_next_empty = True
                        continue

                # Convert all level titles (#, ##, ###, #### etc.) to bold
                # Because section title is added by system, content should have no titles
                cleaned_lines.append(f"**{title_text}**")
                cleaned_lines.append("")  # Add empty line
                continue

            # If previous line was skipped title and current line is empty, also skip
            if skip_next_empty and stripped == "":
                skip_next_empty = False
                continue

            skip_next_empty = False
            cleaned_lines.append(line)

        # Remove leading empty lines
        while cleaned_lines and cleaned_lines[0].strip() == "":
            cleaned_lines.pop(0)

        # Remove leading separators
        while cleaned_lines and cleaned_lines[0].strip() in ["---", "***", "___"]:
            cleaned_lines.pop(0)
            # Also remove empty line after separator
            while cleaned_lines and cleaned_lines[0].strip() == "":
                cleaned_lines.pop(0)

        return "\n".join(cleaned_lines)

    @classmethod
    def update_progress(
        cls,
        report_id: str,
        status: str,
        progress: int,
        message: str,
        current_section: str = None,
        completed_sections: List[str] = None,
    ) -> None:
        """
        Update report generation progress

        Frontend can get real-time progress by reading progress.json
        """
        cls._ensure_report_folder(report_id)

        progress_data = {
            "status": status,
            "progress": progress,
            "message": message,
            "current_section": current_section,
            "completed_sections": completed_sections or [],
            "updated_at": datetime.now().isoformat(),
        }

        with open(cls._get_progress_path(report_id), "w", encoding="utf-8") as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)

    @classmethod
    def get_progress(cls, report_id: str) -> Optional[Dict[str, Any]]:
        """Get report generation progress"""
        path = cls._get_progress_path(report_id)

        if not os.path.exists(path):
            return None

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def get_generated_sections(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Get list of generated sections

        Returns info for all saved section files
        """
        folder = cls._get_report_folder(report_id)

        if not os.path.exists(folder):
            return []

        sections = []
        for filename in sorted(os.listdir(folder)):
            if filename.startswith("section_") and filename.endswith(".md"):
                file_path = os.path.join(folder, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Parse section index from filename
                parts = filename.replace(".md", "").split("_")
                section_index = int(parts[1])

                sections.append(
                    {
                        "filename": filename,
                        "section_index": section_index,
                        "content": content,
                    }
                )

        return sections

    @classmethod
    def assemble_full_report(cls, report_id: str, outline: ReportOutline) -> str:
        """
        Assemble complete report

        Assemble complete report from saved section files, with title cleaning
        """
        folder = cls._get_report_folder(report_id)

        # Build report header
        md_content = f"# {outline.title}\n\n"
        md_content += f"> {outline.summary}\n\n"
        md_content += f"---\n\n"

        # Read all section files in order
        sections = cls.get_generated_sections(report_id)
        for section_info in sections:
            md_content += section_info["content"]

        # Post-process: clean title issues in entire report
        md_content = cls._post_process_report(md_content, outline)

        # Save complete report
        full_path = cls._get_report_markdown_path(report_id)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"Complete report assembled: {report_id}")
        return md_content

    @classmethod
    def _post_process_report(cls, content: str, outline: ReportOutline) -> str:
        """
        Post-process report content

        1. Remove duplicate titles
        2. Keep report main title (#) and section titles (##), remove other levels (###, #### etc.)
        3. Clean extra empty lines and separators

        Args:
            content: Raw report content
            outline: Report outline

        Returns:
            Processed content
        """
        import re

        lines = content.split("\n")
        processed_lines = []
        prev_was_heading = False

        # Collect all section titles from outline
        section_titles = set()
        for section in outline.sections:
            section_titles.add(section.title)

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Check if it's a title line
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)

            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()

                # Check if it's a duplicate title (same content appears within consecutive 5 lines)
                is_duplicate = False
                for j in range(max(0, len(processed_lines) - 5), len(processed_lines)):
                    prev_line = processed_lines[j].strip()
                    prev_match = re.match(r"^(#{1,6})\s+(.+)$", prev_line)
                    if prev_match:
                        prev_title = prev_match.group(2).strip()
                        if prev_title == title:
                            is_duplicate = True
                            break

                if is_duplicate:
                    # Skip duplicate title and following empty lines
                    i += 1
                    while i < len(lines) and lines[i].strip() == "":
                        i += 1
                    continue

                # Title level handling:
                # - # (level=1) only keep report main title
                # - ## (level=2) keep section titles
                # - ### and below (level>=3) convert to bold text

                if level == 1:
                    if title == outline.title:
                        # Keep report main title
                        processed_lines.append(line)
                        prev_was_heading = True
                    elif title in section_titles:
                        # Section title incorrectly used #, fix to ##
                        processed_lines.append(f"## {title}")
                        prev_was_heading = True
                    else:
                        # Other level 1 titles convert to bold
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                elif level == 2:
                    if title in section_titles or title == outline.title:
                        # Keep section title
                        processed_lines.append(line)
                        prev_was_heading = True
                    else:
                        # Non-section level 2 titles convert to bold
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                else:
                    # Level 3 and below titles convert to bold text
                    processed_lines.append(f"**{title}**")
                    processed_lines.append("")
                    prev_was_heading = False

                i += 1
                continue

            elif stripped == "---" and prev_was_heading:
                # Skip separator immediately after title
                i += 1
                continue

            elif stripped == "" and prev_was_heading:
                # Only keep one empty line after title
                if processed_lines and processed_lines[-1].strip() != "":
                    processed_lines.append(line)
                prev_was_heading = False

            else:
                processed_lines.append(line)
                prev_was_heading = False

            i += 1

        # Clean consecutive multiple empty lines (keep max 2)
        result_lines = []
        empty_count = 0
        for line in processed_lines:
            if line.strip() == "":
                empty_count += 1
                if empty_count <= 2:
                    result_lines.append(line)
            else:
                empty_count = 0
                result_lines.append(line)

        return "\n".join(result_lines)

    @classmethod
    def save_report(cls, report: Report) -> None:
        """Save report metadata and complete report"""
        cls._ensure_report_folder(report.report_id)

        # Save metadata JSON
        with open(cls._get_report_path(report.report_id), "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

        # Save outline
        if report.outline:
            cls.save_outline(report.report_id, report.outline)

        # Save complete Markdown report
        if report.markdown_content:
            with open(
                cls._get_report_markdown_path(report.report_id), "w", encoding="utf-8"
            ) as f:
                f.write(report.markdown_content)

        logger.info(f"Report saved: {report.report_id}")

    @classmethod
    def get_report(cls, report_id: str) -> Optional[Report]:
        """Get report"""
        path = cls._get_report_path(report_id)

        if not os.path.exists(path):
            # Backward compatible: check file directly in reports directory
            old_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
            if os.path.exists(old_path):
                path = old_path
            else:
                return None

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Rebuild Report object
        outline = None
        if data.get("outline"):
            outline_data = data["outline"]
            sections = []
            for s in outline_data.get("sections", []):
                sections.append(
                    ReportSection(title=s["title"], content=s.get("content", ""))
                )
            outline = ReportOutline(
                title=outline_data["title"],
                summary=outline_data["summary"],
                sections=sections,
            )

        # If markdown_content is empty, try reading from full_report.md
        markdown_content = data.get("markdown_content", "")
        if not markdown_content:
            full_report_path = cls._get_report_markdown_path(report_id)
            if os.path.exists(full_report_path):
                with open(full_report_path, "r", encoding="utf-8") as f:
                    markdown_content = f.read()

        return Report(
            report_id=data["report_id"],
            simulation_id=data["simulation_id"],
            graph_id=data["graph_id"],
            simulation_requirement=data["simulation_requirement"],
            status=ReportStatus(data["status"]),
            outline=outline,
            markdown_content=markdown_content,
            created_at=data.get("created_at", ""),
            completed_at=data.get("completed_at", ""),
            error=data.get("error"),
        )

    @classmethod
    def get_report_by_simulation(cls, simulation_id: str) -> Optional[Report]:
        """Get report by simulation ID"""
        cls._ensure_reports_dir()

        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # New format: folder
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report and report.simulation_id == simulation_id:
                    return report
            # Backward compatible: JSON file
            elif item.endswith(".json"):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report and report.simulation_id == simulation_id:
                    return report

        return None

    @classmethod
    def list_reports(
        cls, simulation_id: Optional[str] = None, limit: int = 50
    ) -> List[Report]:
        """List reports"""
        cls._ensure_reports_dir()

        reports = []
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # New format: folder
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
            # Backward compatible: JSON file
            elif item.endswith(".json"):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)

        # Sort by creation time descending
        reports.sort(key=lambda r: r.created_at, reverse=True)

        return reports[:limit]

    @classmethod
    def delete_report(cls, report_id: str) -> bool:
        """Delete report (entire folder)"""
        import shutil

        folder_path = cls._get_report_folder(report_id)

        # New format: delete entire folder
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            logger.info(f"Report folder deleted: {report_id}")
            return True

        # Backward compatible: delete individual files
        deleted = False
        old_json_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
        old_md_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.md")

        if os.path.exists(old_json_path):
            os.remove(old_json_path)
            deleted = True
        if os.path.exists(old_md_path):
            os.remove(old_md_path)
            deleted = True

        return deleted
