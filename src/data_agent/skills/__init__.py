"""Reusable analysis skills backing the .claude agent definitions.

Each module here implements the contract of the matching `.claude/skills/*/SKILL.md`
and is invoked both by the Python orchestrator and (directly) by the Claude Code
subagents in `.claude/agents/`. Skills wrap the proven core in
``src/data_agent/{schema,task,features,models,reporting,runner}.py`` rather than
reimplementing it.
"""
