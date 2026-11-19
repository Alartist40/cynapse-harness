# SkillSpector — External Skill & MCP Intake Gate

Scan any third-party skill, prompt pack, or MCP server **before** loading or executing it alongside Myelin.

## Run Static Intake Scan
```bash
./bin/skill-scan <path_to_external_skill_or_repo>
```
*(If NVIDIA's `skillspector` CLI is also installed, run `skillspector scan <path>` for deep OSV/YARA/LLM semantic checks.)*

## Risk Categories & Actions
- **`[HIGH:PROMPT_INJECT]`** (override instructions, hidden actions, safety bypass): **Block immediately** — do not load the skill.
- **`[HIGH:OBFUSCATION_EXEC]`** (`base64 -d`, `| sh`, `eval(`, `exec(`, `rm -rf /`): **Block** unless verified safe at exact `file:line`.
- **`[WARN:SECRET_ACCESS]`** (`.ssh/`, `aws_secret`, `github_token`, `os.environ`, `process.env`): Inspect `file:line` to confirm it is not harvesting credentials.
- **`[WARN:NET_EXFIL]`** (`curl `, `wget `, `nc -`, `requests.post`): Inspect destination URL and payload; reject unknown outbound sinks.

Exit `0` = `CLEAN`. Exit `2` = findings surfaced; inspect every hit before trusting the skill.
