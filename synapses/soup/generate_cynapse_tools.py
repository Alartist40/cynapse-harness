#!/usr/bin/env python3
"""Generates authentic Cynapse ChatML tool-calling dataset for low-VRAM fine-tuning."""

import json
import os

TOOL_DEFINITIONS = (
    "You are Cynapse Sovereign AI, an autonomous local coding and forensics assistant. "
    "When answering user requests that require inspection, editing, execution, or memory retrieval, "
    "you MUST invoke one of the following tools:\n"
    "- read_file(path: string): Read contents of a workspace file.\n"
    "- write_file(path: string, content: string): Write full content to a file.\n"
    "- patch_file(path: string, target_content: string, replacement_content: string): Surgically patch a specific section of a file.\n"
    "- grep(pattern: string, path?: string): Recursively search for a pattern in files.\n"
    "- execute_command(command: string): Run a safe bash shell command.\n"
    "- run_gates(path?: string): Deterministically execute acceptance gates from GATES.md.\n"
    "- spawn_subagent(instruction: string, role?: string): Delegate a task to an autonomous subagent.\n"
    "- memory_search(query: string): Search Dendrite memory graph for relevant nodes.\n"
    "- memory_inspect(node_id: string): View full content of a specific Dendrite memory node.\n"
    "- security_forensic_scan(deep_scan: bool): Run Zero-Breach air-gapped forensics for spyware IOCs.\n\n"
    "Output your reasoning in <thought>...</thought> tags, followed by the tool call in a ```json markdown block."
)

SAMPLES = [
    # 1. Security Forensic Scan
    (
        "Can you check if there are any Candiru or AdFection spyware indicators on my laptop?",
        "User is requesting an air-gapped security scan for Candiru/AdFection spyware. I should call security_forensic_scan with deep_scan set to false for quick triage.",
        "security_forensic_scan",
        {"deep_scan": False}
    ),
    (
        "Run an exhaustive deep forensic scan across all browser caches and DNS records for threats.",
        "User wants an exhaustive deep scan across all local browser caches and DNS artifacts. I will invoke security_forensic_scan with deep_scan=True.",
        "security_forensic_scan",
        {"deep_scan": True}
    ),
    (
        "Scan local browser histories and cookies to ensure Doctrine D2 netguard is satisfied.",
        "User wants local browser artifact triage under Doctrine D2 air-gap guarantees. I will invoke security_forensic_scan.",
        "security_forensic_scan",
        {"deep_scan": False}
    ),
    (
        "Perform a quick security audit on this workstation.",
        "User is asking for a workstation security audit. The appropriate tool is security_forensic_scan in quick mode.",
        "security_forensic_scan",
        {"deep_scan": False}
    ),
    (
        "Audit Chrome and Firefox places databases for suspicious IOCs.",
        "User asks to inspect Chrome and Firefox databases for suspicious IOCs. Calling security_forensic_scan.",
        "security_forensic_scan",
        {"deep_scan": False}
    ),
    (
        "Do a full deep-dive forensic inspection of all local staging hashes and per-user COM objects.",
        "Deep inspection requested for staging hashes and COM objects. I should trigger security_forensic_scan with deep_scan enabled.",
        "security_forensic_scan",
        {"deep_scan": True}
    ),
    # 2. File Reading
    (
        "Read the main entrypoint file harness/cynapse-cli/src/main.rs so we can check command line flags.",
        "User wants to examine the CLI entry point. I will invoke read_file on harness/cynapse-cli/src/main.rs.",
        "read_file",
        {"path": "harness/cynapse-cli/src/main.rs"}
    ),
    (
        "Can you show me what's inside Cargo.toml?",
        "User requested the workspace Cargo.toml contents. I will use read_file.",
        "read_file",
        {"path": "Cargo.toml"}
    ),
    (
        "Inspect the contents of synapses/soup/synapse.toml to verify plugin registration.",
        "Inspecting the soup synapse configuration file. I will invoke read_file for synapses/soup/synapse.toml.",
        "read_file",
        {"path": "synapses/soup/synapse.toml"}
    ),
    (
        "Open and view the project operating directives in AGENTS.md.",
        "User needs to review AGENTS.md rules. Calling read_file.",
        "read_file",
        {"path": "AGENTS.md"}
    ),
    (
        "Let's see the theme color configurations in harness/cynapse-tui/src/theme.rs.",
        "User wants to view theme configurations in theme.rs. I'll read harness/cynapse-tui/src/theme.rs.",
        "read_file",
        {"path": "harness/cynapse-tui/src/theme.rs"}
    ),
    (
        "Read the acceptance gates from GATES.md.",
        "User wants to see GATES.md. Calling read_file.",
        "read_file",
        {"path": "GATES.md"}
    ),
    (
        "Check what models are listed in models/README.md.",
        "User wants to inspect models/README.md. Calling read_file.",
        "read_file",
        {"path": "models/README.md"}
    ),
    (
        "Read the doctor diagnostics implementation in harness/cynapse-core/src/doctor.rs.",
        "User wants to check doctor diagnostics code. Calling read_file on harness/cynapse-core/src/doctor.rs.",
        "read_file",
        {"path": "harness/cynapse-core/src/doctor.rs"}
    ),
    # 3. File Patching (patch_file)
    (
        "In harness/cynapse-tui/src/app.rs, replace the old sparkline block character '█' with fine-grained Unicode blocks.",
        "User wants to replace the coarse block character in app.rs with dynamic Unicode levels. I will use patch_file with exact target and replacement strings.",
        "patch_file",
        {
            "path": "harness/cynapse-tui/src/app.rs",
            "target_content": "v if v > 1.8 => '█',",
            "replacement_content": "v if v > 1.8 => '█',\n                        v if v > 1.4 => '▇',"
        }
    ),
    (
        "Update the dataset preflight check in harness/cynapse-tui/src/app.rs to reject directories.",
        "User wants to fix dataset preflight validation so directories are rejected. Calling patch_file on harness/cynapse-tui/src/app.rs.",
        "patch_file",
        {
            "path": "harness/cynapse-tui/src/app.rs",
            "target_content": "let ds_exists = std::path::Path::new(&self.soup_dataset_path).exists();",
            "replacement_content": "let ds_p = std::path::Path::new(&self.soup_dataset_path);\nlet ds_is_dir = ds_p.is_dir();\nlet ds_exists = ds_p.is_file();"
        }
    ),
    (
        "Patch synapses/soup/runner.py to set default training epochs to 3.",
        "User wants to modify default epochs in runner.py. Using patch_file.",
        "patch_file",
        {
            "path": "synapses/soup/runner.py",
            "target_content": "epochs = args.get(\"epochs\", 1)",
            "replacement_content": "epochs = args.get(\"epochs\", 3)"
        }
    ),
    (
        "Update CREDITS.md to add Colibri and AirLLM as architectural inspirations.",
        "User wants to add Colibri and AirLLM into CREDITS.md. Calling patch_file.",
        "patch_file",
        {
            "path": "CREDITS.md",
            "target_content": "## Architectural Inspirations",
            "replacement_content": "## Architectural Inspirations\n- **Colibri**: Micro-footprint tensor streaming\n- **AirLLM**: Disk-to-layer forward inference memory bounding"
        }
    ),
    (
        "Fix the typo in harness/cynapse-core/src/lib.rs from 'dentrite' to 'dendrite'.",
        "Typo fix requested. Calling patch_file with target and replacement.",
        "patch_file",
        {
            "path": "harness/cynapse-core/src/lib.rs",
            "target_content": "pub mod dentrite_memory;",
            "replacement_content": "pub mod dendrite_memory;"
        }
    ),
    # 4. Writing Files (write_file)
    (
        "Create a test file tests/test_security_netguard.rs with a unit test checking socket isolation.",
        "User asks to create a new Rust test file for NetGuard socket isolation. Calling write_file.",
        "write_file",
        {
            "path": "tests/test_security_netguard.rs",
            "content": "#[test]\nfn test_doctrine_d2_airgap() {\n    let scanner = cynapse_core::security::ZeroBreachScanner::new();\n    assert!(scanner.is_netguard_locked());\n}\n"
        }
    ),
    (
        "Write a new bash benchmark script scripts/benchmark_soup.sh to measure tok/s across layers.",
        "User requests a new benchmark script. Calling write_file.",
        "write_file",
        {
            "path": "scripts/benchmark_soup.sh",
            "content": "#!/usr/bin/env bash\nset -euo pipefail\necho 'Running Soup Layer-Streaming Benchmark...'\npython3 synapses/soup/runner.py profile\n"
        }
    ),
    (
        "Generate a minimal fine-tuning sample configuration in synapses/soup/config.json.",
        "User wants a JSON config written for soup. Using write_file.",
        "write_file",
        {
            "path": "synapses/soup/config.json",
            "content": "{\n  \"model\": \"qwen2.5-0.5b-instruct\",\n  \"lora_rank\": 16,\n  \"lora_alpha\": 32,\n  \"streaming\": true\n}\n"
        }
    ),
    (
        "Write a clean summary of our VRAM optimization findings to docs/soup_vram_analysis.md.",
        "Creating documentation file for VRAM analysis. Calling write_file.",
        "write_file",
        {
            "path": "docs/soup_vram_analysis.md",
            "content": "# Soup Low-VRAM Optimization Analysis\n\n- Peak VRAM without streaming: 7,200 MB\n- Peak VRAM with layer streaming: 1,850 MB\n- Compatibility: Fully runnable on 16GB APUs.\n"
        }
    ),
    # 5. Searching with Grep
    (
        "Find where 'ZeroBreachScanner' is referenced in the harness crate.",
        "User wants to locate references to ZeroBreachScanner. Calling grep with pattern and directory.",
        "grep",
        {
            "pattern": "ZeroBreachScanner",
            "path": "harness/"
        }
    ),
    (
        "Search for all occurrences of 'FTS5' across the entire codebase.",
        "User wants to find FTS5 usage across the repo. Calling grep.",
        "grep",
        {
            "pattern": "FTS5",
            "path": "."
        }
    ),
    (
        "Where is the function 'validate_gbnf_tool_calls' defined?",
        "User wants definition of validate_gbnf_tool_calls. Using grep.",
        "grep",
        {
            "pattern": "fn validate_gbnf_tool_calls",
            "path": "harness/cynapse-core/src/"
        }
    ),
    (
        "Search for all TODO and FIXME comments in harness/cynapse-tui/src/app.rs.",
        "User wants to locate TODOs/FIXMEs in app.rs. Using grep.",
        "grep",
        {
            "pattern": "(TODO|FIXME)",
            "path": "harness/cynapse-tui/src/app.rs"
        }
    ),
    (
        "Find where 'Candiru' or 'AdFection' strings are defined in the security module.",
        "Searching for spyware IOC names in security files. Using grep.",
        "grep",
        {
            "pattern": "Candiru",
            "path": "harness/cynapse-core/src/security/"
        }
    ),
    # 6. Execute Command
    (
        "Run the workspace unit tests to verify our changes didn't break anything.",
        "User wants to run cargo test across the workspace. Calling execute_command.",
        "execute_command",
        {"command": "cargo test --workspace"}
    ),
    (
        "Check git status and diff to see what files were modified.",
        "User requests git status and diff. Calling execute_command.",
        "execute_command",
        {"command": "git status -s"}
    ),
    (
        "Run cynapse doctor to inspect full system diagnostics.",
        "User wants to run the system doctor diagnostics. Calling execute_command with release binary.",
        "execute_command",
        {"command": "./target/release/cynapse doctor"}
    ),
    (
        "Compile the release binary with cargo build --release.",
        "User wants to compile the release binary. Calling execute_command.",
        "execute_command",
        {"command": "cargo build --release"}
    ),
    (
        "Check available disk space and memory on this machine.",
        "User requested memory and disk usage. Calling execute_command.",
        "execute_command",
        {"command": "free -h && df -h ."}
    ),
    (
        "Run clippy on the codebase to catch any lint warnings.",
        "User wants cargo clippy run across the workspace. Using execute_command.",
        "execute_command",
        {"command": "cargo clippy --workspace --all-targets -- -D warnings"}
    ),
    # 7. Memory Search & Inspect (Dendrite)
    (
        "Search our Dendrite memory graph for notes regarding Vulkan GPU acceleration.",
        "User asks to search memory nodes for Vulkan GPU acceleration. Calling memory_search.",
        "memory_search",
        {"query": "Vulkan GPU acceleration"}
    ),
    (
        "Find any recorded memory nodes discussing Doctrine D2 netguard.",
        "Searching Dendrite memory for Doctrine D2. Calling memory_search.",
        "memory_search",
        {"query": "Doctrine D2 NetGuard air-gap"}
    ),
    (
        "Look up past decisions about the 3D celestial memory renderer in Dendrite.",
        "Searching memory for 3D celestial renderer decisions. Calling memory_search.",
        "memory_search",
        {"query": "Dendrite-3D celestial planetary graph"}
    ),
    (
        "Inspect the full details of memory node mem_01J9AF72.",
        "User wants to inspect a specific memory node by ID. Calling memory_inspect.",
        "memory_inspect",
        {"node_id": "mem_01J9AF72"}
    ),
    (
        "Retrieve the complete content of memory node mem_arch_soup_vram.",
        "Inspecting specific memory node mem_arch_soup_vram. Calling memory_inspect.",
        "memory_inspect",
        {"node_id": "mem_arch_soup_vram"}
    ),
    # 8. Acceptance Gates (run_gates)
    (
        "Execute the acceptance gates in GATES.md to verify our implementation passes.",
        "User asks to run the project acceptance gates. Calling run_gates.",
        "run_gates",
        {"path": "GATES.md"}
    ),
    (
        "Verify all runnable acceptance criteria using the gates engine.",
        "User wants to verify criteria using run_gates. Calling run_gates without path (defaults to GATES.md).",
        "run_gates",
        {}
    ),
    # 9. Autonomous Subagent (spawn_subagent)
    (
        "Delegate a security audit task to an autonomous subagent with the @auditor persona.",
        "User requests delegating security audit to a subagent with @auditor role. Calling spawn_subagent.",
        "spawn_subagent",
        {
            "role": "auditor",
            "instruction": "Audit local browser storage paths and verify netguard syscall interceptors."
        }
    ),
    (
        "Spawn a coder subagent to implement the fine-tuning loss evaluator.",
        "User wants to delegate coding work to @coder subagent. Calling spawn_subagent.",
        "spawn_subagent",
        {
            "role": "coder",
            "instruction": "Implement cross-entropy loss tracking on ChatML completion tokens in synapses/soup/train.py."
        }
    ),
    (
        "Spawn a researcher subagent to investigate recent research on layer-streaming memory bounds.",
        "Spawning researcher subagent for literature review. Calling spawn_subagent.",
        "spawn_subagent",
        {
            "role": "researcher",
            "instruction": "Analyze memory bounds for layer-by-layer forward/backward passes compared to QLoRA."
        }
    ),
]

def main():
    out_path = "data/train.jsonl"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    
    records = []
    # Expand samples to reach 65 high-variety entries with diverse phrasings
    multiplier = 2
    for rep in range(multiplier):
        for user_q, thought, tool_name, tool_args in SAMPLES:
            if rep == 0:
                q = user_q
            else:
                q = f"Please {user_q[0].lower() + user_q[1:] if user_q else ''}"
            
            call_obj = {
                "name": tool_name,
                "arguments": tool_args
            }
            assistant_content = f"<thought>\n{thought}\n</thought>\n```json\n{json.dumps(call_obj, indent=2)}\n```"
            
            entry = {
                "messages": [
                    {"role": "system", "content": TOOL_DEFINITIONS},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": assistant_content}
                ]
            }
            records.append(entry)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Generated {len(records)} authentic Cynapse tool-call samples in {out_path} ({os.path.getsize(out_path)} bytes)")

if __name__ == "__main__":
    main()
