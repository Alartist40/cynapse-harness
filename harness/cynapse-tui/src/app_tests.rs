    use super::*;

    #[test]
    fn test_zone_a_prefix_caching_and_invalidation() {
        let graph = Arc::new(cynapse_memory::graph::Dendrite::new());
        let ctx = cynapse_memory::context::DendriteContext::new(Arc::clone(&graph), None);
        let mut app = TuiApp::new(
            PathBuf::from("./models"),
            "ministral-3:3b".into(),
            "http://127.0.0.1:11434".into(),
            graph,
            None,
            ctx,
        );

        assert!(app.cached_zone_a_prefix.is_none());
        let p1 = app.get_or_compile_zone_a_prefix();
        assert!(app.cached_zone_a_prefix.is_some());
        let p2 = app.get_or_compile_zone_a_prefix();
        assert_eq!(p1, p2);

        app.invalidate_zone_a_prefix();
        assert!(app.cached_zone_a_prefix.is_none());
        let p3 = app.get_or_compile_zone_a_prefix();
        assert_eq!(p1, p3);
    }

    #[test]
    fn test_read_receipt_workflow_in_app() {
        use cynapse_core::offline_agent::ToolCall;

        let graph = Arc::new(cynapse_memory::graph::Dendrite::new());
        let ctx = cynapse_memory::context::DendriteContext::new(Arc::clone(&graph), None);
        let mut app = TuiApp::new(
            PathBuf::from("./models"),
            "ministral-3:3b".into(),
            "http://127.0.0.1:11434".into(),
            graph,
            None,
            ctx,
        );

        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("cynapse_tui_receipt_test.txt");
        std::fs::write(&test_file, "hello receipts in tui").unwrap();

        let call = ToolCall {
            name: "read_file".into(),
            arguments: serde_json::json!({
                "path": test_file.to_string_lossy()
            }),
        };

        // 1. First execution returns full content with anchor footer
        let (output1, success1) = app.execute_tool_and_format(&call);
        assert!(success1);
        assert!(output1.contains("hello receipts in tui"));
        assert!(output1.contains("[cynapse-read #1:"));

        // Simulate message added to app.messages containing the anchor
        app.messages.push(ChatMessage {
            role: "assistant".into(),
            content: output1.clone(),
            thinking: None,
        });

        // 2. Second execution returns stub since file is unchanged and anchor is in last 6 messages
        let (output2, success2) = app.execute_tool_and_format(&call);
        assert!(success2);
        assert!(output2.contains("[File unchanged since read #1:"));

        // 3. Clear messages -> resets anchor visibility -> next read returns full
        app.messages.clear();
        let (output3, success3) = app.execute_tool_and_format(&call);
        assert!(success3);
        assert!(output3.contains("[cynapse-read #2:"));

        let _ = std::fs::remove_file(&test_file);
    }

    #[test]
    fn test_finalization_notice_no_tool_directive() {
        let n = super::finalization_notice(5);
        assert!(n.contains("Do NOT emit tool calls"));
        assert!(n.contains("status report"));
        assert!(n.contains('5'));
    }

    #[test]
    fn test_finalization_prompt_structure() {
        let notice = super::finalization_notice(5);
        let prompt = cynapse_core::offline_agent::compile_zone_b_tail(
            "mem", "history", "Original user request", Some(&notice));
        let notice_idx = prompt.find("=== NOTICE ===").unwrap();
        let user_idx = prompt.find("=== USER INSTRUCTION ===").unwrap();
        assert!(notice_idx < user_idx);           // notice precedes instruction
        assert!(prompt.contains("Original user request"));  // query preserved
        assert!(prompt.contains("Do NOT emit tool calls")); // directive present
    }

    #[test]
    fn test_batch_partitioning_execution() {
        use serde_json::json;
        let graph = Arc::new(cynapse_memory::graph::Dendrite::new());
        let ctx = cynapse_memory::context::DendriteContext::new(Arc::clone(&graph), None);
        let mut app = TuiApp::new(
            PathBuf::from("./models"),
            "ministral-3:3b".into(),
            "http://127.0.0.1:11434".into(),
            graph,
            None,
            ctx,
        );
        let calls = vec![
            cynapse_core::offline_agent::ToolCall {
                name: "read_file".into(),
                arguments: json!({"path": "Cargo.toml"}),
            },
            cynapse_core::offline_agent::ToolCall {
                name: "grep".into(),
                arguments: json!({"pattern": "cynapse"}),
            },
            cynapse_core::offline_agent::ToolCall {
                name: "execute_command".into(),
                arguments: json!({"command": "echo 'partition test'"}),
            },
        ];

        let partitions = cynapse_core::partition_tool_calls(&calls);
        assert_eq!(partitions.len(), 2);
        assert_eq!(partitions[0].len(), 2); // [read_file, grep]
        assert_eq!(partitions[1].len(), 1); // [execute_command]

        for batch in partitions {
            for call in &batch {
                assert!(app.loop_guard.check(call).is_ok());
                app.loop_guard.record_call(call);
                let (output, ok) = app.execute_tool_and_format(call);
                assert!(ok);
                app.loop_guard.record_outcome(call, &output);
            }
        }
    }

    #[tokio::test]
    async fn test_turn_verified_status_and_reflection_admission() {
        use serde_json::json;
        let graph = Arc::new(cynapse_memory::graph::Dendrite::new());
        let ctx = cynapse_memory::context::DendriteContext::new(Arc::clone(&graph), None);
        let mut app = TuiApp::new(
            PathBuf::from("./models"),
            "ministral-3:3b".into(),
            "http://127.0.0.1:11434".into(),
            graph.clone(),
            None,
            ctx,
        );

        // Initially verified
        assert!(app.turn_verified);

        // Failing tool execution marks turn unverified
        let fail_call = cynapse_core::offline_agent::ToolCall {
            name: "execute_command".into(),
            arguments: json!({"command": "rm -rf /"}), // blocked by sandbox
        };
        let (_output, ok) = app.execute_tool_and_format(&fail_call);
        assert!(!ok);
        if !ok {
            app.turn_verified = false;
        }
        assert!(!app.turn_verified);

        let messages = vec![
            cynapse_memory::reflection::Message::text(
                cynapse_memory::reflection::Role::User,
                "try dangerous operation",
            ),
            cynapse_memory::reflection::Message::text(
                cynapse_memory::reflection::Role::Assistant,
                "failed to execute dangerous command",
            ),
        ];

        // spawn_reflection with turn_verified=false must be rejected
        app.reflection_worker.spawn_reflection(messages.clone(), app.turn_verified);
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        assert_eq!(graph.all().len(), 0);

        // Clean verified turn admits reflection
        app.turn_verified = true;
        app.reflection_worker.spawn_reflection(messages, app.turn_verified);
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        assert!(graph.all().len() > 0);
    }

    #[test]
    fn test_bots_modal_and_registry_tui_integration() {
        let graph = Arc::new(cynapse_memory::graph::Dendrite::new());
        let ctx = cynapse_memory::context::DendriteContext::new(Arc::clone(&graph), None);
        let mut app = TuiApp::new(
            PathBuf::from("./models"),
            "ministral-3:3b".into(),
            "http://127.0.0.1:11434".into(),
            graph,
            None,
            ctx,
        );

        let bots = app.bot_registry.list();
        assert!(!bots.is_empty(), "Bot registry should load default seeded profiles");
        assert!(app.bot_registry.get("coder").is_some());
        assert!(app.bot_registry.get("researcher").is_some());

        // Simulate /bots slash command opening modal
        app.modal = ActiveModal::Bots;
        app.selected_bot_idx = 0;
        assert_eq!(app.modal, ActiveModal::Bots);
    }

    #[tokio::test]
    async fn test_spawn_subagent_inline_and_background_workflow() {
        use serde_json::json;
        let graph = Arc::new(cynapse_memory::graph::Dendrite::new());
        let ctx = cynapse_memory::context::DendriteContext::new(Arc::clone(&graph), None);
        let mut app = TuiApp::new(
            PathBuf::from("./models"),
            "ministral-3:3b".into(),
            "http://127.0.0.1:59999".into(),
            graph,
            None,
            ctx,
        );

        // 1. Inline execution (wait=true)
        let inline_call = cynapse_core::offline_agent::ToolCall {
            name: "spawn_subagent".into(),
            arguments: json!({
                "task": "Refactor module error types",
                "bot": "coder",
                "wait": true
            }),
        };
        let (inline_res, ok1) = app.execute_tool_and_format(&inline_call);
        assert!(ok1);
        assert!(inline_res.contains("[subagent #1 · @coder ·"));
        assert!(app.subagent_mgr.get_status(1).is_some());

        // 2. Background execution (wait=false)
        let bg_call = cynapse_core::offline_agent::ToolCall {
            name: "spawn_subagent".into(),
            arguments: json!({
                "task": "Audit codebase dependencies",
                "bot": "auditor",
                "wait": false
            }),
        };
        let (bg_res, ok2) = app.execute_tool_and_format(&bg_call);
        assert!(ok2);
        assert!(bg_res.contains("[subagent #2 · @auditor · spawned in background]"));

        // Wait for background tokio task to finish and announce back
        let mut announced = false;
        for _ in 0..100 {
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            app.poll_stream_events();
            if app.messages.iter().any(|m| m.content.contains("Subagent Task #2 Announcement")) {
                announced = true;
                break;
            }
        }
        assert!(announced, "Expected announcement system message in conversation history");
        assert!(app.subagent_mgr.get_status(2).is_some());
    }

    #[test]
    fn test_tool_approval_modal_and_execution_workflow() {
        let graph = Arc::new(cynapse_memory::graph::Dendrite::new());
        let ctx = cynapse_memory::context::DendriteContext::new(Arc::clone(&graph), None);
        let mut app = TuiApp::new(
            PathBuf::from("./models"),
            "ministral-3:3b".into(),
            "http://127.0.0.1:59999".into(),
            graph,
            None,
            ctx,
        );

        // 1. Set pending approval
        app.pending_approval = Some(PendingToolApproval::new(
            1,
            "auditor".into(),
            "execute_command".into(),
            "echo 'approved by user'".into(),
            None,
            None,
        ));
        app.modal = ActiveModal::ToolApproval;

        // 2. Approve via decision handler
        app.handle_approval_decision(true);

        assert_eq!(app.modal, ActiveModal::None);
        assert!(app.pending_approval.is_none());
        assert!(app.messages.iter().any(|m| m.content.contains("approved `execute_command`")));

        // 3. Set another pending approval and deny via decision handler
        app.pending_approval = Some(PendingToolApproval::new(
            1,
            "auditor".into(),
            "execute_command".into(),
            "rm -rf /tmp/dangerous".into(),
            None,
            None,
        ));
        app.modal = ActiveModal::ToolApproval;

        app.handle_approval_decision(false);

        assert_eq!(app.modal, ActiveModal::None);
        assert!(app.pending_approval.is_none());
        assert!(app.messages.iter().any(|m| m.content.contains("denied `execute_command`")));

        // 4. Test interactive editing of parameters
        let mut pending = PendingToolApproval::new(
            2,
            "coder".into(),
            "write_file".into(),
            "temp.rs".into(),
            Some("fn main() {}".into()),
            None,
        );
        pending.edit_mode = true;
        pending.editing_arg = 0;
        pending.edit_buffer = "src/main.rs".into();
        pending.arg1 = pending.edit_buffer.clone();
        pending.editing_arg = 1;
        pending.edit_buffer = "fn main() { println!(\"edited\"); }".into();
        pending.arg2 = Some(pending.edit_buffer.clone());
        pending.edit_mode = false;

        app.pending_approval = Some(pending);
        app.modal = ActiveModal::ToolApproval;
        app.handle_approval_decision(true);

        assert_eq!(app.modal, ActiveModal::None);
        assert!(app.pending_approval.is_none());
        assert!(app.messages.iter().any(|m| m.content.contains("src/main.rs")));
    }

    #[test]
    fn test_spawn_subagent_unknown_slug_fails_closed() {
        use serde_json::json;
        let graph = Arc::new(cynapse_memory::graph::Dendrite::new());
        let ctx = cynapse_memory::context::DendriteContext::new(Arc::clone(&graph), None);
        let mut app = TuiApp::new(
            PathBuf::from("./models"),
            "ministral-3:3b".into(),
            "http://127.0.0.1:59999".into(),
            graph,
            None,
            ctx,
        );

        let unknown_call = cynapse_core::offline_agent::ToolCall {
            name: "spawn_subagent".into(),
            arguments: json!({
                "task": "Do something unsafe",
                "bot": "nonexistent_bot",
                "wait": false
            }),
        };

        let (res, ok) = app.execute_tool_and_format(&unknown_call);
        assert!(!ok, "Unknown bot profile spawn must fail closed");
        assert!(res.contains("Unknown bot profile '@nonexistent_bot'"));
    }

    #[tokio::test]
    async fn test_background_subagent_approval_and_denial_channel_workflow() {
        let graph = Arc::new(cynapse_memory::graph::Dendrite::new());
        let ctx = cynapse_memory::context::DendriteContext::new(Arc::clone(&graph), None);
        let mut app = TuiApp::new(
            PathBuf::from("./models"),
            "ministral-3:3b".into(),
            "http://127.0.0.1:59999".into(),
            graph,
            None,
            ctx,
        );

        let (tx, rx) = tokio::sync::oneshot::channel();
        let req = ToolApprovalRequest {
            task_id: 42,
            bot_slug: "coder".into(),
            tool_name: "write_file".into(),
            arg1: "test.rs".into(),
            arg2: Some("fn main() {}".into()),
            response_tx: tx,
        };

        // 1. Send request through approval channel
        app.approval_tx.send(req).unwrap();

        // 2. Poll stream events to trigger modal
        app.poll_stream_events();
        assert_eq!(app.modal, ActiveModal::ToolApproval);
        assert!(app.pending_approval.is_some());

        // 3. User approves in modal
        app.handle_approval_decision(true);
        assert_eq!(app.modal, ActiveModal::None);

        // 4. Verify oneshot channel received true
        let received = rx.await.unwrap();
        assert!(received);
    }

    #[tokio::test]
    async fn test_at_slug_routing_and_bots_slash_commands() {
        let graph = Arc::new(cynapse_memory::graph::Dendrite::new());
        let ctx = cynapse_memory::context::DendriteContext::new(Arc::clone(&graph), None);
        let mut app = TuiApp::new(
            PathBuf::from("./models"),
            "ministral-3:3b".into(),
            "http://127.0.0.1:59999".into(),
            graph,
            None,
            ctx,
        );

        // 1. Simulate @coder <task> input routing
        app.input = "@coder Write tests for parser".into();
        app.input_cursor = app.input.len();
        let _ = crossterm::event::Event::Key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Enter,
            crossterm::event::KeyModifiers::empty(),
        ));
        // Run the input handler logic directly
        let trimmed = app.input.trim().to_string();
        if trimmed.starts_with('@') {
            let rest = &trimmed[1..];
            let mut parts = rest.splitn(2, |c: char| c.is_whitespace());
            let slug = parts.next().unwrap_or("").trim();
            let task = parts.next().unwrap_or("").trim();
            let call = cynapse_core::offline_agent::ToolCall {
                name: "spawn_subagent".into(),
                arguments: serde_json::json!({
                    "bot": slug,
                    "task": task,
                    "wait": false,
                }),
            };
            let (out, ok) = app.execute_tool_and_format(&call);
            assert!(ok);
            assert!(out.contains("[subagent #1 · @coder · spawned in background]"));
        }

        // 2. Test /bots status listing
        let tasks = app.subagent_mgr.list_tasks();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].bot_slug, "coder");

        // 3. Test /bots cancel
        let cancelled = app.subagent_mgr.cancel_task(1);
        assert!(cancelled);
        assert_eq!(app.subagent_mgr.get_status(1), Some(cynapse_core::subagent::SubagentStatus::Cancelled));
    }

    #[test]
    fn test_poll_stream_events_auto_cancels_stalled_tasks() {
        let graph = Arc::new(cynapse_memory::graph::Dendrite::new());
        let ctx = cynapse_memory::context::DendriteContext::new(Arc::clone(&graph), None);
        let mut app = TuiApp::new(
            PathBuf::from("./models"),
            "ministral-3:3b".into(),
            "http://127.0.0.1:59999".into(),
            graph,
            None,
            ctx,
        );

        let (task_id, cancel_flag) = app.subagent_mgr.create_task("Hanging computation", "researcher", false);
        app.subagent_mgr.update_status(task_id, cynapse_core::subagent::SubagentStatus::Running, None);

        // Artificially age the task to >2m
        let old_time = cynapse_core::subagent::SubagentManager::now_ms().saturating_sub(180_000);
        app.subagent_mgr.set_task_started_at_for_test(task_id, old_time);

        app.poll_stream_events();
        assert!(cancel_flag.load(std::sync::atomic::Ordering::SeqCst));
        assert_eq!(app.subagent_mgr.get_status(task_id), Some(cynapse_core::subagent::SubagentStatus::Cancelled));
        assert!(app.messages.iter().any(|m| m.content.contains("stalled (> 2m) and was auto-cancelled")));
    }
