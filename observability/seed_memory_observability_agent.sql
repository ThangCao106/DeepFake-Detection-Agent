-- Seed data cho hsc.agent_memory_dist — sinh từ memory project_observability_agent.md
-- (2026-08-06). Chạy 1 lần sau khi đã tạo bảng + user agent_smith (xem MEMORY_SETUP.md
-- bước 1-2). Dùng user bất kỳ có quyền INSERT vào hsc.agent_memory_dist (agent_smith
-- hoặc user admin) để chạy file này — không phải agent tự lưu, nên source = 'manual'.

INSERT INTO hsc.agent_memory_dist (memory_type, title, content, source, tags, related_service) VALUES

('decision',
 'Observability Agent — tổng quan project',
 'Service FastAPI dùng claude_agent_sdk (tái dùng Claude Pro/Max login qua Claude Code, KHÔNG dùng ANTHROPIC_API_KEY — mục tiêu xuyên suốt cả project là không phát sinh phí). Deploy bằng Docker trên 1 Ubuntu server (D:/hsc/agent/observability-agent-ubuntu là bản dev/nguồn trên máy Windows, copy sang server để build). Ban đầu chỉ dùng cá nhân, không mở cho teammate/n8n production — sau đó (2026-08-06) đã chủ động mở kết nối từ n8n (qua HTTP Request node gọi /api/chat, hoặc OpenAI Chat Model node trỏ base URL /v1), vượt qua giới hạn tự đặt trước đó, user chấp nhận rủi ro.',
 'manual',
 ['observability-agent', 'overview', 'architecture'],
 'observability-agent'),

('decision',
 'Khoá quyền agent — chỉ phục vụ observability',
 'app/config.py + app/agent.py khoá agent chỉ được: ALLOWED_TOOLS whitelist tường minh mcp__clickhouse__* (đọc log), mcp__clickhouse-memory__* (đọc/ghi memory), + từng tool ĐỌC của Grafana liệt kê tên rõ ràng (KHÔNG dùng mcp__grafana__* wildcard vì sẽ mở luôn update_dashboard/alerting_manage_*/create_incident). DISALLOWED_TOOLS chặn cứng toàn bộ built-in tool Claude Code: Bash, Read/Write/Edit, Glob/Grep, WebSearch/WebFetch, Task/SlashCommand/TodoWrite. KHÔNG set setting_sources=["user"] vì sẽ nạp toàn bộ ~/.claude/settings.json (bash pre-approved của các dự án khác) + hook cấp máy vào agent, phá vỡ mục tiêu lockdown. Lý do: user yêu cầu tường minh agent không được thực thi lệnh hệ thống, không tạo file, không truy cập hệ thống khác ngoài ClickHouse/Grafana đã duyệt. Áp dụng: mọi tool/MCP server mới thêm vào agent này phải liệt kê tên tool rõ ràng, không wildcard cho service có tool ghi/admin.',
 'manual',
 ['observability-agent', 'permissions', 'security', 'lockdown'],
 'observability-agent'),

('known_issue',
 'claude-mem đã bỏ hẳn khỏi observability-agent',
 'Từng thử tích hợp claude-mem (memory chung của Claude Code) rồi bỏ hoàn toàn, thay bằng ClickHouse memory tự xây. Lý do chính: claude-mem cần setting_sources=["user"] để hoạt động (đọc hook/MCP nó đăng ký), nhưng dòng đó xung đột trực tiếp với chủ trương khoá quyền agent (kéo theo toàn bộ settings/hook cấp máy). Đã gỡ RUN npm install -g claude-mem khỏi Dockerfile, gỡ setting_sources khỏi agent.py, gỡ mcp__claude-mem__* khỏi ALLOWED_TOOLS. Node.js vẫn còn trong image chính nhưng KHÔNG phải cho claude-mem — vì @anthropic-ai/claude-code (cài qua service login riêng, ghi vào volume claude-home dùng chung) là 1 CLI chạy bằng Node mà claude_agent_sdk gọi tới ở mọi request.',
 'manual',
 ['observability-agent', 'claude-mem', 'deprecated'],
 'observability-agent'),

('decision',
 'Memory agent dùng ClickHouse (soft-delete) thay vì RAG/claude-mem server',
 'Quyết định cuối sau khi cân nhắc claude-mem (phức tạp, cần LLM riêng để nén) vs RAG DB qua n8n vs ClickHouse tự quản: dùng bảng hsc.agent_memory (ReplicatedMergeTree) + hsc.agent_memory_dist (Distributed, cluster hsc_log_mgmt, 3 shard). Soft-delete bằng is_active/superseded_at/superseded_reason, KHÔNG hard DELETE — khi phát hiện kiến thức cũ sai thì ALTER TABLE ... UPDATE is_active=false thay vì xoá, rồi INSERT bản ghi mới. Agent tự quyết lúc nào lưu (root cause đã confirm, false positive pattern, quyết định vận hành, pattern log khó) + user lưu tay qua chat bình thường. Quyền ghi tách riêng: user ClickHouse mới agent_smith (khác hyperdx dùng đọc log), chỉ có GRANT trên đúng hsc.agent_memory/hsc.agent_memory_dist, không đụng được bảng log/production thật — 2 MCP server riêng (clickhouse read-only, clickhouse-memory read-write) dù cùng host 10.33.3.135. Chi tiết DDL/GRANT đầy đủ ở file MEMORY_SETUP.md trong repo.',
 'manual',
 ['observability-agent', 'memory', 'clickhouse', 'soft-delete'],
 'observability-agent'),

('pattern',
 'Gửi ảnh/file cho agent qua content block, không qua tool Read',
 'app/agent.py có build_user_content(text, attachments): ảnh (image/*) và PDF thành content block chuẩn Anthropic Messages API ({"type":"image"/"document","source":{"type":"base64","media_type":...,"data":...}}), file text khác (.txt/.csv/.log) decode base64 rồi chèn thẳng làm text block kèm tên file. Gửi qua field attachments trong POST /api/chat (media_type/data base64/filename tuỳ chọn) — KHÔNG cấp tool Read cho agent, nên nó vẫn không tự mở được file nào trên server, chỉ thấy nội dung nếu caller chủ động đính kèm trong request. Lưu ý kỹ thuật: client.query()/query() của claude_agent_sdk chỉ nhận str hoặc AsyncIterable[dict] — khi có attachment phải wrap content qua 1 async generator yield đúng 1 message dict dạng {"type":"user","message":{"role":"user","content":[...]}}. Single-message query() KHÔNG hỗ trợ ảnh, chỉ streaming input mode (ClaudeSDKClient.query với AsyncIterable) mới hỗ trợ multimodal — verified qua docs chính thức code.claude.com/docs/en/agent-sdk/streaming-vs-single-mode.',
 'manual',
 ['observability-agent', 'multimodal', 'claude-agent-sdk', 'attachments'],
 'observability-agent');
