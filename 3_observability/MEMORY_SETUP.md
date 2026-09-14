# Setup Memory cho Agent bằng ClickHouse (kế hoạch — chưa làm)

Mục tiêu: agent tự lưu nhận định đúng lúc suy luận, tự đọc lại lúc cần, và khi phát hiện
kiến thức cũ sai thì **đánh dấu lỗi thời (soft-delete)** thay vì xóa cứng. Người dùng cũng
lưu tay được qua chat bình thường ("nhớ giúp anh...").

Quyền ghi được **tách riêng** khỏi kết nối đọc log hiện tại — agent không có khả năng ghi
nhầm vào bảng log/production thật, chỉ ghi được đúng 1 bảng `agent_memory`.

---

## Bước 1 — Tạo bảng `agent_memory` trong ClickHouse (distributed, đúng chuẩn cluster thật)

Đã verify trực tiếp trên `10.33.3.135` (`system.clusters`): cluster tên **`hsc_log_mgmt`**,
3 shard × 2 replica = 6 node. Database dùng chung với log thật là **`hsc`** (không phải
`default`). DDL dưới đây copy đúng convention đang dùng cho `hsc.otel_logs`/`otel_logs_dist`
(xem `SHOW CREATE TABLE hsc.otel_logs` để đối chiếu lại nếu nghi ngờ) — **không tự bịa path
ZooKeeper**, dùng đúng pattern `/clickhouse/tables/{shard}/hsc/<tên bảng>` họ đã dùng sẵn.

**1a. Bảng local (Replicated), tạo trên cả 6 node cùng lúc bằng `ON CLUSTER`:**
```sql
CREATE TABLE hsc.agent_memory ON CLUSTER hsc_log_mgmt
(
    id                UUID DEFAULT generateUUIDv4(),
    created_at        DateTime DEFAULT now(),
    memory_type       LowCardinality(String),   -- 'incident' | 'pattern' | 'decision' | 'known_issue'
    title             String,                   -- tóm tắt 1 dòng
    content           String,                   -- chi tiết đầy đủ
    source            String,                   -- 'ai-agent:chat' | 'ai-agent:auto' | 'manual'
    tags              Array(String),            -- ['order-api', 'clickhouse', '5xx']
    related_service   LowCardinality(String) DEFAULT '',
    is_active         Bool DEFAULT true,
    superseded_at     Nullable(DateTime),
    superseded_reason String DEFAULT ''
)
ENGINE = ReplicatedMergeTree('/clickhouse/tables/{shard}/hsc/agent_memory', '{replica}')
ORDER BY (memory_type, created_at)
```

**1b. Bảng Distributed** — agent đọc/ghi qua bảng này (giống hệt cách `otel_logs_dist` route
tới `otel_logs`), `rand()` để chia đều bản ghi mới vào 3 shard, không cần sharding key riêng
vì memory không lớn tới mức cần route theo key cụ thể:
```sql
CREATE TABLE hsc.agent_memory_dist ON CLUSTER hsc_log_mgmt
AS hsc.agent_memory
ENGINE = Distributed('hsc_log_mgmt', 'hsc', 'agent_memory', rand())
```

**Kiểm tra** (query qua bảng `_dist`, không phải bảng local):
```sql
INSERT INTO hsc.agent_memory_dist (memory_type, title, content, source, tags)
VALUES ('pattern', 'Test entry', 'Kiểm tra bảng tạo đúng chưa', 'manual', ['test']);

SELECT * FROM hsc.agent_memory_dist WHERE is_active;
```
Nếu muốn chắc chắn dữ liệu chia đều 3 shard (không dồn hết 1 node): `SELECT hostName(), count()
FROM hsc.agent_memory_dist GROUP BY hostName()` sau khi insert vài chục dòng test.

---

## Bước 2 — Tạo user ClickHouse riêng, chỉ ghi được bảng này

**Không dùng chung** user `default`/`hyperdx` đang query log — tạo user mới (đã đặt sẵn
`agent_smith`), quyền giới hạn chặt, tạo trên cả cluster bằng `ON CLUSTER` (không phải chỉ 1
node — vì có 6 node, request có thể rơi vào node bất kỳ qua nginx LB):

```sql
-- OR REPLACE (không phải CREATE thường) — vì user này có thể đã lỡ tạo cục bộ trên 1 node
-- trước đó (không có ON CLUSTER), gây ACCESS_ENTITY_ALREADY_EXISTS khi tạo lại theo cluster.
-- OR REPLACE ghi đè sạch, đảm bảo đồng bộ đúng trên cả 6 node bất kể trạng thái cũ thế nào.
CREATE USER OR REPLACE agent_smith ON CLUSTER hsc_log_mgmt IDENTIFIED WITH plaintext_password BY 'whosyourdaddy';

-- INSERT qua bảng _dist thực chất forward xuống bảng local trên từng shard bằng chính user
-- đang kết nối — nên cần GRANT trên CẢ HAI bảng (local + dist), không phải chỉ 1 trong 2.
GRANT SELECT, INSERT, ALTER UPDATE ON hsc.agent_memory ON CLUSTER hsc_log_mgmt TO agent_smith;
GRANT SELECT, INSERT ON hsc.agent_memory_dist ON CLUSTER hsc_log_mgmt TO agent_smith;

-- Đọc log thật (không ghi) — giữ lại nếu muốn agent_smith cũng tự query log lúc suy luận
GRANT SELECT ON hsc.* TO agent_smith ON CLUSTER hsc_log_mgmt;
```

**Kiểm tra quyền bị giới hạn đúng** (dùng user `agent_smith` thử ghi vào 1 bảng log thật — phải
bị từ chối):
```sql
-- Chạy bằng user agent_smith, phải ra lỗi "Not enough privileges"
INSERT INTO hsc.otel_logs_dist VALUES (...);
```
Nếu lệnh trên **không báo lỗi** → cấu hình quyền sai, phải xem lại GRANT trước khi dùng thật.

---

## Bước 3 — Thêm MCP connection thứ 2 (read-write, chỉ cho memory) vào `app/config.py`

Giữ nguyên connection `clickhouse` cũ (đọc log, không đổi gì). Thêm connection mới tên
`clickhouse-memory`, dùng user vừa tạo, **bật quyền ghi**:

```python
CLICKHOUSE_MEMORY_MCP = {
    "clickhouse-memory": {
        "command": "uvx",
        "args": ["--with", "mcp-clickhouse", "--python", "3.10", "mcp-clickhouse"],
        "env": {
            "CLICKHOUSE_HOST": "10.33.3.135",          # giống connection đọc log
            "CLICKHOUSE_PORT": "8123",
            "CLICKHOUSE_USER": "agent_smith",           # user MỚI, quyền giới hạn (bước 2)
            "CLICKHOUSE_PASSWORD": "whosyourdaddy",
            "CLICKHOUSE_SECURE": "false",
            "CLICKHOUSE_ALLOW_WRITE_ACCESS": "true",    # BẬT ghi — chỉ connection này
        },
    }
}
```

Gộp cả 2 dict lại khi truyền vào `ClaudeAgentOptions(mcp_servers=...)` trong `app/agent.py`:
```python
mcp_servers={**CLICKHOUSE_MCP, **CLICKHOUSE_MEMORY_MCP}
```

Cập nhật `ALLOWED_TOOLS` trong `config.py`:
```python
ALLOWED_TOOLS = [
    "mcp__clickhouse__*",           # đọc log (cũ, không đổi)
    "mcp__clickhouse-memory__*",    # đọc/ghi memory (mới)
]
```

---

## Bước 4 — Dặn agent khi nào lưu / khi nào đánh dấu lỗi thời (system prompt)

Thêm vào `CHAT_SYSTEM_PROMPT` (và `ALERT_SYSTEM_PROMPT` nếu muốn agent tự lưu cả lúc triage
alert) trong `app/config.py`:

```
Bạn có quyền đọc/ghi bảng hsc.agent_memory_dist (qua tool clickhouse-memory) để nhớ thông tin
giữa các phiên chat khác nhau. LUÔN đọc/ghi qua bảng _dist (KHÔNG dùng bảng agent_memory local)
để dữ liệu được phân phối đúng qua cả 3 shard của cluster.

KHI NÀO LƯU (dùng INSERT vào hsc.agent_memory_dist):
- Vừa xác nhận được root cause 1 incident đã điều tra xong
- Vừa xác nhận 1 false positive pattern (alert X ở service Y luôn false vì lý do Z)
- Vừa có 1 quyết định vận hành có lý do rõ ràng
- Vừa học được 1 pattern log/schema khó, có khả năng cần lại sau
KHÔNG lưu: số liệu tạm thời, thông tin tự tra lại được từ ClickHouse, việc chỉ xảy ra 1 lần.

TRƯỚC KHI TRẢ LỜI liên quan chủ đề cũ: LUÔN query hsc.agent_memory_dist trước
(SELECT * FROM hsc.agent_memory_dist WHERE is_active AND ...) để không lặp lại điều đã biết.

KHI PHÁT HIỆN MEMORY CŨ SAI: KHÔNG xóa (DELETE) — chạy:
ALTER TABLE hsc.agent_memory_dist UPDATE is_active=false, superseded_at=now(),
  superseded_reason='<lý do cụ thể>' WHERE id='<id bản ghi cũ>'
rồi INSERT bản ghi mới thay thế. Luôn ghi rõ superseded_reason để sau này xem lại được.
```

> Lưu ý cú pháp: `UPDATE`/`DELETE` trên ClickHouse phải viết dạng `ALTER TABLE ... UPDATE ...`
> (mutation), không viết `UPDATE ... SET ...` kiểu SQL thường — cú pháp đó ClickHouse không hỗ
> trợ. Đây cũng chính là lý do quyền cần cấp là `ALTER UPDATE` (bước 2), không phải `UPDATE`.

---

## Bước 5 — Test

1. Chat hỏi agent 1 câu buộc nó phải tự lưu (VD: "root cause lỗi X là do Y, nhớ giúp anh") →
   kiểm tra `SELECT * FROM hsc.agent_memory_dist ORDER BY created_at DESC LIMIT 5` có bản ghi mới.
2. Hỏi lại câu liên quan ở hội thoại/phiên MỚI (client mới, không resume) → xác nhận agent tự
   query lại và dùng đúng memory đã lưu (không hỏi lại từ đầu / không bịa).
3. Nói với agent 1 thông tin **mâu thuẫn** với memory cũ đã lưu → kiểm tra bản ghi cũ có
   `is_active=false` + `superseded_reason` hợp lý, và có bản ghi mới thay thế — **không** bị
   xóa mất khỏi bảng (`SELECT * FROM hsc.agent_memory_dist` vẫn thấy đủ, kể cả bản
   `is_active=false`). Lưu ý: `ALTER TABLE ... UPDATE` là mutation bất đồng bộ, có thể mất vài
   giây mới thấy `is_active` đổi — không phải lỗi nếu chưa thấy ngay lập tức.
4. Thử (bằng tay) bảo agent ghi vào 1 bảng log khác (`hsc.otel_logs_dist`) — phải bị ClickHouse
   từ chối quyền, xác nhận lại kết quả bước 2.
5. Kiểm tra dữ liệu chia đều 3 shard, không dồn 1 node:
   `SELECT hostName(), count() FROM hsc.agent_memory_dist GROUP BY hostName()`.

---

## Việc còn để mở, làm sau nếu cần

- Dọn dữ liệu cũ: thêm `TTL created_at + INTERVAL 180 DAY DELETE` nếu bảng quá lớn theo thời
  gian (chưa cần ngay ở quy mô 1 agent tự dùng).
- Semantic/vector search nếu full-text (`ILIKE`/`hasToken`) không đủ tốt — lúc đó mới cần
  thêm embedding, chưa vội.
- Cân nhắc thêm cột `created_by` (phân biệt agent tự lưu vs người lưu tay) nếu cần audit kỹ
  hơn `source`.
