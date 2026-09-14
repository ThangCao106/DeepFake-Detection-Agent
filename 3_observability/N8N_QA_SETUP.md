# Mở rộng workflow n8n "ai-agent" — thêm luồng Q&A thống kê/tình trạng hệ thống (kế hoạch)

Hiện tại workflow `ai-agent` trong n8n chỉ đọc tin nhắn Teams mới nhất rồi gửi thẳng vào
`/api/alert`. Kế hoạch này thêm 1 luồng thứ 2 (hỏi-đáp tự do qua `/api/chat`) chạy song song
trong cùng workflow, dùng chung cơ chế polling (không tạo bot mới được do policy công ty, nên
không dùng event-trigger/webhook Teams).

---

## Bước 1 — Đổi cách lấy tin nhắn: top 10 gần nhất thay vì chỉ 1 tin cuối

Lấy 1 tin cuối cùng dễ miss tin nhắn nếu:
- Nhiều tin nhắn mới đến giữa 2 lần poll (chỉ tin mới nhất được xử lý, các tin trước đó bị bỏ
  sót vĩnh viễn).
- Tin "cuối cùng" tại thời điểm poll lại chính là tin do bot tự trả lời trước đó.

**Đổi**: node lấy tin nhắn Teams sửa để lấy **top 10 tin gần nhất** (thay vì `limit: 1`), sort
theo thời gian **cũ → mới** (ASC) — thứ tự này quan trọng cho bước 4.

---

## Bước 2 — Lọc bỏ tin đã xử lý (dedup) TRƯỚC khi phân loại

Với "Mark Processed" hiện có: lọc bỏ khỏi danh sách 10 tin bất kỳ `messageId` nào đã có trong
processed-store, **trước khi** đưa vào bước phân loại ở Bước 3. Cần vì 2 lần poll liên tiếp chắc
chắn có tin trùng nhau (cửa sổ 10 tin dịch chuyển dần).

**Ghi processed ngay sau khi xử lý xong TỪNG tin** (không đợi xử lý hết cả 10 mới ghi 1 lần) —
nếu workflow lỗi giữa chừng, các tin đã xử lý xong không bị lặp lại ở lần chạy sau.

---

## Bước 3 — Phân loại từng tin (routing), áp dụng cho MỖI tin riêng lẻ trong danh sách còn lại

Thêm 1 node IF/Switch, chạy theo đúng thứ tự ưu tiên cho từng tin:

```
1. NẾU from = bot(chính ai-agent)                      → bỏ qua (chống bot-loop)
2. NẾU from = bot(khác, VD flow alert Grafana)
   VÀ nội dung khớp pattern alert
   (VD bắt đầu bằng "⚠️ [WARNING]"/"🔴", có "Trạng thái:"/"Lỗi:"/"Thời gian:") → route /api/alert
3. NẾU from = người thật VÀ nội dung chứa "AI-MODE: ON" → route /api/chat
4. NGƯỢC LẠI                                            → bỏ qua, không reply
```

Tag `AI-MODE: ON` là cơ chế **opt-in tường minh** — giải quyết luôn vấn đề "tin nhắn này của
flow nào": các workflow khác (Biometric Downloader, Truy Vết...) không gõ tag này nên `ai-agent`
sẽ không bao giờ vô tình xử lý nhầm tin nhắn của flow khác trong cùng channel.

Regex gợi ý cho bước 2 (Code node, JavaScript):
```js
const isFromSelf = (msg) => msg.from?.application?.id === AI_AGENT_BOT_APP_ID;
const looksLikeAlert = (text) =>
  /^[⚠️🔴]\s*\[?(WARNING|CRITICAL|ERROR)\]?/i.test(text.trim()) &&
  /(Trạng thái|Lỗi|Thời gian):/i.test(text);
const isAiModeOn = (text) => /AI-MODE:\s*ON/i.test(text);
```

---

## Bước 4 — Xử lý TUẦN TỰ theo thread, tin cũ trước

**Quan trọng riêng cho agent này**: `/api/chat` dùng `session_id = conversationId` (Teams thread
id) để giữ ngữ cảnh qua nhiều lượt — 1 `ClaudeSDKClient` sống lâu, gắn với session_id đó (xem
`app/agent.py`). Nếu n8n xử lý nhiều tin **song song** (mặc định của loop/Split-in-Batches nếu
không cấu hình cẩn thận) mà 2+ tin trong đó thuộc cùng 1 thread → nhiều request cùng lúc gọi vào
đúng 1 client đang kết nối, dễ lộn xộn/lỗi vì client không được thiết kế nhận nhiều `.query()`
chồng lên nhau cùng lúc.

**Bắt buộc**: set node loop qua danh sách tin (đã lọc + phân loại) chạy **"Run Once for Each
Item"** theo đúng thứ tự đã sort ở Bước 1 (cũ → mới), **KHÔNG chạy song song**. Nhờ vậy tin sau
trong cùng thread tự động thấy đúng ngữ cảnh của tin trước — không cần logic gộp nhiều tin thành
1 request.

---

## Bước 5 — Gọi API, timeout, và format bảng cho Teams

- **Timeout**: cả `/api/alert` và `/api/chat` chạy đồng bộ (n8n đợi full response, không stream)
  — set **Timeout ≥ 600000ms** (10 phút) ở node HTTP Request cho cả 2 nhánh, để tránh lỗi
  `ECONNABORTED` đã gặp với câu hỏi thống kê phức tạp.
- **Header**: `X-API-Key` đúng giá trị `API_KEY` trong `.env` (credential Header Auth, field
  Name phải đúng `X-API-Key`).
- **Format bảng khi trả lời vào Teams** (mức nhanh, không sửa gì bên agent — agent vẫn trả
  markdown table chuẩn, tốt cho LibreChat): thêm 1 Code node sau khi nhận `reply`, detect nếu có
  markdown table (dòng bắt đầu bằng `|`), nếu có thì đưa cả đoạn đó vào 1 TextBlock của Adaptive
  Card với `"fontType": "monospace"` — Teams hiện ra dạng khối canh cột đọc được, không bị vỡ
  thành ký tự `|` rời rạc như để nguyên markdown pipe-table trong TextBlock thường.

```js
// Gợi ý Code node — tách phần bảng ra khỏi phần text thường
function splitTableBlocks(reply) {
  const lines = reply.split('\n');
  const blocks = [];
  let buf = [], inTable = false;
  for (const line of lines) {
    const isTableLine = /^\s*\|.*\|\s*$/.test(line);
    if (isTableLine !== inTable) {
      if (buf.length) blocks.push({ type: inTable ? 'table' : 'text', content: buf.join('\n') });
      buf = []; inTable = isTableLine;
    }
    buf.push(line);
  }
  if (buf.length) blocks.push({ type: inTable ? 'table' : 'text', content: buf.join('\n') });
  return blocks;
}
// blocks có type "table" -> render TextBlock fontType: "monospace"
// blocks có type "text"  -> render TextBlock thường (markdown chuẩn)
```

*(Nâng cấp sau nếu cần, không bắt buộc ngay)*: parse table thật thành Adaptive Card `Table`
element (v1.5+, có `Table.Row`/`TableCell`) để ra bảng lưới đẹp như Excel thay vì khối monospace.

---

## Checklist trước khi chạy thật

- [ ] Xác nhận `AI_AGENT_BOT_APP_ID` (id bot của chính `ai-agent` khi post vào Teams) để chặn
      bot-loop ở Bước 3.1.
- [ ] Test pattern regex nhận diện alert với vài tin nhắn alert thật gần đây (tránh false
      negative làm alert thật lọt qua route sai, hoặc false positive coi nhầm câu hỏi người dùng
      là alert).
- [ ] Test dedup: chạy poll 2 lần liên tiếp với cùng dữ liệu, xác nhận không bị trả lời trùng.
- [ ] Test tuần tự: gửi liên tiếp 2-3 tin trong cùng thread trước khi n8n kịp poll, xác nhận agent
      trả lời đúng thứ tự và giữ được ngữ cảnh của nhau.
- [ ] Tăng Timeout HTTP Request lên 600000ms cho cả 2 nhánh.
