-- Seed data cho hsc.agent_memory_dist — sinh từ D:\hsc\doc\hsc-system-knowledge.md (2026-08-06).
-- Domain knowledge hệ thống HSC ONE (clusters, schema, order flow, log patterns, baseline...),
-- KHÔNG phải kiến thức về bản thân observability-agent (xem seed_memory_observability_agent.sql
-- cho phần đó). Chạy sau khi đã tạo bảng + user agent_smith (MEMORY_SETUP.md bước 1-2).
-- source = 'manual' vì đây là seed thủ công, không phải agent tự lưu lúc suy luận.
-- Lưu ý: nội dung tránh nhúng chuỗi SQL có dấu nháy đơn literal để không phải escape phức tạp —
-- những chỗ cần trích dẫn giá trị SQL được diễn giải bằng lời thay vì viết code SQL nguyên văn.

INSERT INTO hsc.agent_memory_dist (memory_type, title, content, source, tags, related_service) VALUES

('pattern',
 'HSC K8s clusters & log sources',
 'k8s-one-core: trading chính (order, market data, Kong) — log qua ClickHouse hsc.otel_logs_dist. IBS: banking/auth/client mgmt — log qua Loki IBS (UID efnexppb9epz4d). TSP-2: core gateway sang G2/Sở CK — log qua Loki loki-one&tsp (UID bfnexdhgx2d4wf), CHƯA có OTel nên KHÔNG có trong ClickHouse. DMZ: DMZ services — Loki DMZ (UID bfnspyvf1ux34d). Endpoints production: Grafana https://rancher-ops.hsc.com.vn/grafana (Basic admin:admin), ClickHouse LB http://10.33.3.135:8123 (user default, password trống), N8N https://rancher-ops.hsc.com.vn/n8n.',
 'manual',
 ['hsc-system', 'clusters', 'log-sources'],
 ''),

('pattern',
 'ClickHouse cluster hsc_log_mgmt — physical topology',
 'Cluster hsc_log_mgmt: 3 shard x 2 replica = 6 node. vm-share01 (10.33.3.132): ch-01 port 9000 là S1R1, ch-06 port 9001 là S3R2, có keeper-01. vm-share02 (10.33.3.133): ch-02 port 9000 là S2R1, ch-03 port 9001 là S1R2, có keeper-02. vm-share03 (10.33.3.134): ch-04 port 9000 là S2R2, ch-05 port 9001 là S3R1, có keeper-03. vm-itops (10.33.3.135): nginx LB port 8123/9000, HyperDX port 8080 — đây là endpoint client thực tế kết nối vào, không phải node dữ liệu.',
 'manual',
 ['hsc-system', 'clickhouse', 'infra'],
 ''),

('pattern',
 'ClickHouse hsc.otel_logs_dist — schema và quy ước query',
 'Cột quan trọng: TimestampTime (DateTime, luôn WHERE theo cột này), Body (raw log text), SeverityText (uppercase hoặc lowercase tuỳ service), ResourceAttributes map có key k8s.container.name/k8s.namespace.name/k8s.cluster.name/host.node.name (dùng trước 2026-06-17)/k8s.node.name (dùng sau 2026-06-17), LogAttributes map có key httpStatus/apiPath/responseTime/httpMethod/userId, TraceId là FixedString(16) — đọc bằng lower(hex(TraceId)), filter bằng TraceId = unhex(traceid_hex_32_chars). Quy ước SeverityText: Node.js services (order-api, user-sh-api) dùng lowercase info/warn/error; Java (kube-cash-mgmt-one, Kong) và Go (corporate-data) dùng uppercase INFO/WARN/ERROR — phải kiểm tra đúng case khi WHERE, sai case sẽ không ra kết quả. Template query chuẩn: SELECT TimestampTime, ResourceAttributes[k8s.container.name] AS service, SeverityText, substring(Body,1,500) AS body FROM hsc.otel_logs_dist WHERE TimestampTime trong khoảng cần tìm AND ResourceAttributes[k8s.container.name] = tên service AND SeverityText đúng case ORDER BY TimestampTime DESC LIMIT 50 FORMAT JSONEachRow. Trong Grafana panel dùng macro $__timeFilter(TimestampTime) thay vì tự viết WHERE thời gian.',
 'manual',
 ['hsc-system', 'clickhouse', 'schema', 'otel_logs_dist'],
 'otel_logs_dist'),

('pattern',
 'Order flow (equity) — luồng đặt lệnh cơ sở và Kafka topics',
 'Client App gọi order-api (namespace hsc-dti-trading-ns): (1) POST /equity/v1/getOtp, (2) GET_PERMISSIONS_V3 (Redis cache TTL 600s), (3) getTradingPower qua core-api-gw (TSP-2) tới /buying-power, (4) POST /equity/v1/verifyRiskRules tới lotus-api.hsc.com.vn, (5) POST /equity/v1/placeOrder — Response LUÔN 200 kể cả khi G2 reject sau đó. Sau đó publish Kafka topic ORDER_PLACED, 2 consumer xử lý song song: order-consumer-equity-order-placed gửi lên TSP-2 core-api-gw rồi G2 Core rồi Sở CK; order-consumer-equity-order-place-persisted-to-db ghi Postgres. G2 trả status qua TCP stream: order-listener-status-streaming nhận rồi publish Kafka EQUITY_ORDER_STATUS_CHANGED_BATCH rồi order-consumer-equity-order-status-changed publish Kafka ORDER_STATUS_SYNCED rồi order-consumer-equity-order-status-synced UPDATE Postgres. Trade events tương tự qua order-listener-trade-streaming publish Kafka EQUITY_ORDER_TRADE_CHANGED rồi order-consumer-equity-order-trade-changed UPDATE matched qty/price. Các topic khác: ORDER_CANCEL_REQUESTED (order-api tới consumer-cancel-requested), ORDER_AMEND_REQUESTED (order-api tới order-consumer-equity-order-amend-requested). Log volume/ngày: order-api 4.2M, conditional-order-go 15.3M (cao nhất), order-listener-status-streaming 1.3M, order-consumer-equity-order-status-changed 675K, order-consumer-equity-order-placed 60K, order-pro-api 10K.',
 'manual',
 ['hsc-system', 'order-flow', 'kafka', 'order-api'],
 'order-api'),

('pattern',
 'order-api và order-consumer-equity-order-placed — log pattern G2 rejection',
 'order-api Body format: uuid, timestamp, info: POST /equity/v1/placeOrder, rồi RequestParams (JSON), Response (JSON), Status: 200, ResponseTime tính bằng ms, UserId. Lưu ý: placeOrder LUÔN trả 200, G2 reject sẽ được log riêng ở order-consumer-equity-order-placed, không phải ở order-api. accountId nằm trong RequestParams body, khác với UserId (UserId là người thực hiện). price/quantity không có quotes trong JSON. G2 rejection: SeverityText là error (lowercase), Body chứa cụm "core return error", nội dung dạng JSON có field message, data.orderData (accountId/ticker/bidAsk/quantity/price), và error là chuỗi lỗi từ G2. Các G2 error đã gặp: "Not enough current room" nghĩa là room nước ngoài (foreign room) đã hết cho mã cổ phiếu đó; "This stock is suspended or terminated" nghĩa là cổ phiếu bị đình chỉ giao dịch. Query template tìm G2 rejection theo mã cổ phiếu: SELECT TimestampTime, substring(Body,1,2000) AS body FROM hsc.otel_logs_dist WHERE TimestampTime trong khoảng cần tìm AND ResourceAttributes[k8s.container.name] = order-consumer-equity-order-placed AND SeverityText = error (chữ thường) AND Body LIKE chứa cụm "core return error" AND Body LIKE chứa mã ticker cần lọc ORDER BY TimestampTime ASC FORMAT JSONEachRow.',
 'manual',
 ['hsc-system', 'order-flow', 'g2', 'log-pattern'],
 'order-consumer-equity-order-placed'),

('pattern',
 'corporate-data — Foreign Room pattern và phân biệt lỗi Frontend vs Backend',
 'Container corporate-data, API GET /v1/financial?symbol=<TICKER>, SeverityText uppercase INFO. Response JSON có field f_room_remaining (số lượng còn lại cho NĐTNN mua, LÀ SNAPSHOT CUỐI NGÀY HÔM TRƯỚC, KHÔNG real-time), outstanding_shares, free_float, update_date. Phân biệt lỗi hiện trên app: thông báo kiểu "số lượng lệnh đặt phải bé hơn room còn lại" là lỗi frontend tự check f_room_remaining trước khi gọi API đặt lệnh — KHÔNG có log tương ứng ở backend, đừng nhầm với lỗi thật. Lỗi "G2 Internal Error: Not enough current room" mới là lỗi thật từ G2, xuất hiện trong log order-consumer-equity-order-placed — xảy ra khi order đã vào Kafka rồi, G2 mới reject. Query template tìm foreign room: SELECT TimestampTime, substring(Body,1,500) AS body FROM hsc.otel_logs_dist WHERE ResourceAttributes[k8s.container.name] = corporate-data AND Body LIKE chứa f_room_remaining AND Body LIKE chứa mã ticker ORDER BY TimestampTime DESC LIMIT 3 FORMAT JSONEachRow.',
 'manual',
 ['hsc-system', 'foreign-room', 'log-pattern'],
 'corporate-data'),

('pattern',
 'IBS Cluster — services, audit log pattern, Kong Lotus log format',
 'Kong route ibs-lotus-api.hsc.com.vn vào namespace kong-ibs-lotus rồi tới các banking services trong namespace banking: kube-customer-bff-one (port 9001, bank-accounts CRUD + external bank verification), kube-client-mgmt-one (port 7088, client profile/accounts), kube-iam-client (port 8085, auth/access-token), kube-cash-mgmt-one (port 7083, cash management/deposits/withdrawals). kube-cash-mgmt-one có audit log KHÔNG phải service riêng mà ghi ra stdout bởi Java class v.c.h.c.l.AuditLogging, dạng: timestamp, level INFO, rồi "AuditLogs:" kèm JSON có username/ipAddress/deviceId/deviceName/userAgent/method/responseStatus/requestUri/sessionId/requestData/responseData/channel. Loki query pattern cho audit log: app bằng kube-cash-mgmt-one, filter chứa cụm "AuditLogs:". Kong IBS Lotus log format khác Kong thường: có timestamp, level debug, SN=Kong, URI (method + path), Status, RequestTime (giây), Host (upstream service:port), VisitedId (trace session id), Agent (user agent string), ClientIp2 (real client ip). Field quan trọng: Status, RequestTime, Host (giá trị dấu gạch ngang nghĩa là upstream chết/không resolve được), VisitedId dùng để trace theo session.',
 'manual',
 ['hsc-system', 'ibs', 'banking', 'log-pattern'],
 'kube-customer-bff-one'),

('pattern',
 'Loki query qua Grafana API và tips Grafana + ClickHouse dashboard',
 'Query Loki không gọi Loki trực tiếp mà qua Grafana datasource proxy: GET tới đường dẫn rancher-ops.hsc.com.vn/grafana/api/datasources/proxy/uid/<loki_uid>/loki/api/v1/query_range với param query (LogQL đã URL-encode), start/end (unix nano), limit, direction=backward. Header Authorization Basic admin:admin. Rate limit 429 xảy ra nếu gọi liên tục — phải sleep vài giây giữa các request, không được query dồn dập. Lưu ý timestamp 2026 là khoảng 1785xxxxxx giây (KHÔNG phải 1753xxxxxx là năm 2025) khi tính start/end nano. Loki label schema cluster IBS gồm: app, container, filename, job, namespace, node_name, pod, stream. Về Grafana + ClickHouse dashboard: macro $__timeFilter(col) dùng cho WHERE time filter, $__timeInterval(col) dùng cho GROUP BY bucket (phải set interval ở panel level); KHÔNG dùng INTERVAL $__interval second vì $__interval là string dạng "1m" không phải số; timezone nên dùng formatDateTime(col, format, Asia/Ho_Chi_Minh) explicit, KHÔNG dùng toTimezone() vì gây lệch time range với Grafana; gọi Grafana API từ PowerShell phải encode body bằng UTF8.GetBytes để không mất dấu tiếng Việt; lỗi 409 provenanceMismatch StoredProvenance=file fix bằng xoá dòng tương ứng trong bảng provenance_type của file grafana.db (SQLite).',
 'manual',
 ['hsc-system', 'loki', 'grafana', 'dashboard'],
 ''),

('decision',
 'Baseline metrics ngày giao dịch bình thường (2026-07-07) và các false positive đừng alert nhầm',
 'Baseline: tổng lệnh đặt/ngày (equity) khoảng 8906, lệnh gửi lên G2 thành công khoảng 12683, verifyRiskRules error rate (400) 9.3%, trading-power error rate (403) 4.4%, derivatives/v1/account error (500) 1.6% — dùng làm tham chiếu khi so sánh ngày bất thường. False positive KHÔNG được alert nhầm: streaming-centrifugo "already subscribed" code 105 khoảng 203K/ngày là client reconnect bình thường; order-api SeverityText=error nhưng body thực chất là info khoảng 51K/ngày do OTel parse sai severity; order-api "console.timeEnd() label not found" khoảng 225K warn/ngày do race condition async trong Node.js; kong-ibs-lotus "openidc.lua 404" khoảng 41K/ngày do IBS discovery endpoint không tồn tại — đây là hành vi bình thường, không phải lỗi.',
 'manual',
 ['hsc-system', 'baseline', 'false-positive', 'alerting'],
 ''),

('decision',
 'Real errors cần alert (ưu tiên P0/P1) và account prefix patterns',
 'Lỗi thật cần alert: order-consumer-equity-order-place-sent-to-core log ERROR (khoảng 6/ngày) ưu tiên P0; order-listener STREAM_STATUS "Stream error" nghĩa là G2 TCP bị ngắt kết nối (khoảng 69/ngày) ưu tiên P0; order-consumer-equity-order-trade-changed "Order not found" (khoảng 514/ngày) ưu tiên P1. Account prefix: 011C là tài khoản cơ sở domestic thông thường; KTZ có thể là tài khoản nước ngoài (foreign investor) nhưng CHƯA verify chính thức, cần kiểm tra lại trước khi khẳng định với user.',
 'manual',
 ['hsc-system', 'alerting', 'priority', 'account-pattern'],
 ''),

('decision',
 'Security / Operational constraints khi investigate hệ thống HSC',
 'Không tự thực thi lệnh docker, kubectl, hoặc các lệnh delete/xoá dữ liệu dù được yêu cầu — luôn hỏi lại xác nhận trước. Phải hỏi rõ môi trường đang làm việc (UAT hay production) nếu không chắc chắn, vì hành vi/dữ liệu 2 môi trường khác nhau. Ở production chỉ được GET/View dữ liệu, KHÔNG sửa config — ngoại lệ duy nhất là được phép tạo dashboard và alert trên Grafana. Kết luận root cause bắt buộc phải dựa trên log/dữ liệu thực tế query được, không được suy đoán hoặc bịa. Nên đọc memory (bảng hsc.agent_memory_dist này) trước khi tự đi query lại từ đầu để tiết kiệm round-trip và tránh lặp lại điều đã biết.',
 'manual',
 ['hsc-system', 'constraints', 'operational-rules'],
 ''),

('known_issue',
 'Blind spots — khu vực không có log trong ClickHouse',
 'TSP-2 (service core-api-gw): chưa có OTel collector nên không có trong ClickHouse, chỉ xem được qua Loki datasource loki-one&tsp (UID bfnexdhgx2d4wf). IBS app logs (namespace banking): cluster IBS không có OTel, chỉ xem được qua Loki datasource loki-ibs (UID efnexppb9epz4d). Latency đoạn order-placed tới G2: TSP-2 là blind spot hoàn toàn, chỉ thấy được gián tiếp qua dòng text thuần dạng "SendOrderToCore: <số>ms" trong log của order-consumer-equity-order-placed, không có structured field riêng.',
 'manual',
 ['hsc-system', 'blind-spot', 'tsp-2', 'ibs'],
 '');
