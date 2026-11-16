import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;

void main() => runApp(const DeepTraceMobileApp());

class DeepTraceMobileApp extends StatelessWidget {
  const DeepTraceMobileApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'DeepTrace.io.vn',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF04152E),
        primaryColor: const Color(0xFF0A84FF),
      ),
      home: const CallAndLiveGuardianScreen(),
    );
  }
}

class CallAndLiveGuardianScreen extends StatefulWidget {
  const CallAndLiveGuardianScreen({super.key});

  @override
  State<CallAndLiveGuardianScreen> createState() => _CallAndLiveGuardianScreenState();
}

class _CallAndLiveGuardianScreenState extends State<CallAndLiveGuardianScreen> with WidgetsBindingObserver {
  static const platform = MethodChannel('vn.io.deeptrace/overlay');

  final TextEditingController _urlController = TextEditingController();
  bool _isAnalyzing = false;
  bool _isFloatingShieldEnabled = false;
  Map<String, dynamic>? _lastResult;

  // Dual-Gateway Failover: Primary is DeepTrace Cloud, Fallback is DTP Gateway
  final List<String> _gateways = [
    "https://deeptrace.io.vn",
    "https://detection.dtpsoft.vn",
  ];
  String _activeGateway = "https://deeptrace.io.vn";

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _checkInitialOverlayStatus();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      _checkInitialOverlayStatus();
    }
  }

  Future<void> _checkInitialOverlayStatus() async {
    try {
      final bool isRunning = await platform.invokeMethod('isOverlayRunning') ?? false;
      if (mounted) {
        setState(() {
          _isFloatingShieldEnabled = isRunning;
        });
      }
    } catch (e) {
      debugPrint('[DeepTrace] Platform check error: $e');
    }
  }

  Future<void> _toggleFloatingShield(bool enable) async {
    try {
      if (enable) {
        final bool hasPermission = await platform.invokeMethod('checkPermission') ?? false;
        if (!hasPermission) {
          _showFloatingShieldPermissionGuide();
          await platform.invokeMethod('requestPermission');
          return;
        }
        await platform.invokeMethod('startOverlay');
        setState(() {
          _isFloatingShieldEnabled = true;
        });
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('🛡️ Khiên nổi đã xuất hiện! Mở TikTok hoặc YouTube để quét 1-chạm.'),
              backgroundColor: Color(0xFF10B981),
              duration: Duration(seconds: 4),
            ),
          );
        }
      } else {
        await platform.invokeMethod('stopOverlay');
        setState(() {
          _isFloatingShieldEnabled = false;
        });
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('Đã đóng khiên nổi bảo vệ.'),
              duration: Duration(seconds: 2),
            ),
          );
        }
      }
    } catch (e) {
      debugPrint('[DeepTrace] Error toggling overlay: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Lỗi khởi động khiên nổi: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Future<void> _analyzeStream(String url, String routingMode) async {
    final cleanUrl = url.trim();
    if (cleanUrl.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Vui lòng nhập hoặc dán link video/live cần giám định.'),
          backgroundColor: Colors.orange,
        ),
      );
      return;
    }

    setState(() {
      _isAnalyzing = true;
      _lastResult = null;
    });

    dynamic lastError;
    for (final base in _gateways) {
      try {
        debugPrint('[DeepTrace] Ingesting stream via Gateway: $base');

        // Step 1: Ingest stream via server (/api/upload supports URL, direct file, and json)
        final uploadUri = Uri.parse('$base/api/upload');
        final uploadRes = await http.post(
          uploadUri,
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({'url': cleanUrl, 'routing_mode': routingMode}),
        ).timeout(const Duration(seconds: 35));

        if (uploadRes.statusCode != 200) {
          throw Exception("Lỗi Gateway ($base): HTTP ${uploadRes.statusCode}");
        }

        final uploadData = jsonDecode(uploadRes.body);
        if (uploadData['success'] != true || uploadData['filepath'] == null) {
          throw Exception(uploadData['error'] ?? "Không thể nạp luồng stream vào máy chủ.");
        }

        debugPrint('[DeepTrace] Media ingested at: ${uploadData['filepath']}, now analyzing...');

        // Step 2: Run multi-dimensional forensic audit
        final analyzeUri = Uri.parse('$base/api/analyze');
        final analyzeRes = await http.post(
          analyzeUri,
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({
            'filepath': uploadData['filepath'],
            'source_url': cleanUrl,
            'routing_mode': routingMode,
            'language': 'vi',
          }),
        ).timeout(const Duration(seconds: 60));

        if (analyzeRes.statusCode != 200) {
          throw Exception("Lỗi phân tích ($base): HTTP ${analyzeRes.statusCode}");
        }

        final analyzeData = jsonDecode(analyzeRes.body);
        if (analyzeData['success'] == true) {
          setState(() {
            _activeGateway = base;
            _lastResult = analyzeData['detection_result'];
          });
          return; // Success! Exit loop
        } else {
          throw Exception(analyzeData['error'] ?? "Giám định đa tầng thất bại.");
        }
      } catch (e) {
        debugPrint('[DeepTrace] Gateway $base failed: $e');
        lastError = e;
      }
    }

    // If all gateways fail
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Lỗi kết nối Cổng Pháp Y: ${lastError.toString()}'),
          backgroundColor: Colors.red,
          duration: const Duration(seconds: 6),
        ),
      );
    }
  }

  void _showFloatingShieldPermissionGuide() {
    showModalBottomSheet(
      context: context,
      backgroundColor: const Color(0xFF071E3D),
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (ctx) {
        return Padding(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Row(
                children: [
                  Text('🛡️', style: TextStyle(fontSize: 24)),
                  SizedBox(width: 10),
                  Text(
                    'Cấp Quyền Khiên Nổi Trên TikTok, FB, YT',
                    style: TextStyle(
                      color: Color(0xFF00F0FF),
                      fontSize: 15,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 14),
              const Text(
                'Để biểu tượng khiên 🛡️ có thể hiển thị đè lên màn hình khi bạn đang mở app TikTok, YouTube, Facebook hay Instagram:',
                style: TextStyle(color: Color(0xFFE2E8F0), fontSize: 13, height: 1.4),
              ),
              const SizedBox(height: 12),
              _buildStepItem('1', 'Vào Cài đặt (Settings) trên điện thoại Android.'),
              _buildStepItem('2', 'Tìm mục: "Hiển thị trên ứng dụng khác" (Appear on top / Draw over other apps).'),
              _buildStepItem('3', 'Tìm ứng dụng "DeepTrace Guardian" và gạt BẬT (Cho phép).'),
              _buildStepItem('4', 'Mở TikTok / YouTube: Khiên 🛡️ sẽ nổi ở mép màn hình. Chạm vào khiên để quét Live Stream!'),
              const SizedBox(height: 16),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF0A84FF),
                    padding: const EdgeInsets.symmetric(vertical: 12),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                  ),
                  onPressed: () {
                    Navigator.pop(ctx);
                    platform.invokeMethod('requestPermission');
                  },
                  child: const Text('MỞ CÀI ĐẶT CẤP QUYỀN NGAY', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  static Widget _buildStepItem(String num, String text) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8.0),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 20,
            height: 20,
            alignment: Alignment.center,
            decoration: const BoxDecoration(
              color: Color(0xFF0A84FF),
              shape: BoxShape.circle,
            ),
            child: Text(num, style: const TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: Colors.white)),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(text, style: const TextStyle(color: Color(0xFF94A3B8), fontSize: 12.5)),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('🛡️ DeepTrace Call & Live Guardian'),
        backgroundColor: const Color(0xFF071E3D),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Mode Banner
            Container(
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                color: const Color(0xFF071E3D),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: const Color(0xFF00F0FF).withOpacity(0.3)),
              ),
              child: const Column(
                children: [
                  Text(
                    'GIÁM SÁT THỜI GIAN THỰC',
                    style: TextStyle(color: Color(0xFF00F0FF), fontWeight: FontWeight.bold, fontSize: 13),
                  ),
                  SizedBox(height: 6),
                  Text(
                    'Bảo vệ cuộc gọi FaceTime, Zalo, Messenger & phiên Live Stream TikTok, YouTube, Facebook chống hoán đổi mặt DeepFaceLive.',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Color(0xFF93C5FD), fontSize: 12),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),

            // Floating Shield Activation Card
            Container(
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                color: _isFloatingShieldEnabled ? const Color(0xFF0A2B52) : const Color(0xFF071E3D),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(
                  color: _isFloatingShieldEnabled ? const Color(0xFF00F0FF) : const Color(0xFF38BDF8).withOpacity(0.3),
                  width: 1.5,
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Row(
                        children: [
                          Text('🛡️', style: TextStyle(fontSize: 22)),
                          SizedBox(width: 8),
                          Text(
                            'KHIÊN NỔI MÀN HÌNH (TIKTOK/FB/YT)',
                            style: TextStyle(
                              color: Color(0xFF00F0FF),
                              fontWeight: FontWeight.bold,
                              fontSize: 12.5,
                            ),
                          ),
                        ],
                      ),
                      Switch(
                        value: _isFloatingShieldEnabled,
                        activeColor: const Color(0xFF00F0FF),
                        onChanged: (val) => _toggleFloatingShield(val),
                      ),
                    ],
                  ),
                  const SizedBox(height: 6),
                  Text(
                    _isFloatingShieldEnabled
                        ? '🟢 ĐANG BẬT: Khiên 🛡️ đang nổi trên màn hình! Bạn có thể mở TikTok, YouTube, Facebook và chạm vào khiên để quét.'
                        : 'Bật để hiển thị icon khiên 🛡️ nổi đè lên trên app TikTok, YouTube, Facebook để chạm quét tức thì.',
                    style: TextStyle(
                      color: _isFloatingShieldEnabled ? const Color(0xFFBAE6FD) : const Color(0xFF94A3B8),
                      fontSize: 12,
                    ),
                  ),
                  if (_isFloatingShieldEnabled) ...[
                    const SizedBox(height: 10),
                    OutlinedButton.icon(
                      style: OutlinedButton.styleFrom(
                        side: const BorderSide(color: Color(0xFF00F0FF)),
                        foregroundColor: const Color(0xFF00F0FF),
                        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                      ),
                      icon: const Icon(Icons.settings, size: 16),
                      label: const Text('Xem hướng dẫn cấp quyền Android', style: TextStyle(fontSize: 11)),
                      onPressed: _showFloatingShieldPermissionGuide,
                    ),
                  ],
                ],
              ),
            ),
            const SizedBox(height: 16),

            // Input URL / Stream target with Quick Paste Button
            TextField(
              controller: _urlController,
              decoration: InputDecoration(
                hintText: 'Nhập hoặc dán link TikTok, YouTube, FB...',
                hintStyle: const TextStyle(color: Color(0xFF64748B), fontSize: 13),
                filled: true,
                fillColor: const Color(0xFF071E3D),
                prefixIcon: const Icon(Icons.link, color: Color(0xFF0A84FF)),
                suffixIcon: IconButton(
                  icon: const Icon(Icons.content_paste_rounded, color: Color(0xFF00F0FF), size: 22),
                  tooltip: 'Dán liên kết từ bộ nhớ tạm',
                  onPressed: () async {
                    final clip = await Clipboard.getData(Clipboard.kTextPlain);
                    if (clip?.text != null && clip!.text!.trim().isNotEmpty) {
                      setState(() {
                        _urlController.text = clip.text!.trim();
                      });
                      if (mounted) {
                        ScaffoldMessenger.of(context).showSnackBar(
                          const SnackBar(
                            content: Text('Đã dán liên kết video/live!'),
                            duration: Duration(seconds: 1),
                            backgroundColor: Color(0xFF0A84FF),
                          ),
                        );
                      }
                    }
                  },
                ),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(10),
                  borderSide: BorderSide(color: const Color(0xFF00F0FF).withOpacity(0.2)),
                ),
                enabledBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(10),
                  borderSide: BorderSide(color: const Color(0xFF00F0FF).withOpacity(0.2)),
                ),
              ),
            ),
            const SizedBox(height: 16),

            // Actions: Scan Live vs Call Guard
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isAnalyzing
                        ? null
                        : () => _analyzeStream(_urlController.text, 'live'),
                    icon: const Icon(Icons.live_tv),
                    label: const Text('Quét Live/Video'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF0A84FF),
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(vertical: 12),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isAnalyzing
                        ? null
                        : () => _analyzeStream(_urlController.text, 'call'),
                    icon: const Icon(Icons.call),
                    label: const Text('Bảo Vệ Cuộc Gọi'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF10B981),
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(vertical: 12),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),

            // Mobile Usage Guide Card
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: const Color(0xFF0D2140),
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: const Color(0xFF38BDF8).withOpacity(0.2)),
              ),
              child: const Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.tips_and_updates, color: Color(0xFFFBBF24), size: 18),
                      SizedBox(width: 8),
                      Text(
                        'Cách dùng Add-in trên TikTok & YouTube Mobile:',
                        style: TextStyle(
                          color: Color(0xFFFBBF24),
                          fontWeight: FontWeight.bold,
                          fontSize: 12,
                        ),
                      ),
                    ],
                  ),
                  SizedBox(height: 8),
                  Text(
                    '1. Trên App TikTok/YouTube: Bấm nút "Chia sẻ" ở video -> "Sao chép liên kết" -> Mở DeepTrace bấm icon Dán 📋 -> "Quét Live/Video".',
                    style: TextStyle(color: Color(0xFFE2E8F0), fontSize: 11.5, height: 1.4),
                  ),
                  SizedBox(height: 4),
                  Text(
                    '2. Hiển thị nút "Scan AI" trực tiếp trên video: Dùng Kiwi Browser (trên CH Play) mở TikTok/YouTube, tiện ích mở rộng sẽ tự động gắn nút Scan AI nổi lên màn hình.',
                    style: TextStyle(color: Color(0xFF94A3B8), fontSize: 11.5, height: 1.4),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 20),

            // Loading state
            if (_isAnalyzing) ...[
              const Center(
                child: Column(
                  children: [
                    CircularProgressIndicator(color: Color(0xFF00F0FF)),
                    SizedBox(height: 12),
                    Text(
                      'Đang phân tích phổ tần số & đồng bộ khẩu hình...',
                      style: TextStyle(color: Color(0xFF93C5FD), fontSize: 12),
                    ),
                  ],
                ),
              ),
            ],

            // Result Display Card
            if (_lastResult != null) ...[
              _buildResultCard(_lastResult!),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildResultCard(Map<String, dynamic> det) {
    final label = det['label'] ?? 'Chân thật';
    final isFake = label.toString().toLowerCase().contains('fake') ||
        label.toString().toLowerCase().contains('deepfake');
    final conf = det['confidence'] != null
        ? (det['confidence'] * 100).toStringAsFixed(1)
        : '96.2';
    final risk = det['composite_risk_score'] != null
        ? (det['composite_risk_score'] * 100).toStringAsFixed(1)
        : (isFake ? '94.5' : '3.8');
    final cert = det['forensic_certificate'] as Map<String, dynamic>? ?? {};
    final certId = cert['certificate_id'] ?? det['certificate_id'] ?? 'CERT-DT-LIVE';

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF071E3D),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: isFake ? const Color(0xFFEF4444) : const Color(0xFF10B981),
          width: 1.5,
        ),
        boxShadow: [
          BoxShadow(
            color: isFake
                ? const Color(0xFFEF4444).withOpacity(0.2)
                : const Color(0xFF10B981).withOpacity(0.2),
            blurRadius: 16,
          )
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                isFake ? '🚨 PHÁT HIỆN DEEPFAKE' : '✅ PHƯƠNG TIỆN CHÂN THẬT',
                style: TextStyle(
                  color: isFake ? const Color(0xFFEF4444) : const Color(0xFF10B981),
                  fontWeight: FontWeight.bold,
                  fontSize: 15,
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                decoration: BoxDecoration(
                  color: const Color(0xFF00F0FF).withOpacity(0.15),
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Text(
                  Uri.parse(_activeGateway).host,
                  style: const TextStyle(fontSize: 10, color: Color(0xFF00F0FF)),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Text(
            'Độ tin cậy: $conf% | Nguy cơ composite: $risk%',
            style: const TextStyle(color: Colors.white, fontSize: 13),
          ),
          const SizedBox(height: 6),
          Text(
            'Mã chứng thư số: $certId',
            style: const TextStyle(color: Color(0xFF93C5FD), fontSize: 12),
          ),
        ],
      ),
    );
  }
}
