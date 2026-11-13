import 'dart:convert';
import 'package:flutter/material.dart';
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

class _CallAndLiveGuardianScreenState extends State<CallAndLiveGuardianScreen> {
  final TextEditingController _urlController = TextEditingController();
  bool _isAnalyzing = false;
  Map<String, dynamic>? _lastResult;

  final String apiBase = "https://deeptrace.io.vn";

  Future<void> _analyzeStream(String url, String routingMode) async {
    setState(() {
      _isAnalyzing = true;
      _lastResult = null;
    });

    try {
      // Step 1: Ingest stream via server
      final uploadRes = await http.post(
        Uri.parse('$apiBase/api/upload_url'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'url': url, 'routing_mode': routingMode}),
      );

      final uploadData = jsonDecode(uploadRes.body);
      if (uploadData['success'] != true || uploadData['filepath'] == null) {
        throw Exception(uploadData['error'] ?? "Không thể nạp luồng stream.");
      }

      // Step 2: Run forensic audit
      final analyzeRes = await http.post(
        Uri.parse('$apiBase/api/analyze'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'filepath': uploadData['filepath'],
          'source_url': url,
          'routing_mode': routingMode,
          'language': 'vi',
        }),
      );

      final analyzeData = jsonDecode(analyzeRes.body);
      if (analyzeData['success'] == true) {
        setState(() {
          _lastResult = analyzeData['detection_result'];
        });
      } else {
        throw Exception(analyzeData['error'] ?? "Giám định thất bại.");
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Lỗi: ${e.toString()}'), backgroundColor: Colors.red),
      );
    } finally {
      setState(() {
        _isAnalyzing = false;
      });
    }
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
            const SizedBox(height: 20),

            // Ingestion input
            TextField(
              controller: _urlController,
              decoration: InputDecoration(
                hintText: 'Nhập link Live Stream (TikTok/FB/YT) hoặc Call Link...',
                hintStyle: const TextStyle(color: Color(0xFF64748B), fontSize: 13),
                filled: true,
                fillColor: const Color(0xFF0F172A),
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(10), borderSide: BorderSide.none),
                prefixIcon: const Icon(Icons.link, color: Color(0xFF0A84FF)),
              ),
            ),
            const SizedBox(height: 12),

            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF0A84FF),
                      padding: const EdgeInsets.symmetric(vertical: 12),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                    ),
                    onPressed: _isAnalyzing
                        ? null
                        : () => _analyzeStream(_urlController.text.trim(), 'livestream'),
                    icon: const Icon(Icons.live_tv),
                    label: const Text('Quét Live Stream', style: TextStyle(fontWeight: FontWeight.bold)),
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: ElevatedButton.icon(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF059669),
                      padding: const EdgeInsets.symmetric(vertical: 12),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                    ),
                    onPressed: _isAnalyzing
                        ? null
                        : () => _analyzeStream(_urlController.text.trim(), 'call'),
                    icon: const Icon(Icons.call_end),
                    label: const Text('Bảo Vệ Cuộc Gọi', style: TextStyle(fontWeight: FontWeight.bold)),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 24),

            if (_isAnalyzing)
              const Center(
                child: Column(
                  children: [
                    CircularProgressIndicator(color: Color(0xFF00F0FF)),
                    SizedBox(height: 12),
                    Text('Đang phân tích vi mô khẩu hình, mép mặt & phổ tần số...', style: TextStyle(color: Colors.white70)),
                  ],
                ),
              ),

            if (_lastResult != null) _buildResultCard(_lastResult!),
          ],
        ),
      ),
    );
  }

  Widget _buildResultCard(Map<String, dynamic> det) {
    final double risk = (det['composite_risk_score'] as num?)?.toDouble() ?? 0.0;
    final bool isFake = risk >= 0.7318;
    final String certId = det['certificate_id'] ?? 'CERT-DT-LIVE';

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF071E3D),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: isFake ? Colors.red : Colors.green, width: 2),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                isFake ? '🚨 PHÁT HIỆN GIẢ MẠO!' : '✅ XÁC THỰC AN TOÀN',
                style: TextStyle(
                  color: isFake ? Colors.redAccent : Colors.greenAccent,
                  fontWeight: FontWeight.bold,
                  fontSize: 16,
                ),
              ),
              Text(
                'Rủi ro: ${(risk * 100).toStringAsFixed(1)}%',
                style: const TextStyle(fontWeight: FontWeight.bold, color: Colors.white),
              ),
            ],
          ),
          const Divider(color: Colors.white24, height: 20),
          Text('Mã chứng thư: $certId', style: const TextStyle(color: Color(0xFF00F0FF), fontFamily: 'monospace')),
          const SizedBox(height: 8),
          Text(
            det['auditable_verdict'] ?? (isFake ? 'SYNTHETIC_DETECTED' : 'AUTHENTIC_VERIFIED'),
            style: const TextStyle(color: Colors.white70, fontSize: 12),
          ),
        ],
      ),
    );
  }
}

