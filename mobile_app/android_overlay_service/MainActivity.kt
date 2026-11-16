package vn.io.deeptrace.deeptrace_guard

import android.content.Intent
import android.net.Uri
import android.os.Build
import android.provider.Settings
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity: FlutterActivity() {
    private val CHANNEL = "vn.io.deeptrace/overlay"

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "checkPermission" -> {
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                        result.success(Settings.canDrawOverlays(this))
                    } else {
                        result.success(true)
                    }
                }
                "requestPermission" -> {
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                        if (!Settings.canDrawOverlays(this)) {
                            val intent = Intent(
                                Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                                Uri.parse("package:$packageName")
                            )
                            startActivity(intent)
                        }
                    }
                    result.success(true)
                }
                "isOverlayRunning" -> {
                    result.success(DeepTraceOverlayService.isRunning)
                }
                "startOverlay" -> {
                    try {
                        val intent = Intent(this, DeepTraceOverlayService::class.java)
                        startService(intent)
                        result.success(true)
                    } catch (e: Throwable) {
                        result.error("START_FAILED", e.message, null)
                    }
                }
                "stopOverlay" -> {
                    try {
                        val intent = Intent(this, DeepTraceOverlayService::class.java)
                        stopService(intent)
                        result.success(true)
                    } catch (e: Throwable) {
                        result.error("STOP_FAILED", e.message, null)
                    }
                }
                else -> result.notImplemented()
            }
        }
    }
}
