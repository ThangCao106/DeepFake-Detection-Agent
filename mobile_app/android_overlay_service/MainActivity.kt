package vn.io.deeptrace.deeptrace_guard

import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.util.regex.Pattern

class MainActivity: FlutterActivity() {
    private val CHANNEL = "vn.io.deeptrace/overlay"
    private var sharedUrl: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        handleSendIntent(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleSendIntent(intent)
    }

    private fun handleSendIntent(intent: Intent?) {
        if (intent != null && Intent.ACTION_SEND == intent.action && "text/plain" == intent.type) {
            val text = intent.getStringExtra(Intent.EXTRA_TEXT)
            if (text != null) {
                val m = Pattern.compile("(https?://[\\w\\-\\.\\?\\=\\&\\%\\/\\#\\+\\@\\:\\;\\~]+)").matcher(text)
                if (m.find()) {
                    sharedUrl = m.group(1)?.trim()
                }
            }
        }
    }

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
                "getSharedUrl" -> {
                    val url = sharedUrl
                    sharedUrl = null // Clear after reading
                    result.success(url)
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
