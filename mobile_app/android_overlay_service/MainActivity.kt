package vn.io.deeptrace.deeptrace_guard

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.media.projection.MediaProjectionManager
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
    private val REQUEST_SCREEN_CAPTURE = 2001
    private var sharedUrl: String? = null
    private var pendingOverlayResult: MethodChannel.Result? = null

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

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_SCREEN_CAPTURE) {
            if (resultCode == Activity.RESULT_OK && data != null) {
                try {
                    val serviceIntent = Intent(this, DeepTraceOverlayService::class.java).apply {
                        putExtra("RESULT_CODE", resultCode)
                        putExtra("DATA_INTENT", data)
                    }
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                        startForegroundService(serviceIntent)
                    } else {
                        startService(serviceIntent)
                    }
                    pendingOverlayResult?.success(true)
                } catch (e: Throwable) {
                    pendingOverlayResult?.error("START_FAILED", e.message, null)
                }
            } else {
                try {
                    val fallbackIntent = Intent(this, DeepTraceOverlayService::class.java)
                    startService(fallbackIntent)
                    pendingOverlayResult?.success(true)
                } catch (e: Throwable) {
                    pendingOverlayResult?.error("CANCELLED", "User declined screen capture", null)
                }
            }
            pendingOverlayResult = null
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
                    sharedUrl = null
                    result.success(url)
                }
                "startOverlay" -> {
                    val mgr = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as? MediaProjectionManager
                    if (mgr != null) {
                        pendingOverlayResult = result
                        startActivityForResult(mgr.createScreenCaptureIntent(), REQUEST_SCREEN_CAPTURE)
                    } else {
                        try {
                            val intent = Intent(this, DeepTraceOverlayService::class.java)
                            startService(intent)
                            result.success(true)
                        } catch (e: Throwable) {
                            result.error("START_FAILED", e.message, null)
                        }
                    }
                }
                "isAudioGuardianRunning" -> {
                    result.success(DeepTraceOverlayService.isAudioGuardianRunning)
                }
                "toggleAudioGuardian" -> {
                    val enable = call.argument<Boolean>("enable") ?: false
                    if (enable) {
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                            if (checkSelfPermission(android.Manifest.permission.RECORD_AUDIO) != android.content.pm.PackageManager.PERMISSION_GRANTED) {
                                requestPermissions(arrayOf(android.Manifest.permission.RECORD_AUDIO), 2002)
                            }
                        }
                    }
                    val intent = Intent(this, DeepTraceOverlayService::class.java).apply {
                        putExtra("ACTION_AUDIO_GUARDIAN", enable)
                    }
                    startService(intent)
                    result.success(true)
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
