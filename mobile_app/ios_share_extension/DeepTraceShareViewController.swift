import UIKit
import Social
import MobileCoreServices
import UniformTypeIdentifiers

/**
 * DeepTrace iOS Action & Share Extension
 * Receives shared TikTok Live, YouTube Shorts, Facebook Video URLs or Call recording frames
 * and forwards directly to DeepTrace Forensic Engine (https://deeptrace.io.vn)
 */
class DeepTraceShareViewController: UIViewController {

    private let apiBase = "https://deeptrace.io.vn"

    override func viewDidLoad() {
        super.viewDidLoad()
        self.view.backgroundColor = UIColor(red: 4/255, green: 21/255, blue: 46/255, alpha: 0.95)
        setupUI()
        handleSharedItem()
    }

    private func setupUI() {
        let titleLabel = UILabel(frame: CGRect(x: 20, y: 80, width: view.bounds.width - 40, height: 40))
        titleLabel.text = "🛡️ DeepTrace.io.vn"
        titleLabel.textColor = .white
        titleLabel.font = UIFont.boldSystemFont(ofSize: 22)
        titleLabel.textAlignment = .center
        view.addSubview(titleLabel)

        let statusLabel = UILabel(frame: CGRect(x: 20, y: 130, width: view.bounds.width - 40, height: 60))
        statusLabel.tag = 101
        statusLabel.text = "Đang thẩm tra luồng trực tiếp / video..."
        statusLabel.textColor = UIColor(red: 0, green: 240/255, blue: 255/255, alpha: 1.0)
        statusLabel.font = UIFont.systemFont(ofSize: 14)
        statusLabel.textAlignment = .center
        statusLabel.numberOfLines = 2
        view.addSubview(statusLabel)
    }

    private func handleSharedItem() {
        guard let extensionItem = extensionContext?.inputItems.first as? NSExtensionItem,
              let itemProvider = extensionItem.attachments?.first else {
            self.extensionContext?.completeRequest(returningItems: [], completionHandler: nil)
            return
        }

        // Case 1: Shared URL (TikTok Live / YT Shorts / Zalo Web Link)
        if itemProvider.hasItemConformingToTypeIdentifier(UTType.url.identifier) {
            itemProvider.loadItem(forTypeIdentifier: UTType.url.identifier, options: nil) { [weak self] (item, error) in
                if let url = item as? URL {
                    self?.auditUrlStream(url.absoluteString)
                }
            }
        }
        // Case 2: Screenshot from active call / Video frame
        else if itemProvider.hasItemConformingToTypeIdentifier(UTType.image.identifier) {
            itemProvider.loadItem(forTypeIdentifier: UTType.image.identifier, options: nil) { [weak self] (item, error) in
                var targetImage: UIImage? = nil
                if let image = item as? UIImage {
                    targetImage = image
                } else if let imageUrl = item as? URL, let data = try? Data(contentsOf: imageUrl) {
                    targetImage = UIImage(data: data)
                }

                if let img = targetImage, let base64Str = img.jpegData(compressionQuality: 0.85)?.base64EncodedString() {
                    self?.auditImageBase64("data:image/jpeg;base64,\(base64Str)")
                }
            }
        }
    }

    private func auditUrlStream(_ targetUrl: String) {
        guard let endpoint = URL(string: "\(apiBase)/api/upload_url") else { return }
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body: [String: Any] = ["url": targetUrl, "routing_mode": "auto"]
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)

        URLSession.shared.dataTask(with: request) { [weak self] (data, response, error) in
            guard let data = data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let filepath = json["filepath"] as? String else {
                self?.updateStatus("Lỗi kết nối Cổng Forensic DeepTrace.", isError: true)
                return
            }

            self?.executeAnalyze(filepath: filepath, sourceUrl: targetUrl)
        }.resume()
    }

    private func auditImageBase64(_ base64: String) {
        let payload: [String: Any] = [
            "type": "base64",
            "imageBase64": base64,
            "routing_mode": "call",
            "language": "vi"
        ]
        sendToAnalyzeEndpoint(payload: payload)
    }

    private func executeAnalyze(filepath: String, sourceUrl: String) {
        let payload: [String: Any] = [
            "filepath": filepath,
            "source_url": sourceUrl,
            "language": "vi"
        ]
        sendToAnalyzeEndpoint(payload: payload)
    }

    private func sendToAnalyzeEndpoint(payload: [String: Any]) {
        guard let endpoint = URL(string: "\(apiBase)/api/analyze") else { return }
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: payload)

        URLSession.shared.dataTask(with: request) { [weak self] (data, response, error) in
            DispatchQueue.main.async {
                guard let data = data,
                      let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let det = json["detection_result"] as? [String: Any] else {
                    self?.updateStatus("Phân tích đa chiều thất bại.", isError: true)
                    return
                }

                let risk = det["composite_risk_score"] as? Double ?? 0.0
                let certId = det["certificate_id"] as? String ?? "CERT-DT-LIVE"
                let isFake = risk >= 0.7318

                let msg = isFake
                    ? "🚨 PHÁT HIỆN GIẢ MẠO! (Rủi ro: \(String(format: "%.1f", risk * 100))%)\nChứng thư: \(certId)"
                    : "✅ XÁC THỰC CHÂN THỰC (Độ tin cậy: \(String(format: "%.1f", (1.0 - risk) * 100))%)\nChứng thư: \(certId)"

                self?.updateStatus(msg, isError: isFake)

                DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
                    self?.extensionContext?.completeRequest(returningItems: [], completionHandler: nil)
                }
            }
        }.resume()
    }

    private func updateStatus(_ text: String, isError: Bool = false) {
        if let label = self.view.viewWithTag(101) as? UILabel {
            label.text = text
            label.textColor = isError ? .systemRed : .systemGreen
        }
    }
}
