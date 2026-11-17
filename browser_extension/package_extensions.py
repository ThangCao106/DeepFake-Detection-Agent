#!/usr/bin/env python3
"""
package_extensions.py - Automated Browser Extension Packager for DeepTrace Shield
Generates 100% compliant, zero-warning packages for:
  1. Firefox Add-ons (AMO): strict_min_version 140.0, scripts fallback, zero warnings.
  2. Chrome Web Store: service_worker background, stripped gecko settings.
"""
import json
import os
import zipfile

def build_packages():
    ext_dir = os.path.dirname(os.path.abspath(__file__))
    manifest_path = os.path.join(ext_dir, 'manifest.json')

    with open(manifest_path, 'r', encoding='utf-8') as f:
        base_manifest = json.load(f)

    # 1. Firefox AMO Package: scripts background, strict_min_version 140.0, data_collection_permissions
    firefox_manifest = json.loads(json.dumps(base_manifest))
    firefox_manifest['background'] = {'scripts': ['background.js']}
    firefox_manifest['browser_specific_settings']['gecko']['strict_min_version'] = '140.0'
    firefox_manifest['browser_specific_settings']['gecko']['data_collection_permissions'] = {'required': ['none']}

    # 2. Chrome Web Store Package: service_worker only, remove gecko settings
    chrome_manifest = json.loads(json.dumps(base_manifest))
    chrome_manifest['background'] = {'service_worker': 'background.js'}
    if 'browser_specific_settings' in chrome_manifest:
        del chrome_manifest['browser_specific_settings']

    common_files = [
        'background.js',
        'content_script.js',
        'overlay.css',
        'popup.html',
        'popup.js',
        'icons/icon-16.png',
        'icons/icon-48.png',
        'icons/icon-128.png'
    ]

    def pack(zip_filename, manifest_dict):
        zip_path = os.path.join(ext_dir, zip_filename)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('manifest.json', json.dumps(manifest_dict, indent=2, ensure_ascii=False))
            for rel in common_files:
                abs_f = os.path.join(ext_dir, rel.replace('/', os.sep))
                zf.write(abs_f, rel)
        print(f"Packed: {zip_path} ({os.path.getsize(zip_path)} bytes)")

    pack('deeptrace-browser-extension-v1.2.1.zip', firefox_manifest)
    pack('deeptrace-firefox-addon-v1.2.1.zip', firefox_manifest)
    pack('deeptrace-chrome-extension-v1.2.1.zip', chrome_manifest)

if __name__ == '__main__':
    build_packages()
