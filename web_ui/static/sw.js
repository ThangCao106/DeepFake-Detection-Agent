// sw.js - Minimal PWA Service Worker for DeepTrace.io.vn (enables Mobile Share Target & Install)
const CACHE_NAME = 'deeptrace-pwa-v1';

self.addEventListener('install', (event) => {
    self.skipWaiting();
});

self.addEventListener('activate', (event) => {
    event.waitUntil(self.clients.claim());
});

self.addEventListener('fetch', (event) => {
    // Network-first pass-through so forensic API calls & SSR certificates are always live
    if (event.request.method !== 'GET') return;
    event.respondWith(fetch(event.request));
});

