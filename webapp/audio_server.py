#!/usr/bin/env python3
"""audio_server.py — minimal audio upload + playback webapp.

Reachable over Tailscale so you can listen from any device.

Usage:
    ./webapp/audio_server.py [--port 8080] [--dir /path/to/clips]

Then open http://TAILSCALE_HOST:8080 (or whatever your tailscale IP is).
"""

import argparse, html, os, sys, time, re
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from pathlib import Path

AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.opus', '.weba'}

def human_size(n):
    for unit in ['B','KB','MB','GB']:
        if n < 1024: return f'{n:.0f} {unit}'
        n /= 1024
    return f'{n:.1f} TB'

def wav_duration(path):
    try:
        import wave
        with wave.open(str(path),'rb') as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return None

def file_duration(path):
    suf = path.suffix.lower()
    if suf == '.wav':
        d = wav_duration(path)
        if d: return d
    return None

HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>audiocore clips</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; background: #0e0e10; color: #e4e4e7; margin: 0; padding: 24px; }}
  h1 {{ margin: 0 0 16px; font-size: 20px; font-weight: 600; }}
  h2 {{ margin: 24px 0 8px; font-size: 14px; font-weight: 600; color: #a0a0a8; text-transform: uppercase; letter-spacing: 0.06em; }}
  .topbar {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; }}
  .upload {{ background: #1a1a1d; border: 1px dashed #3f3f46; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
  .upload form {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
  input[type=file] {{ color: #a0a0a8; }}
  input[type=submit], button {{ background: #6366f1; color: white; border: 0; padding: 8px 14px; border-radius: 6px; font-size: 13px; cursor: pointer; font-weight: 500; }}
  input[type=submit]:hover, button:hover {{ background: #4f46e5; }}
  .del {{ background: #b91c1c; padding: 4px 10px; font-size: 12px; }}
  .del:hover {{ background: #991b1b; }}
  .clip {{ background: #1a1a1d; border-radius: 8px; padding: 12px 14px; margin-bottom: 10px; display: flex; align-items: center; gap: 14px; }}
  .clip-info {{ flex: 1; min-width: 0; }}
  .clip-name {{ font-weight: 500; font-size: 14px; word-break: break-all; }}
  .clip-meta {{ font-size: 12px; color: #71717a; margin-top: 2px; }}
  audio {{ width: 320px; max-width: 50vw; height: 32px; }}
  .empty {{ color: #71717a; font-style: italic; padding: 24px; text-align: center; }}
  a {{ color: #6366f1; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .ts {{ font-size: 11px; color: #52525b; }}
</style>
</head>
<body>
<div class="topbar">
  <h1>🎙️ audiocore clips</h1>
  <span class="ts">{host} · {clips_dir}</span>
</div>

<div class="upload">
  <form method="POST" action="/upload" enctype="multipart/form-data">
    <input type="file" name="file" accept="audio/*" multiple required>
    <input type="submit" value="Upload">
  </form>
</div>

<h2>{n_clips} clip{s}</h2>
{clips_html}

<h2>Combo: voice + emotional instruct + text → high-quality output</h2>
<div class="upload">
  <form method="POST" action="/generate">
    <input type="text" name="text" placeholder="Text to speak…" size="50" required><br>
    <input type="text" name="instruct" placeholder="Emotional instruct (optional): 'Whispering, with deep sadness.'" size="50"><br>
    <select name="voice">
{voice_options}
    </select>
    <input type="submit" value="Synthesize & upload">
  </form>
</div>

</body>
</html>
"""

CLIP_ITEM = """
<div class="clip">
  <audio controls preload="metadata" src="/raw/{name}"></audio>
  <div class="clip-info">
    <div class="clip-name">{name}</div>
    <div class="clip-meta">{size} · {duration}{ts}</div>
  </div>
  <form method="POST" action="/delete" onsubmit="return confirm('Delete {name_js}?');">
    <input type="hidden" name="name" value="{name}">
    <button class="del" type="submit">Delete</button>
  </form>
</div>
"""

def render_page(host, clips_dir):
    files = []
    for p in sorted(Path(clips_dir).iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)
    if not files:
        clips_html = '<div class="empty">No clips yet — upload some audio above.</div>'
    else:
        items = []
        for p in files:
            stat = p.stat()
            dur = file_duration(p)
            dur_str = f'{dur:.2f}s · ' if dur else ''
            ts = time.strftime('%Y-%m-%d %H:%M', time.localtime(stat.st_mtime))
            items.append(CLIP_ITEM.format(
                name=html.escape(p.name, quote=True),
                name_js=html.escape(p.name, quote=True),
                size=human_size(stat.st_size),
                duration=dur_str,
                ts=ts,
            ))
        clips_html = ''.join(items)
    # Discover .voice files at repo root + tests/fixtures/
    voice_files = []
    for d in [Path('.'), Path('tests/fixtures')]:
        if d.is_dir():
            voice_files += sorted(p.name for p in d.glob('*.voice'))
    # De-duplicate, keep order
    seen = set(); voice_files = [v for v in voice_files if not (v in seen or seen.add(v))]
    if not voice_files:
        voice_files = ['vivian.voice']
    voice_opts = '\n'.join(
        f'      <option value="{html.escape(v)}"{(" selected" if v == "vivian.voice" else "")}>{html.escape(v)}</option>'
        for v in voice_files)
    return HTML_PAGE.format(
        host=html.escape(host),
        clips_dir=html.escape(str(clips_dir)),
        n_clips=len(files),
        s='' if len(files) == 1 else 's',
        clips_html=clips_html,
        voice_options=voice_opts,
    )

def run_generate(text, clips_dir, audiocore_dir, instruct='', voice_name='vivian.voice', max_new_tokens=220):
    """Call qwen_voice apply to synthesize and drop the WAV into clips_dir.

    Full combo: speaker embedding (.voice) + emotional instruct + text →
    high-quality output. Documented in docs/PROFILING.md
    'Combo (speaker embedding + emotional instruct + text) — WORKING 2026-07-05'.
    """
    import subprocess, os
    out = Path(clips_dir) / f'tts_{int(time.time())}.wav'
    # Prefer the deployed /mnt/data GGUFs; fall back to the in-repo weights.
    deployed = Path('/mnt/data/models/audio/qwen3_tts/0.6b-base')
    model_dir = deployed if (deployed / 'qwen3_tts_talker.gguf').exists() else (
        Path(audiocore_dir) / 'weights/qwen3_tts/qwen3-tts-0.6b-base')
    env = dict(os.environ,
        QWEN3TTS_DIR=str(model_dir),
        QWEN3TTS_TALKER='qwen3_tts_talker.gguf',
        QWEN3TTS_PREDICTOR='qwen3_tts_predictor.gguf',
        QWEN3TTS_CODEC='tokenizer-f16.gguf',
    )
    voice = Path(audiocore_dir) / voice_name
    if not voice.exists():
        # Try tests/fixtures/ as a fallback (committed known-good .voice)
        voice = Path(audiocore_dir) / 'tests/fixtures/vivian_0.6b_base.voice'
    if not voice.exists():
        return False, f'{voice_name} not found at repo root or tests/fixtures/'
    # Prefer build-debug (latest); fall back to build/ if that's all there is.
    qv = Path(audiocore_dir) / 'build-debug/qwen_voice'
    if not qv.exists():
        qv = Path(audiocore_dir) / 'build/qwen_voice'
    cmd = [str(qv), 'apply',
        '--model-dir', str(model_dir),
        '--voice', str(voice),
        '--text', text,
        '--out', str(out),
        '--max-new-tokens', str(max_new_tokens),
        '--temperature', '0.7', '--top-p', '0.9',
    ]
    if instruct.strip():
        cmd += ['--instruct', instruct]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or not out.exists():
        return False, (r.stderr or r.stdout or 'unknown error')[-500:]
    return True, out.name

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write(f'[{self.headers.get("X-Forwarded-For", self.client_address[0])}] {fmt % args}\n')

    def do_GET(self):
        self._handle(send_body=True)

    def do_HEAD(self):
        self._handle(send_body=False)

    def _handle(self, send_body):
        parsed = urlparse(self.path)
        if parsed.path == '/':
            body = render_page(self.server.host_label, self.server.clips_dir).encode()
            self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body))); self.end_headers()
            if send_body: self.wfile.write(body)
            return
        if parsed.path.startswith('/raw/'):
            name = os.path.basename(parsed.path[5:])
            target = Path(self.server.clips_dir) / name
            if not target.is_file() or target.suffix.lower() not in AUDIO_EXTS:
                self.send_error(404); return
            data = target.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'audio/wav' if name.lower().endswith('.wav') else 'application/octet-stream')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Accept-Ranges', 'bytes')
            self.end_headers()
            if send_body: self.wfile.write(data)
            return
        self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        cl = int(self.headers.get('Content-Length', 0))
        if parsed.path == '/upload':
            ctype = self.headers.get('Content-Type','')
            if 'multipart/form-data' not in ctype:
                self.send_error(400, 'expected multipart'); return
            # Parse multipart manually (stdlib cgi is deprecated/removed in 3.13+)
            body = self.rfile.read(cl) if cl else b''
            saved = []
            for name, filename, payload in parse_multipart(body, ctype):
                if not filename or Path(filename).suffix.lower() not in AUDIO_EXTS:
                    continue
                safe = sanitize_filename(filename)
                dest = Path(self.server.clips_dir) / safe
                with open(dest, 'wb') as f: f.write(payload)
                saved.append(safe)
            self._redirect('/')
            sys.stderr.write(f'[upload] saved: {saved}\n')
            return
        if parsed.path == '/delete':
            body = self.rfile.read(cl).decode('utf-8','replace') if cl else ''
            qs = parse_qs(body)
            name = os.path.basename(qs.get('name',[''])[0])
            if name:
                target = Path(self.server.clips_dir) / name
                if target.is_file():
                    target.unlink()
                    sys.stderr.write(f'[delete] removed {name}\n')
            self._redirect('/'); return
        if parsed.path == '/generate':
            body = self.rfile.read(cl).decode('utf-8','replace') if cl else ''
            qs = parse_qs(body)
            text = qs.get('text',[''])[0]
            instruct = qs.get('instruct',[''])[0]
            voice_name = qs.get('voice',['vivian.voice'])[0] or 'vivian.voice'
            # Sanitize voice_name — must end in .voice and have no path separators
            if not re.match(r'^[A-Za-z0-9_\-]+\.voice$', voice_name):
                voice_name = 'vivian.voice'
            if not text.strip():
                self.send_error(400, 'empty text'); return
            sys.stderr.write(f'[generate] voice={voice_name!r} instruct={instruct!r} text={text!r}\n')
            ok, info = run_generate(text, self.server.clips_dir, self.server.audiocore_dir,
                                    instruct=instruct, voice_name=voice_name)
            if ok:
                sys.stderr.write(f'[generate] saved {info}\n')
            else:
                sys.stderr.write(f'[generate] FAILED: {info}\n')
            self._redirect('/'); return
        self.send_error(404)

    def _redirect(self, path):
        self.send_response(303); self.send_header('Location', path); self.end_headers()


def parse_multipart(body, ctype):
    """Yield (field_name, filename, payload_bytes) tuples."""
    m = re.search(r'boundary=(.+)$', ctype)
    if not m: return
    boundary = ('--' + m.group(1)).encode()
    parts = body.split(boundary)
    for part in parts:
        part = part.strip(b'\r\n')
        if not part or part == b'--': continue
        if b'\r\n\r\n' not in part: continue
        header_blob, _, payload = part.partition(b'\r\n\r\n')
        # strip trailing \r\n from payload
        if payload.endswith(b'\r\n'): payload = payload[:-2]
        header_str = header_blob.decode('utf-8','replace')
        name_m = re.search(r'name="([^"]+)"', header_str)
        file_m = re.search(r'filename="([^"]*)"', header_str)
        if not name_m: continue
        yield name_m.group(1), (file_m.group(1) if file_m else None), payload


def sanitize_filename(name):
    # Keep ascii alnum + . - _; replace spaces with _; prefix with ts to avoid collisions
    base = re.sub(r'[^A-Za-z0-9._-]', '_', name)
    return f'{int(time.time())}_{base}' if not base[0].isalnum() else base


def detect_tailscale_ip():
    try:
        import subprocess
        r = subprocess.run(['tailscale','ip','-4'], capture_output=True, text=True, timeout=3)
        if r.returncode == 0:
            ip = r.stdout.strip().split('\n')[0]
            if ip.startswith('100.'): return ip
    except Exception: pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--dir', default=None, help='clips directory (default: webapp/clips)')
    ap.add_argument('--bind', default='0.0.0.0')
    args = ap.parse_args()

    audiocore_dir = Path(__file__).resolve().parent.parent
    clips_dir = Path(args.dir) if args.dir else (Path(__file__).parent / 'clips')
    clips_dir.mkdir(parents=True, exist_ok=True)

    # Copy in the test outputs that already exist so they show up.
    for fn in ['voice_applied_emotional.wav', 'voice_ref.wav']:
        src = audiocore_dir / fn
        if src.exists():
            dest = clips_dir / fn
            if not dest.exists():
                try: dest.symlink_to(src)
                except OSError:
                    import shutil; shutil.copy2(src, dest)

    host = detect_tailscale_ip() or args.bind
    host_label = f'http://{host}:{args.port}'

    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    server.clips_dir = str(clips_dir)
    server.audiocore_dir = audiocore_dir
    server.host_label = host_label

    print(f'\n  🎙️  audiocore clips server')
    print(f'  ─────────────────────────────────────────')
    print(f'  Listening : http://{args.bind}:{args.port}')
    print(f'  Tailscale : {host_label}')
    print(f'  Clips dir : {clips_dir}')
    print(f'  (Ctrl-C to stop)\n')

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n  stopped.')

if __name__ == '__main__':
    main()
