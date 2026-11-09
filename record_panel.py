"""Grava o PAINEL REAL (painel/index.html) como vídeo: o Chrome abre o painel, avança o vídeo fotograma a fotograma
(25 fps) e tira uma captura por fotograma, por isso o resultado é exatamente o que se vê no painel.
Uso: .venv/bin/python record_panel.py [saida.mp4]      (precisa do servidor do painel em http://localhost:8780)
"""
import glob, os, subprocess, sys, time
from playwright.sync_api import sync_playwright

OUT = sys.argv[1] if len(sys.argv) > 1 else "social/painel_video.mp4"
EXE = glob.glob(os.path.expanduser("~/Library/Caches/ms-playwright/chromium-*/chrome-mac-arm64/*/Contents/MacOS/*"))[0]   # Chromium instalado pelo playwright (macOS)
VW, VH, SCALE, FPS = 1440, 810, 1920 / 1440, 25
K = float(os.environ.get('REC_K', '1'))   # <1 encurta as cenas (teste rápido)

CSS = """
.pane{max-height:350px;overflow:hidden}
body{cursor:none}
#rec-cap{margin-top:14px;padding:18px 22px;background:var(--panel);border:1px solid var(--line);border-radius:12px}
#rec-cap b{display:block;font-weight:700;margin-bottom:6px}
#rec-cap span{color:#b8c4de;font-size:14.5px;line-height:1.5}
body.rec-play .vw .ctl{opacity:0!important}
#rec-over{position:fixed;inset:0;z-index:50;background:#0b0f17;display:none;align-items:center;justify-content:center;text-align:center}
#rec-over h1{font-size:46px;margin-bottom:10px}#rec-over p{color:#b8c4de;font-size:18px;margin:6px 0}
"""

def main():
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-c:v", "mjpeg", "-framerate", str(FPS), "-i", "-",
                           "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT], stdin=subprocess.PIPE)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=EXE, headless=True, args=["--autoplay-policy=no-user-gesture-required"])
        pg = b.new_page(viewport={"width": VW, "height": VH}, device_scale_factor=SCALE)
        pg.goto("http://localhost:8780/?present=1"); pg.wait_for_timeout(1500)
        pg.add_style_tag(content=CSS)
        pg.evaluate("async()=>{V.src='jogo_hd.mp4';await new Promise(r=>V.addEventListener('loadedmetadata',r,{once:true}))}")
        pg.evaluate("""()=>{const c=document.createElement('div');c.id='rec-cap';c.innerHTML='<b></b><span></span>';document.querySelector('.main').appendChild(c);
          const o=document.createElement('div');o.id='rec-over';document.body.appendChild(o);V.pause();document.querySelector('.vw').classList.add('paused')}""")
        pg.evaluate("async()=>{await Promise.all(S.idx.segments.map(s=>{ensure(s);return S.loading[s.file]}))}")

        def shot(n=1):
            for _ in range(n): ff.stdin.write(pg.screenshot(type="jpeg", quality=92))
        def caption(t, s): pg.evaluate("([t,s])=>{const c=document.getElementById('rec-cap');c.children[0].textContent=t;c.children[1].textContent=s}", [t, s])
        def tab(name): pg.evaluate("n=>document.querySelector(`[data-p=${n}]`).click()", name)
        def seek(t):
            pg.evaluate("""async t=>{V.currentTime=t;await new Promise(r=>V.addEventListener('seeked',r,{once:true}));
              await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))}""", t)
        def clip(t0, t1, cap, controls=False):
            pg.evaluate("c=>document.body.classList.toggle('rec-play',!c)", controls)
            pg.evaluate("v=>document.getElementById('rec-cap').style.display=v", 'block' if cap else 'none')
            if cap: caption(*cap)
            n = max(2, int((t1 - t0) * FPS * K))
            seek(t0); pg.wait_for_timeout(350)
            for i in range(n):
                seek(t0 + i / FPS); shot()
        def hold(secs, t=None, cap=None, controls=True):
            pg.evaluate("c=>document.body.classList.toggle('rec-play',!c)", controls)
            if cap: caption(*cap)
            if t is not None: seek(t)
            shot(max(1, int(secs * FPS * K)))
        def overlay(html, secs):
            pg.evaluate("h=>{const o=document.getElementById('rec-over');o.innerHTML=h;o.style.display='flex'}", html)
            shot(max(1, int(secs * FPS * K))); pg.evaluate("()=>{document.getElementById('rec-over').style.display='none'}")

        # --- cenas ---
        hold(2.5, 0.5, ("Football analysis dashboard · FC Porto 4–1 Napoli", "Built automatically from broadcast video: every player and the ball are tracked and drawn on a live 2D map."))
        clip(299.5, 306.0, ("48′ · Porto chance: cross into the box", "Players labelled frame by frame, ball path in orange, goalkeeper (GK) and referee (REF) identified."))
        clip(392.0, 399.0, ("63′ · Porto attack", "The 2D tactical map follows the same moment from above, with the offside line when it is meaningful."))
        clip(411.0, 418.0, ("64′ · Chance with identified players", "Where a close-up allows it, shirt numbers and names are read automatically and kept on the player."))
        clip(439.6, 448.4, None)
        clip(448.9, 460.5, None)
        overlay("<div><h1>Automatic football analysis</h1><p>Player & ball tracking · pitch calibration · live 2D tactical map · movement analysis</p>"
                "<p style='color:#8592ad'>YOLO + ByteTrack + homography + OCR · Python</p><p style='margin-top:40px;color:#64708c;font-size:15px'>Footage: DAZN. Analysis and 2D map: own work. No affiliation with the clubs or DAZN.</p></div>", 3.0)
        b.close()
    ff.stdin.close(); ff.wait()

if __name__ == "__main__":
    t = time.time(); main(); print("feito em", round(time.time() - t), "s ->", OUT)
