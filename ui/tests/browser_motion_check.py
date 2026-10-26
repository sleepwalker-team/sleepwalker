"""Immediate panning, elastic edges, direct wheel zoom and reduced motion."""
import argparse
import json
import math
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[2]
GRAPH = "document.querySelector('.js-plotly-plot')"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:6019")
    args = parser.parse_args()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width":1540,"height":1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url + "/ui/")
        expect(page.locator("#connection-status")).to_have_text("Verbunden")
        page.locator("#edf-file").set_input_files(ROOT / "ui/assets/examples/sleepwalker-demo-690s.edf")
        page.locator("#open-signals").click()
        expect(page.locator("#viewer")).to_be_visible()
        expect(page.locator("#plot-loading")).to_be_hidden(timeout=30000)
        session = json.loads(page.evaluate("sessionStorage.getItem('sleepwalker.session')"))["id"]
        frame = next(f for f in page.frames if "/ui/plot/" in f.url)
        frame.wait_for_function(f"{GRAPH}?._fullLayout?.xaxis")
        page.locator("#pan-tool").click()
        frame.wait_for_function(f"{GRAPH}.layout.dragmode==='pan'")
        box = frame.locator(".nsewdrag").first.bounding_box()
        x,y = box["x"] + box["width"]*.5, box["y"] + box["height"]*.5
        page.mouse.move(x,y);page.mouse.down();page.mouse.move(x+200,y,steps=8)
        expect(frame.locator(".signal-edge")).to_be_visible()
        assert frame.evaluate(f"{GRAPH}._fullLayout.xaxis.range") == [0,120]
        assert frame.evaluate(f"""() => {{
            const gd={GRAPH}, a=gd._fullLayout.xaxis, edge=document.querySelector('.signal-edge');
            const rect=edge.getBoundingClientRect(), graphRect=gd.getBoundingClientRect();
            return Math.abs(rect.left-graphRect.left-a._offset)<1 && Math.abs(rect.width-a._length)<1
                && [...gd.querySelectorAll('.cartesianlayer,.gl-container')].every(el=>!el.style.transform);
        }}""")
        page.mouse.up()
        expect(frame.locator(".signal-edge")).to_be_hidden()
        frame.wait_for_function(f"[...{GRAPH}.querySelectorAll('.cartesianlayer,.gl-container')].every(el=>!el.style.transform)")
        # A drag changes the viewport before mouseup and before resampling.
        page.mouse.move(x,y);page.mouse.down();page.mouse.move(x-150,y,steps=8)
        frame.wait_for_function(f"{GRAPH}._fullLayout.xaxis.range[0]>5")
        assert abs(frame.evaluate(f"{GRAPH}._fullLayout.xaxis.range[1]-{GRAPH}._fullLayout.xaxis.range[0]")-120)<1e-6
        page.mouse.up();page.wait_for_timeout(650)
        # Wheel zoom reaches its target directly and does not coast afterward.
        page.mouse.move(x,y);page.mouse.wheel(0,-160)
        frame.wait_for_function(f"{GRAPH}._fullLayout.xaxis.range[1]-{GRAPH}._fullLayout.xaxis.range[0]<119")
        width = frame.evaluate(f"{GRAPH}._fullLayout.xaxis.range[1]-{GRAPH}._fullLayout.xaxis.range[0]")
        assert abs(width-120*math.exp(-.32))<1e-5, width
        expect(frame.locator(".signal-edge")).to_be_hidden()
        page.wait_for_timeout(350)
        assert abs(frame.evaluate(f"{GRAPH}._fullLayout.xaxis.range[1]-{GRAPH}._fullLayout.xaxis.range[0]")-width)<1e-5
        # Reversing the wheel changes direction without finishing an old animation.
        page.mouse.wheel(0,160)
        frame.wait_for_function(f"Math.abs({GRAPH}._fullLayout.xaxis.range[1]-{GRAPH}._fullLayout.xaxis.range[0]-120)<1e-5")
        page.wait_for_timeout(250)
        # Reduced motion omits the edge fade and pan inertia.
        page.emulate_media(reduced_motion="reduce")
        page.locator("#goto-time").fill("00:00:00");page.locator("#goto-form button").click()
        frame.wait_for_function(f"{GRAPH}._fullLayout.xaxis.range[0]===0")
        page.mouse.move(x,y);page.mouse.down();page.mouse.move(x+150,y,steps=5)
        expect(frame.locator(".signal-edge")).to_be_hidden()
        page.mouse.up();page.wait_for_timeout(250)
        assert not errors, errors
        assert page.request.delete(args.url + "/api/v1/sessions/" + session).status == 204
        browser.close()
        print("Motion checks passed: live pan, preserved width, aligned edge fade, direct wheel without drift and reduced motion; no browser errors.")


if __name__ == "__main__":
    main()
