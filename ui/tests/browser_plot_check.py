"""Exercise actual mouse navigation, delayed resampling and responsive sizing."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

EDF = Path(__file__).resolve().parents[1] / "assets/examples/sleepwalker-demo-690s.edf"
GRAPH = "document.querySelector('#signals .js-plotly-plot')"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:6019")
    parser.add_argument("--artifacts", type=Path, default=Path("ui/test-results/plot"))
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(viewport={"width": 1540, "height": 1300})
        page = context.new_page()
        errors, timings = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("requestfinished", lambda request: timings.append(request.timing["responseEnd"])
                if "_dash-update-component" in request.url else None)
        page.goto(args.url + "/ui/")
        expect(page.locator("#connection-status")).to_have_text("Verbunden")
        page.locator("#edf-file").set_input_files(EDF)
        page.locator("#open-signals").click()
        expect(page.locator("#viewer")).to_be_visible()
        expect(page.locator("#plot-loading")).to_be_hidden(timeout=30000)
        page.locator("#pan-tool").click()
        page.locator("#channels-open").click()
        for checkbox in page.locator("#channel-list input[type=checkbox]").all():
            checkbox.check()
        page.locator("#channels-apply").click()
        expect(page.locator("#plot-loading")).to_be_hidden(timeout=30000)
        frame = next(frame for frame in page.frames if "/ui/plot/" in frame.url)
        frame.wait_for_function(f"{GRAPH}?.data?.length === 8")
        frame.wait_for_function(f"{GRAPH}?.getBoundingClientRect().height >= 790")
        frame.evaluate("""() => {
            window.plotSizes = [];
            const monitor = () => {
                const graph = document.querySelector('#signals .js-plotly-plot');
                window.plotSizes.push(graph.getBoundingClientRect().height);
                window.plotSizeFrame = requestAnimationFrame(monitor);
            };
            monitor();
        }""")

        def settled():
            frame.wait_for_function(f"""(() => {{
                const g = {GRAPH}, range = g?.layout?.xaxis?.range;
                return g?.getBoundingClientRect().height >= 790 && g.data.length === 8 &&
                    g._fullData.every(t => t._length > 0 && t._length <= 2000) &&
                    g.data.every(t => Math.abs(t.x[0] - range[0]) < .1 &&
                        Math.abs(t.x[t.x.length-1] - range[1]) < .1);
            }})()""", timeout=10000)
            expect(frame.locator(".js-plotly-plot")).to_be_visible()

        def drag(dx, dy=0):
            box = frame.locator(".nsewdrag").first.bounding_box()
            x, y = box["x"] + box["width"] * .55, box["y"] + box["height"] * .5
            page.mouse.move(x, y)
            page.mouse.down()
            page.mouse.move(x + dx, y + dy, steps=8)
            page.mouse.up()

        # The original responsive Graph collapsed to 0 px after this update.
        # A deep zoom also needs original samples in every lower subplot.
        frame.evaluate(f"Plotly.relayout({GRAPH}, {{'xaxis.range': [120.12, 120.17]}})")
        frame.wait_for_function(f"{GRAPH}.data.every(t => t.x.length >= 4 && t.x.length <= 6)")
        settled()
        frame.evaluate(f"Plotly.relayout({GRAPH}, {{'xaxis.range': [120, 150]}})")
        expect(page.locator("#range-label")).to_have_text("00:02:00 – 00:02:30")
        settled()
        frame.locator(".js-plotly-plot").screenshot(path=str(args.artifacts / "zoom.png"))
        initial_y = frame.evaluate(f"{GRAPH}.layout.yaxis.range")
        drag(0, 18)
        page.wait_for_timeout(200)
        assert frame.evaluate(f"{GRAPH}.layout.yaxis.range") == initial_y

        # Alternate actual mouse drags with wheel zoom, including diagonal drags.
        for index in range(8):
            drag(-65 if index % 2 else 50, 12 if index % 2 else -12)
            settled()
            box = frame.locator(".nsewdrag").first.bounding_box()
            page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            page.mouse.wheel(0, -70 if index % 2 else 70)
            page.wait_for_timeout(220)
            settled()

        # Hold one callback response while a newer viewport arrives. No stale
        # status may roll the toolbar or prediction timeline back to the old range.
        frame.evaluate("""() => {
            const originalFetch = window.fetch;
            let hold = true;
            window.fetch = async (...args) => {
                const response = await originalFetch(...args);
                if (hold && String(args[0]).includes('_dash-update-component')) {
                    hold = false;
                    await response.clone().text();
                    await new Promise(resolve => { window.releasePlotResponse = resolve; });
                }
                return response;
            };
        }""")
        frame.evaluate(f"Plotly.relayout({GRAPH}, {{'xaxis.range': [200, 230]}})")
        frame.wait_for_function("typeof window.releasePlotResponse === 'function'")
        frame.evaluate(f"Plotly.relayout({GRAPH}, {{'xaxis.range': [320, 350]}})")
        expect(page.locator("#range-label")).to_have_text("00:05:20 – 00:05:50")
        settled()
        frame.evaluate("window.releasePlotResponse()")
        page.wait_for_timeout(400)
        expect(page.locator("#range-label")).to_have_text("00:05:20 – 00:05:50")
        settled()

        # Cursor updates must not redraw the figure or emit relayout events.
        frame.evaluate(f"window.cursorRelayouts = 0; {GRAPH}.on('plotly_relayout', () => window.cursorRelayouts++)")
        page.locator("#goto-time").fill("00:05:30")
        # Select time via the same parent message used by a timeline click.
        page.evaluate("""() => document.querySelector('#signal-frame').contentWindow.postMessage(
            {source:'sleepwalker-ui', type:'cursor', time:330}, location.origin)""")
        expect(frame.locator(".signal-cursor")).to_be_visible()
        assert frame.evaluate("window.cursorRelayouts") == 0

        # Responsive updates must retain a nonzero plot on narrow screens too.
        page.set_viewport_size({"width": 800, "height": 1200})
        page.wait_for_timeout(250)
        settled()
        page.set_viewport_size({"width": 1540, "height": 1300})
        page.wait_for_timeout(250)
        settled()
        frame.locator(".js-plotly-plot").screenshot(path=str(args.artifacts / "after-navigation.png"))
        heights = frame.evaluate("cancelAnimationFrame(window.plotSizeFrame); window.plotSizes")
        assert min(heights) >= 790, min(heights)
        assert not errors, errors
        result = {"checks": "8 channels, repeated mouse drag/wheel, vertical lock, stale response, cursor, resize",
                  "minimum_plot_height": min(heights), "max_callback_ms": round(max(timings), 1)}
        (args.artifacts / "result.json").write_text(json.dumps(result, indent=2))
        saved = json.loads(page.evaluate("sessionStorage.getItem('sleepwalker.session')"))
        context.request.delete(f"{args.url}/api/v1/sessions/{saved['id']}")
        print(json.dumps(result))
        browser.close()


if __name__ == "__main__":
    main()
