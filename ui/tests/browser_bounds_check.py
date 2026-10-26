"""Hard recording boundaries during mouse gestures, including session changes."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT=Path(__file__).resolve().parents[2]
DEMO=ROOT/"ui/assets/examples/sleepwalker-demo-690s.edf"
OTHER=ROOT/"tests/data/signals_01.edf"
GRAPH="document.querySelector('.js-plotly-plot')"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url",default="http://localhost:6019")
    args=parser.parse_args()
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=["--no-sandbox"])
        context=browser.new_context(viewport={"width":1540,"height":1400})
        page=context.new_page();errors=[];sessions=[]
        page.on("pageerror",lambda e:errors.append(str(e)))
        page.goto(args.url+"/ui/")
        expect(page.locator("#connection-status")).to_have_text("Verbunden")

        def opened():
            expect(page.locator("#viewer")).to_be_visible()
            expect(page.locator("#plot-loading")).to_be_hidden(timeout=30000)
            return next(f for f in page.frames if "/ui/plot/" in f.url)

        def limits(duration):
            frame.wait_for_function(f"""Object.entries({GRAPH}?._fullLayout||{{}})
                .filter(([k,v])=>/^xaxis\\d*$/.test(k)).length>0 &&
                Object.entries({GRAPH}._fullLayout).filter(([k,v])=>/^xaxis\\d*$/.test(k))
                .every(([k,a])=>a.minallowed===0&&a.maxallowed==={duration}&&a.range[0]>=-1e-6&&a.range[1]<={duration}+1e-6)""")

        def bounds():return frame.evaluate(f"{GRAPH}._fullLayout.xaxis.range")

        def drag(dx, row=0):
            box=frame.locator(".nsewdrag").nth(row).bounding_box()
            x=box["x"]+box["width"]/2;y=box["y"]+box["height"]/2
            page.mouse.move(x,y);page.mouse.down();page.mouse.move(x+dx,y,steps=12);page.mouse.up()

        page.locator("#edf-file").set_input_files(DEMO);page.locator("#open-signals").click();frame=opened()
        sessions.append(json.loads(page.evaluate("sessionStorage.getItem('sleepwalker.session')"))["id"])
        page.locator("#channels-open").click()
        for checkbox in page.locator("#channel-list input[type=checkbox]").all():checkbox.check()
        page.locator("#channels-apply").click();opened()
        frame.wait_for_function(f"{GRAPH}?.data.length===8")
        limits(690)
        frame.evaluate(f"""() => {{
            window.boundViolations=[];
            {GRAPH}.on('plotly_relayouting', data=>{{
                for(const [key,value] of Object.entries(data)){{
                    if(/^xaxis\\d*\\.range\\[0\\]$/.test(key)&&value < -1e-6)window.boundViolations.push(data);
                    if(/^xaxis\\d*\\.range\\[1\\]$/.test(key)&&value > 690+1e-6)window.boundViolations.push(data);
                }}
            }});
        }}""")
        page.locator("#pan-tool").click()
        frame.wait_for_function(f"{GRAPH}.layout.dragmode==='pan'")
        drag(300)
        limits(690)
        assert bounds()==[0,120],bounds()
        expect(page.locator("#previous")).to_be_disabled()
        # The same limit applies to the bottom subplot.
        page.locator("#goto-time").fill("00:09:30");page.locator("#goto-form button").click()
        frame.wait_for_function(f"Math.abs({GRAPH}._fullLayout.xaxis.range[1]-690)<1e-6")
        drag(-300,7)
        limits(690)
        assert bounds()==[570,690],bounds()
        expect(page.locator("#next")).to_be_disabled()

        # Zooming out repeatedly cannot create padding outside the recording.
        box=frame.locator(".nsewdrag").first.bounding_box()
        page.mouse.move(box["x"]+box["width"]*.8,box["y"]+box["height"]/2)
        for _ in range(4):page.mouse.wheel(0,1200);page.wait_for_timeout(160)
        limits(690)
        page.locator("#window-size").select_option("all")
        frame.wait_for_function(f"{GRAPH}._fullLayout.xaxis.range[0]===0&&{GRAPH}._fullLayout.xaxis.range[1]===690")
        drag(250);drag(-250,7)
        assert bounds()==[0,690],bounds()
        page.locator("#select-tool").click()
        page.mouse.wheel(0,1200);page.wait_for_timeout(200)
        limits(690)
        assert not frame.evaluate("window.boundViolations"),frame.evaluate("window.boundViolations")

        # Selecting channels must retain the limits.
        page.locator("#channels-open").click()
        for checkbox in page.locator("#channel-list input[type=checkbox]").all()[1:]:checkbox.uncheck()
        page.locator("#channels-apply").click();opened();limits(690)

        # A different EDF updates the cap rather than retaining the old length.
        page.locator("#close-open").click();page.locator("#close-keep").click()
        page.locator("#edf-file").set_input_files(OTHER);page.locator("#open-signals").click();opened()
        sessions.append(json.loads(page.evaluate("sessionStorage.getItem('sleepwalker.session')"))["id"])
        limits(3600)
        page.locator("#goto-time").fill("00:30:00");page.locator("#goto-form button").click()
        frame.wait_for_function(f"{GRAPH}._fullLayout.xaxis.range[0]>=1800")
        page.locator("#close-open").click();page.locator("#close-keep").click()
        page.locator(f'[data-session-id="{sessions[0]}"] [data-action="join"]').click();opened();limits(690)
        assert bounds()==[0,690],bounds()
        assert not errors,errors
        for session in sessions:assert context.request.delete(f"{args.url}/api/v1/sessions/{session}").status==204
        browser.close()
    print("Bounds checks passed: native pan stops on both ends, all eight axes, repeated wheel zoom, full view, selection tool, channel changes and EDF lengths 690/3600 seconds; no overshoot during drag.")


if __name__=="__main__":main()
