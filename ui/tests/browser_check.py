"""Run against `python -m ui.tests.browser_app`, with real EDF and Plotly in Chromium."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright, expect

EDF = Path(__file__).resolve().parents[2] / "tests" / "data" / "signals_01.edf"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:6019")
    parser.add_argument("--artifacts", type=Path, default=Path("ui/test-results"))
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(viewport={"width": 1540, "height": 1100}, accept_downloads=True)
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url + "/ui/")
        expect(page.locator("#connection-status")).to_have_text("Verbunden")
        page.locator("#edf-file").set_input_files(EDF)
        page.screenshot(path=str(args.artifacts / "entry.png"), full_page=True)
        page.locator("#open-infer").click()
        expect(page.locator("#viewer")).to_be_visible(timeout=30000)
        expect(page.locator("#plot-loading")).to_be_hidden(timeout=30000)
        expect(page.locator("#task-select option")).to_have_count(3, timeout=30000)
        expect(page.locator("#prediction-label")).to_have_text("N2", timeout=30000)
        expect(page.locator("#timeline-legend .legend-task")).to_have_count(3)
        # Verify the rendered design, including transparent space above a thin
        # event baseline, rather than only checking color constants in code.
        pixels=page.locator("#timeline").evaluate("""canvas=>{
            const c=canvas.getContext('2d'),dpr=devicePixelRatio;
            const at=(time,y)=>[...c.getImageData(Math.floor((126+time/120*(canvas.clientWidth-148))*dpr),Math.floor(y*dpr),1,1).data];
            return {sleep:at(10,17),event:at(17,52),gap:at(30,52),baseline:at(30,58)};
        }""")
        assert pixels["sleep"][:3]==[36,92,240],pixels
        assert pixels["event"][:3]==[244,121,22],pixels
        assert pixels["gap"][3]==0,pixels
        assert pixels["baseline"][:3]==[201,205,203],pixels
        expect(page.locator("#model-input-report")).to_be_visible()
        expect(page.locator("#model-input-issues")).to_contain_text("TEST · unmatched inputs")
        expect(page.locator("#model-input-issues")).to_contain_text("C3-M2, E1-M2, Chin EMG, ECG")
        expect(page.locator("#model-input-signals")).to_contain_text("EEG")
        page.locator("#infer-open").click()
        expect(page.locator('#inference-models input[value="test-unmatched"]')).to_be_disabled()
        expect(page.locator("#inference-start")).to_be_disabled()
        page.locator("#inference-dialog .dialog-heading button").click()
        frame = page.frame_locator("#signal-frame")
        expect(frame.locator(".js-plotly-plot")).to_be_visible()
        def timeline_point(time, row=0, bounds=(0,120)):
            box=page.locator("#timeline").bounding_box()
            return (box["x"]+126+(time-bounds[0])/(bounds[1]-bounds[0])*(box["width"]-148), box["y"]+23+row*35)

        x,y=timeline_point(15)
        page.mouse.click(x,y)
        page.mouse.click(x,y,button="right")
        page.locator("#context-edit").click()
        page.locator("#correction-label").select_option("n3")
        page.locator("#apply-correction").click()
        expect(page.locator("#correction-dialog")).not_to_be_visible()
        expect(page.locator("#effective-label")).to_contain_text("Manuell: N3")
        expect(page.locator("#prediction-label")).to_have_text("N2")
        page.screenshot(path=str(args.artifacts / "viewer.png"), full_page=True)

        # Verify model output remains unchanged, not merely the displayed label.
        saved = json.loads(page.evaluate("sessionStorage.getItem('sleepwalker.session')"))
        session_id = saved["id"]
        original = context.request.get(f"{args.url}/api/v1/sessions/{session_id}/predictions/sleep?start=0&end=30").json()
        assert original["predictions"][0]["label"] == "n2"
        page.locator("#history button").filter(has_text="Rückgängig").first.click()
        expect(page.locator("#effective-label")).to_have_text("Modellwahrscheinlichkeit")

        # Plotly zoom must synchronize the parent timeline and retain a small payload.
        plot_frame = next(f for f in page.frames if "/ui/plot/" in f.url)
        plot_frame.evaluate("Plotly.relayout(document.querySelector('.js-plotly-plot'), {'xaxis.range[0]':40,'xaxis.range[1]':70})")
        expect(page.locator("#range-label")).to_have_text("00:00:40 – 00:01:10")
        sample_count = plot_frame.evaluate("document.querySelector('.js-plotly-plot').data[0].x.length")
        assert sample_count <= 2000

        page.locator("#channels-open").click()
        page.locator("#channel-list input").nth(1).uncheck()
        page.locator("#channels-apply").click()
        expect(page.locator("#plot-loading")).to_be_hidden(timeout=30000)
        plot_frame.wait_for_function("document.querySelector('.js-plotly-plot')?.data?.length === 1")

        # Event and continuous tasks support separate human intervals.
        page.locator("#task-select").select_option("arousal")
        x,y=timeline_point(45,1,(40,70))
        end_x,_=timeline_point(49,1,(40,70))
        page.mouse.move(x,y);page.mouse.down();page.mouse.move(end_x,y,steps=8);page.mouse.up()
        page.mouse.click((x+end_x)/2,y,button="right")
        page.locator("#context-edit").click()
        page.locator("#correction-start").fill("45")
        page.locator("#correction-end").fill("49")
        page.locator("#apply-correction").click()
        expect(page.locator("#history")).to_contain_text("Ereignis: Arousal")
        page.locator("#goto-time").fill("00:00:46")
        page.locator("#goto-form button").click()
        x,y=timeline_point(46,1,(46,76))
        page.mouse.click(x+1,y,button="right")
        page.locator("#context-delete").click()
        expect(page.locator("#notice-text")).to_have_text("Änderung gelöscht.")
        assert not context.request.get(f"{args.url}/api/v1/sessions/{session_id}/annotations").json()["events"]

        page.locator("#task-select").select_option("desaturation")
        expect(page.locator("#prediction-confidence")).not_to_have_text("")

        # Full workspace download and import preserve predictions and history.
        page.locator("#workspace-export").click()
        with page.expect_download(timeout=30000) as download_info:
            page.locator("#export-start").click()
        workspace = args.artifacts / "study.sleepwalker.zip"
        download_info.value.save_as(workspace)
        assert workspace.stat().st_size > 1000
        page.locator("#export-dialog .dialog-heading button").click()
        page.locator("#close-open").click()
        page.locator("#close-keep").click()
        expect(page.locator("#entry")).to_be_visible()
        page.locator(f'[data-session-id="{session_id}"] [data-action="join"]').click()
        expect(page.locator("#viewer")).to_be_visible()
        expect(page.locator("#plot-loading")).to_be_hidden(timeout=30000)
        page.reload()
        expect(page.locator("#connection-status")).to_have_text("Verbunden")
        expect(page.locator("#viewer")).to_be_visible()
        expect(page.locator("#task-select option")).to_have_count(3)
        page.locator("#close-open").click()
        page.locator("#close-delete").click()
        expect(page.locator("#entry")).to_be_visible()
        assert context.request.get(f"{args.url}/api/v1/sessions/{session_id}").status == 404
        page.locator("#workspace-file").set_input_files(workspace)
        expect(page.locator("#viewer")).to_be_visible()
        expect(page.locator("#task-select option")).to_have_count(3, timeout=30000)
        expect(page.locator("#history")).to_contain_text("Rückgängig")
        expect(page.locator("#plot-loading")).to_be_hidden(timeout=30000)
        expect(page.locator("#prediction-loading")).to_be_hidden(timeout=30000)
        expect(page.locator("#prediction-label")).to_have_text("N2", timeout=30000)
        page.screenshot(path=str(args.artifacts / "restored.png"), full_page=True)
        # Check the compact layout does not create horizontal scrolling.
        page.set_viewport_size({"width": 800, "height": 1000})
        page.screenshot(path=str(args.artifacts / "compact.png"), full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        assert not errors, errors
        context.close()
        browser.close()
    print("Browser checks passed: model-specific channel errors, compatible models still run, EDF upload, inference status, real resampler zoom, channel changes, corrections, undo, event edits, workspace export/import, reload, deletion, compact layout.")


if __name__ == "__main__":
    main()
