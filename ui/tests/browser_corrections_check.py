"""Real mouse selection and context-menu correction workflows against browser_app."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

EDF=Path(__file__).resolve().parents[2]/"tests/data/signals_01.edf"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url",default="http://localhost:6019")
    parser.add_argument("--artifacts",type=Path,default=Path("ui/test-results/corrections"))
    args=parser.parse_args();args.artifacts.mkdir(parents=True,exist_ok=True)
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=["--no-sandbox"])
        context=browser.new_context(viewport={"width":1540,"height":1200})
        page=context.new_page();errors=[]
        page.on("pageerror",lambda error:errors.append(str(error)))
        page.goto(args.url+"/ui/")
        expect(page.locator("#connection-status")).to_have_text("Verbunden")
        page.locator("#edf-file").set_input_files(EDF);page.locator("#open-infer").click()
        expect(page.locator("#viewer")).to_be_visible()
        expect(page.locator("#plot-loading")).to_be_hidden(timeout=30000)
        expect(page.locator("#task-select option")).to_have_count(3)
        expect(page.locator("#prediction-label")).to_have_text("N2")
        frame=next(f for f in page.frames if "/ui/plot/" in f.url)
        frame.wait_for_function("document.querySelector('.js-plotly-plot')?.layout.dragmode==='select'")
        session=json.loads(page.evaluate("sessionStorage.getItem('sleepwalker.session')"))["id"]
        path=f"{args.url}/api/v1/sessions/{session}"
        snapshot=lambda:context.request.get(path+"/annotations").json()
        original=context.request.get(path+"/predictions/sleep?start=0&end=120").json()

        def point(time,row=0):
            box=page.locator("#timeline").bounding_box()
            return box["x"]+126+time/120*(box["width"]-148),box["y"]+23+row*35

        def menu(time,row=0,click=False):
            x,y=point(time,row)
            if click:page.mouse.click(x,y)
            page.mouse.click(x,y,button="right")
            expect(page.locator("#selection-menu")).to_be_visible()

        def save(label=None):
            if label:page.locator("#correction-label").select_option(label)
            page.locator("#apply-correction").click()
            expect(page.locator("#correction-dialog")).not_to_be_visible()

        # Mark a signal interval with a real horizontal mouse drag.
        box=frame.locator(".nsewdrag").first.bounding_box()
        x=box["x"]+box["width"]*10/120;y=box["y"]+box["height"]/2
        page.mouse.move(x,y);page.mouse.down();page.mouse.move(x+box["width"]*60/120,y,steps=15);page.mouse.up()
        expect(page.locator("#selection-edit")).to_be_enabled()
        expect(frame.locator(".signal-interval")).to_be_visible()
        page.mouse.click(x+box["width"]*20/120,y,button="right")
        expect(page.locator("#selection-menu")).to_be_visible()
        page.screenshot(path=str(args.artifacts/"signal-menu.png"))
        page.locator("#context-edit").click()
        expect(page.locator("#correction-preview")).to_contain_text("3 Epochen")
        page.locator("#correction-cancel").click()
        assert not snapshot()["edits"]
        page.locator("#selection-edit").click()
        page.screenshot(path=str(args.artifacts/"interval-dialog.png"),full_page=True)
        save("n3")
        assert [(edit["start"],edit["label"]) for edit in snapshot()["labels"]]==[(0,"n3"),(30,"n3"),(60,"n3")]
        assert context.request.get(path+"/predictions/sleep?start=0&end=120").json()==original
        assert page.locator("#correction-section").count()==0

        # Click a corrected epoch: selected outline is drawn in the canvas.
        menu(15,click=True)
        expect(page.locator("#selection-range")).to_have_text("00:00:00 – 00:00:30")
        assert page.locator("#timeline").evaluate("""canvas=>{
            const data=canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data;
            let selected=0;for(let i=0;i<data.length;i+=4)if(data[i]===54&&data[i+1]===93&&data[i+2]===104)selected++;
            return selected>100;
        }""")
        page.locator("#context-edit").click();save("n1")
        assert snapshot()["labels"][0]["label"]=="n1"
        menu(15);page.locator("#context-delete").click()
        expect(page.locator("#effective-label")).to_contain_text("Manuell: N3")
        menu(15);page.locator("#context-delete").click()
        expect(page.locator("#effective-label")).to_have_text("Modellwahrscheinlichkeit")
        menu(15)
        expect(page.locator("#context-delete-prediction")).to_be_disabled()
        expect(page.locator("#context-limit")).to_be_visible()
        page.keyboard.press("Escape")
        expect(page.locator("#selection-menu")).to_be_hidden()
        page.keyboard.press("Shift+F10")
        expect(page.locator("#selection-menu")).to_be_visible()
        page.keyboard.press("Escape")

        # Mark an interval directly in the event row, then edit and delete it.
        x,y=point(40,1);end_x,_=point(50,1)
        page.mouse.move(x,y);page.mouse.down();page.mouse.move(end_x,y,steps=8);page.mouse.up()
        menu(45,1);page.locator("#context-edit").click()
        expect(page.locator("#correction-task")).to_have_value("arousal")
        page.locator("#correction-start").fill("40");page.locator("#correction-end").fill("50")
        save()
        first=snapshot()["events"][0]
        assert (first["start"],first["end"])==(40,50)
        menu(45,1,click=True);page.locator("#context-edit").click()
        page.locator("#correction-start").fill("42");page.locator("#correction-end").fill("54")
        save()
        events=snapshot()["events"]
        assert len(events)==1 and (events[0]["start"],events[0]["end"])==(42,54)
        assert events[0]["event_id"]!=first["event_id"]

        # A failed replacement retains the original and keeps the dialog open.
        menu(45,1);page.locator("#context-edit").click()
        page.locator("#correction-end").fill("56")
        def reject_replacement_delete(route):
            body=route.request.post_data_json or {}
            if body.get("operation")=="delete_event":
                route.fulfill(status=503,json={"error":{"message":"Test: Löschen fehlgeschlagen."}})
            else:route.continue_()
        page.route("**/annotations",reject_replacement_delete)
        page.locator("#apply-correction").click()
        expect(page.locator("#correction-error")).to_contain_text("Test: Löschen fehlgeschlagen.")
        page.unroute("**/annotations",reject_replacement_delete)
        events=snapshot()["events"]
        assert len(events)==1 and events[0]["end"]==54
        page.locator("#correction-cancel").click()
        menu(45,1);page.locator("#context-delete").click()
        expect(page.locator("#notice-text")).to_have_text("Änderung gelöscht.")
        assert not snapshot()["events"]

        # Continuous tasks also accept arbitrary human event intervals.
        x,y=point(72,2);end_x,_=point(79,2)
        page.mouse.move(x,y);page.mouse.down();page.mouse.move(end_x,y,steps=8);page.mouse.up()
        menu(75,2);page.locator("#context-edit").click()
        save("desaturation")
        assert any(event["task"]=="desaturation" for event in snapshot()["events"])
        # Expanded class rows use the corrected class for hit testing too.
        box=page.locator("#timeline").bounding_box()
        page.mouse.click(box["x"]+30,box["y"]+23)
        menu(105,2,click=True)
        expect(page.locator("#selection-range")).to_have_text("00:01:30 – 00:02:00")
        expect(page.locator("#context-delete-prediction")).to_be_disabled()
        page.keyboard.press("Escape")
        page.screenshot(path=str(args.artifacts/"corrected-view.png"),full_page=True)
        # A saved interval can be removed in one action, including all its epochs.
        x,y=point(31,3);end_x,_=point(89,3)
        page.mouse.move(x,y);page.mouse.down();page.mouse.move(end_x,y,steps=8);page.mouse.up()
        menu(45,3)
        expect(page.locator("#context-delete")).to_have_text("2 Änderungen löschen")
        page.locator("#context-delete").click()
        expect(page.locator("#notice-text")).to_have_text("2 Änderungen gelöscht.")
        assert not snapshot()["labels"]
        assert context.request.get(path+"/predictions/sleep?start=0&end=120").json()==original
        # Wheel zoom also remains available in the default selection tool.
        box=frame.locator(".nsewdrag").first.bounding_box()
        page.mouse.move(box["x"]+box["width"]/2,box["y"]+box["height"]/2)
        page.mouse.wheel(0,-100)
        frame.wait_for_function("document.querySelector('.js-plotly-plot').layout.xaxis.range[1]-document.querySelector('.js-plotly-plot').layout.xaxis.range[0]<119")
        expect(page.locator("#select-tool")).to_have_attribute("aria-pressed","true")
        expect(frame.locator(".js-plotly-plot")).to_be_visible()
        assert not errors,errors
        context.request.delete(path)
        browser.close()
    print("Correction browser checks passed: signal and timeline intervals, epoch snapping, class selection and outline, context menu, cancel, multi-epoch correction, editing/deleting changes, event replacement and rollback, continuous events, unchanged model output, keyboard menu.")


if __name__=="__main__":main()
