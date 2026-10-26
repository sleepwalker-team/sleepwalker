"""Light/dark theme persists and recolors plots without changing session data."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import expect, sync_playwright

ROOT=Path(__file__).resolve().parents[2]
GRAPH="document.querySelector('.js-plotly-plot')"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://localhost:6019')
    parser.add_argument('--artifacts',type=Path,default=Path('/tmp/sleepwalker-theme-check'))
    args=parser.parse_args();args.artifacts.mkdir(parents=True,exist_ok=True)
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1540,'height':1200})
        errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(args.url+'/ui/')
        expect(page.locator('#connection-status')).to_have_text('Verbunden')
        expect(page.locator('#theme-toggle')).to_have_attribute('aria-pressed','false')
        page.locator('#theme-toggle').click()
        expect(page.locator('html')).to_have_attribute('data-theme','dark')
        expect(page.locator('#theme-toggle .theme-moon')).to_be_visible()
        expect(page.locator('#theme-toggle .theme-sun')).to_be_hidden()
        page.locator('#language-select').select_option('en')
        expect(page.locator('#theme-toggle')).to_have_attribute('title','Switch to light mode')
        page.screenshot(path=str(args.artifacts/'entry-dark.png'),full_page=True)
        page.reload()
        expect(page.locator('#connection-status')).to_have_text('Connected')
        expect(page.locator('html')).to_have_attribute('data-theme','dark')
        page.locator('#edf-file').set_input_files(ROOT/'tests/data/signals_01.edf')
        page.locator('#models input[value="test-unmatched"]').uncheck()
        page.locator('#open-infer').click()
        expect(page.locator('#viewer')).to_be_visible()
        expect(page.locator('#plot-loading')).to_be_hidden(timeout=30000)
        expect(page.locator('#prediction-label')).to_have_text('N2',timeout=30000)
        expect(page.locator('#prediction-loading')).to_be_hidden(timeout=30000)
        frame=next(f for f in page.frames if '/ui/plot/' in f.url)
        def colors(paper,ink):
            frame.wait_for_function(f"{GRAPH}?.layout.paper_bgcolor==='{paper}' && {GRAPH}?.data.length>0 && {GRAPH}.data.every(trace=>trace.line.color==='{ink}')")
            if paper=='#20262b':
                assert frame.evaluate(f"Object.entries({GRAPH}.layout).filter(([key])=>/^[xy]axis\\d*$/.test(key)).every(([,axis])=>axis.gridcolor==='#303a42')")
        colors('#20262b','#d5dfe5')
        page.locator('#goto-time').fill('00:02:00');page.locator('#goto-form button').click()
        frame.wait_for_function(f"{GRAPH}._fullLayout.xaxis.range[0]===120")
        expect(page.locator('#prediction-loading')).to_be_hidden(timeout=30000)
        frame.wait_for_function(f"{GRAPH}.data[0].x[0]>=120 && {GRAPH}.data[0].x.at(-1)<=240")
        box=frame.locator('.nsewdrag').first.bounding_box()
        x,y=box['x']+box['width']*.1,box['y']+box['height']*.5
        page.mouse.move(x,y);page.mouse.down();page.mouse.move(x+box['width']*.35,y,steps=8);page.mouse.up()
        expect(page.locator('#selection-edit')).to_be_enabled()
        before=json.loads(page.evaluate("sessionStorage.getItem('sleepwalker.session')"))
        data=frame.evaluate(f"JSON.stringify({GRAPH}.data.map(trace=>[trace.x,trace.y]))")
        page.screenshot(path=str(args.artifacts/'viewer-dark.png'),full_page=True)
        for dark in [False,True,False,True]:
            page.locator('#theme-toggle').click()
            colors('#20262b' if dark else '#fcfcfb','#d5dfe5' if dark else '#343c40')
            assert frame.evaluate(f"{GRAPH}._fullLayout.xaxis.range")==[120,240]
            expect(frame.locator('.signal-interval')).to_be_visible()
            assert frame.evaluate(f"JSON.stringify({GRAPH}.data.map(trace=>[trace.x,trace.y]))")==data
            assert json.loads(page.evaluate("sessionStorage.getItem('sleepwalker.session')"))==before
        # Native plot selection remains visible against the dark paper.
        assert frame.locator('.signal-interval').evaluate('(el)=>getComputedStyle(el).borderTopColor')=='rgb(237, 243, 245)'
        page.locator('#selection-edit').click()
        page.screenshot(path=str(args.artifacts/'dialog-dark.png'))
        assert page.locator('#correction-dialog').evaluate('(el)=>getComputedStyle(el).backgroundColor')=='rgb(32, 38, 43)'
        page.locator('#correction-label').select_option('n3');page.locator('#apply-correction').click()
        expect(page.locator('#correction-dialog')).to_be_hidden()
        page.reload()
        expect(page.locator('#viewer')).to_be_visible()
        expect(page.locator('#plot-loading')).to_be_hidden(timeout=30000)
        frame=next(f for f in page.frames if '/ui/plot/' in f.url)
        colors('#20262b','#d5dfe5')
        expect(page.locator('#theme-toggle')).to_have_attribute('aria-pressed','true')
        page.locator('#channels-open').click()
        for checkbox in page.locator('#channel-list input').all():checkbox.check()
        page.locator('#channels-apply').click()
        frame.wait_for_function(f"{GRAPH}.data.length===3")
        colors('#20262b','#d5dfe5')
        page.locator('#theme-toggle').click();colors('#fcfcfb','#343c40')
        page.screenshot(path=str(args.artifacts/'viewer-light.png'),full_page=True)
        page.set_viewport_size({'width':800,'height':1000})
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        assert not errors, errors
        assert page.request.delete(args.url+'/api/v1/sessions/'+before['id']).status==204
        browser.close()
        print('Theme checks passed: icons, language, persistence, actual Plotly colors, stable ranges/data/selection, corrections, F5 and compact layout.')


if __name__=='__main__':main()
