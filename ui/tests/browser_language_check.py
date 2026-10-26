"""Switch UI language without resetting sessions, selections or API identifiers."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import expect, sync_playwright

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://localhost:6019')
    args=parser.parse_args()
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1540,'height':1200})
        errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(args.url+'/ui/')
        expect(page.locator('#connection-status')).to_have_text('Verbunden')
        page.locator('#edf-file').set_input_files(ROOT/'tests/data/signals_01.edf')
        page.locator('#models input[value="test-unmatched"]').uncheck()
        selected=page.locator('#models input:checked').evaluate_all('(inputs)=>inputs.map(input=>input.value)')
        page.locator('#language-select').select_option('en')
        expect(page.locator('html')).to_have_attribute('lang','en')
        expect(page.locator('#connect')).to_have_text('Connect')
        expect(page.locator('#connection-status')).to_have_text('Connected')
        expect(page.locator('#file-name')).to_have_text('signals_01.edf')
        assert page.locator('#models input:checked').evaluate_all('(inputs)=>inputs.map(input=>input.value)')==selected
        assert not page.locator('button').evaluate_all("buttons=>buttons.some(button=>/[.…]{3}|…/.test(button.textContent))")
        page.locator('#help-open').click()
        expect(page.locator('#help-dialog h2')).to_have_text('Controls')
        page.keyboard.press('Escape')
        page.locator('#open-infer').click()
        expect(page.locator('#viewer')).to_be_visible()
        expect(page.locator('#plot-loading')).to_be_hidden(timeout=30000)
        expect(page.locator('#task-select option')).to_have_count(3,timeout=30000)
        expect(page.locator('#prediction-label')).to_have_text('N2')
        expect(page.locator('#task-select option[value="sleep"]')).to_have_text('Sleep stage')
        page.locator('#goto-time').fill('00:02:00');page.locator('#goto-form button').click()
        frame=next(frame for frame in page.frames if '/ui/plot/' in frame.url)
        frame.wait_for_function("document.querySelector('.js-plotly-plot')._fullLayout.xaxis.range[0]===120")
        box=frame.locator('.nsewdrag').first.bounding_box()
        x,y=box['x']+box['width']*.15,box['y']+box['height']*.5
        page.mouse.move(x,y);page.mouse.down();page.mouse.move(x+box['width']*.3,y,steps=10);page.mouse.up()
        expect(page.locator('#selection-edit')).to_be_enabled()
        before=json.loads(page.evaluate("sessionStorage.getItem('sleepwalker.session')"))
        path=args.url+'/api/v1/sessions/'+before['id']
        page.locator('#language-select').select_option('de')
        expect(page.locator('#task-select option[value="sleep"]')).to_have_text('Schlafstadium')
        expect(page.locator('#selection-edit')).to_be_enabled()
        after=json.loads(page.evaluate("sessionStorage.getItem('sleepwalker.session')"))
        assert before==after,(before,after)
        page.locator('#language-select').select_option('en')
        page.locator('#selection-edit').click()
        expect(page.locator('#correction-dialog h2')).to_have_text('Correct selection')
        expect(page.locator('#correction-preview')).to_contain_text('epochs')
        page.locator('#correction-label').select_option('n3')
        page.locator('#apply-correction').click()
        expect(page.locator('#correction-dialog')).to_be_hidden()
        annotations=page.request.get(path+'/annotations').json()
        assert annotations['labels'] and all(label['label']=='n3' and label['task']=='sleep' for label in annotations['labels'])
        expect(page.locator('#history')).to_contain_text('Undo')
        page.reload()
        expect(page.locator('#viewer')).to_be_visible()
        expect(page.locator('#plot-loading')).to_be_hidden(timeout=30000)
        expect(page.locator('#language-select')).to_have_value('en')
        expect(page.locator('#close-open')).to_have_text('Close')
        assert json.loads(page.evaluate("sessionStorage.getItem('sleepwalker.session')"))['id']==before['id']
        page.locator('#language-select').select_option('de')
        expect(page.locator('#close-open')).to_have_text('Schließen')
        page.locator('#language-select').select_option('en')
        page.set_viewport_size({'width':800,'height':1000})
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        assert not errors, errors
        assert page.request.delete(path).status==204
        browser.close()
        print('Language checks passed: live DE/EN, unchanged file/model choices, session/zoom/selection, translated correction/history, original API labels, F5 persistence, compact layout and no button ellipses.')


if __name__=='__main__':
    main()
