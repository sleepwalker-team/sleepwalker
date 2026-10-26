"""Reload, multiple recordings, separate tabs and stale session references."""

import argparse
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import expect, sync_playwright

EDF=Path(__file__).resolve().parents[2]/"tests/data/signals_01.edf"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url",default="http://localhost:6019")
    parser.add_argument("--artifacts",type=Path,default=Path("ui/test-results/sessions"))
    args=parser.parse_args();args.artifacts.mkdir(parents=True,exist_ok=True)
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=["--no-sandbox"])
        context=browser.new_context(viewport={"width":1540,"height":1200})
        page=context.new_page();errors=[]
        page.on("pageerror",lambda e:errors.append(str(e)))

        def card(session_id):return page.locator(f'[data-session-id="{session_id}"]')
        def visible_plot(target=page):
            expect(target.locator("#viewer")).to_be_visible(timeout=30000)
            expect(target.locator("#plot-loading")).to_be_hidden(timeout=30000)
            frame=next(f for f in target.frames if "/ui/plot/" in f.url)
            expect(frame.locator(".js-plotly-plot")).to_be_visible()
            return frame
        def home():
            page.locator("#close-open").click();page.locator("#close-keep").click()
            expect(page.locator("#entry")).to_be_visible()
            assert "session" not in parse_qs(urlparse(page.url).query)

        page.goto(args.url+"/ui/")
        expect(page.locator("#connection-status")).to_have_text("Verbunden")
        page.locator("#edf-file").set_input_files(EDF);page.locator("#open-signals").click()
        visible_plot()
        first=json.loads(page.evaluate("sessionStorage.getItem('sleepwalker.session')"))["id"]
        assert parse_qs(urlparse(page.url).query)["session"]==[first]
        page.locator("#window-size").select_option("30")
        page.locator("#goto-time").fill("00:00:40");page.locator("#goto-form button").click()
        page.locator("#pan-tool").click()
        page.locator("#channels-open").click();page.locator("#channel-list input").nth(1).uncheck();page.locator("#channels-apply").click()
        visible_plot()
        page.reload()
        frame=visible_plot()
        expect(page.locator("#study-title")).to_have_text("signals_01.edf")
        expect(page.locator("#range-label")).to_have_text("00:00:40 – 00:01:10")
        expect(page.locator("#pan-tool")).to_have_attribute("aria-pressed","true")
        frame.wait_for_function("document.querySelector('.js-plotly-plot')?.data.length===1")
        assert json.loads(page.evaluate("sessionStorage.getItem('sleepwalker.session')"))["id"]==first

        # A separate tab can follow the URL with no preexisting sessionStorage.
        other=context.new_page();other.goto(page.url);visible_plot(other)
        expect(other.locator("#range-label")).to_have_text("00:00:40 – 00:01:10")
        assert context.request.get(f"{args.url}/api/v1/sessions/{first}").status==200
        home();page.reload()
        expect(page.locator("#entry")).to_be_visible()
        expect(card(first).locator('[data-action="join"]')).to_be_enabled()
        assert page.evaluate("sessionStorage.getItem('sleepwalker.session')") is None

        # Keep two recordings, and switch back through the recent list.
        page.locator("#edf-file").set_input_files({"name":"second-night.edf","mimeType":"application/octet-stream","buffer":EDF.read_bytes()})
        page.locator("#open-signals").click();visible_plot()
        second=json.loads(page.evaluate("sessionStorage.getItem('sleepwalker.session')"))["id"]
        assert first!=second
        home()
        expect(page.locator(".recent-session")).to_have_count(2)
        expect(card(first)).to_contain_text("signals_01.edf")
        expect(card(second)).to_contain_text("second-night.edf")
        card(first).locator('[data-action="join"]').click();visible_plot()
        expect(page.locator("#range-label")).to_have_text("00:00:40 – 00:01:10")
        # Leaving the first tab has not invalidated the second tab's view.
        other.reload();visible_plot(other);other.close()
        home()
        page.screenshot(path=str(args.artifacts/"recent-sessions.png"),full_page=True)

        # Forgetting is local only; deletion/expiration disables joining.
        card(first).locator('[data-action="forget"]').click()
        expect(card(first)).to_have_count(0)
        assert context.request.get(f"{args.url}/api/v1/sessions/{first}").status==200
        assert context.request.delete(f"{args.url}/api/v1/sessions/{second}").status==204
        page.locator("#refresh-sessions").click()
        expect(card(second)).to_contain_text("Nicht mehr verfügbar")
        expect(card(second).locator('[data-action="join"]')).to_be_disabled()

        # Direct URL joins also work when this browser has forgotten the entry.
        page.goto(f"{args.url}/ui/?session={first}");visible_plot()
        expect(page.locator("#viewer")).to_be_visible()
        # A temporary outage preserves the URL and reference for a later retry.
        page.route(f"**/api/v1/sessions/{first}",lambda route:route.abort())
        page.reload()
        expect(page.locator("#notice-text")).to_contain_text("Verbindung zum Server unterbrochen")
        assert parse_qs(urlparse(page.url).query)["session"]==[first]
        expect(card(first).locator('[data-action="join"]')).to_be_enabled()
        page.unroute(f"**/api/v1/sessions/{first}")
        page.reload();visible_plot()

        # An expired session returns to the start page once, without a reload loop.
        assert context.request.delete(f"{args.url}/api/v1/sessions/{first}").status==204
        page.reload()
        expect(page.locator("#entry")).to_be_visible()
        expect(page.locator("#notice-text")).to_contain_text("abgelaufen oder wurde gelöscht")
        assert "session" not in parse_qs(urlparse(page.url).query)
        expect(card(first).locator('[data-action="join"]')).to_be_disabled()
        page.reload();expect(page.locator("#entry")).to_be_visible()
        expect(card(first).locator('[data-action="join"]')).to_be_disabled()
        page.set_viewport_size({"width":800,"height":1000})
        page.screenshot(path=str(args.artifacts/"compact-sessions.png"),full_page=True)
        if page.evaluate("document.documentElement.scrollWidth>innerWidth"):
            print(page.evaluate("[...document.querySelectorAll('body *')].filter(e=>e.getBoundingClientRect().right>innerWidth+1).map(e=>({tag:e.tagName,id:e.id,cls:e.className,width:e.getBoundingClientRect().width})).slice(0,20)"))
        assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
        assert not errors,errors
        browser.close()
    print("Session browser checks passed: F5 restores view and channels, direct URL, separate tabs, multiple recordings, explicit home stays home, local forgetting, expired sessions, temporary outage, retry and compact layout.")


if __name__=="__main__":main()
