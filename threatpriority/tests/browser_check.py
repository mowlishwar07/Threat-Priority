"""Browser checks use the local server; install dev dependencies to rerun."""
import json
import os
import re
from pathlib import Path
from playwright.sync_api import sync_playwright, expect
ROOT=Path(__file__).resolve().parents[1]
results=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=os.environ.get('TP_BROWSER_EXECUTABLE'),headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1100})
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(os.environ.get('TP_BASE_URL','http://127.0.0.1:5050'))
    expect(page.locator('#source')).to_contain_text('Synthetic')
    page.click('#demo')
    expect(page.locator('#total')).to_have_text('300')
    page.wait_for_selector('.eval-table')
    results.append('Demo queue and ranking evaluation displayed')
    page.locator('#events tr').first.click()
    page.wait_for_selector('.shap-row')
    assert page.locator('.shap-row').count()==9
    results.append('Nine genuine SHAP feature contributions rendered')
    page.get_by_role('button',name='Benign',exact=True).click()
    expect(page.locator('#notice')).to_contain_text('does not alter')
    results.append('Feedback updates review state')
    page.evaluate('window.scrollTo(0,0)')
    page.screenshot(path=str(ROOT/'reports'/'dashboard.png'),full_page=False)
    page.fill('#budget','10')
    page.click('#evaluate')
    expect(page.locator('.eval-table th').first).to_have_text('Budget K=10')
    results.append('Custom investigation budget evaluated')
    page.set_input_files('#file',str(ROOT/'data'/'demo_missing_columns.csv'))
    page.click('#upload')
    expect(page.locator('#notice')).to_contain_text('Missing columns imputed')
    page.locator('#events tr').first.click()
    page.wait_for_selector('.caution')
    assert 'dst_bytes' in page.locator('.caution').inner_text()
    results.append('Missing-column upload and imputation warning explained')
    page.set_input_files('#file',str(ROOT/'data'/'demo_unlabeled.csv'))
    page.click('#upload')
    expect(page.locator('#evaluation')).to_contain_text('ground-truth label')
    results.append('Unlabeled file scored without fabricated evaluation')
    page.fill('#search','ssh')
    assert all('ssh' in text for text in page.locator('#events tr').all_text_contents())
    page.fill('#search','')
    page.click('#next')
    assert '31' in page.locator('#pages').inner_text()
    results.append('Queue search and pagination')
    with page.expect_download() as info:
        page.get_by_role('link',name='Export CSV').click()
    assert info.value.suggested_filename=='ranked_alerts.csv'
    results.append('Ranked CSV export')
    page.click('#stream')
    expect(page.locator('#stream')).to_have_text('Stop replay')
    expect(page.locator('#total')).to_have_text('1')
    expect(page.locator('#total')).to_have_text(re.compile(r'^(?:[2-9]|[1-9][0-9]+)$'))
    page.click('#stream')
    assert page.locator('#stream').inner_text()=='Start recorded replay'
    results.append('Recorded replay start, event ingestion, and stop')
    current=int(page.locator('#total').inner_text())
    page.request.post(os.environ.get('TP_BASE_URL','http://127.0.0.1:5050')+'/api/events',headers={'X-Requested-With':'ThreatPriority'},data={'event':{'duration':1,'src_bytes':200,'dst_bytes':80,'protocol':'tcp','service':'http'}})
    expect(page.locator('#total')).to_have_text(str(current+1))
    results.append('Externally posted API event appears through automatic refresh')
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
    page.screenshot(path=str(ROOT/'reports'/'mobile.png'),full_page=True)
    results.append('390-pixel layout has no page overflow')
    page.set_viewport_size({'width':1440,'height':1100})
    page.evaluate("document.documentElement.style.fontSize='200%'")
    assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
    results.append('200% root font size has no page overflow')
    assert not errors,errors
    results.append('No JavaScript page errors')
    browser.close()
(ROOT/'reports'/'browser_checks.json').write_text(json.dumps({'passed':results,'page_errors':errors},indent=2))
print(json.dumps(results,indent=2))
