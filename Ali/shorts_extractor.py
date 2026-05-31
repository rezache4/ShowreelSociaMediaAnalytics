"""
Standalone YouTube Shorts overlay extractor.

Run as its own process so Playwright's sync API gets a proper main-thread
event loop (Jupyter on Windows forces a SelectorEventLoop that cannot spawn
the browser subprocess -> NotImplementedError). Invoked by the notebook via:

    python shorts_extractor.py <input_csv> <output_csv>

Input  CSV columns : videoId, title, channelTitle, publishedAt, year
Output CSV columns : the full connection record per short.
"""
import sys
import io
import re
import json
import subprocess

import pandas as pd
from playwright.sync_api import sync_playwright

# Force UTF-8 stdout so emoji / accented titles don't crash on Windows cp1252
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

YT_CONSENT_COOKIE = {
    "name": "SOCS",
    "value": "CAISEwgDEgk0ODE3Nzk3MjQaAmVuIAEaBgiA_LyaBg",
    "domain": ".youtube.com",
    "path": "/",
}
MULTIFORMAT_SELECTOR = "a.ytReelMultiFormatLinkViewModelEndpoint"


def start_browser(playwright, headless=True):
    browser = playwright.chromium.launch(headless=headless)
    context = browser.new_context(
        user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"),
        locale="en-US",
    )
    context.add_cookies([YT_CONSENT_COOKIE])
    return browser, context


def parse_link_target(href):
    if not href:
        return None, None
    m = re.search(r"/watch\?v=([a-zA-Z0-9_-]{11})", href)
    if m:
        return "long", m.group(1)
    m = re.search(r"/shorts/([a-zA-Z0-9_-]{11})", href)
    if m:
        return "short", m.group(1)
    return None, None


def get_multiformat_link(page, short_id, timeout_ms=15000):
    url = f"https://www.youtube.com/shorts/{short_id}"
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        if "consent.youtube.com" in page.url:
            return None
        page.wait_for_selector(MULTIFORMAT_SELECTOR, state="attached", timeout=timeout_ms)
        rows = page.eval_on_selector_all(
            MULTIFORMAT_SELECTOR,
            "els => els.map(e => ({href: e.getAttribute('href'), "
            "label: e.getAttribute('aria-label')}))"
        )
        chosen = None
        for r in rows:
            lt, vid = parse_link_target(r.get("href"))
            if lt == "long":
                chosen = r
                break
        if chosen is None:
            for r in rows:
                lt, vid = parse_link_target(r.get("href"))
                if vid and vid != short_id:
                    chosen = r
                    break
        if chosen is None and rows:
            chosen = rows[0]
        if chosen is None:
            return None
        href = chosen.get("href")
        link_type, linked_video_id = parse_link_target(href)
        return {
            "href": href,
            "label": chosen.get("label"),
            "link_type": link_type,
            "linked_video_id": linked_video_id,
        }
    except Exception:
        return None


def download_video_json(video_id):
    try:
        cmd = ['yt-dlp', '--dump-json', '--skip-download',
               f'https://www.youtube.com/watch?v={video_id}']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception:
        pass
    return None


def extract_youtube_links_from_description(text, exclude_id=None):
    if not text:
        return []
    links = []
    patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?youtu\.be/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})'
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            vid_id = match.group(1)
            if exclude_id is None or vid_id != exclude_id:
                links.append(vid_id)
    return list(dict.fromkeys(links))


def main(input_csv, output_csv):
    shorts_df = pd.read_csv(input_csv)
    total = len(shorts_df)
    print(f"Processing {total} shorts...")

    results = []
    with sync_playwright() as pw:
        browser, context = start_browser(pw, headless=True)
        page = context.new_page()

        for idx, short in shorts_df.iterrows():
            video_id = str(short['videoId'])
            title = short.get('title', '')
            channel = short.get('channelTitle', '')
            published = short.get('publishedAt', 'N/A')
            year = short.get('year', 'N/A')
            short_url = f"https://www.youtube.com/shorts/{video_id}"

            print(f"[{idx+1}/{total}] {video_id} ({year}) {short_url}")

            link_info = get_multiformat_link(page, video_id)

            overlay_href = overlay_label = link_type = linked_video_id = None
            is_self_link = False
            if link_info:
                overlay_href = link_info['href']
                overlay_label = link_info['label']
                link_type = link_info['link_type']
                linked_video_id = link_info['linked_video_id']
                is_self_link = (linked_video_id == video_id)
                if link_type == 'long':
                    print(f"    LONG -> {linked_video_id}")
                elif link_type == 'short':
                    print(f"    SHORT ({'self' if is_self_link else 'related'}) -> {linked_video_id}")
            else:
                print("    no overlay")

            source_id_fallback = None
            if linked_video_id is None or is_self_link:
                vj = download_video_json(video_id)
                if vj:
                    links = extract_youtube_links_from_description(vj.get('description', ''),
                                                                   exclude_id=video_id)
                    if links:
                        source_id_fallback = links[0]

            if linked_video_id and not is_self_link:
                connected_id = linked_video_id
                connection_type = link_type
                detection_method = 'overlay_link'
            elif source_id_fallback:
                connected_id = source_id_fallback
                connection_type = 'long'
                detection_method = 'description'
            else:
                connected_id = None
                connection_type = 'none'
                detection_method = 'not_found'

            if connected_id is not None:
                base = "watch?v=" if connection_type == 'long' else "shorts/"
                connected_url = f"https://www.youtube.com/{base}{connected_id}"
                status = 'connected'
            else:
                connected_url = None
                status = 'no_connection'

            results.append({
                'year': year,
                'published_date': published,
                'short_id': video_id,
                'short_title': title,
                'short_channel': channel,
                'short_url': short_url,
                'overlay_href': overlay_href,
                'overlay_label': overlay_label,
                'overlay_link_type': link_type,
                'is_self_link': is_self_link,
                'connected_id': connected_id,
                'connected_url': connected_url,
                'connection_type': connection_type,
                'detection_method': detection_method,
                'status': status,
            })

        browser.close()

    pd.DataFrame(results).to_csv(output_csv, index=False, encoding='utf-8')
    print(f"DONE: wrote {len(results)} rows to {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python shorts_extractor.py <input_csv> <output_csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
