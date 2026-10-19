"""
Site Probe - live health check of any website or web app, independent of the SRS.

    python engine/site_probe.py <url> <out_dir> [--headed] [--timeout-ms 30000]

Opens the URL in a real Chromium browser (desktop and mobile viewport), saves screenshots into
<out_dir> and prints one line:  PROBE_RESULT: {json}

The JSON holds: reachability + HTTP status, load time, a list of PASS / WARN / FAIL checks
(console errors, failed requests, broken internal links, missing alt text / form labels,
mobile overflow, security headers ...), a 0-100 health score, and an inventory of the live DOM
(forms, inputs, buttons, links) that the Test Design agent uses to ground its test cases.
"""
import argparse
import json
import os
import sys
import time
from urllib.parse import urljoin, urlparse

DESKTOP_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/124.0.0.0 Safari/537.36")

INVENTORY_JS = r"""() => {
  const text = (el) => (el.innerText || el.value || el.getAttribute('aria-label') || '').trim().replace(/\s+/g, ' ').slice(0, 60);
  const visible = (el) => { const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
  const labelOf = (el) => {
    if (el.labels && el.labels.length) return el.labels[0].innerText.trim().slice(0, 60);
    const by = el.getAttribute('aria-labelledby');
    if (by) { const n = document.getElementById(by); if (n) return n.innerText.trim().slice(0, 60); }
    return (el.getAttribute('aria-label') || el.getAttribute('title') || '').trim();
  };
  const skip = ['hidden', 'submit', 'button', 'image', 'reset'];
  const inputs = [...document.querySelectorAll('input, textarea, select')]
    .filter(e => !skip.includes((e.type || '').toLowerCase()) && visible(e))
    .map(e => ({tag: e.tagName.toLowerCase(), type: e.type || '', id: e.id || '', name: e.name || '',
                placeholder: e.placeholder || '', label: labelOf(e), required: !!e.required}));
  const buttons = [...document.querySelectorAll('button, input[type=submit], input[type=button], [role=button]')]
    .filter(visible).map(e => ({text: text(e), id: e.id || '', type: e.type || ''}));
  const links = [...document.querySelectorAll('a[href]')].map(a => ({text: text(a), href: a.href}));
  const imgs = [...document.images];
  return {
    title: document.title || '',
    lang: document.documentElement.lang || '',
    viewport_meta: !!document.querySelector('meta[name=viewport]'),
    headings: [...document.querySelectorAll('h1, h2, h3')].filter(visible).map(h => h.innerText.trim().slice(0, 80)).slice(0, 10),
    forms: document.forms.length,
    inputs: inputs,
    buttons: buttons,
    links: links,
    images_total: imgs.length,
    images_without_alt: imgs.filter(i => !i.hasAttribute('alt')).length,
    inputs_without_label: inputs.filter(i => !i.label).length,
    scroll_width: document.documentElement.scrollWidth,
    inner_width: window.innerWidth,
  };
}"""


def _first_line(err) -> str:
    return (str(err).strip().splitlines() or [type(err).__name__])[0][:200]


def probe(url: str, out_dir: str, headless: bool = True, timeout_ms: int = 30000) -> dict:
    from playwright.sync_api import sync_playwright

    os.makedirs(out_dir, exist_ok=True)
    result = {"url": url, "reachable": False, "status": None, "final_url": url, "title": "", "load_ms": None,
              "checks": [], "score": 0, "screenshots": {}, "inventory": {}, "error": None}
    checks = result["checks"]

    def add(name, status, detail=""):
        checks.append({"name": name, "status": status, "detail": detail})

    console_errors, page_errors, failed_requests = [], [], []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        try:
            ctx = browser.new_context(viewport={"width": 1280, "height": 720}, user_agent=DESKTOP_UA)
            page = ctx.new_page()
            page.on("console", lambda m: console_errors.append(m.text[:160]) if m.type == "error" else None)
            page.on("pageerror", lambda e: page_errors.append(_first_line(e)))
            page.on("requestfailed", lambda r: failed_requests.append(f"{r.method} {r.url[:110]} ({r.failure})"))
            page.on("response", lambda r: failed_requests.append(f"HTTP {r.status} {r.url[:110]}")
                    if r.status >= 400 and r.request.resource_type != "document" else None)

            started = time.time()
            try:
                response = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            except Exception as err:
                result["error"] = _first_line(err)
                add("Server responds", "FAIL", result["error"])
                return result
            for state, wait in (("load", 10000), ("networkidle", 4000)):
                try:
                    page.wait_for_load_state(state, timeout=wait)
                except Exception:
                    pass
            result["load_ms"] = int((time.time() - started) * 1000)
            result["reachable"] = True
            result["status"] = response.status if response else None
            result["final_url"] = page.url
            headers = {k.lower(): v for k, v in (response.headers.items() if response else [])}

            page.screenshot(path=os.path.join(out_dir, "site_desktop.png"))
            result["screenshots"]["desktop"] = os.path.join(out_dir, "site_desktop.png")
            try:
                page.screenshot(path=os.path.join(out_dir, "site_full.png"), full_page=True)
                result["screenshots"]["full_page"] = os.path.join(out_dir, "site_full.png")
            except Exception:
                pass

            inv = page.evaluate(INVENTORY_JS)
            result["title"] = inv["title"]

            # ---- checks ---------------------------------------------------------------------
            add("Server responds", "PASS", f"{url} answered")
            status = result["status"] or 0
            add("HTTP status", "PASS" if status < 400 else "FAIL", f"HTTP {status}")
            host = urlparse(result["final_url"]).hostname or ""
            if result["final_url"].startswith("https://"):
                add("Served over HTTPS", "PASS")
            elif host in ("localhost", "127.0.0.1"):
                add("Served over HTTPS", "PASS", "local development URL")
            else:
                add("Served over HTTPS", "WARN", "page is served over plain HTTP")
            add("Page title present", "PASS" if inv["title"] else "WARN", inv["title"] or "empty <title>")
            ms = result["load_ms"]
            add("Load time", "PASS" if ms < 3000 else ("WARN" if ms < 8000 else "FAIL"), f"{ms} ms")
            add("No JavaScript console errors", "PASS" if not console_errors else "WARN",
                f"{len(console_errors)} error(s), first: {console_errors[0]}" if console_errors else "")
            add("No uncaught page exceptions", "PASS" if not page_errors else "FAIL",
                f"{len(page_errors)}, first: {page_errors[0]}" if page_errors else "")
            add("No failed network requests", "PASS" if not failed_requests else "WARN",
                f"{len(failed_requests)} failed, first: {failed_requests[0]}" if failed_requests else "")
            add("Images have alt text", "PASS" if not inv["images_without_alt"] else "WARN",
                f"{inv['images_without_alt']} of {inv['images_total']} images missing alt" if inv["images_without_alt"]
                else f"{inv['images_total']} image(s)")
            add("Form fields are labelled", "PASS" if not inv["inputs_without_label"] else "WARN",
                f"{inv['inputs_without_label']} of {len(inv['inputs'])} fields have no label / aria-label"
                if inv["inputs_without_label"] else f"{len(inv['inputs'])} field(s)")
            add("Viewport meta tag", "PASS" if inv["viewport_meta"] else "WARN",
                "" if inv["viewport_meta"] else "no <meta name=viewport>: page is not mobile-ready")

            wanted = ["content-security-policy", "x-content-type-options", "referrer-policy"]
            if result["final_url"].startswith("https://"):
                wanted.append("strict-transport-security")
            if "frame-ancestors" not in headers.get("content-security-policy", ""):
                wanted.append("x-frame-options")
            missing = [h for h in wanted if h not in headers]
            add("Security headers", "PASS" if not missing else "WARN",
                ("missing: " + ", ".join(missing)) if missing else "all present")

            # ---- internal links ---------------------------------------------------------------
            origin = urlparse(result["final_url"]).netloc
            seen, internal = set(), []
            for link in inv["links"]:
                href = urljoin(result["final_url"], link["href"]).split("#")[0]
                if href.startswith("http") and urlparse(href).netloc == origin and href not in seen:
                    seen.add(href)
                    internal.append(href)
            broken = []
            for href in internal[:12]:
                try:
                    code = ctx.request.get(href, timeout=8000).status
                    if code >= 400:
                        broken.append(f"{code} {href[:90]}")
                except Exception as err:
                    broken.append(f"{_first_line(err)[:40]} {href[:70]}")
            add("Internal links resolve", "PASS" if not broken else "FAIL",
                f"checked {min(len(internal), 12)} of {len(internal)}" if not broken else f"{len(broken)} broken, first: {broken[0]}")

            # ---- mobile viewport ----------------------------------------------------------------
            try:
                device = p.devices.get("iPhone 13") or {"viewport": {"width": 390, "height": 844}}
                mctx = browser.new_context(**device)
                mpage = mctx.new_page()
                mpage.goto(result["final_url"], wait_until="domcontentloaded", timeout=timeout_ms)
                try:
                    mpage.wait_for_load_state("load", timeout=8000)
                except Exception:
                    pass
                mpage.screenshot(path=os.path.join(out_dir, "site_mobile.png"))
                result["screenshots"]["mobile"] = os.path.join(out_dir, "site_mobile.png")
                m = mpage.evaluate("() => [document.documentElement.scrollWidth, window.innerWidth]")
                add("Mobile layout (no horizontal scroll)", "PASS" if m[0] <= m[1] + 2 else "WARN",
                    "" if m[0] <= m[1] + 2 else f"content is {m[0]}px wide on a {m[1]}px screen")
                mctx.close()
            except Exception as err:
                add("Mobile layout (no horizontal scroll)", "WARN", f"could not run: {_first_line(err)}")

            result["inventory"] = {
                "title": inv["title"], "forms": inv["forms"], "headings": inv["headings"],
                "inputs": inv["inputs"][:25], "buttons": inv["buttons"][:20],
                "links": [{"text": l["text"], "href": l["href"]} for l in inv["links"][:15]],
            }
        finally:
            browser.close()

    points = {"PASS": 1.0, "WARN": 0.5, "FAIL": 0.0}
    result["score"] = round(100 * sum(points[c["status"]] for c in checks) / len(checks)) if checks else 0
    return result


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("out_dir")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--timeout-ms", type=int, default=30000)
    args = parser.parse_args()
    try:
        outcome = probe(args.url, args.out_dir, headless=not args.headed, timeout_ms=args.timeout_ms)
    except Exception as exc:
        outcome = {"url": args.url, "reachable": False, "error": _first_line(exc), "checks": [], "score": 0,
                   "screenshots": {}, "inventory": {}}
    print("PROBE_RESULT: " + json.dumps(outcome))
