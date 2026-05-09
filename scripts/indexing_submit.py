#!/usr/bin/env python3
"""
Search engine indexing submission: Google Indexing API + Bing IndexNow.

Usage:
  # Submit to both Google and Bing
  python scripts/indexing_submit.py --credentials sa.json --bing-key YOUR_KEY

  # Bing IndexNow only (no extra auth, no quota limit)
  python scripts/indexing_submit.py --bing-only --bing-key YOUR_KEY

  # Google only
  python scripts/indexing_submit.py --credentials sa.json --google-only

  # Submit only recently changed files
  python scripts/indexing_submit.py --credentials sa.json --bing-key YOUR_KEY --changed-only

  # Submit index pages only (homepage, conference indexes, domain indexes)
  python scripts/indexing_submit.py --bing-only --bing-key YOUR_KEY --index-pages

Google prerequisites:
  1. Enable Indexing API in Google Cloud project
  2. Create a Service Account and download the JSON key
  3. Add the Service Account email as Owner in Google Search Console
  4. pip install google-auth google-auth-httplib2 google-api-python-client

Bing IndexNow prerequisites:
  1. Generate a key (any UUID)
  2. Place a {key}.txt file at the site root (content = the key itself)
  3. No extra dependencies needed
"""

import argparse
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

NS = "http://www.sitemaps.org/schemas/sitemap/0.9"
SITE_URL = "https://en.papernotes.org/"
HOST = "en.papernotes.org"
DAILY_QUOTA = 200
INDEXNOW_BATCH_SIZE = 10000  # IndexNow max 10,000 per request
PROGRESS_FILE = "logs/indexing_progress.json"


def parse_args():
    p = argparse.ArgumentParser(description="Google Indexing API + Bing IndexNow batch submit")
    p.add_argument("--credentials", default=None,
                   help="Path to Google Service Account JSON key")
    p.add_argument("--bing-key", default=None,
                   help="Bing IndexNow key")
    p.add_argument("--google-only", action="store_true",
                   help="Submit to Google only")
    p.add_argument("--bing-only", action="store_true",
                   help="Submit to Bing IndexNow only")
    p.add_argument("--sitemap", default=None,
                   help="Path to sitemap.xml (default: fetch from remote)")
    p.add_argument("--changed-only", action="store_true",
                   help="Submit only recently changed files via git diff")
    p.add_argument("--dry-run", action="store_true",
                   help="Print URLs without actually submitting")
    p.add_argument("--limit", type=int, default=DAILY_QUOTA,
                   help=f"Max URLs per Google submission (default: {DAILY_QUOTA})")
    p.add_argument("--index-pages", action="store_true",
                   help="Submit index pages only (homepage, conference/domain indexes)")
    p.add_argument("--action", choices=["URL_UPDATED", "URL_DELETED"],
                   default="URL_UPDATED",
                   help="Submission action type (default: URL_UPDATED)")
    return p.parse_args()


def load_progress() -> dict:
    """Load submission progress from disk."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"submitted": {}, "total_submitted": 0}


def save_progress(progress: dict):
    """Save submission progress to disk."""
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def get_urls_from_sitemap(sitemap_path: str) -> list[str]:
    """Parse all URLs from a local sitemap or sitemap index."""
    urls = []
    tree = ET.parse(sitemap_path)
    root = tree.getroot()

    # Check if this is a sitemap index
    sitemaps = root.findall(f"{{{NS}}}sitemap")
    if sitemaps:
        # Sitemap index: recursively parse each child sitemap
        site_dir = os.path.dirname(sitemap_path)
        for sm in sitemaps:
            loc = sm.find(f"{{{NS}}}loc")
            if loc is not None:
                filename = loc.text.split("/")[-1]
                sub_path = os.path.join(site_dir, filename)
                if os.path.exists(sub_path):
                    urls.extend(get_urls_from_sitemap(sub_path))
    else:
        # Regular sitemap
        for url_elem in root.findall(f"{{{NS}}}url"):
            loc = url_elem.find(f"{{{NS}}}loc")
            if loc is not None:
                urls.append(loc.text)

    return urls


def get_urls_from_remote_sitemap(sitemap_url: str) -> list[str]:
    """Fetch and parse URLs from a remote sitemap."""
    import urllib.request
    urls = []
    try:
        with urllib.request.urlopen(sitemap_url, timeout=30) as resp:
            content = resp.read().decode("utf-8")
        root = ET.fromstring(content)

        sitemaps = root.findall(f"{{{NS}}}sitemap")
        if sitemaps:
            for sm in sitemaps:
                loc = sm.find(f"{{{NS}}}loc")
                if loc is not None:
                    urls.extend(get_urls_from_remote_sitemap(loc.text))
        else:
            for url_elem in root.findall(f"{{{NS}}}url"):
                loc = url_elem.find(f"{{{NS}}}loc")
                if loc is not None:
                    urls.append(loc.text)
    except Exception as e:
        print(f"[ERROR] Failed to fetch remote sitemap: {e}")
    return urls


def get_changed_urls() -> list[str]:
    """Get URLs of recently changed paper pages via git diff."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD~1", "HEAD",
             "--", "docs/"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            print(f"[WARN] git diff failed: {result.stderr}")
            return []

        urls = []
        for line in result.stdout.strip().split("\n"):
            if not line or not line.endswith(".md"):
                continue
            # docs/ICLR2026/image_generation/xxx.md -> URL
            rel = line.replace("docs/", "")
            rel = rel.replace(".md", "/")
            if rel.endswith("index/"):
                rel = rel.replace("index/", "")
            url = SITE_URL + rel
            urls.append(url)

        return urls
    except Exception as e:
        print(f"[WARN] Failed to get changed files: {e}")
        return []


def filter_index_urls(urls: list[str]) -> list[str]:
    """Filter for index pages (homepage, conference indexes, domain indexes).
    Excludes paper detail pages and TODO pages."""
    filtered = []
    for url in urls:
        path = url.replace(SITE_URL, "").strip("/")
        # Homepage
        if not path:
            filtered.append(url)
            continue
        parts = path.split("/")
        # Exclude TODO pages
        if parts[-1] == "TODO":
            continue
        # Conference index: CVPR2025/ (depth 1)
        # Domain index: CVPR2025/object_detection/ (depth 2)
        if len(parts) <= 2:
            filtered.append(url)
            continue
    return filtered


def filter_paper_urls(urls: list[str]) -> list[str]:
    """Filter for paper note pages (exclude index/TODO pages)."""
    filtered = []
    for url in urls:
        path = url.replace(SITE_URL, "")
        # Exclude homepage, TODO, and pure index pages
        if not path or path == "/":
            continue
        parts = path.strip("/").split("/")
        if len(parts) < 3:
            continue  # Need at least conference/domain/paper
        if parts[-1] in ("TODO", "index"):
            continue
        filtered.append(url)
    return filtered


def submit_urls(urls: list[str], credentials_path: str, action: str,
                dry_run: bool = False) -> tuple[int, int]:
    """Submit URLs via Google Indexing API. Returns (success, fail) counts."""
    if dry_run:
        for url in urls:
            print(f"  [DRY-RUN] Google {action}: {url}")
        return len(urls), 0

    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ImportError:
        print("[ERROR] Missing dependencies, run:")
        print("   pip install google-auth google-auth-httplib2 google-api-python-client")
        sys.exit(1)

    SCOPES = ["https://www.googleapis.com/auth/indexing"]
    credentials = service_account.Credentials.from_service_account_file(
        credentials_path, scopes=SCOPES
    )
    service = build("indexing", "v3", credentials=credentials)

    success = 0
    fail = 0
    progress = load_progress()

    for i, url in enumerate(urls):
        try:
            body = {"url": url, "type": action}
            service.urlNotifications().publish(body=body).execute()
            success += 1
            progress["submitted"][url] = {
                "time": datetime.now().isoformat(),
                "action": action,
                "engine": "google",
            }
            progress["total_submitted"] = len(progress["submitted"])

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(urls)} (ok: {success}, fail: {fail})")
                save_progress(progress)

            # Rate limiting
            time.sleep(0.5)

        except Exception as e:
            fail += 1
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                print(f"  [WARN] API quota reached, submitted {success} URLs")
                break
            print(f"  [ERROR] Submit failed [{url}]: {error_msg}")

    save_progress(progress)
    return success, fail


def submit_urls_indexnow(urls: list[str], key: str,
                         dry_run: bool = False,
                         endpoint: str = "https://www.bing.com/indexnow") -> tuple[int, int]:
    """Submit URLs via IndexNow protocol (batch, up to 10,000 per request).

    By default submits to Bing's direct endpoint (faster than the api.indexnow.org
    aggregator, since Bing is our primary target). Use endpoint to switch.
    Per IndexNow spec, response 200 = received, 202 = received + key validation pending.
    Returns (success, fail) counts.
    """
    import urllib.error
    import urllib.request

    if dry_run:
        for url in urls[:5]:
            print(f"  [DRY-RUN] IndexNow: {url}")
        if len(urls) > 5:
            print(f"  [DRY-RUN] ... {len(urls)} URLs total")
        return len(urls), 0

    success = 0
    fail = 0
    key_location = f"https://{HOST}/{key}.txt"

    # Submit in batches (max 10,000 per batch)
    for batch_start in range(0, len(urls), INDEXNOW_BATCH_SIZE):
        batch = urls[batch_start:batch_start + INDEXNOW_BATCH_SIZE]
        payload = json.dumps({
            "host": HOST,
            "key": key,
            "keyLocation": key_location,
            "urlList": batch,
        }).encode("utf-8")

        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                status = resp.status
                body = resp.read(512).decode("utf-8", errors="replace").strip()
        except urllib.error.HTTPError as e:
            status = e.code
            body = e.read(512).decode("utf-8", errors="replace").strip() if e.fp else ""
        except Exception as e:
            fail += len(batch)
            print(f"  [ERROR] IndexNow submit failed: {e}")
            continue

        # 200 OK / 202 Accepted (key validation pending) both count as success
        if status in (200, 202):
            success += len(batch)
            note = "" if status == 200 else " (key validation pending)"
            print(f"  [OK] IndexNow batch submitted: {len(batch)} URLs (HTTP {status}){note}")
        else:
            fail += len(batch)
            # Diagnostic hints per IndexNow spec
            hint = {
                400: "Invalid format",
                403: "Key not valid (file missing or content mismatch)",
                422: "URL not under host, or key/keyLocation mismatch",
                429: "Too many requests",
            }.get(status, "")
            extra = f" - {hint}" if hint else ""
            body_snippet = f" body={body!r}" if body else ""
            print(f"  [WARN] IndexNow returned HTTP {status}{extra}{body_snippet}")

    return success, fail


def main():
    args = parse_args()

    # Determine submission engines
    do_google = not args.bing_only and args.credentials
    do_bing = not args.google_only and args.bing_key

    if not do_google and not do_bing:
        print("[ERROR] Must specify --credentials (Google) or --bing-key (Bing)")
        sys.exit(1)

    engines = []
    if do_google:
        engines.append("Google")
    if do_bing:
        engines.append("Bing IndexNow")
    print(f"Engines: {' + '.join(engines)}")

    # 1. Collect URLs
    if args.changed_only:
        print("Fetching changed files from git...")
        all_urls = get_changed_urls()
        print(f"  Changed pages: {len(all_urls)}")
    elif args.sitemap:
        print(f"Parsing local sitemap: {args.sitemap}")
        all_urls = get_urls_from_sitemap(args.sitemap)
        print(f"  Total URLs: {len(all_urls)}")
    else:
        print(f"Fetching remote sitemap: {SITE_URL}sitemap.xml")
        all_urls = get_urls_from_remote_sitemap(f"{SITE_URL}sitemap.xml")
        print(f"  Total URLs: {len(all_urls)}")

    # 2. Filter URLs
    if args.index_pages:
        target_urls = filter_index_urls(all_urls)
        label = "index pages"
    else:
        target_urls = filter_paper_urls(all_urls)
        label = "paper pages"
    print(f"  {label.capitalize()}: {len(target_urls)}")

    if not target_urls:
        print(f"[WARN] No {label} found to submit")
        return

    # === Bing IndexNow ===
    if do_bing:
        print(f"\n{'='*50}")
        print(f"Bing IndexNow: submitting {len(target_urls)} {label} (no quota limit)")
        if args.dry_run:
            print("[DRY-RUN mode]\n")
        bing_ok, bing_fail = submit_urls_indexnow(
            target_urls, args.bing_key, args.dry_run
        )
        print(f"  Bing result: ok={bing_ok}, fail={bing_fail}")

    # === Google Indexing API ===
    if do_google:
        # Google has quota, use checkpoint-based progress
        progress = load_progress()
        already_submitted = set(progress.get("submitted", {}).keys())
        pending = [u for u in target_urls if u not in already_submitted]
        print(f"\n{'='*50}")
        print(f"Google Indexing API:")
        print(f"  Pending: {len(pending)} (already submitted: {len(already_submitted)})")

        if not pending:
            print(f"  [OK] Google: all {label} submitted!")
        else:
            batch = pending[:args.limit]
            print(f"  This batch: {len(batch)} (limit: {args.limit})")
            if args.dry_run:
                print("  [DRY-RUN mode]\n")

            google_ok, google_fail = submit_urls(
                batch, args.credentials, args.action, args.dry_run
            )
            print(f"  Google result: ok={google_ok}, fail={google_fail}")
            remaining = len(pending) - len(batch)
            if remaining > 0:
                days = remaining // args.limit + 1
                print(f"  Remaining: {remaining}, ~{days} day(s) to finish")


if __name__ == "__main__":
    main()
