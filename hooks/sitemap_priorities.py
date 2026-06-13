"""Post-process MkDocs sitemaps with PaperNotes SEO priorities.

MkDocs generates ``sitemap.xml`` after copying static files. This hook updates
that generated sitemap in-place and emits the two supplemental sitemaps used by
Search Console:

- sitemap.xml: all pages
- sitemap-sections.xml: home + conference/field index pages
- sitemap-focus.xml: home + conference/field index pages + featured papers
"""

from __future__ import annotations

from pathlib import Path
import re
import xml.etree.ElementTree as ET


NS = "http://www.sitemaps.org/schemas/sitemap/0.9"
FEATURED_PATTERNS = [
    r"\bbest\s+paper\b",
    r"\bbest\s+student\s+paper\b",
    r"\(\s*oral\s*\)",
    r"\boral\s+paper\b",
    r"\bspotlight\b",
    r"\bhighlight\b",
    r"最佳论文",
    r"口头报告",
    r"（口头）",
]


def _site_url(config) -> str:
    return str(config.get("site_url") or "").rstrip("/") + "/"


def _docs_dir(config) -> Path:
    return Path(config["docs_dir"]).resolve()


def _conference_dirs(config) -> set[str]:
    docs_dir = _docs_dir(config)
    conferences = set()
    for child in docs_dir.iterdir():
        if child.is_dir() and (child / "index.md").exists():
            conferences.add(child.name)
    return conferences


def _url_path(loc: str, site_url: str) -> str:
    if loc.startswith(site_url):
        return loc[len(site_url):].strip("/")
    return loc.strip("/")


def _level(loc: str, site_url: str, conferences: set[str]) -> str:
    path = _url_path(loc, site_url)
    if not path:
        return "home"

    parts = path.split("/")
    if parts[0] not in conferences:
        return "other"
    if len(parts) == 1:
        return "conf_index"
    if parts[-1].upper() == "TODO":
        return "other"
    if len(parts) == 2:
        return "domain_index"
    return "paper"


def _priority(level: str) -> str:
    return {
        "home": "1.0",
        "conf_index": "0.9",
        "domain_index": "0.8",
        "paper": "0.6",
    }.get(level, "0.3")


# Domain-index priority overrides that apply ONLY to sitemap-sections.xml.
# sitemap.xml and sitemap-focus.xml keep the generic domain_index priority (0.8).
# New conferences not listed here default to 0.8.
_SECTION_DOMAIN_PRIORITY_OVERRIDES = {
    "CVPR2026": "0.8",
    "ICML2026": "0.8",
    "ACL2026": "0.8",
    "ICLR2026": "0.7",
    "AAAI2026": "0.7",
    "NeurIPS2025": "0.7",
    "ECCV2024": "0.6",
    "ICCV2025": "0.6",
    "CVPR2025": "0.6",
    "ICML2025": "0.6",
    "ACL2025": "0.6",
}
_SECTION_DOMAIN_PRIORITY_DEFAULT = "0.8"  # for any new conference not listed above


def _section_domain_priority(conference: str) -> str:
    return _SECTION_DOMAIN_PRIORITY_OVERRIDES.get(
        conference, _SECTION_DOMAIN_PRIORITY_DEFAULT
    )


def _changefreq(level: str) -> str:
    return {
        "home": "daily",
        "conf_index": "weekly",
        "domain_index": "weekly",
        "paper": "monthly",
    }.get(level, "monthly")


def _set_child(url_elem: ET.Element, tag: str, text: str) -> None:
    qname = f"{{{NS}}}{tag}"
    existing = url_elem.find(qname)
    if existing is None:
        existing = ET.SubElement(url_elem, qname)
    existing.text = text


def _doc_path_for_url(loc: str, site_url: str, docs_dir: Path) -> Path | None:
    path = _url_path(loc, site_url)
    if not path:
        return docs_dir / "index.md"
    candidates = []
    if loc.endswith("/"):
        candidates.append(docs_dir / path / "index.md")
        candidates.append(docs_dir / f"{path}.md")
    else:
        candidates.append(docs_dir / f"{path}.md")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _is_featured_paper(loc: str, site_url: str, docs_dir: Path) -> bool:
    doc_path = _doc_path_for_url(loc, site_url, docs_dir)
    if not doc_path or not doc_path.exists():
        return False
    try:
        content = doc_path.read_text(encoding="utf-8").lower()
    except UnicodeDecodeError:
        return False
    return any(re.search(pattern, content, flags=re.I) for pattern in FEATURED_PATTERNS)


def _copy_url_elem(url_elem: ET.Element) -> ET.Element:
    clone = ET.Element(f"{{{NS}}}url")
    for child in list(url_elem):
        copied = ET.SubElement(clone, child.tag)
        copied.text = child.text
    return clone


def _write_urlset(path: Path, url_elems: list[ET.Element]) -> None:
    urlset = ET.Element(f"{{{NS}}}urlset")
    for url_elem in url_elems:
        urlset.append(_copy_url_elem(url_elem))
    tree = ET.ElementTree(urlset)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def on_post_build(config, **kwargs):
    ET.register_namespace("", NS)

    site_dir = Path(config["site_dir"]).resolve()
    sitemap_path = site_dir / "sitemap.xml"
    if not sitemap_path.exists():
        print(f"[sitemap_priorities] skip: {sitemap_path} not found")
        return

    site_url = _site_url(config)
    docs_dir = _docs_dir(config)
    conferences = _conference_dirs(config)

    tree = ET.parse(sitemap_path)
    root = tree.getroot()
    if root.tag.endswith("sitemapindex"):
        print("[sitemap_priorities] skip: sitemap.xml is a sitemap index")
        return

    all_url_elems = root.findall(f"{{{NS}}}url")
    section_elems: list[ET.Element] = []
    focus_elems: list[ET.Element] = []
    counts = {"home": 0, "conf_index": 0, "domain_index": 0, "paper": 0, "other": 0}

    for url_elem in all_url_elems:
        loc_elem = url_elem.find(f"{{{NS}}}loc")
        if loc_elem is None or not loc_elem.text:
            continue
        level = _level(loc_elem.text, site_url, conferences)
        counts[level] = counts.get(level, 0) + 1
        # Extract conference name for domain-level section priority overrides
        conf = (
            _url_path(loc_elem.text, site_url).split("/")[0]
            if level == "domain_index"
            else ""
        )
        _set_child(url_elem, "changefreq", _changefreq(level))
        _set_child(url_elem, "priority", _priority(level))

        if level in {"home", "conf_index", "domain_index"}:
            if level == "domain_index":
                # sitemap-sections.xml gets a conference-specific domain priority;
                # the shared url_elem (sitemap.xml / sitemap-focus.xml) keeps 0.8.
                section_clone = _copy_url_elem(url_elem)
                _set_child(section_clone, "priority", _section_domain_priority(conf))
                section_elems.append(section_clone)
            else:
                section_elems.append(url_elem)
            focus_elems.append(url_elem)
        elif level == "paper" and _is_featured_paper(loc_elem.text, site_url, docs_dir):
            focus_elems.append(url_elem)

    ET.indent(tree, space="  ")
    tree.write(sitemap_path, encoding="utf-8", xml_declaration=True)
    _write_urlset(site_dir / "sitemap-sections.xml", section_elems)
    _write_urlset(site_dir / "sitemap-focus.xml", focus_elems)

    print(
        "[sitemap_priorities] "
        f"home={counts.get('home', 0)}, "
        f"conference={counts.get('conf_index', 0)}, "
        f"field={counts.get('domain_index', 0)}, "
        f"paper={counts.get('paper', 0)}, "
        f"other={counts.get('other', 0)}; "
        f"sections={len(section_elems)}, focus={len(focus_elems)}"
    )