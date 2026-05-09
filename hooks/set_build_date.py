"""Inject today's ISO date into config.extra.build_date at build time.

Used by overrides/main.html as a fallback for JSON-LD datePublished/dateModified
when a page does not declare its own `date:` frontmatter. Generating dynamically
avoids checking a hardcoded date into git (which would otherwise make every page
look "updated on 2026-04-14" forever).
"""

from datetime import date


def on_config(config, **kwargs):
    extra = config.setdefault("extra", {})
    extra["build_date"] = date.today().isoformat()
    return config
