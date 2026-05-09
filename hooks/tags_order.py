"""Preserve frontmatter tag order instead of alphabetical sorting."""


def on_page_context(context, page, config, **kwargs):
    """Reorder tags in context to match frontmatter order."""
    tags = context.get("tags")
    if not tags or not page.meta.get("tags"):
        return context

    # Build order map from frontmatter
    order = {name: i for i, name in enumerate(page.meta["tags"])}

    # Sort tag references by frontmatter position (unknown tags go to end)
    context["tags"] = sorted(tags, key=lambda t: order.get(t.name, len(order)))
    return context
