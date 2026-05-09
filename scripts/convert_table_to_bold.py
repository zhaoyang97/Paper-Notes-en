"""Convert metadata tables (| 属性 | 值 |) to bold format (**key**: value)."""

import re
from pathlib import Path

DOCS_DIR = Path(__file__).parent.parent / "docs"


def convert_table_to_bold(content: str) -> str:
    """Find all metadata tables and convert them to bold format."""
    # Pattern to match the full table block:
    # | 属性 | 值 |
    # |------|------|
    # | key1 | val1 |
    # | key2 | val2 |
    # ...
    pattern = re.compile(
        r"^\| 属性 \| 值 \|\n"
        r"\|[-]+\|[-]+\|\n"
        r"((?:\|[^\n]+\|\n?)+)",
        re.MULTILINE,
    )

    def replacer(m: re.Match) -> str:
        rows_block = m.group(1)
        lines = []
        for row in rows_block.strip().split("\n"):
            # Parse | key | value |
            parts = row.split("|")
            # parts: ['', ' key ', ' value ', '']
            if len(parts) >= 3:
                key = parts[1].strip()
                value = parts[2].strip()
                lines.append(f"**{key}**: {value}  ")
        return "\n".join(lines) + "\n"

    return pattern.sub(replacer, content)


def main():
    changed = 0
    for md_file in sorted(DOCS_DIR.rglob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        if "| 属性 | 值 |" not in text:
            continue
        new_text = convert_table_to_bold(text)
        if new_text != text:
            md_file.write_text(new_text, encoding="utf-8")
            print(f"  converted: {md_file.relative_to(DOCS_DIR)}")
            changed += 1
    print(f"\nTotal: {changed} files converted.")


if __name__ == "__main__":
    main()
