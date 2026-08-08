import argparse
 
def normalize_line(line):
    """Wraps a single token following ^ or _ in { } if not already braced.
 
    e.g. "x _ 1"      -> "x _ { 1 }"
         "e ^ x"       -> "e ^ { x }"
         "x _ { 1 }"   -> "x _ { 1 }"   (unchanged, already braced)
         "x _ { a b }" -> "x _ { a b }" (unchanged, multi-token group)
    """
    tokens = line.split()
    out = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        out.append(tok)
        if tok in ("^", "_") and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt != "{":
                out.append("{")
                out.append(nxt)
                out.append("}")
                i += 2
                continue
        i += 1
    return " ".join(out)
 
 
def normalize_file(in_path, out_path, has_img_id=False):
    """Processes a caption.txt file line by line.
 
    has_img_id: if True, the first whitespace-separated field on each line
    is treated as an image id and left untouched (matches the
    "{img_id} {caption}" format used in finetune_custom_author.py).
    """
    changed = 0
    total = 0
 
    with open(in_path, "r", encoding="utf-8") as f_in, \
         open(out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            stripped = line.strip()
            if not stripped:
                continue
            total += 1
 
            if has_img_id:
                parts = stripped.split(" ", 1)
                if len(parts) != 2:
                    # No caption content on this line, write through unchanged
                    f_out.write(stripped + "\n")
                    continue
                img_id, caption = parts
                new_caption = normalize_line(caption)
                if new_caption != caption:
                    changed += 1
                f_out.write(f"{img_id} {new_caption}\n")
            else:
                new_line = normalize_line(stripped)
                if new_line != stripped:
                    changed += 1
                f_out.write(new_line + "\n")
 
    print(f"Processed {total} lines, modified {changed}.")
    print(f"Output written to: {out_path}")
 
if __name__ == "__main__":
    normalize_file("data/custom/caption.txt", "data/custom/caption_braced.txt")
