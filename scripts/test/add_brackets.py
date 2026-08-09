import argparse

# "Big operators" where \limits changes rendering: without it, a sub/super-
# script on these renders beside the symbol; with it, above/below. \sum vs
# \sum \limits are NOT the same glyph layout in general -- BUT in display-
# style math (which is what CROHME/HME formulas are rendered as), most
# renderers already default \sum/\prod/\int etc. to the limits-below style,
# making the explicit \limits redundant in that context. That's exactly the
# equivalence seen in the reported errors (\sum _ {...} vs \sum \limits _
# {...} rendering identically), so we normalize the ground truth to always
# include \limits here, matching what the model consistently predicts.
BIG_OPERATORS = {
    "\\sum", "\\prod", "\\coprod",
    "\\int", "\\oint", "\\iint", "\\iiint",
    "\\bigcup", "\\bigcap", "\\bigsqcup",
    "\\bigvee", "\\bigwedge",
    "\\bigodot", "\\bigoplus", "\\bigotimes", "\\biguplus",
}

# Operators that render with limits below by default, in EVERY style, with
# or without an explicit \limits (per amsmath: these are defined as limit
# operators, unlike e.g. \det, \arg, \dim, \log which are NOT). Adding
# \limits here is a strict no-op on rendering, so it's always safe to
# normalize -- included in case the model shows the same \limits-insertion
# habit on these (no evidence of that in the current error samples, but
# harmless to cover).
DEFAULT_LIMIT_FUNCS = {
    "\\lim", "\\max", "\\min", "\\sup", "\\inf",
    "\\gcd", "\\Pr", "\\liminf", "\\limsup",
}

LIMIT_OPS = BIG_OPERATORS | DEFAULT_LIMIT_FUNCS

# Deliberately NOT included: \log, \det, \arg, \dim, \ker, \hom, \exp, \sin,
# \cos, etc. -- for these, \limits DOES change rendering (forces the
# sub/superscript below instead of beside), so inserting it would be a real
# edit, not a harmless normalization. Example #2 in the error report is
# exactly this: the model predicted "\lim" where the ground truth was
# "\log" -- a genuine recognition error that should stay flagged as wrong.


def normalize_line(line):
    """Wraps a single token following ^ or _ in { } if not already braced,
    and inserts \\limits after big operators (\\sum, \\prod, \\int, ...)
    when followed by ^ or _ and not already present.

    e.g. "x _ 1"                 -> "x _ { 1 }"
         "e ^ x"                 -> "e ^ { x }"
         "x _ { 1 }"             -> "x _ { 1 }"             (unchanged, already braced)
         "x _ { a b }"           -> "x _ { a b }"           (unchanged, multi-token group)
         "\\sum _ { i = 1 } ^ { n }" -> "\\sum \\limits _ { i = 1 } ^ { n }"
         "\\sum \\limits _ { i } "   -> "\\sum \\limits _ { i }"   (unchanged, already has \\limits)
         "\\log _ { 2 }"             -> "\\log _ { 2 }"             (unchanged, \\log is not a limits operator)
    """
    tokens = line.split()
    out = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        out.append(tok)

        # Insert \limits right after a big/limit operator if it's directly
        # followed by ^ or _ and doesn't already have \limits.
        if tok in LIMIT_OPS and i + 1 < len(tokens) and tokens[i + 1] in ("^", "_"):
            out.append("\\limits")

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
    limits_inserted = 0
    lines_with_limits_inserted = 0

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
                added = new_caption.count("\\limits") - caption.count("\\limits")
                if added > 0:
                    limits_inserted += added
                    lines_with_limits_inserted += 1
                f_out.write(f"{img_id} {new_caption}\n")
            else:
                new_line = normalize_line(stripped)
                if new_line != stripped:
                    changed += 1
                added = new_line.count("\\limits") - stripped.count("\\limits")
                if added > 0:
                    limits_inserted += added
                    lines_with_limits_inserted += 1
                f_out.write(new_line + "\n")

    print(f"Processed {total} lines, modified {changed}.")
    print(f"Inserted \\limits {limits_inserted} time(s) across {lines_with_limits_inserted} line(s).")
    print(f"Output written to: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Normalize brace-wrapping and \\limits placement in CoMER caption files"
    )
    parser.add_argument("--in-path", default="data/custom/caption.txt",
                         help="Input caption.txt path")
    parser.add_argument("--out-path", default="data/custom/caption_braced.txt",
                         help="Output path for the normalized captions")
    parser.add_argument("--has-img-id", action="store_true",
                         help="Set if each line is '{img_id} {caption}' rather than just '{caption}'")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    normalize_file(args.in_path, args.out_path, has_img_id=args.has_img_id)