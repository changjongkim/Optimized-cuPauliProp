#!/usr/bin/env python3
"""Style check of the SectorQ paper (the writing rules of the lab, as for the STATOR paper).

  python3 scripts/check_style.py [file.tex ...]     (default: every section file)

Prints the findings per file and the word count of every prose paragraph; exit code 1 when there are findings.
Commented lines (starting with %) are skipped: the plans in the skeleton are comments.
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULES = [
    (r"\binstead\b", "instead -> In contrast,"),
    (r"\bUnlike\b", "Unlike -> In contrast,"),
    (r"\blets?\b", "let(s) -> allows ... to / enables ... to"),
    (r"\bstays?\b", "stay(s) -> remains"),
    (r"\bfr(?:ozen|eeze)\b", "frozen/freeze"),
    (r"\bgives?\b|\bgiving\b", "give -> assigns / provides"),
    (r"\b[Ii]n (?:this|the same) way\b", "in this/the same way"),
    (r"\bremedy\b", "remedy"),
    (r"\bdown to\b", "down to"),
    (r"\bunmodified\b", "unmodified"),
    (r"\bsuch an? \b|\bAs such\b", "such a / As such"),
    (r",\s*so\b(?! that)", ", so (result clause)"),
    (r",\s*thus\b|\band therefore\b|,\s*therefore\b", "thus/therefore in mid-sentence"),
    (r"(?<![\w.])(?:Of the|While)\s", "inverted opening (Of the / While)"),
    (r"\d--\d", "number range with -- (use one hyphen)"),
    (r"\b(?:does|do|did)\b(?! not)(?!n't)", "do/does/did as a verb"),
    (r"\b(?:gives? up|calls? for|falls? behind|carr(?:y|ies) over|turns? \w+ into|draws? from)\b", "phrasal verb"),
    (r"\\texttt\{", "\\texttt (describe the action, not the identifier)"),
    (r"\bcommunication-free\b", "communication-free"),
    (r"\b[Pp]lanner\b", "Planner"),
]


def prose(text):
    lines = [l for l in text.splitlines() if not l.lstrip().startswith("%")]
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", l) for l in lines)


def words(par):
    t = re.sub(r"\\(?:cite|ref|label|url)\{[^}]*\}", "X", par)
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-$\\]*", t))


def sentences(par):
    t = re.sub(r"\s+", " ", par)
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\\])", t) if s.strip()]


def check(path):
    text = prose(open(path).read())
    found = []
    for i, line in enumerate(text.splitlines(), 1):
        for pat, why in RULES:
            for m in re.finditer(pat, line):
                found.append(f"{os.path.basename(path)}:{i}: {why}: ...{line[max(0, m.start() - 30):m.end() + 30]}...")
    blocks = [re.sub(r"(?m)^\\(?:sub)*section\*?\{.*\}\s*$|^\\label\{.*\}\s*$", "", b).strip() for b in re.split(r"\n\s*\n", text)]
    pars = [b for b in blocks if b and not b.startswith(("\\begin", "\\end", "\\centering", "\\caption", "\\subfloat", "\\figsource"))]
    report = []
    because = 0
    for p in pars:
        if p.startswith("\\") and not p.startswith(("\\noindent", "\\SectorQ", "\\cuPP")):
            continue
        n = words(p)
        ss = sentences(p)
        head = re.match(r"\\noindent\\textbf\{([^}]*)\}", p)
        report.append((head.group(1) if head else p[:40].replace("\n", " "), n, len(ss)))
        because += len(re.findall(r"This is (?:possible )?because", p))
        for s in ss:
            if re.search(r",\s*(?:which|that)\s+\w+\s+\w+[^,]*\.\s*$", s) and len(s.split(",")[-1].split()) <= 8:
                found.append(f"{os.path.basename(path)}: sentence ends with a relative clause: ...{s[-80:]}")
            if re.match(r"[A-Z][a-z]+ing\b(?![^,]*,)[^.]*\b(?:is|are|was|were)\b", s) and not s.startswith(("During", "Using", "Following", "Assuming", "Given")):
                found.append(f"{os.path.basename(path)}: gerund subject?: {s[:80]}")
    if because and pars and because > max(1, len(pars) // 4):
        found.append(f"{os.path.basename(path)}: 'This is because' {because} times in {len(pars)} paragraphs")
    return found, report


if __name__ == "__main__":
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, "[0-9]*.tex")))
    total = 0
    for f in files:
        found, report = check(f)
        total += len(found)
        print(f"== {os.path.basename(f)}: {len(found)} findings")
        for line in found:
            print("  " + line)
        for name, n, s in report:
            print(f"     {n:4d} words, {s:2d} sentences  {name}")
    print(f"total findings: {total}")
    sys.exit(1 if total else 0)
