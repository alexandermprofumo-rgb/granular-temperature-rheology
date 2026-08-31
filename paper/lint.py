"""Static checks on main.tex, standing in for a compile we cannot run here.

There is no TeX on this machine and no network to install one, so this covers
the error classes pdflatex would catch: unbalanced environments and braces,
unbalanced math mode, undefined references and citations, duplicate labels,
wrong column counts, unknown macros, unescaped specials, and missing figures.

It is not a substitute for compiling.  Run pdflatex before submission.

Usage:  python3 lint.py
"""
import os
import re
import sys
from collections import Counter

TEX = 'main.tex'
BIB = 'refs.bib'


def strip_comments(s):
    """Drop % comments, honouring \\% escapes."""
    out = []
    for line in s.split('\n'):
        i, esc = None, False
        for k, ch in enumerate(line):
            if ch == '\\':
                esc = not esc
                continue
            if ch == '%' and not esc:
                i = k
                break
            esc = False
        out.append(line if i is None else line[:i])
    return '\n'.join(out)


def main():
    raw = open(TEX).read()
    s = strip_comments(raw)
    bib = open(BIB).read() if os.path.exists(BIB) else ''
    fails, warns = [], []

    # environments
    begins = Counter(re.findall(r'\\begin\{([^}]*)\}', s))
    ends = Counter(re.findall(r'\\end\{([^}]*)\}', s))
    for env in set(begins) | set(ends):
        if begins[env] != ends[env]:
            fails.append(f'environment {env}: {begins[env]} begin, {ends[env]} end')

    # braces
    depth, esc = 0, False
    for ch in s:
        if ch == '\\':
            esc = not esc
            continue
        if not esc:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth < 0:
                    fails.append('unmatched closing brace')
                    break
        esc = False
    if depth > 0:
        fails.append(f'{depth} unclosed brace(s)')

    # math mode
    # count $ that are neither escaped nor part of $$
    body = s.split(r'\begin{document}')[-1]
    n_dollar = len(re.findall(r'(?<!\\)\$', body.replace('$$', '')))
    if n_dollar % 2:
        fails.append(f'odd number of $ ({n_dollar}): unbalanced math mode')

    # labels and refs
    labels = re.findall(r'\\label\{([^}]*)\}', s)
    dup = [k for k, v in Counter(labels).items() if v > 1]
    if dup:
        fails.append(f'duplicate labels: {dup}')
    refs = set(re.findall(r'\\(?:ref|eqref)\{([^}]*)\}', s))
    missing = sorted(refs - set(labels))
    if missing:
        fails.append(f'undefined \\ref: {missing}')
    unused = sorted(l for l in set(labels)
                    if l not in refs and l.split(':')[0] in ('fig', 'tab'))
    if unused:
        warns.append(f'float labels never referenced: {unused}')

    # citations
    keys = set()
    for m in re.findall(r'\\cite[a-z]*\{([^}]*)\}', s):
        keys.update(k.strip() for k in m.split(','))
    nobib = sorted(k for k in keys if '{' + k + ',' not in bib)
    if nobib:
        fails.append(f'citations absent from {BIB}: {nobib}')

    # tabular column counts
    for m in re.finditer(r'\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}',
                         s, re.S):
        cols = sum(1 for c in m.group(1) if c in 'lcrp')
        for row in m.group(2).split(r'\\'):
            row = row.strip()
            if not row or row.startswith(r'\colrule') or r'\multicolumn' in row:
                continue
            got = row.count('&') + 1
            if got != cols:
                fails.append(f'tabular row has {got} cells, header declares '
                             f'{cols}: {row[:45]}')

    # macros used but never defined
    defined = set(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', s))
    known = set("""begin end section subsection label ref eqref cite citep title
        author affiliation email date maketitle documentclass usepackage
        newcommand includegraphics caption bibliography appendix textwidth
        subsubsection paragraph subparagraph ell hat otimes colon
        textit textbf emph frac sqrt sum int left right qquad quad text mathrm
        times approx leq geq neq sim pm ll gg to Longleftrightarrow colrule
        multicolumn hline centering item bm dot alpha beta gamma delta Delta
        epsilon varepsilon theta Theta lambda mu nu pi rho sigma Sigma tau phi
        chi psi omega Omega partial infty ldots dots cdot cdots langle rangle
        acknowledgments appendix par noindent tfrac boldmath unboldmath
        ensuremath protect relax hspace vspace footnote and thanks
        Pi S deg equiv ge le ln propto rm today kappa varphi Gamma Lambda
        Phi Psi Xi eta zeta iota xi upsilon nabla exp log sin cos tan max min
        lim sup inf det dim ker deg arg gcd hom Pr""".split())
    used = set(re.findall(r'\\([A-Za-z]+)', s))
    unknown = sorted(used - defined - known)
    if unknown:
        warns.append(f'macros not in the known list (verify these exist): {unknown}')

    # unescaped specials outside math
    nomath = re.sub(r'(?<!\\)\$[^$]*\$', '', body, flags=re.S)
    nomath = re.sub(r'\\begin\{(equation|tabular)\}.*?\\end\{\1\}', '',
                    nomath, flags=re.S)
    # \newcommand bodies legitimately contain #1, #2 ... as parameter slots
    nomath = re.sub(r'\\newcommand\{[^}]*\}(\[\d\])?\{[^\n]*\}', '', nomath)
    for ch in ('&', '#'):
        hits = [m.start() for m in re.finditer(r'(?<!\\)' + re.escape(ch), nomath)]
        if hits:
            warns.append(f'{len(hits)} unescaped "{ch}" outside math/tabular')

    # graphics files
    for g in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}', s):
        if not any(os.path.exists(g + e) for e in ('', '.png', '.pdf', '.eps')):
            fails.append(f'missing graphic: {g}')

    # REVTeX specifics
    if r'\documentclass' in s:
        cls = re.search(r'\\documentclass\[(.*?)\]\{(.*?)\}', s, re.S)
        if cls:
            opts = [o.strip() for o in cls.group(1).split(',') if o.strip()]
            if 'reprint' in opts and r'\begin{figure}' in s:
                warns.append('single-column figure in a reprint (two-column) '
                             'document: wide plots need figure*')
            jour = [o for o in opts if o in ('prl', 'pra', 'prb', 'prc', 'prd',
                                             'pre', 'prfluids', 'prx')]
            print(f'  class: {cls.group(2)}, journal option: '
                  f'{jour[0] if jour else "NONE"}')
    if r'\affiliation' not in s:
        fails.append('no \\affiliation: REVTeX requires one')

    # report
    print(f'  words: {len(s.split())}   sections: '
          f'{len(re.findall(r"\\section\{", s))}   figures: '
          f'{len(re.findall(r"includegraphics", s))}   '
          f'tables: {begins["table"]}')
    print()
    for f in fails:
        print(f'  FAIL   {f}')
    for w in warns:
        print(f'  warn   {w}')
    if not fails:
        print('  no blocking errors found')
    print('\n  This is not a compile.  Run pdflatex before submission.')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
