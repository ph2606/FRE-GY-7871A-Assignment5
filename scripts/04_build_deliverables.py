"""Build the shared narrative, six-page LaTeX PDF source and runnable notebook."""
from pathlib import Path
import json
import re
import sys
import nbformat as nbf
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_content import ROOT, REPO, sections
from src.figures import build as figures


def esc(value):
    mapping = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
        '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(mapping.get(char, char) for char in str(value))


def prose(text):
    saved = []
    def keep(value):
        saved.append(value)
        return f'ZZTOKEN{len(saved)-1}ZZ'
    text = re.sub(r'\[([^]]+)\]\((https?://[^)]+)\)', lambda m: keep(r'\href{' + esc(m[2]) + '}{' + esc(m[1]) + '}'), text)
    text = re.sub(r'\*\*(.*?)\*\*', lambda m: keep(r'\textbf{' + esc(m[1]) + '}'), text)
    text = esc(text)
    for i, value in enumerate(saved):
        text = text.replace(f'ZZTOKEN{i}ZZ', value)
    return text


def latex_table(item):
    count = len(item['headers'])
    widths = item.get('widths') or [1 / count] * count
    layout = ''.join(r'>{\raggedright\arraybackslash}p{' + str(width) + r'\dimexpr\linewidth-' + str(2 * count) + r'\tabcolsep\relax}' for width in widths)
    lines = [r'\par\smallskip\noindent{\fontsize{8.9}{10.8}\selectfont', r'\begin{tabular}{' + layout + '}', r'\toprule',
        ' & '.join(r'\textbf{' + esc(h) + '}' for h in item['headers']) + r' \\', r'\midrule']
    lines += [' & '.join(esc(x) for x in row) + r' \\[2pt]' for row in item['rows']]
    lines += [r'\bottomrule\end{tabular}}\par\smallskip']
    return '\n'.join(lines)


PREAMBLE = r'''\documentclass[11pt,letterpaper]{article}
\usepackage[margin=0.65in,headheight=15pt,headsep=13pt,footskip=19pt]{geometry}
\usepackage{fontspec}
\IfFontExistsTF{Cambria}{\setmainfont{Cambria}}{\setmainfont{Latin Modern Roman}}
\IfFontExistsTF{Calibri}{\setsansfont{Calibri}}{\setsansfont{Latin Modern Sans}}
\usepackage{amsmath,booktabs,array,graphicx,xcolor,fancyhdr,titlesec,hyperref}
\definecolor{navy}{HTML}{17334D}
\definecolor{teal}{HTML}{168C8C}
\hypersetup{colorlinks=true,urlcolor=teal,linkcolor=navy,pdfauthor={Panagiotis Housos},pdftitle={Midterm Topics, Sentiment and Market Expectations}}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\sffamily\small FRE-GY 7871A | Assignment 5}
\fancyhead[R]{\sffamily\small Panagiotis Housos | ph2606}
\fancyfoot[L]{\sffamily\footnotesize August 1--October 8, 2026 | As of October 9, 2026}
\fancyfoot[R]{\sffamily\small\thepage\ / 6}
\renewcommand{\headrulewidth}{0.3pt}
\titleformat{\section}{\sffamily\large\bfseries\color{navy}}{}{0pt}{}
\titlespacing*{\section}{0pt}{0pt}{6pt}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.02}
\emergencystretch=2em
\begin{document}
\fontsize{10.1}{11.8}\selectfont
'''


def build_report(content):
    tex = [PREAMBLE]
    markdown = ['# Midterm Topics, Sentiment and Market Expectations', 'Panagiotis Housos | ph2606 | FRE-GY 7871A | Fall 2026', '']
    for page, (title, items) in enumerate(content):
        if page:
            tex.append(r'\newpage')
        if page == 0:
            tex.append(r'{\sffamily\fontsize{20}{23}\selectfont\bfseries\color{navy} Midterm Topics, Sentiment and Market Expectations}\par')
            tex.append(r'{\sffamily\small NLP and the Investment Process | Fall 2026}\par\smallskip')
        if page == 5:
            tex.append(r'\fontsize{9.7}{11.4}\selectfont')
        tex.append(r'\section*{' + esc(title) + '}')
        markdown += ['## ' + title, '']
        for item in items:
            if item['type'] == 'p':
                tex.append(prose(item['text']) + '\n' + r'\par')
                markdown += [item['text'], '']
            elif item['type'] == 'table':
                tex.append(latex_table(item))
                markdown += ['| ' + ' | '.join(item['headers']) + ' |', '| ' + ' | '.join(['---'] * len(item['headers'])) + ' |']
                markdown += ['| ' + ' | '.join(map(str, row)) + ' |' for row in item['rows']]
                markdown.append('')
            else:
                tex += [r'\begin{center}\includegraphics[width=' + item['width'] + r'\linewidth]{outputs/' + item['name'] + r'.pdf}\end{center}',
                    r'{\footnotesize\color{navy} ' + prose(item['caption']) + r'\par}']
                markdown += ['![' + item['caption'] + '](outputs/' + item['name'] + '.png)', '']
    tex.append(r'\end{document}')
    (ROOT / 'report.tex').write_text('\n'.join(tex), encoding='utf-8')
    (ROOT / 'REPORT.md').write_text('\n'.join(markdown), encoding='utf-8')


def build_notebook(content):
    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell('# Assignment 5 — Applied topic modeling and sentiment around the U.S. midterms\n\n**Panagiotis Housos | ph2606 | FRE-GY 7871A | Fall 2026**\n\nInformation through October 8, 2026 at 16:00 New York. Model training ends August 21; assessment begins August 22. The separate six-page PDF shares this computed narrative. Raw and derived datasets are intentionally excluded from Git. See README for acquisition and the pinned contextual model.'),
        nbf.v4.new_code_cell("from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT = Path.cwd()\nassert (ROOT / 'src/text_model.py').exists(), 'Run from the repository root.'\npd.set_option('display.max_columns', 15)\npd.set_option('display.max_colwidth', 65)\npd.set_option('display.precision', 4)"),
        nbf.v4.new_markdown_cell('## Recalculate from the acquired observations\n\nAcquisition is explicit in the README and runs separately. The cell below reconstructs scores and all statistical exhibits from local caches; it does not create observations. The contextual file must match every current corpus ID. The saved outputs below contain aggregate results only.'),
        nbf.v4.new_code_cell("from src.text_model import score_documents\nfrom src.analysis import analyze\nfrom src.issue_sentiment import build as issue_indices\nfrom src.figures import build as draw_figures\nassert (ROOT / 'data/processed/contextual_scores.csv').exists(), 'Run the contextual scorer after acquisition and initial text modeling.'\nscored = score_documents()\ncalendar, market, tests = analyze()\nissue_indices()\ndraw_figures()\nsummary = json.loads((ROOT / 'outputs/summary.json').read_text())\ndisplay(pd.Series(summary, name='observed_result'))\nprint('All indices and associations recalculated from acquired observations.')")]
    exhibits = [['text_filter_audit'], ['topic_model_selection', 'topic_summary', 'issue_summary'],
        ['sentiment_review_summary', 'contextual_review_summary'],
        ['correlations', 'flagged_diagnostics', 'level_correlations', 'sensitivities', 'issue_correlations', 'probability_equity_associations'],
        ['equity_performance', 'individual_equity_correlations'], []]
    for page, (title, items) in enumerate(content):
        cells.append(nbf.v4.new_markdown_cell('## ' + title))
        for item in items:
            if item['type'] == 'p':
                cells.append(nbf.v4.new_markdown_cell(item['text']))
            elif item['type'] == 'table':
                text = '| ' + ' | '.join(item['headers']) + ' |\n| ' + ' | '.join(['---'] * len(item['headers'])) + ' |\n'
                text += '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in item['rows'])
                cells.append(nbf.v4.new_markdown_cell(text))
            else:
                cells.append(nbf.v4.new_code_cell(f"display(Image(filename=str(ROOT / 'outputs/{item['name']}.png')))"))
                cells.append(nbf.v4.new_markdown_cell(item['caption']))
        for name in exhibits[page]:
            if name.endswith('_summary') and name in ['sentiment_review_summary', 'contextual_review_summary']:
                # Frozen labels are private local data. Fresh clones require fresh
                # blind labels; the submitted notebook retains aggregate evidence.
                audit_path = ROOT / f'outputs/{name}.json'
                if audit_path.exists():
                    aggregate = json.loads(audit_path.read_text())
                    excluded_ids = aggregate.pop('excluded_review_ids', None)
                    if excluded_ids is not None:
                        aggregate['excluded_review_n'] = len(excluded_ids)
                    cells.append(nbf.v4.new_markdown_cell(f'### {name.replace("_", " ")}\n\nFrozen aggregate diagnostic (AI reference; no independent human accuracy):\n\n```json\n' + json.dumps(aggregate, indent=2) + '\n```'))
                else:
                    cells.append(nbf.v4.new_markdown_cell('The local blind-label audit is unavailable in this checkout. Recreate independent labels to recompute the submitted aggregate diagnostic; labels are excluded from Git under the no-datasets policy.'))
            else:
                cells.append(nbf.v4.new_code_cell(f"print('{name}')\ndisplay(pd.read_csv(ROOT / 'outputs/{name}.csv'))"))
    cells += [nbf.v4.new_markdown_cell('## Measurement, model and timing audit\n\nThe checks below audit implementation and acquired-record alignment. They do not establish voter identity, sentiment accuracy or causal effects. Detailed labels and texts remain local. The October poll never supplies historical index weights.'),
        nbf.v4.new_code_cell("collection = json.loads((ROOT / 'outputs/text_collection_summary.json').read_text())\nprint(json.dumps(collection, indent=2))\nprint('Source collection counts precede the final analysis availability cutoff.')"),
        nbf.v4.new_code_cell("assert scored.id.is_unique\nassert scored.available_at.max() <= pd.Timestamp('2026-10-08T20:00:00Z')\nassert market.index.is_unique and calendar.index.is_unique\nassert market[['long_only_minus_spy','signed_beta_adjusted']].notna().all().all()\nassert scored.token_truncation.eq(False).all()\nassert scored.discarded_tokens.eq(0).all()\nmethod = json.loads((ROOT / 'outputs/text_method.json').read_text())\nprint(json.dumps(method, indent=2))\nprint('No post-cutoff text, duplicate IDs, truncated text or incomplete basket returns entered the submitted results.')"),
        nbf.v4.new_markdown_cell('## Reproduction boundary\n\nA fresh clone must run acquisition and contextual model download before executing this notebook. APIs can revise rankings, text and adjusted prices; a fresh run reproduces the method rather than guaranteeing the exact saved snapshot. Local SHA256 manifests bind this submission to its actual acquired inputs. All synthetic data are confined to software tests. No trade or institutional submission was made.')]
    nb.cells = cells
    nb.metadata = {'kernelspec': {'display_name': 'Python 3 (Assignment 5)', 'language': 'python', 'name': 'assignment5'},
        'language_info': {'name': 'python', 'version': '3.12.0'}}
    nbf.validate(nb)
    nbf.write(nb, ROOT / 'assignment5.ipynb')


if __name__ == '__main__':
    content = sections()
    figures()
    build_report(content)
    build_notebook(content)
    print('Built report source, shared narrative and executable notebook.')
