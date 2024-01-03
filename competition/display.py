from IPython.display import Markdown, display
from typing import Callable, List, Dict, Optional
import matplotlib.pyplot as plt
import base64
from io import BytesIO


class DisplayCompetition:
    styles = {
        'h1': 'background-color:#eef; color:#006; padding:20px; border-radius:10px;',
        'h2': 'background-color:#fff; color:#006; padding:20px; border-bottom:solid 2px #006;',
        'h3': 'background-color:#fff; color:#006; padding-left:20px;',
        'h4': 'background-color:#fff; color:#006; padding-left:20px;',
        'p': '',
        '.tbl_selected': 'background-color:#fa9',
        '.column': 'padding:20px; margin:20px; box-shadow: 0 0 1em #006;',
    }

    def __init__(self):
        self.__in_section = False
        self.__html = ''

    def __getattr__(self, tag) -> Optional[Callable]:
        if tag in self.styles:
            def tag_generator(text):
                self._p(f'<{tag} style="{self.styles[tag]}">{text}</{tag}>')

            return tag_generator

    def fig(self, fig: plt.figure) -> None:
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png')
        plt.close(fig)
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        self._p('<img src=\'data:image/png;base64,{}\'>'.format(encoded))

    def table(self, headings: List[str], data: Dict[str, List[str]], select: Optional[Callable] = None) -> None:
        if select is None:
            def select(_): return False

        html = '<table>'
        html += '<tr><td></td><th>' + '</th><th>'.join(headings) + '</th></tr>'

        for name, row in data.items():
            if select is None:
                html += '<tr><th>' + name + '</th>' + ''.join(f'<td>{item}</td>' for item in row) + '</tr>'
            else:
                html += '<tr><th>' + name + '</th>' + ''.join(
                    f'<td style="' + (self.styles['.tbl_selected'] if select(item) else '') + f'">{item}</td>' for item
                    in row) + '</tr>'
        html += '<table>'
        self._p(html)

    def table2level(self, headings: Dict[str, List[str]], row_headings: Dict[str, List[str]], data: List,
                    select: Optional[Callable] = None) -> None:
        if select is None:
            def select(_): return False

        html = '<table>'

        # Headers
        h1line = ''
        h2line = ''
        for h1, h2s in headings.items():
            h1line += f'<th colspan="{len(h2s)}">{h1}</th>'
            h2line += '<th>' + '</th><th>'.join(h2s) + '</th>'
        html += f'<tr><td></td><td></td>{h1line}</tr>'
        html += f'<tr><td></td><td></td>{h2line}</tr>'

        # Row headers and data
        rows = ''
        rn = 0
        for h1, h2s in row_headings.items():
            for n, h2 in enumerate(h2s):
                rows += '<tr>'
                if n == 0:
                    rows += f'<th rowspan="{len(h2s)}">{h1}</th>'
                rows += f'<th>{h2}</th><td>' + '</td><td>'.join(data[rn]) + '</td></tr>'
                rn += 1
        html += rows
        html += '<table>'
        self._p(html)

    def ol(self, data: List[str]) -> None:
        html = '<ol>'
        if len(data):
            html += '<li>' + ('</li><li>'.join(data)) + '</li>'
        html += '</ol>'
        self._p(html)

    def ul(self, data: List[str]) -> None:
        html = '<ul>'
        if len(data):
            html += '<li>' + ('</li><li>'.join(data)) + '</li>'
        html += '</ul>'
        self._p(html)

    def start_section(self, style=None) -> None:
        self.__html = f'<div style="{self.styles[style]}">' if style in self.styles else '<div>'
        self.__in_section = True

    def stop_section(self) -> None:
        self.__in_section = False
        self.__html += '<div>'
        self._p()

    def _p(self, html: Optional[str] = None) -> None:
        if self.__in_section:
            self.__html += html
        elif html is None:
            self._p(self.__html)
            self.__html = ''
        else:
            display(Markdown(html))
