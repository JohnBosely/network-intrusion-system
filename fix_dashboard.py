replacements = [
    ('\u00f0\u009f\u009b\u00a1\u00ef\u00b8\u008f', 'NIDS'),
    ('\u00f0\u009f\u0094\u00b4', 'X'),
    ('\u00e2\u009c\u0085', 'OK'),
    ('\u00e2\u0096\u00b6', 'Start'),
    ('\u00e2\u0096\u00b9', 'Stop'),
    ('\u00f0\u009f\u0094\u0084', 'Reset'),
    ('\u00f0\u009f\u0093\u00a1', 'Live'),
    ('\u00e2\u009a\u00a0\u00ef\u00b8\u008f', 'WARN'),
    ('\u00c2\u00b7', '-'),
    ('\u00e2\u0080\u0094', '--'),
]

with open('dashboard/streamlit_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

for bad, good in replacements:
    content = content.replace(bad, good)

with open('dashboard/streamlit_app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done')
