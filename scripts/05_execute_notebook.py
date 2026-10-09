from pathlib import Path
import os
import nbformat
from nbclient import NotebookClient
ROOT = Path(__file__).resolve().parents[1]
runtime = ROOT / 'outputs/jupyter_runtime'
runtime.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('JUPYTER_RUNTIME_DIR', str(runtime))
os.environ.setdefault('IPYTHONDIR', str(ROOT / 'outputs/ipython'))
path = ROOT / 'assignment5.ipynb'
notebook = nbformat.read(path, as_version=4)
NotebookClient(notebook, timeout=1200, kernel_name='assignment5', resources={'metadata': {'path': str(ROOT)}}).execute()
assert not [output for cell in notebook.cells if cell.cell_type == 'code' for output in cell.get('outputs', []) if output.output_type == 'error']
nbformat.validate(notebook)
nbformat.write(notebook, path)
print(f'Executed {sum(cell.cell_type == "code" for cell in notebook.cells)} code cells without errors.')
