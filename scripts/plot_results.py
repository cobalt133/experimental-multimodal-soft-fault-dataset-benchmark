from pathlib import Path
import runpy
r=Path(__file__).resolve().parents[1]/'src/figures'
for name in ['plot_controls.py','plot_results.py','plot_single_column.py','plot_core_analysis.py']:runpy.run_path(str(r/name),run_name='__main__')
