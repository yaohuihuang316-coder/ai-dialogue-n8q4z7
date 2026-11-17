"""Rebuild Figures 2, 6, 7, 8 and S6; export retained artwork for the other figures."""
from pathlib import Path
import shutil,subprocess,sys
P=Path(__file__).resolve().parents[1]
out=P/'outputs/figures';out.mkdir(parents=True,exist_ok=True)
for f in (P/'06_Artwork').iterdir():
    if f.is_file():shutil.copy2(f,out/f.name)
subprocess.run([sys.executable,str(P/'03_Code/recolor_figures.py')],check=True)
for n in [2,6,7]:
    subprocess.run([sys.executable,str(P/f'03_Code/figures/current_figure{n}.py')],check=True)


subprocess.run([sys.executable,str(P/'03_Code/normalize_supplementary_labels.py')],check=True)

subprocess.run([sys.executable,str(P/"03_Code/reproduce_embedding_summary.py")],check=True)
subprocess.run([sys.executable,str(P/"03_Code/figures/current_figure8.py")],check=True)
