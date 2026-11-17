# AI Dialogue

Replication materials for **Composite requests, local connections, and recorded progress: User follow-up behavior in generative AI dialogues**.

The main dataset contains 2,405 source conversations, 5,100 tasks, and 10,850 follow-up positions. The external prediction check uses 542 ShareGPT source conversations.

## Reproduce the analyses

Use Python 3.12 and install `requirements.txt`. From this directory run:

```sh
python 03_Code/reproduce.py --refit --output outputs/prediction
python 03_Code/reproduce_outcomes.py --output outputs/outcomes
python 03_Code/reproduce_sensitivity.py --output outputs/sensitivity
python 03_Code/reproduce_agreement.py
python 03_Code/reproduce_human_agreement.py
python 03_Code/reproduce_embedding_summary.py
python 03_Code/reproduce_figures.py
```

Outputs go to `outputs`; supplied results remain in `05_Results`. No command above calls a model API. The first three programs retain their smoke checks and resume behavior.

`02_Data` contains derived task, function, agreement and comparison data. `04_Prompts` contains V13 and the response schema. `03_Code/source_preprocessing` contains the original source-selection code; the source releases and optional dependencies are needed to rerun that stage. Raw dialogue text is not included.

The model–model agreement entry uses 321 aligned positions in 50 source conversations. Human–model agreement uses 244 aligned positions from 50 designated tasks with the retained human rejudgments and common five-function mapping (C04 to G1). Each coefficient retains its own paired denominator.

The comparison uses 3,847 pairs. `comparison_pairs.json` supplies semantic-strategy judgments and word-token operation codes. `comparison_embedding_metrics.csv` supplies content-word measures, sentence-embedding cosines and fitted cluster assignments. Keys align exactly between files. `embedding_method.json` records the model revision and preprocessing. `reproduce_embedding_summary.py` recomputes the reported means, source-bootstrap intervals and length-adjusted coefficient without embedding inference. The optional `reconstruct_embeddings.py` entry reruns embedding inference and the feature analysis when the matching source texts and local model files are available; see its `--help`. Model weights are obtained from the named public release.

Figures 2, 6, 7, 8 and S6 have Python/vector sources. Figure 8a–b retains the strategy analysis; c–d and S6 use the explicit embedding implementation. Figures 3 and 4 retain the supplied geometry with the blue–white–red palette; the figure command applies that transform and the supplementary outcome-label correction. The final comparison figures are generated after the retained-artwork export. Numerical origins are listed in `05_Results/analysis_provenance.json`.

Repository: https://github.com/yaohuihuang316-coder/ai-dialogue-n8q4z7

`SHA256SUMS` records the checksums of the supplied files. Raw source datasets and optional model weights are obtained from their original releases under their respective terms.

Figure 1 retains its original diagram with two revised headings. The final PNG is exported by the figure command above. To reapply only those headings, run `python 03_Code/figures/relabel_figure1.py --font /path/to/arialbd.ttf`; the original artwork is in `06_Artwork/label_sources/Figure_1_before_titles.png`.
