"""Reproduce the two archived human/model comparisons without API calls."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from reproduce_agreement import layer
P=Path(__file__).resolve().parents[1]
data=json.loads((P/'02_Data/human_model_action_pairs.json').read_text('utf-8'))
expected=json.loads((P/'05_Results/human_model_agreement_expected.json').read_text('utf-8'))
groups=[[0,1,3,10,14],[4,5,6],[8,11,12,13],[2],[7,9]]
def mapped(a):
    return np.array([[1. if np.any(row[g]==1) else np.nan if np.isnan(row[g]).any() else 0. for g in groups] for row in a])
results={}
for model in ['GPT6','DeepSeekFlash']:
    rows=[r for r in data['rows'] if r['model']==model]
    assert len(rows)==244 and len({r['case_id'] for r in rows})==50
    a=mapped(np.array([r['human'] for r in rows],float))
    b=mapped(np.array([r['model_labels'] for r in rows],float))
    actual=layer(a,b,[r['case_id'] for r in rows],[f'G{i}' for i in range(1,6)])
    exp=expected['pairs'][model]['G5_common_legacy_mapping']
    for k in ['positions','sources','paired_cells','excluded_cells','macro_alpha','micro_agreement','macro_alpha_ci95']:
        assert np.allclose(actual[k],exp[k]),(model,k)
    for r in actual['per_label']:
        r['human_yes']=r.pop('GPT_yes');r['model_yes']=r.pop('DeepSeek_yes')
    results[model]=actual
out=P/'outputs/agreement';out.mkdir(parents=True,exist_ok=True)
f=out/'human_model_agreement.json'
f.write_text(json.dumps(results,indent=2,allow_nan=False),'utf-8')
assert len(json.loads(f.read_text('utf-8')))==2
print(json.dumps({'passed':True,'pairs':{k:{'macro_alpha':v['macro_alpha'],'paired_cells':v['paired_cells'],'positions':v['positions']} for k,v in results.items()}}))
