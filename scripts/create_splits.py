import argparse, json
from pathlib import Path
import numpy as np
from src.data_utils_v3 import load_data

def make_split(n, seed=42):
    ids=np.random.default_rng(seed).permutation(n).tolist()
    n_train=int(n*.70); n_val=int(n*.15)
    return {'train':ids[:n_train], 'val':ids[n_train:n_train+n_val], 'test':ids[n_train+n_val:]}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,default=42); args=p.parse_args()
    Path('data/splits').mkdir(parents=True,exist_ok=True)
    for name in ('rs','csd'):
        data=load_data(f'data/processed/{name}_v3.pkl')
        split=make_split(len(data),args.seed)
        Path(f'data/splits/{name}_seed{args.seed}.json').write_text(json.dumps({'dataset':name,'seed':args.seed,'n_sequences':len(data),'indices':split},indent=2),encoding='utf-8')
        print(name,{k:len(v) for k,v in split.items()})
