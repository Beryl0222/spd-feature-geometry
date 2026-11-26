"""按固定种子生成容量输入，不计算算子输出。"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
BASE=Path(__file__).resolve().parent

def generate(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(261126)
    arrays={};meta={'format_version':1,'source':'本地合成容量输入','seed':261126}
    nsite,dim,ncal=8,16,32
    a=rng.normal(0,.05,(nsite,ncal,dim,dim));arrays['calibration']=a@a.swapaxes(-1,-2)+np.eye(dim)
    arrays['weights']=np.ones((nsite,ncal));arrays['bases']=np.broadcast_to(np.eye(dim),(nsite,dim,dim)).copy()
    b=rng.normal(0,.05,(2048,dim,dim));arrays['descriptors']=b@b.swapaxes(-1,-2)+np.eye(dim)*1.1
    meta.update(sites=['site'+str(i) for i in range(nsite)],channel_ids=['c'+str(i) for i in range(dim)],record_id_rule='site与行号联合唯一',descriptor_id_rule='按数组行号唯一')
    np.savez(output/'arrays.npz',**arrays)
    meta['arrays']={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in arrays.items()}
    (output/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return meta

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='生成可复现的合成容量输入')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();meta=generate(args.output)
    print(json.dumps({'数组数':len(meta['arrays']),'资料SHA256':hashlib.sha256((Path(args.output)/'arrays.npz').read_bytes()).hexdigest()},ensure_ascii=False))
