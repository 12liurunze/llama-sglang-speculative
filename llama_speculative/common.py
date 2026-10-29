import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def method_config(method,threshold):
    if method == 'ar':
        return None
    match = re.fullmatch(r'(chain|adaptive)([1-9][0-9]*)',method)
    if not match:
        raise ValueError('Method must be ar, chainK or adaptiveK, for example chain5/adaptive5')
    budget = int(match[2]);adaptive = match[1] == 'adaptive'
    if not 1 <= budget <= 16:
        raise ValueError('This benchmark restricts the node budget to1..16')
    if not 0 <= threshold <= 1:
        raise ValueError('threshold must lie in [0,1]')
    return dict(speculative_algorithm='EAGLE3',speculative_num_steps=budget,
                speculative_num_draft_tokens=budget+1,speculative_eagle_topk=2 if adaptive else 1,
                speculative_adaptive_tree=adaptive,speculative_adaptive_confidence='top1',
                speculative_adaptive_threshold=threshold if adaptive else 0.)


def engine_config(options,method):
    gpus = options['gpus'];tp = options['tp']
    if tp not in [1,2,4,8] or len(gpus) != tp or len(set(gpus)) != tp:
        raise ValueError('Use exactly tp distinct GPUs; supported TP sizes:1,2,4,8')
    if any(not str(x).isdigit() for x in gpus):
        raise ValueError('Use local integer GPU indices, for example 0,1,2,3')
    if options['bs'] < 1 or options['max_new_tokens'] < 1 or options['context'] <= options['max_new_tokens']:
        raise ValueError('Invalid BS/context/output cap')
    if not 0 < options['memory_fraction'] < 1:
        raise ValueError('memory_fraction must lie in (0,1)')
    cfg = dict(model_path=options['model'],tp_size=tp,dp_size=1,pp_size=1,nnodes=1,node_rank=0,
        dtype='bfloat16',trust_remote_code=False,max_running_requests=options['bs'],
        context_length=options['context'],mem_fraction_static=options['memory_fraction'],
        attention_backend=options['attention_backend'],sampling_backend='pytorch',
        disable_cuda_graph=False,disable_decode_cuda_graph=False,
        cuda_graph_backend_prefill='breakable',disable_radix_cache=True,random_seed=options['seed'],
        cuda_graph_bs_decode=sorted(set([x for x in [1,2,4,8,16,24,32,48,64,96,128] if x <= options['bs']]+[options['bs']])))
    spec = method_config(method,options['threshold'])
    if spec:
        if not options['draft']:
            raise ValueError('Speculative methods need a matching draft model')
        cfg.update(spec,speculative_draft_model_path=options['draft'])
    return cfg


def check_runtime(runtime):
    runtime = Path(runtime).resolve()
    manifest = load(ROOT/'vendor/runtime_manifest.json')
    actual = {p.relative_to(runtime).as_posix():sha(p) for p in runtime.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.pyo']}
    if actual != manifest:
        changed = sorted(k for k in set(actual)|set(manifest) if actual.get(k) != manifest.get(k))
        raise RuntimeError('Runtime differs from the frozen original: '+str(changed[:20]))
    return digest(actual)


def model_audit(path):
    root = Path(path).resolve()
    if not root.is_dir() or not (root/'config.json').is_file():
        raise ValueError('Actual experiments require a downloaded local checkpoint directory: '+str(path))
    metadata = {p.name:sha(p) for p in root.iterdir() if p.is_file() and p.suffix in ['.json','.model','.tiktoken']}
    weights = []
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.suffix in ['.safetensors','.bin']:
            stat = p.stat()
            weights.append(dict(file=p.relative_to(root).as_posix(),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns))
    if not weights:
        raise ValueError('No local model weight files found: '+str(root))
    return dict(path=str(root),metadata_sha256=metadata,weights=weights,
        note='Metadata SHA256 plus weight size/mtime; full weight-content hashes are not computed.')


def clean_environment(gpus,runtime,audit=''):
    env = dict(os.environ)
    for key in ['ADAPTIVE_PROFILE_OUT','FUSION_AUDIT_OUT','GDN_CAPTURE_OUT','TREE_AUDIT_OUT','LIGHT_OUT','ACTIVE_TRACE_OUT','PREFILL_AUDIT_OUT']:
        env.pop(key,None)
    env.update(CUDA_VISIBLE_DEVICES=','.join(gpus),PYTHONDONTWRITEBYTECODE='1',
        LLAMA_RUNTIME_ROOT=str(Path(runtime).resolve().parent),LLAMA_TP_AUDIT_OUT=audit,
        SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN='1',SGLANG_CRASH_ON_TRITON_LOAD_AFTER_READY='0',
        LLAMA_EXPERIMENT_SKIP_LOAD_MEM_QUERY='1')
    return env

