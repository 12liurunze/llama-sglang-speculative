"""One coordinator creates Engine(tp_size=N); SGLang spawns the TP ranks."""
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
if os.environ.get('LLAMA_RUNTIME_ROOT'):
    sys.path.insert(0,os.environ['LLAMA_RUNTIME_ROOT'])
if os.environ.get('LLAMA_TP_AUDIT_OUT'):
    from llama_speculative.audit import install
    install()

from llama_speculative.common import load,save,digest
from llama_speculative.metrics import recompute


def main():
    root = Path(sys.argv[1]).resolve();name = sys.argv[2]
    plan = load(root/'plan.json');run = next(r for r in plan['runs'] if r['name'] == name)
    folder = root/'runs'/name;folder.mkdir(parents=True,exist_ok=False)
    import sglang as sgl
    config = plan['configs'][run['method']]
    expected_source = Path(plan['runtime'])/'__init__.py'
    assert Path(sgl.__file__).resolve() == expected_source.resolve(),sgl.__file__
    prepared = load(root/'prepared.json')
    rows = plan['pilot_rows'] if run['pilot'] else prepared['subsets'][run['subset']]
    ids = [r['input_ids'] for r in rows]
    cap = min(128,plan['options']['max_new_tokens']) if run['pilot'] else plan['options']['max_new_tokens']
    save(folder/'config.json',dict(run=run,engine=config,input_sha256=digest(ids),count=len(ids),source=str(expected_source),
        gpu_ids=plan['options']['gpus'],plan_sha256=digest(plan),sampling=dict(temperature=0,max_new_tokens=cap)))
    engine = sgl.Engine(**config)
    try:
        info = engine.get_server_info();effective = info.get('server_args',info)
        keys = ['model_path','tp_size','dp_size','pp_size','nnodes','max_running_requests','context_length',
                'speculative_algorithm','speculative_draft_model_path','speculative_num_steps','speculative_num_draft_tokens',
                'speculative_eagle_topk','speculative_adaptive_tree','speculative_adaptive_threshold',
                'disable_cuda_graph','disable_decode_cuda_graph','cuda_graph_backend_prefill']
        observed = {k:effective.get(k) for k in keys}
        save(folder/'effective_args.json',observed)
        for key in keys:
            if key in config:
                assert observed[key] == config[key],(key,observed[key],config[key])
        if run['method'] == 'ar':
            assert observed['speculative_algorithm'] is None and observed['speculative_draft_model_path'] is None
        engine.generate(input_ids=ids[:plan['options']['bs']],sampling_params=dict(temperature=0,max_new_tokens=32))
        engine.flush_cache()
        save(folder/'timing_start.json',dict(time=time.time()))
        start = time.perf_counter()
        outputs = engine.generate(input_ids=ids,sampling_params=dict(temperature=0,max_new_tokens=cap))
        elapsed = time.perf_counter()-start
        save(folder/'timing_end.json',dict(time=time.time(),elapsed_seconds=elapsed))
        save(folder/'model_work_complete.json',dict(time=time.time()))
        save(folder/'outputs.json',outputs)
        summary = recompute(outputs,elapsed,cap,run['method'])
        assert summary['count'] == len(ids)
        save(folder/'summary.json',dict(complete=True,**summary))
        print('RUN_COMPLETE',name,summary,flush=True)
    finally:
        engine.shutdown()


if __name__ == '__main__':
    main()

