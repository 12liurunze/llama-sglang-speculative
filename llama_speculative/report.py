import json
import math
import statistics
from pathlib import Path
from .common import digest,load,save
from .metrics import recompute


def summarize(output,allow_partial=False):
    root = Path(output);plan = load(root/'plan.json');data = load(root/'prepared.json')
    assert digest(data) == plan['prepared_sha256']
    monitors = [json.loads(line) for line in (root/'gpu_monitor.jsonl').read_text().splitlines()] if (root/'gpu_monitor.jsonl').exists() else []
    rows = [];missing = [];raw = {}
    for run in plan['runs']:
        if run['pilot']:
            continue
        folder = root/'runs'/run['name']
        if not (folder/'summary.json').exists() or (folder/'interference.json').exists():
            missing.append(run['name']);continue
        cfg = load(folder/'config.json');outputs = load(folder/'outputs.json');saved = load(folder/'summary.json')
        assert cfg['run'] == run and cfg['engine'] == plan['configs'][run['method']]
        assert cfg['plan_sha256'] == digest(plan) and cfg['gpu_ids'] == plan['options']['gpus']
        effective = load(folder/'effective_args.json')
        for key in ['model_path','tp_size','dp_size','pp_size','nnodes','max_running_requests','context_length','speculative_num_steps','speculative_num_draft_tokens','speculative_eagle_topk','speculative_adaptive_tree','speculative_adaptive_threshold','disable_cuda_graph','disable_decode_cuda_graph','cuda_graph_backend_prefill']:
            if key in cfg['engine']:
                assert effective[key] == cfg['engine'][key],(run['name'],key)
        assert cfg['input_sha256'] == data['audit'][run['subset']]['input_sha256']
        start = load(folder/'timing_start.json')['time'];end = load(folder/'timing_end.json')
        actual = recompute(outputs,end['elapsed_seconds'],plan['options']['max_new_tokens'],run['method'])
        assert abs(end['time']-start-end['elapsed_seconds']) < .1
        assert actual['count'] == data['audit'][run['subset']]['requests']
        for k,v in actual.items():
            assert math.isclose(v,saved[k],rel_tol=1e-12,abs_tol=1e-10) if isinstance(v,float) else v == saved[k]
        samples = [x for x in monitors if x['run'] == run['name'] and start <= x['time'] <= end['time']]
        assert samples and not any(x['outsiders'] for x in samples)
        rows.append(dict(**run,**actual));raw[run['name']] = outputs
    if missing and not allow_partial:
        raise RuntimeError('Incomplete experiment; use report --allow-partial: '+str(missing))
    groups = {};repeatability = {}
    for subset in data['subsets']:
        groups[subset] = {}
        for method in plan['options']['methods']:
            group = [r for r in rows if r['subset'] == subset and r['method'] == method]
            if not group:
                continue
            stats = dict(repetitions=len(group))
            for key in ['elapsed_seconds','tokens_per_second','mean_accepted_draft','acceptance','output_tokens']:
                values = [r[key] for r in group]
                stats[key] = dict(mean=statistics.mean(values),sample_sd=statistics.stdev(values) if len(values)>1 else None) if values[0] is not None else dict(mean=None,sample_sd=None)
            groups[subset][method] = stats
            first = group[0]
            repeatability[subset+'/'+method] = [dict(repeat=r['repeat'],same_output_ids=sum(a['output_ids'] == b['output_ids'] for a,b in zip(raw[first['name']],raw[r['name']]))) for r in group]
    result = dict(complete=not missing,completed_formal=len(rows),missing=missing,subsets=groups,runs=rows,repeatability=repeatability)
    save(root/'results.json',result)
    lines = ['# Llama多卡Chain / Adaptive2对照','',f"正式结果{len(rows)}/{len(rows)+len(missing)}；TP={plan['options']['tp']}；设备={plan['options']['gpus']}。",'',
        '|子集|方法|已完成遍数|整批耗时s|吞吐token/s|接受draft长度|接受率|输出tokens|','|---|---|---:|---:|---:|---:|---:|---:|']
    for subset,methods in groups.items():
        for method,stats in methods.items():
            cells = []
            for key in ['elapsed_seconds','tokens_per_second','mean_accepted_draft','acceptance','output_tokens']:
                x = stats[key];factor = 100 if key == 'acceptance' else 1;precision = 3 if key == 'mean_accepted_draft' else 0 if key == 'output_tokens' else 2
                cell = '—' if x['mean'] is None else f"{factor*x['mean']:.{precision}f}"
                if x['sample_sd'] is not None:
                    cell += f"±{factor*x['sample_sd']:.{precision}f}"
                if key == 'acceptance' and x['mean'] is not None:
                    cell += '%'
                cells.append(cell)
            lines.append('|'+ '|'.join([subset,method,str(stats['repetitions'])]+cells)+'|')
    lines += ['', '±为样本标准差；只有一次时不显示标准差。接受长度仅计接受的draft，不含target额外token；不同预算的接受率分母不同。AR的接受指标不适用。', '',
        '耗时包含prefill、调度和decode，不包含加载、Graph捕获、预热及写盘。比较吞吐时同时检查输出量和上限命中数，不能把生成成功等同答案正确。', '',
        'BS是上限；真实BS和各TP进程的Graph调用见pilot_*_audit.json。GPU每2秒采样，未观测干扰不等于连续隔离证明。', '',
        '输入重复情况：'+json.dumps(data.get('prompt_overlap',[]),ensure_ascii=False)]
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result

