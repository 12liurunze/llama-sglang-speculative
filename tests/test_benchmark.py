import copy
import json
import math
import subprocess
import sys
from pathlib import Path
import pytest
from llama_speculative.common import clean_environment,digest,engine_config,method_config,model_audit,save
from llama_speculative.data import prepare
from llama_speculative.metrics import recompute
from llama_speculative.runner import build_plan
from llama_speculative.report import summarize

ROOT = Path(__file__).resolve().parents[1]


def options(tp=4):
    return dict(model='target',draft='matching-draft',tp=tp,gpus=[str(i) for i in range(tp)],bs=32,
        methods=['chain5','adaptive5'],threshold=.5,repeat=3,seed=42,context=4096,max_new_tokens=1024,
        memory_fraction=.8,attention_backend='triton')


class Tokenizer:
    name_or_path = 'target';chat_template = 'test-template'
    def apply_chat_template(self,messages,**kw):
        assert kw['return_dict'] is False and kw['add_generation_prompt'] is True
        return [1]+[ord(c) for c in messages[0]['content']]+[2]


def prepared(tmp_path):
    p = tmp_path/'qa.jsonl';p.write_text('\n'.join(json.dumps(dict(prompt=s)) for s in ['abc','def']),encoding='utf-8')
    return prepare([p],Tokenizer(),tmp_path/'prepared.json',2,2,4096,1024)


@pytest.mark.parametrize('tp',[1,2,4,8])
def test_tensor_parallel_config_preserves_budget(tp):
    cfg = engine_config(options(tp),'adaptive5')
    assert cfg['tp_size'] == tp and cfg['dp_size'] == cfg['pp_size'] == cfg['nnodes'] == 1
    assert cfg['speculative_num_steps'] == 5 and cfg['speculative_num_draft_tokens'] == 6
    assert cfg['speculative_eagle_topk'] == 2 and cfg['speculative_adaptive_threshold'] == .5
    assert cfg['cuda_graph_backend_prefill'] == 'breakable' and not cfg['disable_decode_cuda_graph']


@pytest.mark.parametrize('gpus',[['0','1'],['0','0','1','2'],['a','1','2','3']])
def test_invalid_device_assignments(gpus):
    x = options();x['gpus'] = gpus
    with pytest.raises(ValueError):
        engine_config(x,'chain7')


def test_ar_has_no_draft_or_speculative_arguments():
    cfg = engine_config(options(),'ar')
    assert not any(k.startswith('speculative_') for k in cfg)


def test_chain_and_tree_use_equal_node_budget():
    for k in [1,3,5,7]:
        c = method_config('chain'+str(k),.5);a = method_config('adaptive'+str(k),.5)
        assert c['speculative_num_draft_tokens'] == a['speculative_num_draft_tokens'] == k+1
        assert c['speculative_eagle_topk'] == 1 and a['speculative_eagle_topk'] == 2


def test_prepare_repeats_are_explicit_and_no_answers_injected(tmp_path):
    data = prepared(tmp_path);rows = data['subsets']['qa']
    assert len(rows) == 4 and [r['input_ids'] for r in rows[:2]] == [r['input_ids'] for r in rows[2:]]
    assert data['audit']['qa']['original_count'] == 2 and data['audit']['qa']['copies'] == 2
    assert {r['copy_index'] for r in rows} == {0,1}


def test_prepare_rejects_context_overflow(tmp_path):
    p = tmp_path/'large.jsonl';p.write_text(json.dumps(dict(prompt='x'*4000)))
    with pytest.raises(ValueError,match='no silent truncation'):
        prepare([p],Tokenizer(),tmp_path/'bad.json',1,1,4096,1024)


def test_plan_alternates_methods_and_rejects_tokenizer_mismatch(tmp_path):
    data = prepared(tmp_path);plan = build_plan(options(),data,ROOT/'runtime/sglang')
    formal = [r for r in plan['runs'] if not r['pilot']]
    assert len(formal) == 6 and [r['method'] for r in formal[:4]] == ['chain5','adaptive5','adaptive5','chain5']
    wrong = copy.deepcopy(data);wrong['tokenizer_name'] = 'other'
    with pytest.raises(ValueError,match='tokenizer'):
        build_plan(options(),wrong,ROOT/'runtime/sglang')


def output(accepted,steps=5):
    h = [0]*(steps+1);h[accepted] = 1
    return dict(output_ids=list(range(accepted+1)),meta_info=dict(completion_tokens=accepted+1,spec_verify_ct=1,
        spec_num_correct_drafts=accepted,spec_num_proposed_drafts=steps,spec_correct_drafts_histogram=h))


def test_metrics_use_correct_denominator():
    m = recompute([output(0),output(5)],2,1024,'adaptive5')
    assert m['output_tokens'] == 7 and m['tokens_per_second'] == 3.5
    assert m['mean_accepted_draft'] == 2.5 and m['acceptance'] == .5
    bad = output(5);bad['meta_info']['spec_num_proposed_drafts'] = 7
    with pytest.raises(AssertionError):
        recompute([bad],1,1024,'adaptive5')


def test_ar_metrics_not_zero_acceptance():
    m = recompute([dict(output_ids=[1,2],meta_info=dict(completion_tokens=2))],1,1024,'ar')
    assert m['tokens_per_second'] == 2 and m['acceptance'] is None and m['mean_accepted_draft'] is None


def test_local_checkpoint_audit_detects_changed_metadata_and_weights(tmp_path):
    model = tmp_path/'model';model.mkdir();save(model/'config.json',dict(hidden_size=8))
    (model/'weight.safetensors').write_bytes(b'fixture')
    first = model_audit(model)
    save(model/'config.json',dict(hidden_size=16));second = model_audit(model)
    assert first['metadata_sha256'] != second['metadata_sha256']
    (model/'weight.safetensors').write_bytes(b'longer-fixture')
    assert model_audit(model)['weights'] != second['weights']
    with pytest.raises(ValueError,match='local checkpoint'):
        model_audit('not-downloaded-model')


def test_clean_environment_sets_all_visible_gpus_and_drops_instrumentation(monkeypatch):
    monkeypatch.setenv('FUSION_AUDIT_OUT','old-experiment')
    env = clean_environment(['0','1','2','3'],ROOT/'runtime/sglang')
    assert env['CUDA_VISIBLE_DEVICES'] == '0,1,2,3' and 'FUSION_AUDIT_OUT' not in env
    assert env['LLAMA_TP_AUDIT_OUT'] == ''


def test_dry_run_works_without_torch_sglang_or_gpu_execution(tmp_path):
    prepared(tmp_path);out = tmp_path/'plan'
    cmd = [sys.executable,str(ROOT/'main.py'),'run','--data',str(tmp_path/'prepared.json'),'--model','target',
        '--draft','matching-draft','--gpus','0,1,2,3,4,5,6,7','--tp','8','--methods','chain5','adaptive5','--output',str(out),'--dry-run']
    result = subprocess.run(cmd,text=True,capture_output=True,check=True)
    assert 'DRY_RUN_ONLY' in result.stdout
    assert json.loads((out/'plan.json').read_text())['configs']['adaptive5']['tp_size'] == 8


def test_report_recomputes_raw_counters_and_requires_complete_repeats(tmp_path):
    data = prepared(tmp_path);opts = options();opts['repeat'] = 2
    plan = build_plan(opts,data,ROOT/'runtime/sglang');root = tmp_path/'results';root.mkdir()
    save(root/'plan.json',plan);save(root/'prepared.json',data)
    monitors = []
    formal = [r for r in plan['runs'] if not r['pilot']]
    for i,run in enumerate(formal[:-1]):
        folder = root/'runs'/run['name'];folder.mkdir(parents=True)
        outputs = [output(0),output(5),output(1),output(4)]
        elapsed = i+1
        save(folder/'config.json',dict(run=run,engine=plan['configs'][run['method']],gpu_ids=opts['gpus'],
            plan_sha256=digest(plan),input_sha256=data['audit']['qa']['input_sha256']))
        save(folder/'effective_args.json',plan['configs'][run['method']])
        save(folder/'outputs.json',outputs);save(folder/'summary.json',recompute(outputs,elapsed,1024,run['method']))
        save(folder/'timing_start.json',dict(time=100));save(folder/'timing_end.json',dict(time=100+elapsed,elapsed_seconds=elapsed))
        monitors.append(dict(run=run['name'],time=100.5,outsiders=[]))
    (root/'gpu_monitor.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in monitors))
    with pytest.raises(RuntimeError,match='Incomplete'):
        summarize(root)
    result = summarize(root,allow_partial=True)
    assert result['complete'] is False and result['completed_formal'] == 3
    c = result['subsets']['qa']['chain5']['elapsed_seconds']
    assert c['mean'] == 1 and c['sample_sd'] is None
