import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from .common import ROOT,check_runtime,clean_environment,digest,engine_config,load,model_audit,save,sha
from .hardware import selected_processes,stop_own_group,wait_for_free


def build_plan(options,data,runtime):
    if options['repeat'] < 1 or not options['methods'] or len(set(options['methods'])) != len(options['methods']):
        raise ValueError('Need positive repeat and distinct methods')
    if data['context'] != options['context'] or data['max_new_tokens'] != options['max_new_tokens']:
        raise ValueError('Prepare and run must use identical context/output caps')
    if data['tokenizer_name'] != options['model']:
        raise ValueError('Prepared tokenizer model differs from --model; re-prepare with the target tokenizer')
    configs = {m:engine_config(options,m) for m in options['methods']}
    flat = [r for rows in data['subsets'].values() for r in rows if r['copy_index'] == 0]
    if not flat:
        raise ValueError('No input records')
    ordered = sorted(flat,key=lambda r:len(r['input_ids']))
    pilot = ordered[:16]+ordered[-16:] if len(ordered) >= 32 else ordered
    runs = [dict(name='pilot_'+m,method=m,pilot=True) for m in options['methods']]
    for repeat in range(1,options['repeat']+1):
        for i,subset in enumerate(data['subsets']):
            methods = options['methods'];shift = (repeat-1+i)%len(methods)
            for method in methods[shift:]+methods[:shift]:
                runs.append(dict(name=f'{subset}_r{repeat}_{method}',method=method,pilot=False,subset=subset,repeat=repeat))
    return dict(format_version=1,options=options,configs=configs,runtime=str(Path(runtime).resolve()),
        prepared_sha256=digest(data),pilot_rows=pilot,runs=runs,
        harness_files={p.relative_to(ROOT).as_posix():sha(p) for p in [ROOT/'main.py',*sorted((ROOT/'llama_speculative').glob('*.py'))]},
        timing='One generation call after warm-up and cache flush; includes prefill/scheduling/decode; excludes engine load, graph capture, warm-up, serialization.',
        caveats=['BS is an upper bound, not the BS of every round.','Repeated records are not independent questions.',
                 'Output equality is not task accuracy.','No multi-GPU speedup is established until the benchmark is actually run.'])


def pilot_audit(root,run,tp):
    records = [json.loads(line) for path in (root/'audits'/run['name']).glob('*.jsonl') for line in path.read_text().splitlines()]
    result = {}
    for event in ['decode_graph','prefill_graph','draft_graph']:
        rows = [r for r in records if r['event'] == event and (event == 'draft_graph' or not r['draft'])]
        if event != 'draft_graph' or run['method'] not in ['ar','chain1']:
            assert rows,(run['name'],'Missing actual Graph calls',event)
            assert len({r['pid'] for r in rows}) == tp,(event,'Graph not observed on all TP ranks')
        result[event] = dict(calls=len(rows),worker_processes=len({r['pid'] for r in rows}),max_actual_bs=max((r['bs'] for r in rows),default=None))
    vocab = [r for r in records if r['event'] == 'full_vocab_logits']
    assert vocab,'Full-vocabulary logit audit missing'
    assert all(r['width'] == r['vocab_size'] for r in vocab)
    result['full_vocab_logits'] = dict(observations=len(vocab),widths=sorted({r['width'] for r in vocab}),
        note='This checks full-vocabulary assembly; it is not an end-to-end equivalence proof.')
    save(root/(run['name']+'_audit.json'),result)


def run(plan,data,output,resume=False,wait_free=False,pilot_only=False):
    if os.name != 'posix':
        raise RuntimeError('Actual GPU execution requires Linux; --dry-run and CPU tests work on other systems')
    import fcntl
    root = Path(output).resolve()
    if root.exists() and not resume:
        raise FileExistsError('Use a new output directory or explicit --resume: '+str(root))
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if resume:
            assert load(root/'plan.json') == plan,'Resume configuration changed'
            assert load(root/'prepared.json') == data,'Resume inputs changed'
        else:
            save(root/'plan.json',plan);save(root/'prepared.json',data)
        runtime_hash = check_runtime(plan['runtime'])
        models = dict(target=model_audit(plan['options']['model']))
        if any(method != 'ar' for method in plan['options']['methods']):
            models['draft'] = model_audit(plan['options']['draft'])
        if data.get('tokenizer_file_hashes'):
            assert data['tokenizer_file_hashes'] == models['target']['metadata_sha256'],'Tokenizer/model metadata changed after prepare'
        if resume and (root/'model_audit.json').exists():
            assert load(root/'model_audit.json') == models,'Model metadata or weight stat changed'
        else:
            save(root/'model_audit.json',models)
        def verify_sources():
            assert check_runtime(plan['runtime']) == runtime_hash
            for name,expected in plan['harness_files'].items():
                assert sha(ROOT/name) == expected,'Harness changed: '+name
            assert model_audit(plan['options']['model']) == models['target']
            if 'draft' in models:
                assert model_audit(plan['options']['draft']) == models['draft']
        verify_sources()
        save(root/'runtime_verified.json',dict(manifest_sha256=sha(ROOT/'vendor/runtime_manifest.json'),runtime_hash=runtime_hash))
        child = None;done = []
        def state(status,**kw):
            save(root/'progress.json',dict(status=status,time=time.time(),completed=done,
                formal_completed=sum(not r['pilot'] and r['name'] in done for r in plan['runs']),
                formal_total=sum(not r['pilot'] for r in plan['runs']),**kw))
        def interrupted(signum,frame):
            raise KeyboardInterrupt('Stopped by signal '+str(signum))
        previous = {sig:signal.signal(sig,interrupted) for sig in [signal.SIGINT,signal.SIGTERM]}
        gpu_locks = []
        try:
            state('checking_devices')
            devices = wait_for_free(plan['options']['gpus'],wait_free)
            for device in sorted(devices,key=lambda d:d['uuid']):
                handle = (Path('/tmp')/('llama_speculative_'+device['uuid']+'.lock')).open('a')
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);gpu_locks.append(handle)
            save(root/'devices.json',devices)
            for item in plan['runs']:
                if pilot_only and not item['pilot']:
                    continue
                name = item['name'];folder = root/'runs'/name
                if (folder/'summary.json').exists() and load(folder/'summary.json').get('complete') and not (folder/'interference.json').exists():
                    if item['pilot']:
                        pilot_audit(root,item,plan['options']['tp'])
                    done.append(name);continue
                if folder.exists():
                    if not resume:
                        raise RuntimeError('Incomplete run exists: '+name)
                    archive = root/'incomplete'/f'{name}_{time.time_ns()}';archive.mkdir(parents=True)
                    for path in [folder,root/'logs'/(name+'.log'),root/'audits'/name]:
                        if path.exists():
                            path.rename(archive/(path.name+('_audit' if path.parent.name == 'audits' else '')))
                state('waiting_for_gpus',current=name)
                wait_for_free(plan['options']['gpus'],wait_free)
                verify_sources()
                log = root/'logs'/(name+'.log');log.parent.mkdir(exist_ok=True)
                env = clean_environment(plan['options']['gpus'],plan['runtime'],str(root/'audits'/name) if item['pilot'] else '')
                with log.open('ab') as f:
                    child = subprocess.Popen([sys.executable,'-u',str(ROOT/'llama_speculative/worker.py'),str(root),name],
                        cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                state('running',current=name,pid=child.pid)
                while child.poll() is None:
                    _,processes,other = selected_processes(plan['options']['gpus'],child.pid)
                    finished = (folder/'model_work_complete.json').exists()
                    with (root/'gpu_monitor.jsonl').open('a') as f:
                        f.write(json.dumps(dict(time=time.time(),run=name,group=child.pid,processes=processes,outsiders=other,model_work_complete=finished))+'\n')
                    if other and not finished:
                        stop_own_group(child)
                        save(folder/'interference.json',dict(time=time.time(),outsiders=other,excluded=True))
                        raise RuntimeError('External GPU process detected; only our process group was stopped. Inspect and use --resume to retry.')
                    time.sleep(2)
                assert child.returncode == 0 and (folder/'summary.json').exists(),('Run failed; preserved logs',name,child.returncode)
                verify_sources()
                if item['pilot']:
                    pilot_audit(root,item,plan['options']['tp'])
                done.append(name);state('between_runs',last_complete=name)
            from .report import summarize
            summarize(root,allow_partial=pilot_only)
            state('pilot_complete' if pilot_only else 'complete')
        except KeyboardInterrupt:
            stop_own_group(child);state('stopped_by_user');print('Stopped; existing results preserved.',flush=True)
            raise SystemExit(130)
        except Exception as exc:
            stop_own_group(child);state('failed',error=repr(exc));raise
        finally:
            for sig,handler in previous.items():
                signal.signal(sig,handler)
            for handle in gpu_locks:
                handle.close()

