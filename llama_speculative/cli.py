import argparse
from pathlib import Path
from .common import ROOT,load,save


def parser():
    p = argparse.ArgumentParser(description='Llama single-node TP Chain / original Adaptive2 benchmark')
    sub = p.add_subparsers(dest='action',required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--files',nargs='+',required=True);prep.add_argument('--model',required=True)
    prep.add_argument('--samples',type=int,default=80);prep.add_argument('--copies',type=int,default=1)
    prep.add_argument('--context',type=int,default=4096);prep.add_argument('--max-new-tokens',type=int,default=1024)
    prep.add_argument('--prompt-field',default='prompt');prep.add_argument('--output',required=True)
    run = sub.add_parser('run')
    run.add_argument('--data',required=True);run.add_argument('--model',required=True);run.add_argument('--draft',default='')
    run.add_argument('--gpus',required=True,help='Physical GPU indices; exactly tp entries, e.g.0,1,2,3')
    run.add_argument('--tp',type=int,required=True);run.add_argument('--bs',type=int,default=32)
    run.add_argument('--methods',nargs='+',default=['chain7','adaptive7']);run.add_argument('--threshold',type=float,default=.5)
    run.add_argument('--repeat',type=int,default=3);run.add_argument('--seed',type=int,default=42)
    run.add_argument('--context',type=int,default=4096);run.add_argument('--max-new-tokens',type=int,default=1024)
    run.add_argument('--memory-fraction',type=float,default=.8);run.add_argument('--attention-backend',default='triton')
    run.add_argument('--runtime',default=str(ROOT/'runtime/sglang'));run.add_argument('--output',required=True)
    run.add_argument('--dry-run',action='store_true');run.add_argument('--resume',action='store_true')
    run.add_argument('--wait-free',action='store_true');run.add_argument('--pilot-only',action='store_true')
    report = sub.add_parser('report');report.add_argument('--output',required=True);report.add_argument('--allow-partial',action='store_true')
    return p


def main():
    args = parser().parse_args()
    for name in ['model','draft']:
        value = getattr(args,name,None)
        if value and Path(value).expanduser().is_dir():
            setattr(args,name,str(Path(value).expanduser().resolve()))
    if args.action == 'prepare':
        from transformers import AutoTokenizer
        from .data import prepare
        tokenizer = AutoTokenizer.from_pretrained(args.model,trust_remote_code=False)
        data = prepare(args.files,tokenizer,args.output,args.samples,args.copies,args.context,args.max_new_tokens,args.prompt_field)
        print('PREPARED',list(data['subsets']),data['audit']);return
    if args.action == 'report':
        from .report import summarize
        result = summarize(args.output,args.allow_partial);print('REPORT',result['completed_formal'],'complete',result['complete']);return
    from .runner import build_plan,run
    options = {key:getattr(args,key) for key in ['model','draft','tp','bs','methods','threshold','repeat','seed','context','max_new_tokens','memory_fraction','attention_backend']}
    options['gpus'] = [x.strip() for x in args.gpus.split(',')]
    data = load(args.data);plan = build_plan(options,data,args.runtime)
    if args.dry_run:
        target = Path(args.output)
        if target.exists():
            raise FileExistsError(target)
        target.mkdir(parents=True);save(target/'plan.json',plan);save(target/'prepared.json',data)
        save(target/'progress.json',dict(status='dry_run',formal_completed=0,formal_total=sum(not r['pilot'] for r in plan['runs'])))
        print('DRY_RUN_ONLY',sum(not r['pilot'] for r in plan['runs']),'formal runs; no model/GPU execution');return
    run(plan,data,args.output,resume=args.resume,wait_free=args.wait_free,pilot_only=args.pilot_only)
