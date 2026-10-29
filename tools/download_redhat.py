"""Optional public dataset download; no model weights or credentials in this repo."""
import argparse
import hashlib
import json
from pathlib import Path

FILES = ['HumanEval.jsonl','math_reasoning.jsonl','qa.jsonl','question.jsonl',
         'rag.jsonl','summarization.jsonl','translation.jsonl','writing.jsonl']
REVISION = '2ae86affa2cb97a972b7fc681dd51c04fbff083e'


def main():
    p = argparse.ArgumentParser();p.add_argument('--output',default='data/raw');args = p.parse_args()
    from huggingface_hub import hf_hub_download
    root = Path(args.output);root.mkdir(parents=True,exist_ok=True)
    audit = {}
    for name in FILES:
        path = Path(hf_hub_download(repo_id='RedHatAI/speculator_benchmarks',repo_type='dataset',
                                   filename=name,revision=REVISION,local_dir=str(root)))
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        assert len(rows) >= 80 and all(isinstance(row.get('prompt'),str) for row in rows)
        audit[name] = dict(count=len(rows),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (root/'download_audit.json').write_text(json.dumps(dict(revision=REVISION,files=audit),indent=2))
    print('DOWNLOADED',len(audit),'files; question/writing may duplicate, see prepare audit.')


if __name__ == '__main__':
    main()
