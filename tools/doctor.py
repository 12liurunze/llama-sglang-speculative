"""Read-only diagnostics. Does not load weights or run inference."""
import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from llama_speculative.common import check_runtime,load


def main():
    p = argparse.ArgumentParser();p.add_argument('--runtime',default=str(ROOT/'runtime/sglang'));p.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    args = p.parse_args()
    observed = {}
    expected = load(ROOT/'vendor/environment_versions.json')
    for name,wanted in expected.items():
        try:
            value = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            value = None
        observed[name] = dict(installed=value,recorded=wanted,match=value == wanted)
    print(json.dumps(dict(python=sys.version,packages=observed),indent=2))
    print('RUNTIME_HASH',check_runtime(args.runtime))
    if os.name != 'posix':
        print('GPU experiments require Linux.');return
    from llama_speculative.hardware import selected_processes
    devices,processes,other = selected_processes([x.strip() for x in args.gpus.split(',')])
    print(json.dumps(dict(selected_devices=devices,processes=processes),indent=2))
    subprocess.run(['nvidia-smi','topo','-m'],check=True)
    print('Read-only checks complete; run --pilot-only next. No throughput or correctness guarantee is implied.')


if __name__ == '__main__':
    main()
