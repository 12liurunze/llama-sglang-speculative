"""Verify the complete bundled snapshot before extracting it to a new directory."""
import argparse
import hashlib
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path,PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from llama_speculative.common import check_runtime,load,sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output',default=str(ROOT/'runtime'))
    p.add_argument('--verify-only',action='store_true')
    args = p.parse_args();dest = Path(args.output).resolve()
    archive = ROOT/'vendor/original_runtime.tar.gz';manifest = load(ROOT/'vendor/runtime_manifest.json')
    expected = load(ROOT/'vendor/provenance.json')['archive_sha256']
    assert sha(archive) == expected,'Archive SHA256 mismatch'
    found = {}
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar.getmembers():
            path = PurePosixPath(member.name)
            assert path.parts and path.parts[0] == 'sglang' and '..' not in path.parts and not path.is_absolute()
            assert member.isdir() or member.isfile(),'Links/devices are not allowed'
            if member.isfile():
                key = '/'.join(path.parts[1:]);assert key not in found
                found[key] = hashlib.sha256(tar.extractfile(member).read()).hexdigest()
        assert found == manifest,'Archive contents differ from the recorded snapshot'
        if args.verify_only:
            print('ARCHIVE_VERIFIED',len(found),'files',expected);return
        if os.name != 'posix':
            raise RuntimeError('Extract the runtime on the Linux GPU node. Windows supports --verify-only, --dry-run and CPU tests; long kernel-configuration paths exceed Windows limits.')
        if dest.exists():
            raise FileExistsError('Refusing to overwrite an existing runtime: '+str(dest))
        dest.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='llama_runtime_',dir=dest.parent) as work:
            temporary = Path(work).resolve()
            assert temporary.parent == dest.parent
            # All archive entries were validated above; this supports Python3.10+.
            for member in tar.getmembers():
                tar.extract(member,path=temporary)
            check_runtime(temporary/'sglang')
            temporary.rename(dest)
    print('RUNTIME_INSTALLED',dest/'sglang')


if __name__ == '__main__':
    main()
