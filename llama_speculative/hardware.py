import os
import signal
import subprocess
import time


def query():
    def smi(fields,scope):
        value = subprocess.check_output(['nvidia-smi',f'--query-{scope}='+fields,'--format=csv,noheader,nounits'],text=True,timeout=20)
        return [[v.strip() for v in line.split(',')] for line in value.splitlines() if line.strip()]
    devices = {x[0]:dict(index=x[0],uuid=x[1],name=x[2],memory_mib=int(x[3])) for x in smi('index,uuid,name,memory.total','gpu')}
    processes = [dict(uuid=x[0],pid=int(x[1]),memory_mib=x[2]) for x in smi('gpu_uuid,pid,used_memory','compute-apps')]
    return devices,processes


def selected_processes(gpus,group=None):
    devices,processes = query()
    if any(i not in devices for i in gpus):
        raise ValueError('GPU index not present: '+str(gpus))
    uuids = {devices[i]['uuid'] for i in gpus}
    selected = [p for p in processes if p['uuid'] in uuids]
    outsiders = []
    for row in selected:
        try:
            if group is None or os.getpgid(row['pid']) != group:
                outsiders.append(row)
        except ProcessLookupError:
            pass
    return [devices[i] for i in gpus],selected,outsiders


def stop_own_group(child):
    if child is None or child.poll() is not None:
        return
    assert os.getpgid(child.pid) == child.pid
    os.killpg(child.pid,signal.SIGTERM)
    try:
        child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGKILL)
        child.wait(timeout=20)


def wait_for_free(gpus,wait):
    while True:
        devices,_,other = selected_processes(gpus)
        if not other:
            return devices
        if not wait:
            raise RuntimeError('Selected GPUs already have compute processes: '+str(other))
        time.sleep(10)
