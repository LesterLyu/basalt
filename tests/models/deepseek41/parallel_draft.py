"""Check a DSpark pack with two GPU-backed sessions and fixed expert placement."""
import queue
import subprocess
import sys
import threading
from pathlib import Path

root = Path(__file__).resolve().parents[3]
prompts = [(root / f'data/bench/deepseek41/{name}.ids').read_text().strip() for name in ('conv', 'prose')]
engine, model, *options = sys.argv[1:]
cmd = [engine, '--serve', '--model', model, '--batch', '2', '--spec', '3', '--diag-tier-frozen', *options]
limit = 48
p = subprocess.Popen(cmd, cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
lines = queue.Queue()
def read():
    for line in p.stdout:
        lines.put(line.rstrip())
    lines.put(None)
threading.Thread(target=read, daemon=True).start()

def send(line):
    p.stdin.write(line + '\n')
    p.stdin.flush()

def recv():
    line = lines.get(timeout=180)
    if line is None or line.startswith('ERR'):
        raise RuntimeError(line or f'engine exited: {p.poll()}')
    return line.split()

def request(verb, slot):
    return f'{verb} {slot} {limit} temperature=0 seed=42 {prompts[slot]}'

def solo(slot):
    send(request('SGEN', slot))
    ids = []
    while True:
        f = recv()
        if f[0] == 'T':
            ids.append(int(f[1]))
        if f[0] == 'DONE':
            assert len(ids) == limit and int(f[6]) > 0, f
            print('solo', slot, f, flush=True)
            return ids

def concurrent(reference, cancel):
    send(request('BGEN', 0) + '\n' + request('BGEN', 1))
    ids = [[], []]
    done = {}
    admitted = set()
    stopped = False
    overlap = False
    while len(done) < 2:
        f = recv()
        if f[0] == 'T':
            ids[len(admitted)].append(int(f[1]))
        elif f[0] == 'BADM':
            assert f[2] in ('1', '2'), f
            admitted.add(int(f[1]))
        elif f[0] == 'BT':
            b = int(f[1])
            ids[b].extend(map(int, f[2:]))
            if len(admitted) == 2 and not done and all(ids):
                overlap = True
            if cancel and not stopped and min(map(len, ids)) >= 8:
                send('BSTOP 0')
                stopped = True
        elif f[0] == 'BDONE':
            b = int(f[1])
            done[b] = f
            print('batch', b, f, flush=True)
    assert overlap
    for b, f in done.items():
        n = len(ids[b])
        assert int(f[2]) == n and int(f[6]) >= int(f[5]) > 0, f
        if cancel and b == 0:
            assert stopped and 8 <= n < limit and f[3] == 'cancel', f
        else:
            assert n == limit and f[3] == 'length', f
        assert ids[b] == reference[b][:n], (b, next(i for i in range(n) if ids[b][i] != reference[b][i]))
    print('greedy parity:', list(map(len, ids)), 'cancel:', cancel, flush=True)

try:
    while recv()[0] != 'READY':
        pass
    reference = [solo(0), solo(1)]
    concurrent(reference, False)
    concurrent(reference, True)
    assert solo(0) == reference[0]
    send('QUIT')
    assert p.wait(timeout=45) == 0
finally:
    if p.poll() is None:
        p.kill()
        p.wait()
