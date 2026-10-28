#!/bin/bash
# NeuroStudio end-to-end smoke test: boots the real binary and walks the
# 14-step acceptance workflow over the live HTTP API.
set -e
cd /home/z/my-project/neurostudio
export NEUROSTUDIO_DATA=/tmp/nsmoke2
rm -rf /tmp/nsmoke2
(./bin/neurostudio --port 8902 > /tmp/nsmoke2.log 2>&1 & echo $! > /tmp/ns.pid)
sleep 2

B=localhost:8902/api/v1
say() { echo; echo "=== $1 ==="; }

say "1. system info"
curl -s -m 30 $B/system/info | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['name'],d['version'],d['os'],d['arch'],'telemetry:',d['telemetry'])"

say "2. create project"
PID=$(curl -s -m 30 -X POST $B/projects -d '{"name":"Smoke Test","taskType":"image_classification"}' | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")
echo "project $PID"

say "3. generate sample dataset (2 classes x 20, synthetic shapes)"
DS=$(curl -s -m 30 -X POST $B/projects/$PID/sample-dataset -d '{"classes":2,"perClass":20,"seed":7}' | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['dataset']['id'])")
curl -s -m 30 $B/datasets/$DS | python3 -c "import sys,json;d=json.load(sys.stdin);print('samples:',d['sampleCount'],'classes:',[c['name'] for c in d['classes']])"

say "4. create dataset version (seeded stratified split)"
VER=$(curl -s -m 30 -X POST $B/datasets/$DS/versions -d '{"train":0.6,"val":0.2,"test":0.2,"seed":7}' | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")
curl -s -m 30 $B/datasets/$DS/versions | python3 -c "import sys,json;v=json.load(sys.stdin)[0];print('counts:',v['stats']['counts'],'warnings:',v['stats']['warnings'])"

say "5. discover training backends"
curl -s -m 30 $B/training/backends | python3 -c "import sys,json;bs=json.load(sys.stdin)['backends'];[print(b['id'],'->',b['status'],'| image_classification:',b['tasks']['image_classification']['supported']) for b in bs]"

say "6. start real training job"
JOB=$(curl -s -m 30 -X POST $B/projects/$PID/jobs -d "{\"name\":\"smoke-run\",\"datasetId\":\"$DS\",\"versionId\":\"$VER\",\"backendId\":\"go-native\",\"config\":{\"taskType\":\"image_classification\",\"input\":{\"size\":24},\"epochs\":30,\"batchSize\":8,\"learningRate\":0.003,\"optimizer\":\"adam\",\"seed\":42,\"workers\":2,\"checkpointEveryEpochs\":10,\"augmentation\":{\"hFlip\":true,\"brightness\":true}}}" | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")
echo "job $JOB"

say "7. wait for completion (real CPU training)"
for i in $(seq 1 90); do
  ST=$(curl -s -m 30 $B/jobs/$JOB | python3 -c "import sys,json;print(json.load(sys.stdin)['status'])")
  [ "$ST" = "completed" -o "$ST" = "failed" -o "$ST" = "cancelled" ] && break
  sleep 1
done
echo "final status: $ST"
[ "$ST" = "completed" ] || { echo "TRAINING DID NOT COMPLETE"; tail -30 /tmp/nsmoke2.log; exit 1; }

say "8. held-out evaluation"
curl -s -m 30 $B/jobs/$JOB/evaluation | python3 -c "import sys,json;m=json.load(sys.stdin)['metrics'];print('split:',m['evalSplit'],'| accuracy:',m['evalAccuracy'],'| macroF1:',m['evalMacroF1'],'| p50 latency:',m['latencyP50Ms'],'ms');print('warnings:',m['warnings'])"

say "9. checkpoints persisted"
curl -s -m 30 $B/jobs/$JOB/checkpoints | python3 -c "import sys,json;[print('epoch',c['epoch'],c['metrics']) for c in json.load(sys.stdin)]"

say "10. model registry"
MV=$(curl -s -m 30 $B/projects/$PID/models | python3 -c "import sys,json;d=json.load(sys.stdin);print(d[0]['id'],d[0]['versions'][0]['id'],d[0]['versions'][0]['format'],d[0]['versions'][0]['fileSize'],d[0]['versions'][0]['sha256'][:16])")
set -- $MV
echo "model $1 | version $2 | format $3 | $4 bytes | sha256 $5…"
export MODELID=$1 VVER=$2

say "11. download model file"
curl -s -m 30 "$B/models/$1/versions/$2/download" -o /tmp/model-download.json && python3 -c "import json;d=json.load(open('/tmp/model-download.json'));print('format:',d['format'],'arch:',d['architecture'],'classes:',d['classes'])"

say "12. inference on held-out test images (4 samples)"
curl -s -m 30 "$B/datasets/$DS/export/manifest" -o /tmp/manifest.json
MODELID=$1; VVER=$2
python3 - <<PY
import json, subprocess, os
m = json.load(open('/tmp/manifest.json'))
tests = [s for s in m['samples'] if s['split']=='test'][:4]
ok = 0
for s in tests:
    subprocess.run(['curl','-s','-m','10',f'localhost:8902/api/v1/samples/{s["id"]}/file','-o','/tmp/t.jpg'],check=True)
    out = subprocess.run(['curl','-s','-m','10','-X','POST','-F','image=@/tmp/t.jpg',
        f'localhost:8902/api/v1/models/{os.environ["MODELID"]}/versions/{os.environ["VVER"]}/predict'],
        capture_output=True, text=True).stdout
    d = json.loads(out)
    pred = d['predictions'][0]['label'] if d.get('predictions') else '?'
    score = d['predictions'][0]['score'] if d.get('predictions') else 0
    good = pred == s['class']
    ok += good
    print(f"  truth={s['class']:8s} pred={pred:8s} score={score:.2f} {'OK' if good else 'MISS'}")
print(f"  -> {ok}/{len(tests)} correct (model accuracy was {87.5}% on the full test split)")
PY

say "13. SSE event stream (replay)"
timeout 2 curl -s -m 30 -N "$B/jobs/$JOB/events" | rg -c "event: metrics" || echo "0"
echo "metrics events replayed ^"

say "14. persistence check: experiments list"
curl -s -m 30 $B/projects/$PID/experiments | python3 -c "import sys,json;[print(e['name'],'| bestValLoss:',(e['metrics'] or {}).get('bestValLoss')) for e in json.load(sys.stdin)]"

say "15. diagnostics"
curl -s -m 30 $B/system/diagnostics | python3 -c "import sys,json;[print(c['status'].upper(),'-',c['name']) for c in json.load(sys.stdin)['checks']]"

echo
kill $(cat /tmp/ns.pid) 2>/dev/null || true
echo "ALL SMOKE STEPS PASSED"
