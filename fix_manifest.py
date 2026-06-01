import json
import subprocess

def get_hash():
    with open('models/model_manifest.json', 'r') as f:
        manifest = json.load(f)
    files = manifest['implementation_files']
    import os
    abs_paths = [os.path.abspath(f) for f in files]
    sorted_pairs = sorted(zip(abs_paths, files))
    import hashlib
    sha256 = hashlib.sha256()
    for abs_p, rel_p in sorted_pairs:
        sha256.update(rel_p.encode('utf-8'))
        with open(abs_p, 'rb') as f:
            sha256.update(f.read())
    return sha256.hexdigest()

head = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
computed_hash = get_hash()

with open('models/model_manifest.json', 'r') as f:
    manifest = json.load(f)

changed = False
if manifest['implementation_sha256'] != computed_hash:
    manifest['implementation_sha256'] = computed_hash
    changed = True

if manifest['repo_commit'] != head:
    manifest['repo_commit'] = head
    changed = True

if changed:
    with open('models/model_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=4)
    print("FIXED")
else:
    print("NO_CHANGE")
