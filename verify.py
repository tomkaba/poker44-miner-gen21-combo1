import json
import subprocess
import os
import hashlib

def get_hash():
    with open('models/model_manifest.json', 'r') as f:
        manifest = json.load(f)
    files = manifest['implementation_files']
    abs_paths = [os.path.abspath(f) for f in files]
    sorted_pairs = sorted(zip(abs_paths, files))
    sha256 = hashlib.sha256()
    for abs_p, rel_p in sorted_pairs:
        sha256.update(rel_p.encode('utf-8'))
        with open(abs_p, 'rb') as f:
            sha256.update(f.read())
    return sha256.hexdigest()

head = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
head_minus_1 = subprocess.check_output(['git', 'rev-parse', 'HEAD~1']).decode().strip()

with open('models/model_manifest.json', 'r') as f:
    manifest = json.load(f)

manifest_repo_commit = manifest['repo_commit']
manifest_implementation_sha256 = manifest['implementation_sha256']
recomputed_sha = get_hash()

print(f"HEAD: {head}")
print(f"HEAD~1: {head_minus_1}")
print(f"manifest repo_commit: {manifest_repo_commit}")
print(f"manifest implementation_sha256: {manifest_implementation_sha256}")
print(f"recomputed implementation sha: {recomputed_sha}")
print(f"repo_commit == HEAD~1: {manifest_repo_commit == head_minus_1}")
print(f"implementation_sha256 matches recomputed: {manifest_implementation_sha256 == recomputed_sha}")
