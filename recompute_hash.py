import hashlib
import json
import os

def get_hash():
    with open('models/model_manifest.json', 'r') as f:
        manifest = json.load(f)
    files = manifest['implementation_files']
    # Sort resolved absolute paths
    abs_paths = [os.path.abspath(f) for f in files]
    sorted_pairs = sorted(zip(abs_paths, files))
    
    sha256 = hashlib.sha256()
    for abs_p, rel_p in sorted_pairs:
        # relative path (utf-8)
        sha256.update(rel_p.encode('utf-8'))
        # file bytes
        with open(abs_p, 'rb') as f:
            sha256.update(f.read())
    return sha256.hexdigest()

print(get_hash())
