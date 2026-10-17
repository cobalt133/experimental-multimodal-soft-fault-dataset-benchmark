from pathlib import Path
import shutil
r=Path(__file__).resolve().parents[1]/'data/tensors'
parts=sorted(r.glob('shard.h5.part-*'))
if len(parts)!=3:raise RuntimeError('Expected all three public tensor parts')
target=r/'shard.h5'
if target.exists():raise RuntimeError('Output already exists; not overwriting')
with target.open('wb') as out:
 for p in parts:
  with p.open('rb') as src:shutil.copyfileobj(src,out)
print(target)
