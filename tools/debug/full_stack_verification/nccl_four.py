import os
rank=int(os.environ['RANK'])
case=os.environ['PROFILE_CASE']
server=rank>=2
os.environ['NCCL_ALGO']='allreduce:tree' if server or case=='channels' else 'Ring'
if server or case in ('matched','algorithm'):
 os.environ['NCCL_MIN_NCHANNELS']=os.environ['NCCL_MAX_NCHANNELS']='8'
else:
 os.environ.pop('NCCL_MIN_NCHANNELS',None);os.environ.pop('NCCL_MAX_NCHANNELS',None)
import json
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from miles.utils.distributed_utils import init_process_group
from sglang.srt.utils.common import init_custom_process_group
root=Path('/tmp/results')
receipt={'rank':rank,'case':case,'stage':'imports_passed','checks':0,'torch':torch.__version__,'nccl':torch.cuda.nccl.version()}
def save(): (root/f'four-{case}-{rank}.json').write_text(json.dumps(receipt,indent=2))
save()
try:
 torch.cuda.set_device(rank)
 dist.init_process_group('gloo',timeout=timedelta(seconds=60))
 train=dist.new_group([0,1],backend='nccl',timeout=timedelta(seconds=60))
 serve=dist.new_group([2,3],backend='nccl',timeout=timedelta(seconds=60))
 x=torch.tensor([float(rank)],device=f'cuda:{rank}')
 dist.all_reduce(x,group=serve if server else train)
 assert x.item()==(5 if server else 1)
 receipt['stage']='role_tp_allreduce_passed';save()
 if rank!=1:
  store=dist.PrefixStore('transfer',dist.distributed_c10d._get_default_store())
  group=(init_custom_process_group if server else init_process_group)(backend='nccl',store=store,rank=rank-1 if server else 0,world_size=3,group_name='miles-pp_0',timeout=timedelta(seconds=60))
  for repeat in range(3):
   expected=(torch.arange(8192,device=f'cuda:{rank}')%127+repeat).to(torch.bfloat16)
   tensor=expected.clone() if rank==0 else torch.zeros_like(expected)
   dist.broadcast(tensor,0,group=group,async_op=True).wait()
   assert torch.equal(tensor,expected)
   receipt['checks']+=1;save()
  dist.destroy_process_group(group)
 receipt['stage']='passed';save()
 dist.barrier()
 dist.destroy_process_group()
except Exception as exc:
 receipt['stage']='failed';receipt['error']=repr(exc);save();raise
