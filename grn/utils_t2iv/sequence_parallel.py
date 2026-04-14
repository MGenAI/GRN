import torch
import torch.nn as nn
import torch.distributed as dist
from .comm.pg_utils import ProcessGroupManager
from .comm.comm import set_sp_comm_group, split_sequence, gather_sequence, all_to_all_comm
from .comm.operation import gather_forward_split_backward

class SequenceParallelManager:
    _SP_GROUP = None
    _SP_SIZE = 0

    @staticmethod
    def sp_on():
        return SequenceParallelManager._SP_GROUP is not None

    @staticmethod
    def init_sp(sp_size):
        if SequenceParallelManager._SP_GROUP is not None:
            print("WARN: sequence parallel group is already initialized")
            return

        if sp_size <= 1:
            print(f"WARN: sequence parallel size must > 1 but got {sp_size}")
            return

        world_size = dist.get_world_size()
        assert world_size % sp_size == 0, f"world_size {world_size} must be divisible by sp_size({sp_size})"
        SequenceParallelManager._SP_SIZE = sp_size

        pm = ProcessGroupManager(
            world_size // sp_size,
            sp_size,
            dp_axis=0,
            sp_axis=1,
        )
        pm_group = pm.sp_group
        set_sp_comm_group(pm_group)
        SequenceParallelManager._SP_GROUP = pm_group
        return

    @staticmethod
    def get_sp_group():
        return SequenceParallelManager._SP_GROUP

    @staticmethod
    def get_sp_size():
        return SequenceParallelManager._SP_SIZE

    @staticmethod
    def get_sp_group_nums():
        # if 2 sp_size, 8 ranks, group nums is 4
        if SequenceParallelManager.sp_on():
            world_size = torch.distributed.get_world_size()
            return world_size // SequenceParallelManager._SP_SIZE
        else:
            return 0

    @staticmethod
    def get_sp_rank():
        if SequenceParallelManager.sp_on():
            global_rank = torch.distributed.get_rank()
            sp_rank = global_rank % SequenceParallelManager._SP_SIZE
            return sp_rank
        else:
            return 0

    def get_sp_group_rank():
        if SequenceParallelManager.sp_on():
            global_rank = torch.distributed.get_rank()
            sp_group_rank = global_rank // SequenceParallelManager._SP_SIZE
            return sp_group_rank
        else:
            return 0

def sp_split_sequence_by_dim(seq, seqlen_dim=1) -> torch.Tensor:
    """
    split the raw sequence by seqlen_dim
    """
    return split_sequence(seq, SequenceParallelManager.get_sp_group(), seqlen_dim, 'down')

def sp_gather_sequence_by_dim(seq, seqlen_dim=1) -> torch.Tensor:
    """
    gather seqlen_dim to recover raw sequence
    """
    return gather_sequence(seq, SequenceParallelManager.get_sp_group(), seqlen_dim, 'up')

def sp_all_to_all(ts, scatter_dim, gather_dim):
    """
    reorder the tensor's dimension, like [raw_seq_len/sp_size, hidden_dim] to [raw_seq_len, hidden_dim/sp_size]

    scatter_dim: the dimension to split the tensor
    gather_dim: the dimension to concatenate
    """

    return all_to_all_comm(ts, SequenceParallelManager.get_sp_group(), scatter_dim, gather_dim)

