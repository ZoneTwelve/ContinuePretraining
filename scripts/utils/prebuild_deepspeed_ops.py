from deepspeed.ops.op_builder.cpu_adam import CPUAdamBuilder
from deepspeed.ops.op_builder.fused_adam import FusedAdamBuilder

CPUAdamBuilder().load()
FusedAdamBuilder().load()
