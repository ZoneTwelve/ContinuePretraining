from deepspeed.ops.op_builder.cpu_adam import CPUAdamBuilder
from deepspeed.ops.op_builder.fused_adam import FusedAdamBuilder
from deepspeed.ops.op_builder.quantizer import QuantizerBuilder

CPUAdamBuilder().load()
FusedAdamBuilder().load()
QuantizerBuilder().load()
