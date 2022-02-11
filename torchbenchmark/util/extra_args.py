import argparse
from typing import List
from torchbenchmark.util.backends.fx2trt import enable_fx2trt
from torchbenchmark.util.backends.flops import enable_flops

# Dispatch arguments based on model type
def parse_args(model: 'torchbenchmark.util.model.BenchmarkModel', extra_args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flops", action='store_true', help="enable flops counting")
    parser.add_argument("--fx2trt", action='store_true', help="enable fx2trt")
    args = parser.parse_args(extra_args)
    args.device = model.device
    args.jit = model.jit
    args.batch_size = model.batch_size
    if not (model.device == "cuda" and model.test == "eval"):
        args.fx2trt = False
    if hasattr(model, 'TORCHVISION_MODEL') and model.TORCHVISION_MODEL:
        args.cudagraph = False
        if model.test == "train" and args.flops:
            args.flops = False
            raise NotImplementedError("Flops is only enabled for TorchVision model inference tests")
    elif args.flops:
        args.flops = False
        raise NotImplementedError("Flops is only enabled for TorchVision model inference tests")
    return args

def apply_args(model: 'torchbenchmark.util.model.BenchmarkModel', args: argparse.Namespace):
    # apply fx2trt
    if args.fx2trt:
        assert not args.jit, "fx2trt with JIT is not available."
        module, exmaple_inputs = model.get_module()
        model.set_module(enable_fx2trt(args.batch_size, fp16=False, model=module, example_inputs=exmaple_inputs))
    if args.flops:
        enable_flops(model)
