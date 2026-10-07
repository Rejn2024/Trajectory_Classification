"""Capture a fixed-size PPO forward/backward pass without changing optimizer steps."""

import torch


class CUDABatchBackward:
    """Reuse GPU launches for full minibatches; keep Adam and AMP scaling eager.

    The caller supplies a deterministic loss/backward function and owns clipping,
    optimizer steps, and scaler updates. Warmup computes gradients only: it must
    not advance parameters, the optimizer, or the scaler. Inputs and gradients
    retain their capture-time addresses, including after eager remainder batches.
    Construct once per PPO update so changed shapes/settings cannot reuse a stale
    graph and memory use stays bounded to one minibatch graph.
    """

    def __init__(self, backward, parameters, batch):
        self.parameters = tuple(parameters)
        self.inputs = tuple(tensor.detach().clone() for tensor in batch)
        device = self.inputs[0].device
        with torch.cuda.device(device):
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._clear_gradients()
                    backward(self.inputs)
            torch.cuda.current_stream(device).wait_stream(stream)
            self._clear_gradients()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                # Drop the autograd graph; replay only needs captured GPU work.
                self.loss = backward(self.inputs).detach()
            self.gradients = tuple(parameter.grad for parameter in self.parameters)

    def _clear_gradients(self):
        for parameter in self.parameters:
            parameter.grad = None

    def __call__(self, batch):
        for target, source in zip(self.inputs, batch):
            target.copy_(source)
        # An eager remainder batch can replace .grad. Restore graph-owned buffers.
        for parameter, gradient in zip(self.parameters, self.gradients):
            parameter.grad = gradient
        self.graph.replay()
        # Each replay overwrites self.loss. Preserve each minibatch's diagnostic.
        return self.loss.clone()
