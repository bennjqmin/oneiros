"""
Graph compiler: converts Oneiros graph JSON into a PyTorch nn.Module.
No exec() or eval() is used – the model is built entirely from Python objects.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Union


# ── Activation helpers ────────────────────────────────────────────────────────

ACTIVATION_FNS: dict[str, Any] = {
    "relu": F.relu,
    "gelu": F.gelu,
    "silu": F.silu,
    "elu": F.elu,
    "leaky_relu": F.leaky_relu,
    "mish": F.mish,
    "softplus": F.softplus,
    "hardswish": F.hardswish,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "softmax": lambda x: F.softmax(x, dim=-1),
    "none": None,
}


# ── Topological sort ──────────────────────────────────────────────────────────

def topological_sort(nodes: list[dict], edges: list[dict]) -> list[dict] | None:
    node_map = {n["id"]: n for n in nodes}
    in_degree: dict[str, int] = {n["id"]: 0 for n in nodes}
    adj: dict[str, list[str]] = {n["id"]: [] for n in nodes}

    for e in edges:
        adj[e["source"]].append(e["target"])
        in_degree[e["target"]] += 1

    queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
    sorted_nodes: list[dict] = []

    while queue:
        nid = queue.popleft()
        sorted_nodes.append(node_map[nid])
        for neighbor in adj[nid]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return sorted_nodes if len(sorted_nodes) == len(nodes) else None


# ── Shape types ───────────────────────────────────────────────────────────────

@dataclass
class FlatShape:
    kind: Literal["flat"] = "flat"
    features: int = 0


@dataclass
class SpatialShape:
    kind: Literal["spatial"] = "spatial"
    channels: int = 1
    height: int = 1
    width: int = 1


NodeShape = Union[FlatShape, SpatialShape]


def flat_features(shape: NodeShape) -> int:
    if shape.kind == "flat":
        return shape.features
    return shape.channels * shape.height * shape.width


def conv_dim(dim: int, kernel: int, stride: int, padding: int) -> int:
    return max(1, math.floor((dim + 2 * padding - kernel) / stride + 1))


# ── Shape inference ───────────────────────────────────────────────────────────

def infer_shapes(sorted_nodes: list[dict], edges: list[dict]) -> dict[str, NodeShape]:
    parents: dict[str, list[str]] = {n["id"]: [] for n in sorted_nodes}
    for e in edges:
        parents[e["target"]].append(e["source"])

    shapes: dict[str, NodeShape] = {}

    for node in sorted_nodes:
        ntype = node.get("type", "")
        data = node.get("data", {})
        nid = node["id"]
        node_parents = parents[nid]
        first_parent_shape = shapes.get(node_parents[0]) if node_parents else None

        if ntype == "inputNode":
            shapes[nid] = SpatialShape(
                channels=int(data.get("channels", 1)),
                height=int(data.get("height", 28)),
                width=int(data.get("width", 28)),
            )

        elif ntype == "conv2dNode":
            in_c = first_parent_shape.channels if isinstance(first_parent_shape, SpatialShape) else 1
            in_h = first_parent_shape.height if isinstance(first_parent_shape, SpatialShape) else 1
            in_w = first_parent_shape.width if isinstance(first_parent_shape, SpatialShape) else 1
            out_c = int(data.get("outChannels", 32))
            k, s, p = int(data.get("kernelSize", 3)), int(data.get("stride", 1)), int(data.get("padding", 0))
            shapes[nid] = SpatialShape(channels=out_c, height=conv_dim(in_h, k, s, p), width=conv_dim(in_w, k, s, p))

        elif ntype == "conv1dNode":
            out_c = int(data.get("outChannels", 64))
            shapes[nid] = FlatShape(features=out_c)

        elif ntype in ("maxPool2dNode", "avgPool2dNode"):
            in_c = first_parent_shape.channels if isinstance(first_parent_shape, SpatialShape) else 1
            in_h = first_parent_shape.height if isinstance(first_parent_shape, SpatialShape) else 1
            in_w = first_parent_shape.width if isinstance(first_parent_shape, SpatialShape) else 1
            k = int(data.get("kernelSize", 2))
            s = int(data.get("stride", k))
            p = int(data.get("padding", 0))
            shapes[nid] = SpatialShape(channels=in_c, height=conv_dim(in_h, k, s, p), width=conv_dim(in_w, k, s, p))

        elif ntype == "adaptiveAvgPool2dNode":
            in_c = first_parent_shape.channels if isinstance(first_parent_shape, SpatialShape) else 1
            out_sz = int(data.get("outputSize", 1))
            shapes[nid] = FlatShape(features=in_c) if out_sz == 1 else SpatialShape(channels=in_c, height=out_sz, width=out_sz)

        elif ntype == "flattenNode":
            features = flat_features(first_parent_shape) if first_parent_shape else 0
            shapes[nid] = FlatShape(features=features)

        elif ntype == "convTranspose2dNode":
            in_h = first_parent_shape.height if isinstance(first_parent_shape, SpatialShape) else 1
            in_w = first_parent_shape.width if isinstance(first_parent_shape, SpatialShape) else 1
            out_c = int(data.get("outChannels", 32))
            k     = int(data.get("kernelSize", 2))
            s     = int(data.get("stride", 2))
            p     = int(data.get("padding", 0))
            op    = int(data.get("outputPadding", 0))
            out_h = (in_h - 1) * s - 2 * p + k + op
            out_w = (in_w - 1) * s - 2 * p + k + op
            shapes[nid] = SpatialShape(channels=out_c, height=out_h, width=out_w)

        elif ntype == "upsampleNode":
            in_c = first_parent_shape.channels if isinstance(first_parent_shape, SpatialShape) else 1
            in_h = first_parent_shape.height if isinstance(first_parent_shape, SpatialShape) else 1
            in_w = first_parent_shape.width if isinstance(first_parent_shape, SpatialShape) else 1
            sf = float(data.get("scaleFactor", 2))
            shapes[nid] = SpatialShape(channels=in_c, height=int(in_h * sf), width=int(in_w * sf))

        elif ntype == "backboneNode":
            model_name = str(data.get("model", "resnet18"))
            _BACKBONE_FEATS = {"resnet18": 512, "resnet34": 512, "resnet50": 2048,
                               "mobilenet_v2": 1280, "efficientnet_b0": 1280, "vgg16": 4096}
            shapes[nid] = FlatShape(features=_BACKBONE_FEATS.get(model_name, 512))

        elif ntype in ("dropoutNode", "batchNormNode", "activationNode", "gaussianNoiseNode", "layerNormNode"):
            if first_parent_shape:
                shapes[nid] = first_parent_shape

        elif ntype == "reshapeNode":
            target = int(data.get("targetFeatures", -1))
            if target == -1 and first_parent_shape:
                shapes[nid] = first_parent_shape
            else:
                shapes[nid] = FlatShape(features=target)

        elif ntype == "embeddingNode":
            emb_dim = int(data.get("embeddingDim", 128))
            seq_len = flat_features(first_parent_shape) if first_parent_shape else 1
            shapes[nid] = FlatShape(features=seq_len * emb_dim)

        elif ntype == "positionalEncodingNode":
            shapes[nid] = FlatShape(features=int(data.get("dModel", 256)))

        elif ntype == "feedForwardNode":
            shapes[nid] = FlatShape(features=int(data.get("dModel", 256)))

        elif ntype == "multiHeadAttentionNode":
            shapes[nid] = FlatShape(features=int(data.get("embedDim", 256)))

        elif ntype == "concatenateNode":
            parent_shapes = [shapes[pid] for pid in node_parents if pid in shapes]
            if parent_shapes and all(s.kind == "flat" for s in parent_shapes):
                shapes[nid] = FlatShape(features=sum(flat_features(s) for s in parent_shapes))
            elif parent_shapes and all(isinstance(s, SpatialShape) for s in parent_shapes):
                ref = parent_shapes[0]
                if isinstance(ref, SpatialShape) and all(
                    isinstance(s, SpatialShape) and s.height == ref.height and s.width == ref.width
                    for s in parent_shapes
                ):
                    shapes[nid] = SpatialShape(
                        channels=sum(s.channels for s in parent_shapes if isinstance(s, SpatialShape)),
                        height=ref.height,
                        width=ref.width,
                    )
                elif first_parent_shape:
                    shapes[nid] = first_parent_shape
            elif first_parent_shape:
                shapes[nid] = first_parent_shape

        elif ntype in ("addNode", "multiplyNode"):
            if first_parent_shape:
                shapes[nid] = first_parent_shape

        elif ntype in ("rnnNode", "gruNode", "lstmNode"):
            hidden = int(data.get("hiddenSize", 128))
            bidir = bool(data.get("bidirectional", False))
            shapes[nid] = FlatShape(features=hidden * (2 if bidir else 1))

        elif ntype == "transformerEncoderNode":
            shapes[nid] = FlatShape(features=int(data.get("dModel", 256)))

        elif ntype == "denseNode":
            units = int(data.get("units", 128))
            in_features = sum(flat_features(shapes[pid]) for pid in node_parents if pid in shapes)
            shapes[nid] = FlatShape(features=units)
            shapes[f"{nid}__in"] = FlatShape(features=in_features)

        elif ntype == "outputNode":
            classes = int(data.get("classes", 10))
            in_features = sum(flat_features(shapes[pid]) for pid in node_parents if pid in shapes)
            shapes[nid] = FlatShape(features=classes)
            shapes[f"{nid}__in"] = FlatShape(features=in_features)

    return shapes


# ── Operation descriptors ─────────────────────────────────────────────────────

@dataclass
class Op:
    kind: str
    node_id: str
    layer_attr: str | None = None
    activation: str | None = None
    input_node_ids: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)  # node-type-specific extra state


# ── Positional encoding helper ────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ── Dynamic graph model ───────────────────────────────────────────────────────

class GraphModel(nn.Module):
    def __init__(self, ops: list[Op]):
        super().__init__()
        self._ops = ops

    def _activate(self, tensor: torch.Tensor, activation: str | None) -> torch.Tensor:
        if not activation:
            return tensor
        fn: Callable | None = ACTIVATION_FNS.get(activation)
        return fn(tensor) if fn is not None else tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var: dict[str, torch.Tensor] = {}

        for op in self._ops:
            if op.kind == "input":
                var[op.node_id] = x

            elif op.kind == "conv2d":
                inp = self._single(var, op.input_node_ids, x)
                layer = getattr(self, op.layer_attr)  # type: ignore[arg-type]
                var[op.node_id] = self._activate(layer(inp), op.activation)

            elif op.kind == "conv1d":
                inp = self._single(var, op.input_node_ids, x)
                layer = getattr(self, op.layer_attr)  # type: ignore[arg-type]
                result = self._activate(layer(inp.unsqueeze(-1)).squeeze(-1), op.activation)
                var[op.node_id] = result

            elif op.kind in ("pool", "avgpool", "adapool"):
                inp = self._single(var, op.input_node_ids, x)
                layer = getattr(self, op.layer_attr)  # type: ignore[arg-type]
                out = layer(inp)
                if op.kind == "adapool" and op.extra.get("flatten"):
                    out = out.view(out.size(0), -1)
                var[op.node_id] = out

            elif op.kind == "flatten":
                inp = self._single(var, op.input_node_ids, x)
                var[op.node_id] = inp.view(inp.size(0), -1)

            elif op.kind in ("dropout", "batchnorm"):
                inp = self._single(var, op.input_node_ids, x)
                layer = getattr(self, op.layer_attr)  # type: ignore[arg-type]
                var[op.node_id] = layer(inp)

            elif op.kind == "activation":
                inp = self._single(var, op.input_node_ids, x)
                var[op.node_id] = self._activate(inp, op.activation)

            elif op.kind in ("rnn", "gru"):
                inp = self._single(var, op.input_node_ids, x)
                rnn_layer = getattr(self, op.layer_attr)  # type: ignore[arg-type]
                _, hn = rnn_layer(inp.unsqueeze(1))
                bidir = op.extra.get("bidirectional", False)
                var[op.node_id] = hn.transpose(0, 1).reshape(inp.size(0), -1) if bidir else hn[-1]

            elif op.kind == "lstm":
                inp = self._single(var, op.input_node_ids, x)
                lstm_layer = getattr(self, op.layer_attr)  # type: ignore[arg-type]
                _, (hn, _) = lstm_layer(inp.unsqueeze(1))
                bidir = op.extra.get("bidirectional", False)
                var[op.node_id] = hn.transpose(0, 1).reshape(inp.size(0), -1) if bidir else hn[-1]

            elif op.kind == "transformer_encoder":
                inp = self._single(var, op.input_node_ids, x)
                proj = getattr(self, op.layer_attr)  # type: ignore[arg-type]
                enc = getattr(self, op.extra["encoder_attr"])
                var[op.node_id] = enc(proj(inp).unsqueeze(1)).squeeze(1)

            elif op.kind == "linear":
                inp = self._gather(var, op.input_node_ids, x)
                layer = getattr(self, op.layer_attr)  # type: ignore[arg-type]
                var[op.node_id] = self._activate(layer(inp), op.activation)

            elif op.kind == "conv_transpose2d":
                inp = self._single(var, op.input_node_ids, x)
                layer = getattr(self, op.layer_attr)  # type: ignore[arg-type]
                var[op.node_id] = self._activate(layer(inp), op.activation)

            elif op.kind == "upsample":
                inp = self._single(var, op.input_node_ids, x)
                layer = getattr(self, op.layer_attr)  # type: ignore[arg-type]
                var[op.node_id] = layer(inp)

            elif op.kind == "backbone":
                inp = self._single(var, op.input_node_ids, x)
                backbone = getattr(self, op.layer_attr)  # type: ignore[arg-type]
                out_layer = op.extra.get("out_layer", "avgpool")
                if out_layer == "avgpool":
                    # Standard feature extraction: run full backbone, flatten
                    out = backbone(inp)
                    if out.dim() > 2:
                        out = out.view(out.size(0), -1)
                else:
                    # Tap intermediate layer (ResNet only)
                    import torchvision.models.resnet as _rn  # type: ignore[import]
                    if hasattr(backbone, out_layer):
                        # Run up to that layer manually
                        h = backbone.conv1(inp)
                        h = backbone.bn1(h)
                        h = backbone.relu(h)
                        h = backbone.maxpool(h)
                        h = backbone.layer1(h)
                        h = backbone.layer2(h)
                        if out_layer in ("layer3", "layer4"):
                            h = backbone.layer3(h)
                        if out_layer == "layer4":
                            h = backbone.layer4(h)
                        out = h
                    else:
                        out = backbone(inp)
                var[op.node_id] = out

            elif op.kind == "output":
                inp = self._gather(var, op.input_node_ids, x)
                layer = getattr(self, op.layer_attr)  # type: ignore[arg-type]
                var[op.node_id] = layer(inp)

            elif op.kind == "gaussian_noise":
                inp = self._single(var, op.input_node_ids, x)
                std = float(op.extra.get("std", 0.1))
                var[op.node_id] = inp + torch.randn_like(inp) * std if self.training else inp

            elif op.kind == "reshape":
                inp = self._single(var, op.input_node_ids, x)
                target = int(op.extra.get("target_features", -1))
                var[op.node_id] = inp.view(inp.size(0), -1) if target == -1 else inp.view(inp.size(0), target)

            elif op.kind == "layer_norm":
                inp = self._single(var, op.input_node_ids, x)
                layer = getattr(self, op.layer_attr)  # type: ignore[arg-type]
                var[op.node_id] = layer(inp)

            elif op.kind == "embedding":
                inp = self._single(var, op.input_node_ids, x)
                layer = getattr(self, op.layer_attr)  # type: ignore[arg-type]
                out = layer(inp.long())
                var[op.node_id] = out.view(out.size(0), -1)

            elif op.kind == "positional_encoding":
                inp = self._single(var, op.input_node_ids, x)
                layer = getattr(self, op.layer_attr)  # type: ignore[arg-type]
                var[op.node_id] = layer(inp.unsqueeze(1)).squeeze(1)

            elif op.kind == "feed_forward":
                inp = self._single(var, op.input_node_ids, x)
                layer = getattr(self, op.layer_attr)  # type: ignore[arg-type]
                var[op.node_id] = layer(inp)

            elif op.kind == "multi_head_attention":
                inp = self._single(var, op.input_node_ids, x)
                layer = getattr(self, op.layer_attr)  # type: ignore[arg-type]
                seq = inp.unsqueeze(1)
                out, _ = layer(seq, seq, seq)
                var[op.node_id] = out.squeeze(1)

            elif op.kind == "concatenate":
                dim = int(op.extra.get("dim", -1))
                tensors = [var.get(nid, x) for nid in op.input_node_ids]
                if len(tensors) == 1:
                    var[op.node_id] = tensors[0]
                else:
                    flat_tensors = []
                    for t in tensors:
                        flat_tensors.append(t if t.dim() > 2 or dim != -1 else t.view(t.size(0), -1))
                    var[op.node_id] = torch.cat(flat_tensors, dim=dim)

            elif op.kind == "add":
                a = var.get(op.input_node_ids[0], x)
                b = var.get(op.input_node_ids[1], x)
                var[op.node_id] = a + b

            elif op.kind == "multiply":
                a = var.get(op.input_node_ids[0], x)
                b = var.get(op.input_node_ids[1], x)
                var[op.node_id] = a * b

        return var[self._ops[-1].node_id]

    @staticmethod
    def _single(var: dict[str, torch.Tensor], node_ids: list[str], fallback: torch.Tensor) -> torch.Tensor:
        if not node_ids:
            return fallback
        return var.get(node_ids[0], fallback)

    @staticmethod
    def _gather(var: dict[str, torch.Tensor], node_ids: list[str], fallback: torch.Tensor) -> torch.Tensor:
        if not node_ids:
            return fallback.view(fallback.size(0), -1)
        tensors = [var.get(nid, fallback).view(var.get(nid, fallback).size(0), -1) for nid in node_ids]
        return tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=-1)


def _ordered_parents(nid: str, edges: list[dict], handle_order: list[str] | None = None) -> list[str]:
    incoming = [e for e in edges if e["target"] == nid]
    if handle_order:
        result: list[str] = []
        for handle in handle_order:
            matched = [e for e in incoming if e.get("targetHandle") == handle]
            if matched:
                result.append(matched[0]["source"])
        return result
    return [e["source"] for e in incoming]


# ── Public entry point ────────────────────────────────────────────────────────

def build_model(graph: dict) -> tuple[GraphModel, list[str]]:
    nodes: list[dict] = graph.get("nodes", [])
    edges: list[dict] = graph.get("edges", [])

    if not nodes:
        return GraphModel([]), ["Graph is empty"]

    sorted_nodes = topological_sort(nodes, edges)
    if sorted_nodes is None:
        return GraphModel([]), ["Cycle detected in graph"]

    shapes = infer_shapes(sorted_nodes, edges)

    parents: dict[str, list[str]] = {n["id"]: [] for n in sorted_nodes}
    for e in edges:
        parents[e["target"]].append(e["source"])

    model = GraphModel.__new__(GraphModel)
    nn.Module.__init__(model)
    ops: list[Op] = []

    for node in sorted_nodes:
        ntype = node.get("type", "")
        nid = node["id"]
        safe_id = nid.replace("-", "_")
        data = node.get("data", {})
        node_parents = parents[nid]
        first_parent_shape = shapes.get(node_parents[0]) if node_parents else None

        if ntype == "inputNode":
            ops.append(Op(kind="input", node_id=nid))

        elif ntype == "conv2dNode":
            in_c = first_parent_shape.channels if isinstance(first_parent_shape, SpatialShape) else 1
            out_c = int(data.get("outChannels", 32))
            k, s, p = int(data.get("kernelSize", 3)), int(data.get("stride", 1)), int(data.get("padding", 0))
            dil = int(data.get("dilation", 1))
            grp = int(data.get("groups", 1))
            bias = bool(data.get("bias", True))
            pmode = str(data.get("paddingMode", "zeros"))
            attr = f"conv_{safe_id}"
            setattr(model, attr, nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p,
                                           dilation=dil, groups=grp, bias=bias, padding_mode=pmode))
            ops.append(Op(kind="conv2d", node_id=nid, layer_attr=attr, activation=str(data.get("activation", "relu")), input_node_ids=node_parents))

        elif ntype == "conv1dNode":
            in_f = flat_features(first_parent_shape) if first_parent_shape else 1
            out_c = int(data.get("outChannels", 64))
            k, s, p = int(data.get("kernelSize", 3)), int(data.get("stride", 1)), int(data.get("padding", 1))
            dil = int(data.get("dilation", 1))
            grp = int(data.get("groups", 1))
            bias = bool(data.get("bias", True))
            attr = f"conv1d_{safe_id}"
            setattr(model, attr, nn.Conv1d(in_f, out_c, kernel_size=k, stride=s, padding=p,
                                           dilation=dil, groups=grp, bias=bias))
            ops.append(Op(kind="conv1d", node_id=nid, layer_attr=attr, activation=str(data.get("activation", "relu")), input_node_ids=node_parents))

        elif ntype == "maxPool2dNode":
            k = int(data.get("kernelSize", 2))
            s = int(data.get("stride", k))
            pad = int(data.get("padding", 0))
            dil = int(data.get("dilation", 1))
            ceil_mode = bool(data.get("ceilMode", False))
            attr = f"pool_{safe_id}"
            setattr(model, attr, nn.MaxPool2d(kernel_size=k, stride=s, padding=pad, dilation=dil, ceil_mode=ceil_mode))
            ops.append(Op(kind="pool", node_id=nid, layer_attr=attr, input_node_ids=node_parents))

        elif ntype == "avgPool2dNode":
            k = int(data.get("kernelSize", 2))
            s = int(data.get("stride", k))
            pad = int(data.get("padding", 0))
            ceil_mode = bool(data.get("ceilMode", False))
            count_pad = bool(data.get("countIncludePad", True))
            attr = f"avgpool_{safe_id}"
            setattr(model, attr, nn.AvgPool2d(kernel_size=k, stride=s, padding=pad,
                                              ceil_mode=ceil_mode, count_include_pad=count_pad))
            ops.append(Op(kind="avgpool", node_id=nid, layer_attr=attr, input_node_ids=node_parents))

        elif ntype == "adaptiveAvgPool2dNode":
            out_sz = int(data.get("outputSize", 1))
            attr = f"adapool_{safe_id}"
            setattr(model, attr, nn.AdaptiveAvgPool2d(out_sz))
            ops.append(Op(kind="adapool", node_id=nid, layer_attr=attr, input_node_ids=node_parents, extra={"flatten": out_sz == 1}))

        elif ntype == "flattenNode":
            ops.append(Op(kind="flatten", node_id=nid, input_node_ids=node_parents))

        elif ntype == "convTranspose2dNode":
            in_c = first_parent_shape.channels if isinstance(first_parent_shape, SpatialShape) else 1
            out_c = int(data.get("outChannels", 32))
            k     = int(data.get("kernelSize", 2))
            s     = int(data.get("stride", 2))
            p     = int(data.get("padding", 0))
            out_p = int(data.get("outputPadding", 0))
            grp   = int(data.get("groups", 1))
            bias  = bool(data.get("bias", True))
            attr  = f"convt_{safe_id}"
            setattr(model, attr, nn.ConvTranspose2d(in_c, out_c, kernel_size=k, stride=s,
                                                     padding=p, output_padding=out_p,
                                                     groups=grp, bias=bias))
            ops.append(Op(kind="conv_transpose2d", node_id=nid, layer_attr=attr,
                          activation=str(data.get("activation", "none")), input_node_ids=node_parents))

        elif ntype == "upsampleNode":
            sf   = float(data.get("scaleFactor", 2))
            mode = str(data.get("mode", "nearest"))
            attr = f"up_{safe_id}"
            kwargs: dict = {"scale_factor": sf, "mode": mode}
            if mode != "nearest":
                kwargs["align_corners"] = False
            setattr(model, attr, nn.Upsample(**kwargs))
            ops.append(Op(kind="upsample", node_id=nid, layer_attr=attr, input_node_ids=node_parents))

        elif ntype == "backboneNode":
            model_name  = str(data.get("model", "resnet18"))
            pretrained  = bool(data.get("pretrained", True))
            freeze      = bool(data.get("freeze", False))
            out_layer   = str(data.get("outputLayer", "avgpool"))
            attr = f"backbone_{safe_id}"
            try:
                import torchvision.models as tv_models  # type: ignore[import]
                _weights = "DEFAULT" if pretrained else None
                _backbone = getattr(tv_models, model_name)(weights=_weights)
                # Remove final classifier / fc
                if hasattr(_backbone, "fc"):
                    _backbone.fc = nn.Identity()
                elif hasattr(_backbone, "classifier"):
                    _backbone.classifier = nn.Identity()
                if freeze:
                    for param in _backbone.parameters():
                        param.requires_grad_(False)
                setattr(model, attr, _backbone)
            except ImportError:
                raise RuntimeError(
                    "torchvision is required for BackboneNode. "
                    "Run: pip install torchvision"
                )
            ops.append(Op(kind="backbone", node_id=nid, layer_attr=attr,
                          input_node_ids=node_parents,
                          extra={"out_layer": out_layer}))

        elif ntype == "dropoutNode":
            attr = f"drop_{safe_id}"
            setattr(model, attr, nn.Dropout(p=float(data.get("p", 0.5))))
            ops.append(Op(kind="dropout", node_id=nid, layer_attr=attr, input_node_ids=node_parents))

        elif ntype == "batchNormNode":
            eps = float(data.get("eps", 1e-5))
            mom = float(data.get("momentum", 0.1))
            affine = bool(data.get("affine", True))
            track = bool(data.get("trackRunningStats", True))
            attr = f"bn_{safe_id}"
            if isinstance(first_parent_shape, SpatialShape):
                setattr(model, attr, nn.BatchNorm2d(first_parent_shape.channels, eps=eps, momentum=mom,
                                                    affine=affine, track_running_stats=track))
            else:
                nf = flat_features(first_parent_shape) if first_parent_shape else 1
                setattr(model, attr, nn.BatchNorm1d(nf, eps=eps, momentum=mom,
                                                    affine=affine, track_running_stats=track))
            ops.append(Op(kind="batchnorm", node_id=nid, layer_attr=attr, input_node_ids=node_parents))

        elif ntype == "activationNode":
            ops.append(Op(kind="activation", node_id=nid, activation=str(data.get("fn", "relu")), input_node_ids=node_parents))

        elif ntype == "rnnNode":
            in_f = flat_features(first_parent_shape) if first_parent_shape else 1
            hidden = int(data.get("hiddenSize", 128))
            layers = int(data.get("numLayers", 1))
            nonlin = str(data.get("nonlinearity", "tanh"))
            drop = float(data.get("dropout", 0))
            bidir = bool(data.get("bidirectional", False))
            attr = f"rnn_{safe_id}"
            setattr(model, attr, nn.RNN(in_f, hidden, num_layers=layers, nonlinearity=nonlin, dropout=drop, bidirectional=bidir, batch_first=True))
            ops.append(Op(kind="rnn", node_id=nid, layer_attr=attr, input_node_ids=node_parents, extra={"bidirectional": bidir}))

        elif ntype == "gruNode":
            in_f = flat_features(first_parent_shape) if first_parent_shape else 1
            hidden = int(data.get("hiddenSize", 128))
            layers = int(data.get("numLayers", 1))
            drop = float(data.get("dropout", 0))
            bidir = bool(data.get("bidirectional", False))
            attr = f"gru_{safe_id}"
            setattr(model, attr, nn.GRU(in_f, hidden, num_layers=layers, dropout=drop, bidirectional=bidir, batch_first=True))
            ops.append(Op(kind="gru", node_id=nid, layer_attr=attr, input_node_ids=node_parents, extra={"bidirectional": bidir}))

        elif ntype == "lstmNode":
            in_f = flat_features(first_parent_shape) if first_parent_shape else 1
            hidden = int(data.get("hiddenSize", 128))
            layers = int(data.get("numLayers", 1))
            drop = float(data.get("dropout", 0))
            bidir = bool(data.get("bidirectional", False))
            proj = int(data.get("projSize", 0))
            attr = f"lstm_{safe_id}"
            setattr(model, attr, nn.LSTM(in_f, hidden, num_layers=layers, dropout=drop,
                                         bidirectional=bidir, batch_first=True, proj_size=proj))
            ops.append(Op(kind="lstm", node_id=nid, layer_attr=attr, input_node_ids=node_parents, extra={"bidirectional": bidir}))

        elif ntype == "transformerEncoderNode":
            in_f = flat_features(first_parent_shape) if first_parent_shape else 1
            d_model = int(data.get("dModel", 256))
            nhead = int(data.get("nhead", 8))
            ff_dim = int(data.get("dimFeedforward", 512))
            num_layers = int(data.get("numLayers", 2))
            drop = float(data.get("dropout", 0.1))
            proj_attr = f"te_proj_{safe_id}"
            enc_attr = f"te_{safe_id}"
            setattr(model, proj_attr, nn.Linear(in_f, d_model))
            te_act = str(data.get("activation", "relu"))
            norm_first = bool(data.get("normFirst", False))
            enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
                                                   dropout=drop, activation=te_act, norm_first=norm_first,
                                                   batch_first=True)
            setattr(model, enc_attr, nn.TransformerEncoder(enc_layer, num_layers=num_layers))
            ops.append(Op(kind="transformer_encoder", node_id=nid, layer_attr=proj_attr, input_node_ids=node_parents, extra={"encoder_attr": enc_attr}))

        elif ntype == "denseNode":
            in_shape = shapes.get(f"{nid}__in")
            out_shape = shapes.get(nid)
            if in_shape is None or out_shape is None:
                continue
            attr = f"layer_{safe_id}"
            setattr(model, attr, nn.Linear(flat_features(in_shape), flat_features(out_shape)))
            ops.append(Op(kind="linear", node_id=nid, layer_attr=attr, activation=str(data.get("activation", "relu")), input_node_ids=node_parents))

        elif ntype == "outputNode":
            in_shape = shapes.get(f"{nid}__in")
            out_shape = shapes.get(nid)
            if in_shape is None or out_shape is None:
                continue
            attr = f"layer_{safe_id}"
            setattr(model, attr, nn.Linear(flat_features(in_shape), flat_features(out_shape)))
            ops.append(Op(kind="output", node_id=nid, layer_attr=attr, input_node_ids=node_parents))

        elif ntype == "gaussianNoiseNode":
            ops.append(Op(kind="gaussian_noise", node_id=nid, input_node_ids=node_parents, extra={"std": float(data.get("std", 0.1))}))

        elif ntype == "reshapeNode":
            ops.append(Op(kind="reshape", node_id=nid, input_node_ids=node_parents, extra={"target_features": int(data.get("targetFeatures", -1))}))

        elif ntype == "layerNormNode":
            nf = flat_features(first_parent_shape) if first_parent_shape else 1
            eps = float(data.get("eps", 1e-5))
            attr = f"ln_{safe_id}"
            setattr(model, attr, nn.LayerNorm(nf, eps=eps))
            ops.append(Op(kind="layer_norm", node_id=nid, layer_attr=attr, input_node_ids=node_parents))

        elif ntype == "embeddingNode":
            num_emb = int(data.get("numEmbeddings", 1000))
            emb_dim = int(data.get("embeddingDim", 128))
            attr = f"emb_{safe_id}"
            setattr(model, attr, nn.Embedding(num_emb, emb_dim))
            ops.append(Op(kind="embedding", node_id=nid, layer_attr=attr, input_node_ids=node_parents))

        elif ntype == "positionalEncodingNode":
            d_model = int(data.get("dModel", 256))
            max_len = int(data.get("maxLen", 512))
            attr = f"pe_{safe_id}"
            setattr(model, attr, PositionalEncoding(d_model, max_len=max_len))
            ops.append(Op(kind="positional_encoding", node_id=nid, layer_attr=attr, input_node_ids=node_parents))

        elif ntype == "feedForwardNode":
            d_model = int(data.get("dModel", 256))
            ff_dim = int(data.get("ffDim", 512))
            drop = float(data.get("dropout", 0.1))
            act_name = str(data.get("activation", "relu"))
            act = nn.GELU() if act_name == "gelu" else nn.ReLU()
            attr = f"ff_{safe_id}"
            setattr(model, attr, nn.Sequential(
                nn.Linear(d_model, ff_dim),
                act,
                nn.Dropout(drop),
                nn.Linear(ff_dim, d_model),
            ))
            ops.append(Op(kind="feed_forward", node_id=nid, layer_attr=attr, input_node_ids=node_parents))

        elif ntype == "multiHeadAttentionNode":
            embed_dim = int(data.get("embedDim", 256))
            num_heads = int(data.get("numHeads", 8))
            drop = float(data.get("dropout", 0.1))
            attr = f"mha_{safe_id}"
            setattr(model, attr, nn.MultiheadAttention(embed_dim, num_heads, dropout=drop, batch_first=True))
            ops.append(Op(kind="multi_head_attention", node_id=nid, layer_attr=attr, input_node_ids=node_parents))

        elif ntype == "concatenateNode":
            ops.append(Op(kind="concatenate", node_id=nid, input_node_ids=node_parents, extra={"dim": int(data.get("dim", -1))}))

        elif ntype == "addNode":
            ordered = _ordered_parents(nid, edges, ["in_a", "in_b"])
            ops.append(Op(kind="add", node_id=nid, input_node_ids=ordered))

        elif ntype == "multiplyNode":
            ordered = _ordered_parents(nid, edges, ["in_a", "in_b"])
            ops.append(Op(kind="multiply", node_id=nid, input_node_ids=ordered))

    model._ops = ops
    return model, []
