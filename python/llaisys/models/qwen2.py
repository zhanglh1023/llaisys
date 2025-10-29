from typing import Sequence
from pathlib import Path
import json
import ctypes
import torch
from safetensors import safe_open
import numpy as np
from ..libllaisys import LIB_LLAISYS as C
from ..libllaisys import DeviceType
from ..libllaisys.llaisys_types import DataType
from ..libllaisys.qwen2 import LlaisysQwen2Meta


import safetensors

def _make_tensor_from_numpy(arr: np.ndarray, device: DeviceType) :
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    
    ndim = arr.ndim
    shape_ctypes = (ctypes.c_size_t* ndim)(*arr.shape)

    if arr.dtype == np.float32:
        dtype = DataType.F32
    elif arr.dtype == np.float16:
        dtype = DataType.F16
    elif arr.dtype == np.uint16:
        dtype = DataType.BF16
    else:
        raise TypeError(f"Unsupported numpy dtype: {arr.dtype}")
    
    t = C.tensorCreate(shape_ctypes, ctypes.c_size_t(ndim), dtype, device, 0)
    C.tensorLoad(t, ctypes.c_void_p(arr.ctypes.data))
    return t
def _torch_to_numpy(t: torch.Tensor)->np.ndarray:
    t = t.contiguous()

    if t.dtype == torch.bfloat16:
        arr = t.cpu().view(torch.uint16).numpy()
        return arr
    
    if t.dtype == torch.float16:
        arr = t.cpu().numpy().astype(np.float16, copy=False)
        return arr
    if t.dtype == torch.float32:
        arr = t.cpu().numpy().astype(np.float32, copy=False)
    
    raise TypeError(f"Unsupported tensor dtype: {t.dtype}")

class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        # TODO: Implement model constructor

        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self.weights = None
        self.meta = None

        cfg_path = self.model_path / "config.json"

        if not cfg_path.exists():
            raise FileNotFoundError(f"config.json not found in {self.model_path}")
        
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        
        eos_token_id = int(cfg.get("eos_token_id", 151643))
        hidden_size = int(cfg.get("hidden_size", 1536))
        intermediate_size = int(cfg.get("intermediate_size", 8960))
        max_position_embeddings = int(cfg.get("max_position_embeddings", 131072))
        num_attention_heads = int(cfg.get("num_attention_heads", 12))
        num_hidden_layers = int(cfg.get("num_hidden_layers", 28))
        num_key_value_heads = int(cfg.get("num_key_value_heads", 2))
        rms_norm_eps = float(cfg.get("rms_norm_eps", 1e-06))
        rope_theta = float(cfg.get("rop_theta", 10000.0))
        vocab_size = int(cfg.get("vocab_size", 151936))

        head_dimension = hidden_size // num_attention_heads

        torch_dtype = str(cfg.get("torch_dtype", "float32")).lower()

        if "bfloat16" in torch_dtype or "bf16" in torch_dtype :
            dtype = DataType.BF16
            print("[Qwen2] Loading weights as BF16 (using uint16 carrier)...")
        elif "float16" in torch_dtype or "fp16" in torch_dtype:
            dtype = DataType.F16
            print("[Qwen2] Loading weights as FP16 ...")
        else :
            dtype = DataType.F32
            print("[Qwen2] Loading weights as FP32 ...")
        
        meta = LlaisysQwen2Meta()
        meta.dtype = dtype
        meta.nlayer = num_hidden_layers
        meta.hs = hidden_size
        meta.nh = num_attention_heads
        meta.nkvh = num_key_value_heads
        meta.dh = head_dimension
        meta.di = intermediate_size
        meta.maxseq = max_position_embeddings
        meta.voc = vocab_size
        meta.epsilon = rms_norm_eps
        meta.theta = rope_theta
        meta.end_token = eos_token_id

        self.meta = meta

        self.model = C.llaisysQwen2ModelCreate(meta, device, None, 0)

        if not self.model :
            raise RuntimeError("Failed to create Qwen2 model.")
        self.weights = C.llaisysQwen2ModelWeights(self.model)

        for file in sorted(self.model_path.glob("*.safetensors")):
            data_ = safetensors.safe_open(file, framework="pt", device="cpu")
            for name_ in data_.keys():
                ## TODO: load the model weights
                tt = data_.get_tensor(name_)
                arr = _torch_to_numpy(tt)
                t = _make_tensor_from_numpy(arr, device)
                self._assign_weight_by_name(name_, t)
    
    def _assign_weight_by_name(self, name: str, t):
        weights = self.weights.contents
        nlayer = self.meta.nlayer

        if name == "model.embed_tokens.weight":
            weights.in_embed = t
            return
        if name == "lm_head.weight":
            weights.out_embed = t
            return
        if name == "model.norm.weight":
            weights.out_norm_w = t
            return
        prefix = "model.layers."
        if not name.startswith(prefix):
            print(f" [SKIP] not a layer param")
            return
        rest = name[len(prefix):]
        try:
            i_str, tail = rest.split(".", 1)
            i = int(i_str)
        except Exception:
            print(f" [SKIP] bad name: {rest}")
            return
        if i < 0 or i >= nlayer:
            print(f" [SKIP] layer {i} out of range (nlayer={nlayer})")
            return
        if tail == "input_layernorm.weight": weights.attn_norm_w[i] = t;  return
        if tail == "post_attention_layernorm.weight": weights.mlp_norm_w[i] = t;  return
        if tail == "self_attn.q_proj.weight": weights.attn_q_w[i] = t;  return
        if tail == "self_attn.q_proj.bias":   weights.attn_q_b[i] = t;  return
        if tail == "self_attn.k_proj.weight": weights.attn_k_w[i] = t;  return
        if tail == "self_attn.k_proj.bias":   weights.attn_k_b[i] = t;  return
        if tail == "self_attn.v_proj.weight": weights.attn_v_w[i] = t;  return
        if tail == "self_attn.v_proj.bias":   weights.attn_v_b[i] = t;  return
        if tail == "self_attn.o_proj.weight": weights.attn_o_w[i] = t;  return
        if tail == "mlp.gate_proj.weight":    weights.mlp_gate_w[i] = t;  return
        if tail == "mlp.up_proj.weight":      weights.mlp_up_w[i] = t;  return
        if tail == "mlp.down_proj.weight":    weights.mlp_down_w[i] = t;  return

        print(f"  [UNUSED] {tail}")

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):

        # TODO: Implement generate function
        eos_id = self.meta.end_token
        if eos_id is None:
            print("[generate] eos token setting wrong, stop.")
            return []

        ids = np.asarray(list(inputs), dtype=np.int64)
        ids_ptr = ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        nxt = C.llaisysQwen2ModelInfer(self.model, ids_ptr, ctypes.c_size_t(ids.size))

        out: list[int] = list(inputs)
        new_tokens = 0

        if nxt is None or nxt < 0:
            print("[generate] prefill returned None/<0>, stop.")
            return out

        token = int(nxt)
        out.append(token)
        
        last = token
        new_tokens += 1
        #max_new_tokens = 3
        limit = max_new_tokens if max_new_tokens is not None else 512
        while last != eos_id and new_tokens < limit :
            nxt = C.llaisysQwen2ModelForwardOne(self.model, ctypes.c_int64(last))
            if nxt is None or nxt < 0:
                print("[generate] prefill returned None/<0>, stop.")
                return out
            token = int(nxt)
            out.append(token)
            last = token
            new_tokens += 1
        return out
