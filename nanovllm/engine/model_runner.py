import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        if config.fp8_mlp:
            self.enable_fp8_mlp()
        self.warmup_model()
        self.allocate_kv_cache()
        self.use_flashinfer = (
            config.attn_backend == "flashinfer"
            or (config.attn_backend == "auto" and torch.cuda.get_device_capability(rank)[0] >= 10)
        )
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager and hasattr(self, "graphs"):
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def enable_fp8_mlp(self):
        # Match Qwen-style module naming: *.mlp.(gate_up_proj|down_proj)
        for name, module in self.model.named_modules():
            if name.endswith(".mlp.gate_up_proj") or name.endswith(".mlp.down_proj"):
                if hasattr(module, "enable_fp8"):
                    module.enable_fp8()

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None

        # FlashInfer paged-kv metadata (page indices are kv-cache blocks)
        fi_seq_lens = []
        fi_seq_lens_q = []
        fi_kv_indptr = [0]
        fi_kv_indices = []
        fi_kv_last_page_len = []

        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            # FlashInfer: block-level paged-kv indptr/indices
            fi_seq_lens.append(seqlen_k)
            fi_seq_lens_q.append(seqlen_q)
            n_pages = len(seq.block_table)
            fi_kv_indptr.append(fi_kv_indptr[-1] + n_pages)
            fi_kv_indices.extend(seq.block_table)
            fi_kv_last_page_len.append(seq.last_block_num_tokens if n_pages > 0 else 0)

            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)

        cu_seqlens_q_cpu = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True)
        cu_seqlens_k_cpu = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True)
        cu_seqlens_q = cu_seqlens_q_cpu.cuda(non_blocking=True)
        cu_seqlens_k = cu_seqlens_k_cpu.cuda(non_blocking=True)

        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        fi_qo_indptr = cu_seqlens_q_cpu
        fi_kv_indptr = torch.tensor(fi_kv_indptr, dtype=torch.int32, pin_memory=True)
        fi_kv_indices = torch.tensor(fi_kv_indices, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        fi_kv_last_page_len = torch.tensor(fi_kv_last_page_len, dtype=torch.int32, pin_memory=True)
        fi_seq_lens = torch.tensor(fi_seq_lens, dtype=torch.int32, pin_memory=True)
        fi_seq_lens_q = torch.tensor(fi_seq_lens_q, dtype=torch.int32, pin_memory=True)

        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
            attn_backend=self.config.attn_backend,
            fi_qo_indptr=fi_qo_indptr,
            fi_kv_indptr=fi_kv_indptr,
            fi_kv_indices=fi_kv_indices,
            fi_kv_last_page_len=fi_kv_last_page_len,
            fi_seq_lens=fi_seq_lens,
            fi_seq_lens_q=fi_seq_lens_q,
        )
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        use_flashinfer = (
            self.config.attn_backend == "flashinfer"
            or (self.config.attn_backend == "auto" and torch.cuda.get_device_capability(self.rank)[0] >= 10)
        )

        bs = len(seqs)
        use_cudagraph = (
            (not self.enforce_eager)
            and hasattr(self, "graphs")
            and bs <= 512
        )

        if use_cudagraph:
            input_ids = []
            positions = []
            slot_mapping = []
            context_lens = [] if not use_flashinfer else None

            fi_seq_lens = [] if use_flashinfer else None
            fi_kv_indptr = [0] if use_flashinfer else None
            fi_kv_indices = [] if use_flashinfer else None
            fi_kv_last_page_len = [] if use_flashinfer else None

            for seq in seqs:
                input_ids.append(seq.last_token)
                positions.append(len(seq) - 1)
                slot_mapping.append(
                    seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
                )
                if use_flashinfer:
                    assert fi_seq_lens is not None
                    assert fi_kv_indptr is not None
                    assert fi_kv_indices is not None
                    assert fi_kv_last_page_len is not None
                    seqlen_k = len(seq)
                    fi_seq_lens.append(seqlen_k)
                    n_pages = len(seq.block_table)
                    fi_kv_indptr.append(fi_kv_indptr[-1] + n_pages)
                    fi_kv_indices.extend(seq.block_table)
                    fi_kv_last_page_len.append(seq.last_block_num_tokens if n_pages > 0 else 0)
                else:
                    assert context_lens is not None
                    context_lens.append(len(seq))

            # Return pinned CPU buffers so run_model() can copy directly into the CUDA-graph inputs.
            input_ids_cpu = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True)
            positions_cpu = torch.tensor(positions, dtype=torch.int64, pin_memory=True)
            slot_mapping_cpu = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True)

            if use_flashinfer:
                assert fi_kv_indptr is not None
                assert fi_kv_indices is not None
                assert fi_kv_last_page_len is not None
                assert fi_seq_lens is not None
                if not hasattr(self, "fi_graph_indices"):
                    raise RuntimeError("FlashInfer cudagraph buffers were not initialized.")

                fi_kv_indptr_cpu = torch.tensor(fi_kv_indptr, dtype=torch.int32, pin_memory=True)
                fi_kv_last_page_len_cpu = torch.tensor(fi_kv_last_page_len, dtype=torch.int32, pin_memory=True)
                fi_seq_lens_cpu = torch.tensor(fi_seq_lens, dtype=torch.int32, pin_memory=True)

                # Indices must live on device for CUDAGraphBatchDecodeWithPagedKVCacheWrapper.plan(...).
                indices_cpu = torch.tensor(fi_kv_indices, dtype=torch.int32, pin_memory=True)
                total_pages = int(fi_kv_indptr_cpu[-1])
                indices_gpu = self.fi_graph_indices[:total_pages]
                indices_gpu.copy_(indices_cpu, non_blocking=True)

                set_context(
                    False,
                    slot_mapping=slot_mapping_cpu,
                    context_lens=None,
                    block_tables=None,
                    attn_backend=self.config.attn_backend,
                    fi_kv_indptr=fi_kv_indptr_cpu,
                    fi_kv_indices=indices_gpu,
                    fi_kv_last_page_len=fi_kv_last_page_len_cpu,
                    fi_seq_lens=fi_seq_lens_cpu,
                )
            else:
                assert context_lens is not None
                context_lens_gpu = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
                block_tables = self.prepare_block_tables(seqs)
                set_context(
                    False,
                    slot_mapping=slot_mapping_cpu,
                    context_lens=context_lens_gpu,
                    block_tables=block_tables,
                    attn_backend=self.config.attn_backend,
                    fi_kv_indptr=None,
                    fi_kv_indices=None,
                    fi_kv_last_page_len=None,
                    fi_seq_lens=None,
                )
            return input_ids_cpu[:bs], positions_cpu[:bs]

        # Eager / non-graph fallback (allocates per-step).
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = [] if not use_flashinfer else None

        fi_seq_lens = [] if use_flashinfer else None
        fi_kv_indptr = [0] if use_flashinfer else None
        fi_kv_indices = [] if use_flashinfer else None
        fi_kv_last_page_len = [] if use_flashinfer else None

        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
            if use_flashinfer:
                seqlen_k = len(seq)
                fi_seq_lens.append(seqlen_k)
                n_pages = len(seq.block_table)
                fi_kv_indptr.append(fi_kv_indptr[-1] + n_pages)
                fi_kv_indices.extend(seq.block_table)
                fi_kv_last_page_len.append(seq.last_block_num_tokens if n_pages > 0 else 0)
            else:
                assert context_lens is not None
                context_lens.append(len(seq))

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        if use_flashinfer:
            context_lens = None
            block_tables = None
        else:
            assert context_lens is not None
            context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            block_tables = self.prepare_block_tables(seqs)

        if use_flashinfer:
            assert fi_kv_indptr is not None
            assert fi_kv_indices is not None
            assert fi_kv_last_page_len is not None
            assert fi_seq_lens is not None
            fi_kv_indptr = torch.tensor(fi_kv_indptr, dtype=torch.int32, pin_memory=True)
            fi_kv_indices = torch.tensor(fi_kv_indices, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            fi_kv_last_page_len = torch.tensor(fi_kv_last_page_len, dtype=torch.int32, pin_memory=True)
            fi_seq_lens = torch.tensor(fi_seq_lens, dtype=torch.int32, pin_memory=True)
        else:
            fi_kv_indptr = None
            fi_kv_indices = None
            fi_kv_last_page_len = None
            fi_seq_lens = None

        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            attn_backend=self.config.attn_backend,
            fi_kv_indptr=fi_kv_indptr,
            fi_kv_indices=fi_kv_indices,
            fi_kv_last_page_len=fi_kv_last_page_len,
            fi_seq_lens=fi_seq_lens,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or not hasattr(self, "graphs") or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        bs = input_ids.size(0)
        context = get_context()
        graph_bs = next(x for x in self.graph_bs if x >= bs)
        graph = self.graphs[graph_bs]
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:bs].copy_(input_ids, non_blocking=True)
        graph_vars["positions"][:bs].copy_(positions, non_blocking=True)
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs].copy_(context.slot_mapping, non_blocking=True)

        if getattr(self, "use_flashinfer", False):
            if (
                context.fi_kv_indptr is None
                or context.fi_kv_indices is None
                or context.fi_kv_last_page_len is None
                or context.fi_seq_lens is None
            ):
                raise RuntimeError("FlashInfer metadata must be set for cudagraph decode.")
            if not hasattr(self, "fi_graph_wrappers"):
                raise RuntimeError("FlashInfer cudagraph wrappers were not captured.")

            wrapper = self.fi_graph_wrappers[graph_bs]
            meta_cpu = self.fi_graph_meta_cpu[graph_bs]
            indptr_cpu = meta_cpu["indptr"]
            last_page_len_cpu = meta_cpu["last_page_len"]
            seq_lens_cpu = meta_cpu["seq_lens"]

            fi_kv_indptr_cpu = context.fi_kv_indptr
            fi_kv_last_page_len_src = context.fi_kv_last_page_len
            fi_seq_lens_src = context.fi_seq_lens
            fi_kv_indices_src = context.fi_kv_indices

            total_pages = int(fi_kv_indptr_cpu[-1])
            pad = graph_bs - bs

            if pad == 0:
                indptr_cpu[: graph_bs + 1].copy_(fi_kv_indptr_cpu)
                last_page_len_cpu[:graph_bs].copy_(fi_kv_last_page_len_src)
                seq_lens_cpu[:graph_bs].copy_(fi_seq_lens_src)
                indices = fi_kv_indices_src
            else:
                indptr_cpu[: bs + 1].copy_(fi_kv_indptr_cpu)
                for i in range(pad):
                    indptr_cpu[bs + 1 + i] = total_pages + i + 1

                last_page_len_cpu[:bs].copy_(fi_kv_last_page_len_src)
                last_page_len_cpu[bs:graph_bs].fill_(1)
                seq_lens_cpu[:bs].copy_(fi_seq_lens_src)
                seq_lens_cpu[bs:graph_bs].fill_(1)

                indices_buf = self.fi_graph_indices
                indices_buf[total_pages : total_pages + pad].zero_()
                indices = indices_buf[: total_pages + pad]

            params = self.fi_graph_params
            wrapper.plan(
                indptr=indptr_cpu,
                indices=indices,
                last_page_len=last_page_len_cpu,
                num_qo_heads=params["num_qo_heads"],
                num_kv_heads=params["num_kv_heads"],
                head_dim=params["head_dim"],
                page_size=params["page_size"],
                pos_encoding_mode="NONE",
                sm_scale=params["sm_scale"],
                q_data_type=params["q_dtype"],
                kv_data_type=params["kv_dtype"],
                non_blocking=True,
                seq_lens=seq_lens_cpu,
            )
        else:
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables

        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.full((max_bs,), -1, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        use_flashinfer = getattr(self, "use_flashinfer", False)
        if use_flashinfer:
            from flashinfer import CUDAGraphBatchDecodeWithPagedKVCacheWrapper

            # Shared workspace buffers (match mini-sglang pattern).
            fi_float_workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=outputs.device)
            fi_int_workspace = torch.empty((8 * 1024 * 1024,), dtype=torch.uint8, device=outputs.device)
            fi_pin_int_workspace = torch.empty(
                (8 * 1024 * 1024,),
                dtype=torch.uint8,
                pin_memory=True,
                device="cpu",
            )

            tp_size = self.world_size
            num_qo_heads = hf_config.num_attention_heads // tp_size
            num_kv_heads = hf_config.num_key_value_heads // tp_size
            head_dim = getattr(
                hf_config,
                "head_dim",
                hf_config.hidden_size // hf_config.num_attention_heads,
            )
            sm_scale = float(head_dim**-0.5)
            use_tensor_cores = (num_qo_heads // num_kv_heads) >= 4

            # Graph metadata buffers (device): indptr/indices/last_page_len.
            fi_kv_indptr = torch.empty(max_bs + 1, dtype=torch.int32, device=outputs.device)
            fi_kv_last_page_len = torch.empty(max_bs, dtype=torch.int32, device=outputs.device)
            fi_kv_indices = torch.empty(
                max_bs * max_num_blocks, dtype=torch.int32, device=outputs.device
            )

            self.fi_graph_wrappers = {}
            self.fi_graph_indices = fi_kv_indices
            self.fi_graph_meta_cpu = {}
            self.fi_graph_params = dict(
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                page_size=self.block_size,
                sm_scale=sm_scale,
                q_dtype=hf_config.torch_dtype,
                kv_dtype=hf_config.torch_dtype,
            )

            # Plan once with "max" metadata so the captured kernel supports the full range.
            n_pages = max_num_blocks
            last_len = config.max_model_len - (n_pages - 1) * self.block_size
            if not (1 <= last_len <= self.block_size):
                raise RuntimeError(
                    f"Invalid last_page_len={last_len} for page_size={self.block_size}, max_model_len={config.max_model_len}."
                )

            for bs in reversed(self.graph_bs):
                graph = torch.cuda.CUDAGraph()
                wrapper = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
                    fi_float_workspace,
                    kv_layout="NHD",
                    use_tensor_cores=use_tensor_cores,
                    indptr_buffer=fi_kv_indptr[: bs + 1],
                    indices_buffer=fi_kv_indices,
                    last_page_len_buffer=fi_kv_last_page_len[:bs],
                )
                wrapper._int_workspace_buffer = fi_int_workspace
                wrapper._pin_memory_int_workspace_buffer = fi_pin_int_workspace
                self.fi_graph_wrappers[bs] = wrapper

                # Cache CPU pinned buffers for padded planning at replay time.
                self.fi_graph_meta_cpu[bs] = dict(
                    indptr=torch.empty(bs + 1, dtype=torch.int32, pin_memory=True, device="cpu"),
                    last_page_len=torch.empty(bs, dtype=torch.int32, pin_memory=True, device="cpu"),
                    seq_lens=torch.empty(bs, dtype=torch.int32, pin_memory=True, device="cpu"),
                )

                indptr_host = torch.arange(
                    0,
                    (bs + 1) * n_pages,
                    step=n_pages,
                    dtype=torch.int32,
                    pin_memory=True,
                    device="cpu",
                )
                fi_kv_indices[: bs * n_pages].zero_()
                last_host = torch.full(
                    (bs,),
                    last_len,
                    dtype=torch.int32,
                    pin_memory=True,
                    device="cpu",
                )
                seq_lens_host = torch.full(
                    (bs,),
                    config.max_model_len,
                    dtype=torch.int32,
                    pin_memory=True,
                    device="cpu",
                )
                wrapper.plan(
                    indptr=indptr_host,
                    indices=fi_kv_indices[: bs * n_pages],
                    last_page_len=last_host,
                    num_qo_heads=num_qo_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    page_size=self.block_size,
                    pos_encoding_mode="NONE",
                    sm_scale=sm_scale,
                    q_data_type=hf_config.torch_dtype,
                    kv_data_type=hf_config.torch_dtype,
                    non_blocking=True,
                    seq_lens=seq_lens_host,
                )

                set_context(
                    False,
                    slot_mapping=slot_mapping[:bs],
                    attn_backend="flashinfer",
                    fi_decode_wrapper=wrapper,
                    fi_planned=True,
                )
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
                with torch.cuda.graph(graph, self.graph_pool):
                    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
                if self.graph_pool is None:
                    self.graph_pool = graph.pool()
                self.graphs[bs] = graph
                torch.cuda.synchronize()
                reset_context()

            self.graph_vars = dict(
                input_ids=input_ids,
                positions=positions,
                slot_mapping=slot_mapping,
                outputs=outputs,
            )
            return

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
