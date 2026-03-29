[lerobot/smolvla_base](https://hf.co/lerobot/smolvla_base) — SmolVLA action expert decoder for robotics.

Benchmark config: chunk_size=50, vlm_seq_len=16, float32, random weights, MSE loss.

| CPU / GPU | Framework | Compile (s) | Forward (ms) | Backward (ms) | Loss |
|-----------|-----------|:-----------:|:------------:|:--------------:|:----:|
| Intel Xeon @ 2.10GHz | [PyTorch 2.11.0+cu130](https://github.com/pytorch/pytorch) | 36.0 | 15410 | 7660 | 0.00 |
| Intel Xeon @ 2.10GHz (Lavapipe) | [Meganeura](https://github.com/kvark/meganeura/tree/550bb6c) | **0.16** | **805** | **3948** | 0.00 |
| | [Burn](https://github.com/tracel-ai/burn/tree/ed72d2b) | ✗ | ✗ | ✗ | |
| | [Luminal](https://github.com/luminal-ai/luminal/tree/f32161d) | ✗ | ✗ | ✗ | |

**Correctness:** PyTorch vs Meganeura: **CLOSE** (loss diff 1e-5, max error 4.6e-3).

<details><summary>Caveats</summary>

- **PyTorch** and **Meganeura** implement the full action expert architecture
  and should produce matching outputs.
- **Burn** and **Luminal** do not implement this architecture yet (reported as ✗).
- Inputs are synthetic: random noisy actions, sinusoidal timestep, random VLM context.

</details>
