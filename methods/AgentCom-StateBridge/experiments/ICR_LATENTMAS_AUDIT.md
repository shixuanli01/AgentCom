# ICR LatentMAS communication-channel audit

## Upstream implementation

- Repository: `https://github.com/Gen-Verse/LatentMAS`
- Inspected commit: `9a9e4d331eb11430bd9e64754c6b252b06d73031`
- Backend: Hugging Face Transformers, matching the upstream reproduction path.
- Relevant upstream code: `models.py::generate_latent_batch`,
  `models.py::generate_text_batch`, and `methods/latent_mas.py::run_batch`.

## Exact state and injection

LatentMAS transmits the causal model's layer-wise `past_key_values`: key and
value tensors from every transformer layer and every source-history position.
For Qwen3-4B this is 36 layers, with 8 KV heads of dimension 128, stored in the
model dtype. Upstream then performs autoregressive latent steps. Each step takes
the final layer's last-position hidden state, applies the upstream default
identity realignment plus input-embedding mean-norm matching, feeds that vector
as one `inputs_embeds` position, and appends its K/V state to the cache.

The receiver receives this complete cache as a causal prefix. Its visible prompt
is processed after the cache, with positions and attention mask extended by the
cache length. The receiver performs exactly one sampled text generation.

## Frozen-ICR adaptation

The original four-agent pipeline is not reproduced. The communication mechanism
is placed into the frozen ICR protocol:

1. Re-render the deterministic cached Phase-1 sender prompt and verify its SHA256
   and token count.
2. Append the exact cached generated token IDs and teacher-force that sequence;
   never resample the sender.
3. Append 10 latent positions (the upstream method constructor default).
4. Inject the resulting complete, all-layer KV cache before the exact same clean
   receiver revision prompt used by StateBridge.
5. Use the same `(replication, item, direction)` revision seed and make exactly
   one receiver revision generation.

`true_latentmas`, `self_latentmas`, and `other_latentmas` select the same source
records as their Text/StateBridge counterparts. Tensor caches are ephemeral
because a single ordinary trajectory is hundreds of MB. Each saved message is a
reconstructible reference containing source prompt hash, generation seed and
token IDs, plus measured cache positions/layers/dtype/payload bytes, teacher
forcing steps, latent steps, and wall-clock time.

The implementation keeps upstream semantics while adapting to Transformers
4.51's cache-position API: inert placeholder IDs represent already-cached
positions in `generate`; those IDs are discarded before model forwarding.
