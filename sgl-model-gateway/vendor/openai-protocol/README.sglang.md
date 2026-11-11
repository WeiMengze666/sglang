# SGLang protocol compatibility backport

Source: `openai-protocol` 1.0.0, Apache-2.0,
https://crates.io/crates/openai-protocol/1.0.0

Published archive SHA256:
`b365334e1e57a6f57ed932951878b31c1fe9087a90149db6f5d071c3504f6dff`.
The `src/` tree and normalized `Cargo.toml` come from that archive. `LICENSE`
contains the Apache-2.0 license. The only production source change is in
`src/chat.rs`: retain the two Boolean SGLang extensions `return_token_ids` and
`return_prompt_token_ids` during request deserialization and serialization.
The appended regression tests cover their round trip through validation and
normalization, defaults, and malformed requests.

Without this backport, the HTTP gateway accepts these fields but drops them
before forwarding to either worker. Both ordinary and disaggregated chat
requests consequently lose the token IDs required for inference diagnostics.
The worker remains responsible for token capture and response construction.

The gateway workspace patches its existing exact 1.0.0 dependency to this
copy. A broader protocol upgrade also changes defaulting, normalization, and
validation; this backport deliberately retains the pinned protocol behavior.
Remove this copy once a qualified upstream dependency includes these fields
or preserves engine-specific extensions. It does not enable token-ID output
for gRPC engines that do not implement that output.

Run the protocol regression independently of the gateway build:

```bash
cargo test --manifest-path sgl-model-gateway/vendor/openai-protocol/Cargo.toml --lib
```
