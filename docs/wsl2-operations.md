# Codex and maintenance on WSL2

See [deployment setup](wsl2.md) for the model disk, build, memory and server configuration.

## Codex and MCP discovery

Basalt implements `/v1/responses` directly. No protocol-conversion proxy is needed.
In the tested Codex CLI 0.163.0 profile-file layout, the provider settings for
`~/.codex/basalt.config.toml` are:

```toml
model = "Qwen3.8-Flash-Next-NVFP4"
model_provider = "basalt"
model_context_window = 262144
model_auto_compact_token_limit = 220000
model_reasoning_effort = "high"
model_reasoning_summary = "none"
service_tier = "default"
web_search = "disabled"

[model_providers.basalt]
name = "Basalt (local)"
base_url = "http://127.0.0.1:18235/v1"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false
stream_idle_timeout_ms = 600000
```

This shows provider settings, not a complete model-catalog installation. For deferred tool discovery,
use a model catalog compatible with the installed Codex version and set `model_catalog_json` to its absolute
WSL path. The matching entry should advertise `supports_search_tool: true` and the correct context limit.
Advertise image input only when vision is enabled. Keep the default OpenAI profile separate.

`use_responses_lite` selects the Codex request format; setting it to `false` is **not a requirement for
deferred tool discovery**. With `supports_search_tool: true`, both values passed local checks against this
Basalt deployment using Codex CLI 0.163.0 on 2026-10-18:

- One local shell call followed by replay of its result.
- Client-side MCP tool discovery followed by replay of the discovered schemas, without invoking the external tool.

The captured requests used `/v1/responses` in both cases. With `false`, definitions were in the top-level
`tools` array. With `true`, Codex added the `x-openai-internal-codex-responses-lite: true` header and placed
definitions in `additional_tools` input items. Both completed the `tool_search_call` / `tool_search_output`
round trip. Initial model input was approximately 12.3K tokens in either mode, differing by less than 0.4%
across these two prompts; this was not evidence of a large context reduction or a throughput improvement.

The installed Basalt profile was subsequently changed to `true` after these checks; the checks themselves
used temporary catalogs. They do not cover
long-session continuation or compaction. The inspected, locally patched Strata deployment uses `true`
with the same client-search mechanism.

This fork preserves client-executed `tool_search_call` / `tool_search_output`, replays discovered tool schemas,
and handles MCP namespace aliases. With a compatible client, tools can be loaded on demand instead of
placing all MCP definitions in the initial context. The catalog flag alone is insufficient if the server
drops these calls. Restart the server after Python API changes.

Responses requests may combine `text.format` JSON output with tools, including client-side discovery.
The schema applies to the final answer after tool results, not to tool-call arguments or interim commentary.
With tools present, final JSON is buffered and validated before delivery; malformed or truncated answers
fail with `structured_output_failed`. Install `jsonschema` for schema validation; without it only the JSON
object shape is checked, with a server warning. Tool arguments use the engine's tool grammar, and JSON
requests without tools use its JSON grammar. This covers Codex automatic approval requests without changing
the approval policy or the reviewer's decision.

`web_search = "disabled"` removes unsupported hosted Responses web search. Separately connected Apps/MCP
search tools can still work according to the client's connections and account. Basalt itself does not
provide a hosted search service.

```sh
codex --profile basalt
codex resume --profile basalt --last
```

`--last` selects the most recent session for the current directory, not the most recent session for that profile.
References: [Codex profiles](https://learn.chatgpt.com/docs/config-file/config-advanced#profiles),
[client tool search](https://developers.openai.com/api/docs/guides/tools-tool-search).

## Updating and validating

Keep this fork as `origin` and the original repository as `upstream`. For a local branch that has not been
shared, commit/stash local edits, fetch upstream, then rebase:

```sh
git remote add upstream https://github.com/jesdga95/basalt.git
git fetch upstream main
git rebase upstream/main
cmake --build build -j 8
```

Add `upstream` only once. For a published/shared branch, avoid force-pushing rewritten history:
use a new branch for the rebased work or merge upstream into the shared branch.
A textual rebase can succeed while bypassing the registration fallback; inspect overlapping memory changes.

Useful checks, with sufficient spare RAM/VRAM:

```sh
.venv/bin/python -m unittest discover -s serve/tests -p 'test_*.py'
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --target file_expert_source_test platform_memory_test direct_upload_test -j 8
ctest --test-dir build --output-on-failure \
  -R '^(file_expert_source_test|platform_memory_test|direct_upload_test)$'
```

The memory test needs the raised lock limit. Do not start another full model instance beside a running server
just to validate an update. Faster disk reads and successful registration do not establish an equal
percentage improvement in decode speed; measure model startup, prefill and decode separately.
