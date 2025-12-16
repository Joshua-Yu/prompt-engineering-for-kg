# prompt_factory.py quick guide

## Purpose
`prompt_factory.py` centralizes all prompt engineering variants used by the KG extractor. Each method (zero-shot, few-shot, CoT, etc.) returns the same nodes/edges output format so downstream parsers can remain consistent.

## Basic usage
```python
from kgextractor.prompt_factory import KGPromptFactory

factory = KGPromptFactory()
prompt_text = factory.build_prompt("few_shot", sample_text)
```
`build_prompt` accepts any method named in `KGPromptFactory.methods`. To preview every prompt for a passage run the module directly:
```bash
python kgextractor/prompt_factory.py
```
Set `PROMPT_FACTORY_RUN_LLM=1` to actually call the configured models; otherwise it only prints the prompts.

## Environment variables
- `PROMPT_FACTORY_MODEL`: generation model (defaults to `gpt-5-mini` when running the script, `gpt-4o-mini` inside helper calls).
- `PROMPT_FACTORY_EVAL_MODEL`: for now it is the same as `PROMPT_FACTORY_MODEL`.
- `PROMPT_FACTORY_TEMPERATURE`: optional float forwarded to OpenAI. Leave unset for models that enforce their built-in temperature (e.g., GPT‑5).
- `PROMPT_FACTORY_RUN_LLM`: set to `1` to enable live LLM calls inside `main()`.
- `PROMPT_FACTORY_API_BASE`: base REST endpoint for OpenAI-compatible services (falls back to `OPENAI_BASE_URL` / `OPENAI_API_BASE` if unset).
- `PROMPT_FACTORY_API_KEY`: alternative env name for the API key; otherwise `OPENAI_API_KEY` is used.
- `PROMPT_FACTORY_API_HEADERS`: Optional. JSON object string with extra headers (e.g., `{"api-key": "..."}`) for custom gateways.
- `PROMPT_FACTORY_ORGANIZATION`: Optional. overrides the organization header (falls back to `OPENAI_ORGANIZATION`).

## Evaluation loop
When `PROMPT_FACTORY_RUN_LLM=1`, the script:
1. Builds each prompt for a sample passage.
2. Calls the generation model via `run_prompt_via_openai`.
3. Parses nodes/edges and re-sends them to the evaluator prompt (`evaluate_graph_with_llm`).
4. Stores prompt, generation, and feedback under `results/<model>/prompt_test_<method>_<model>.txt`.

## Extending
Add new prompt styles by:
1. Implementing a `get_<name>_prompt(self, text: str) -> str`.
2. Registering it in the `mapping` dict inside `build_prompt`.
3. Optionally add examples to `_init_examples` to reuse across strategies.
