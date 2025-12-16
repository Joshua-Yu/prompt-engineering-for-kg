import os
import re
import json
from dataclasses import dataclass
from typing import Dict, Optional, Any, List, Tuple
from textwrap import dedent

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[arg-type]


@dataclass
class KGPromptConfig:
    """
    Configuration for knowledge extraction prompts.
    """
    task_description: str = (
        "Generate a knowledge graph as complete as possible for the text below. "
        "The output should be a list of unique nodes and edges suitable for "
        "knowledge graph construction in KGQA and Graph RAG. Nodes typically "
        "represent entities (stages, organisms, objects, concepts) and edges "
        "represent relations (transformations, interactions, attributes, locations)."
    )
    output_format: str = (
        "Output format:\n"
        "nodes:\n"
        "name|label\n"
        "...\n"
        "\n"
        "edges:\n"
        "relation|from node|to node\n"
        "...\n"
        "\n"
        "Use the exact node names from the nodes list when specifying relationships.\n"
        "Fields are separated by '|'.\n"
        "No explanation, no comments, no code fences: just the 'nodes:' section "
        "and the 'edges:' section in plain text."
    )


class KGPromptFactory:
    """
    Prompt factory for AMR-style node/edge extraction.
    All methods produce the same unified response format:

        nodes:
        name|label
        ...

        edges:
        relation|from node|to node
        ...

    Usage:
        factory = KGPromptFactory()
        prompt_obj = factory.build_prompt("few_shot", text)
        prompt_text = prompt_obj["prompt"]
    """

    methods = [
        "zero_shot",
        "one_shot",
        "few_shot",
        "cot",
        "decomposition",
        "self_ask",
        "role",
        "auto_fact",
        "genknow",
        "react",
        "rag",
        "self_consistency",
    ]
    
    def __init__(self, config: KGPromptConfig | None = None):
        self.config = config or KGPromptConfig()
        self._init_examples()


    # ------------------------------------------------------------------
    # Few-shot examples (now in nodes/edges format)
    # ------------------------------------------------------------------
    def _init_examples(self) -> None:
        self.one_shot_example_text = "Water evaporates from the ocean surface."
        self.one_shot_example_output = (
            "nodes:\n"
            "Water|substance\n"
            "Ocean surface|location\n"
            "\n"
            "edges:\n"
            "evaporates_from|Water|Ocean surface\n"
        )

        self.few_shot_examples = [
            {
                "text": "Leaves absorb sunlight.",
                "output": (
                    "nodes:\n"
                    "Leaves|plant_part\n"
                    "Sunlight|energy\n"
                    "\n"
                    "edges:\n"
                    "absorb|Leaves|Sunlight\n"
                ),
            },
            {
                "text": "A caterpillar turns into a butterfly.",
                "output": (
                    "nodes:\n"
                    "Caterpillar|organism\n"
                    "Butterfly|organism\n"
                    "\n"
                    "edges:\n"
                    "transforms_into|Caterpillar|Butterfly\n"
                ),
            },
            {
                "text": "Birds lay eggs in nests.",
                "output": (
                    "nodes:\n"
                    "Birds|organism\n"
                    "Eggs|object\n"
                    "Nests|location\n"
                    "\n"
                    "edges:\n"
                    "lay|Birds|Eggs\n"
                    "located_in|Eggs|Nests\n"
                ),
            },
        ]

        # Example scaffolds for advanced prompting strategies
        self.genknow_example = {
            "Text": "Leaves absorb sunlight.",
            "Entity Types": "Leaves: plant_part; Sunlight: energy",
            "Triples": "(Leaves, absorb, Sunlight)",
        }
        self.react_example = {
            "Text": "Leaves absorb sunlight.",
            "Entities": "Leaves; Sunlight",
            "Entity Types": "Leaves: plant_part; Sunlight: energy",
            "Relations": "Leaves absorb Sunlight",
            "Triples": "(Leaves, absorb, Sunlight)",
        }
        self.rag_example = {
            "Text": "Birds lay eggs in nests.",
            "Triples": "(Birds, lay, Eggs); (Eggs, located_in, Nests)",
        }
        self.self_consistency_example = {
            "Text": "A caterpillar turns into a butterfly.",
            "Entities": "Caterpillar; Butterfly",
            "Entity Types": "Caterpillar: organism; Butterfly: organism",
            "Relations": "Caterpillar transforms into Butterfly",
            "Mixed Triples": "(Caterpillar, transforms_into, Butterfly); "
                             "(Butterfly, transforms_into, Caterpillar)",
            "Corrupted Triples": "(Butterfly, transforms_into, Caterpillar)",
            "Explanations": "Butterflies do not transform back into caterpillars.",
            "Triples": "(Caterpillar, transforms_into, Butterfly)",
        }

    @staticmethod
    def _fill_placeholders(template: str, values: Dict[str, str]) -> str:
        """Replace simple {placeholder} tokens using the provided mappings."""
        result = template
        for key in sorted(values.keys(), key=len, reverse=True):
            result = result.replace(f"{{{key}}}", values[key])
        return result

    # ------------------------------------------------------------------
    # 1. Zero-shot
    # ------------------------------------------------------------------
    def get_zero_shot_prompt(self, text: str) -> str:
        return (
            "You are a knowledge graph extraction assistant.\n\n"
            f"{self.config.task_description}\n\n"
            f"{self.config.output_format}\n\n"
            f"Text:\n\"\"\"{text}\"\"\""
        )

    # ------------------------------------------------------------------
    # 2. One-shot
    # ------------------------------------------------------------------
    def get_one_shot_prompt(self, text: str) -> str:
        return (
            "You are an expert knowledge graph extractor.\n\n"
            f"{self.config.task_description}\n\n"
            "Follow the format and style of this example:\n\n"
            f"Example text:\n\"{self.one_shot_example_text}\"\n\n"
            "Example output:\n"
            f"{self.one_shot_example_output}\n"
            "Now process the following text.\n\n"
            f"{self.config.output_format}\n\n"
            f"Text:\n\"\"\"{text}\"\"\""
        )

    # ------------------------------------------------------------------
    # 3. Few-shot
    # ------------------------------------------------------------------
    def get_few_shot_prompt(self, text: str) -> str:
        parts = []
        for i, ex in enumerate(self.few_shot_examples, start=1):
            parts.append(
                f"Example {i} text:\n\"{ex['text']}\"\n\n"
                f"Example {i} output:\n{ex['output']}\n"
            )
        examples = "\n".join(parts)

        return (
            "You are an expert system for extracting entities and relations "
            "as nodes and edges for a knowledge graph.\n\n"
            f"{self.config.task_description}\n\n"
            "Here are some examples of the expected format:\n\n"
            f"{examples}\n"
            "Now process the following text.\n\n"
            f"{self.config.output_format}\n\n"
            f"Text:\n\"\"\"{text}\"\"\""
        )

    # ------------------------------------------------------------------
    # 4. Chain-of-Thought (internal, but output still lists only)
    # ------------------------------------------------------------------
    def get_cot_prompt(self, text: str) -> str:
        return (
            "You are an expert in knowledge graph extraction.\n\n"
            f"{self.config.task_description}\n\n"
            "Think through the following steps internally (do NOT write them out):\n"
            "1. Identify all important entities in the text.\n"
            "2. Group them into unique nodes (name + label).\n"
            "3. Identify all meaningful relations between the nodes.\n"
            "4. Map them into edges using exact node names.\n\n"
            "Only output the final result in this format, with no explanation:\n\n"
            f"{self.config.output_format}\n\n"
            f"Text:\n\"\"\"{text}\"\"\""
        )

    # ------------------------------------------------------------------
    # 5. Decomposition
    # ------------------------------------------------------------------
    def get_decomposition_prompt(self, text: str) -> str:
        return (
            "We will conceptually decompose the task into steps, but you will only "
            "output the final nodes and edges.\n\n"
            f"{self.config.task_description}\n\n"
            "Decomposition steps (for your internal reasoning):\n"
            "- Step 1: Identify all candidate entities (stages, organisms, objects, concepts).\n"
            "- Step 2: Identify all semantic relations (transformations, interactions, attributes, locations).\n"
            "- Step 3: Deduplicate entities and construct nodes.\n"
            "- Step 4: Construct edges using exact node names.\n\n"
            "Do NOT print the steps. Only print the final lists.\n\n"
            f"{self.config.output_format}\n\n"
            f"Text:\n\"\"\"{text}\"\"\""
        )

    # ------------------------------------------------------------------
    # 6. Self-Ask / Self-Reflect
    # ------------------------------------------------------------------
    def get_self_ask_prompt(self, text: str) -> str:
        return (
            "You are a knowledge graph extraction assistant.\n\n"
            f"{self.config.task_description}\n\n"
            "Internally, you should ask yourself questions such as:\n"
            "- What are the key entities in this text?\n"
            "- What lifecycle events or transformations occur?\n"
            "- How should they be connected as relations?\n\n"
            "However, do NOT write these questions or reasoning out.\n"
            "Only output the final nodes and edges as follows:\n\n"
            f"{self.config.output_format}\n\n"
            f"Text:\n\"\"\"{text}\"\"\""
        )

    # ------------------------------------------------------------------
    # 7. Role prompting
    # ------------------------------------------------------------------
    def get_role_prompt(self, text: str) -> str:
        return (
            "You are a senior ontologist specializing in biological knowledge graphs "
            "for Question Answering (KGQA) and Graph Retrieval-Augmented Generation (Graph RAG).\n\n"
            f"{self.config.task_description}\n\n"
            "Use concise but informative labels for each node (e.g., organism, stage, plant_part, behavior, location).\n"
            "Use relation names that are suitable for real-world KG schemas "
            "(e.g., hatches_into, transforms_into, feeds_on, lays_on, located_in).\n\n"
            "Output only in the following format:\n\n"
            f"{self.config.output_format}\n\n"
            f"Text:\n\"\"\"{text}\"\"\""
        )
    
    # ------------------------------------------------------------------
    # 7. Automic Fact prompting
    # ------------------------------------------------------------------
    def get_auto_fact_prompt(self, text: str) -> str:
        # Use dedent to avoid accidental indentation inside the multi-line template.
        template_instruction = dedent(
            """
You are an expert extractor that converts natural-language text into atomic semantic facts.
Your goal is to extract MEANING, not surface wording.

Each fact must follow the rules of:
- canonicalization
- event detection
- verb normalization
- modifier / qualifier extraction
- negation handling
- semantic role assignment

##CANONICALIZATION RULES

1. SUBJECTS & OBJECTS (Concepts)
- Convert to lower case and singular form.
- Reduce to the minimal meaningful noun phrase.
- Resolve pronouns to their referents.
- Remove non-semantic determiners (“the”, “a”, “its”, “their”, “his”, “her”, “own”).
- Preserve meaningful descriptors (e.g., “gravitational pull” rather than “gravitational” and “pull”).
- Normalize equivalent phrasings that express the same concept:
  - Possessive form (“X’s Y”)
  - Prepositional form (“Y of X”)

For example:

“butterfly’s lifecycle”  
“lifecycle of butterflies”  
→ canonical concept: “butterfly lifecycle”

You must generalize this pattern to all similar constructions.


2. PREDICATES
- Use only verbs (or nominalizations) from the input text.
- Canonicalize verbs to base form, third-person present tense.
- Convert nominalizations to verbs
  (“discovery of X” → predicate “discover”).

3. EVENT DETECTION
- If the predicate describes an action, process, or change → `predicate_is_event` = true.
- Linking verbs (“is/are/was/were”) → treated as events ONLY if expressing identity or definition.
- Otherwise, linking verbs generate non-event relational facts.

4. MODIFIERS & QUALIFIERS
- MODIFIERS: descriptive meaning (adjectives/adverbs).
- QUALIFIERS: degree, extent, hedging.
- Attach them to subject, predicate, or object as appropriate.

5. NEGATION
- Detect “not”, “never”, “no longer”, “doesn’t”, etc.
- Set polarity = “negative”.
- Do NOT output a concept for negation words.

6. TIME & LOCATION
- Extract explicit time or location expressions.
- Canonicalize them to simple lower-case noun phrases.

7. FACT INDEX
- Within each sentence, assign a 0-based index to each fact:
  - The first fact in a sentence has "fact_index": 0
  - The second fact in the same sentence has "fact_index": 1
  - And so on.
- This index is local to the sentence (it resets for each sentence).
"""
        )

        template_output = dedent("""
Given INPUT_TEXT, produce a list of atomic semantic facts in strict JSON format. Do NOT explain anything outside the JSON.
##OUTPUT JSON SCHEMA

{
  "facts": [
    {
      "id": "fact_1",
      "subject": "___",
      "predicate": "___",
      "object": "___" | null,

      "subject_kind": "concept",
      "object_kind": "concept" | "event" | null,

      "predicate_is_event": true | false,

      "subject_modifiers": [],
      "subject_qualifiers": [],
      "predicate_modifiers": [],
      "predicate_qualifiers": [],
      "object_modifiers": [],
      "object_qualifiers": [],

      "time": null | "___",
      "location": null | "___",
      "polarity": "positive" | "negative",

      "source_sentence": 0,
      "fact_index": 0,
      "source_span": "___"
    }
  ]
}

Fields must always exist. Empty lists are allowed.
""")

        template_example = dedent("""
##EXAMPLE##

INPUT_TEXT:
"Gravity strongly warps spacetime."

OUTPUT:

{
  "facts": [
    {
      "id": "fact_1",
      "subject": "gravity",
      "predicate": "warp",
      "object": "spacetime",
      "subject_kind": "concept",
      "object_kind": "concept",
      "predicate_is_event": true,
      "subject_modifiers": [],
      "subject_qualifiers": [],
      "predicate_modifiers": ["strongly"],
      "predicate_qualifiers": [],
      "object_modifiers": [],
      "object_qualifiers": [],
      "time": null,
      "location": null,
      "polarity": "positive",
      "source_sentence": 0,
      "fact_index": 0,
      "source_span": "warps spacetime"
    }
  ]
}\n
""")

        template_input = dedent("""
##INPUT_TEXT:
<<<
{INPUT_TEXT}
>>>
""")
        return (template_instruction + self.config.output_format + template_input).replace("{INPUT_TEXT}", text)

    # ------------------------------------------------------------------
    # 8. Generative Knowledge prompting
    # ------------------------------------------------------------------
    def get_genknow_prompt(self, text: str) -> str:
        formatter = (
            "Text: {Text} Knowledge: {Entity Types} The following triples can be extracted "
            "considering the knowledge. Triples: {Triples} "
        )
        example = self._fill_placeholders(formatter, self.genknow_example)
        return (
            "Your task is extracting knowledge triples from text. A knowledge triple consists of three "
            "elements: subject - predicate - object. Subjects and objects are entities and the predicate "
            "is the relation between them. Before extracting triples, generate knowledge about the "
            "entities in the text and potential relations between them. Here is an example:\n"
            f"{example}\n"
            "Example ends here. Generate knowledge as shown in the example and extract knowledge triples "
            "from the input passage. Reason freely, but only output the final nodes and edges using this format:\n\n"
            f"{self.config.output_format}\n\n"
            f"Text:\n\"\"\"{text}\"\"\""
        )

    # ------------------------------------------------------------------
    # 9. ReAct prompting
    # ------------------------------------------------------------------
    def get_react_prompt(self, text: str) -> str:
        formatter = (
            "Text: {Text} Thought 1: I need to determine the entities. Act 1: Named entity extraction. "
            "Observation 1: {Entities} Thought 2: What type of entities do I have? "
            "Act 2: Named entity tagging Observation 2: {Entity Types} "
            "Thought 3: What are the potential relations between these entities? "
            "Act3: List the potential relations Observation3: {Relations} "
            "Thought 4: What are the triples? Act4: Form the triples Observation4: {Triples} "
            "Thought5: I have extracted knowledge triples from the input text. Act5: Finish "
            "Observation5: Task is completed. "
        )
        example = self._fill_placeholders(formatter, self.react_example)
        return (
            "Your task is extracting knowledge triples from text. A knowledge triple consists of three "
            "elements: subject - predicate - object. Subjects and objects are entities and the predicate "
            "is the relation between them. Let's use an example:\n"
            f"{example}\n"
            "Before answering a query, think and decide your act as shown above. Extract the knowledge "
            "triples from the following text, but only output the final nodes and edges in this format:\n\n"
            f"{self.config.output_format}\n\n"
            f"Text:\n\"\"\"{text}\"\"\""
        )

    # ------------------------------------------------------------------
    # 10. Retrieval-Augmented prompting
    # ------------------------------------------------------------------
    def get_rag_prompt(self, text: str) -> str:
        formatter = "Text: {Text} Triples: {Triples} "
        example = self._fill_placeholders(formatter, self.rag_example)
        return (
            "Extract knowledge triples from the text. Here are a few examples:\n"
            f"{example}\n"
            "Examples end here. Apply the same reasoning to the input and output only the nodes and edges "
            "using this format:\n\n"
            f"{self.config.output_format}\n\n"
            f"Text:\n\"\"\"{text}\"\"\""
        )

    # ------------------------------------------------------------------
    # 11. Self-Consistency prompting
    # ------------------------------------------------------------------
    def get_self_consistency_prompt(self, text: str) -> str:
        formatter = (
            "Text: {Text} Let's extract the entities first. Here is the list of the entities in this text: "
            "{Entities} What do you know about the entities? {Entity Types} Now we think about the potential "
            "relations between these entities: {Relations} Let's make a draft of the triples. {Mixed Triples} "
            "Now it is time to think and filter out incorrect triples if there is any. The following triples "
            "seem to be incorrect: {Corrupted Triples}. Here is the reason why I think these triples are "
            "incorrect: {Explanations} Therefore, final triples should be: {Triples}."
        )
        example = self._fill_placeholders(formatter, self.self_consistency_example)
        return (
            "Your task is extracting knowledge triples from text. A knowledge triple consists of three "
            "elements: subject - predicate - object. Subjects and objects are entities and the predicate "
            "is the relation between them. Let's use a few examples:\n"
            f"{example}\n"
            "Think like a domain expert and check the validity of the triples. Keep track of your thinking "
            "as shown in the example and extract triples from the following text. Only output the final "
            "nodes and edges in this format:\n\n"
            f"{self.config.output_format}\n\n"
            f"Text:\n\"\"\"{text}\"\"\""
        )
        
    
    # ------------------------------------------------------------------
    # Unified Prompt Builder
    # ------------------------------------------------------------------
    def build_prompt(self, method: str, text: str) -> Dict[str, str]:
        method = method.strip().lower()

        mapping = {
            "zero_shot": self.get_zero_shot_prompt,
            "one_shot": self.get_one_shot_prompt,
            "few_shot": self.get_few_shot_prompt,
            "cot": self.get_cot_prompt,
            "decomposition": self.get_decomposition_prompt,
            "self_ask": self.get_self_ask_prompt,
            "role": self.get_role_prompt,
            "auto_fact": self.get_auto_fact_prompt,
            "genknow": self.get_genknow_prompt,
            "react": self.get_react_prompt,
            "rag": self.get_rag_prompt,
            "self_consistency": self.get_self_consistency_prompt,
        }

        if method not in mapping:
            raise ValueError(
                f"Unknown prompt method '{method}'. "
                f"Valid methods: {list(mapping.keys())}"
            )

        '''return {
            "method": method,
            "prompt": mapping[method](text),
        }'''
        return mapping[method](text)

    def get_all_prompts(self, text: str) -> Dict[str, Dict[str, str]]:
        
        return {m: self.build_prompt(m, text) for m in KGPromptFactory.methods}
    

    # ------------------------------------------------------------------
    # (Optional) legacy JSON reasoning-strip helpers kept for compatibility
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_json_block(text: str) -> Optional[str]:
        """
        Legacy helper for older JSON-style prompts.
        Not used for the new nodes/edges format, but kept for backward compatibility.
        """
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence:
            return fence.group(1).strip()

        start = text.find("{")
        if start == -1:
            return None

        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1].strip()
        return None

    @classmethod
    def parse_json_from_response(cls, text: str) -> Optional[Any]:
        """
        Legacy helper for JSON outputs.
        For the current AMR-style format, you should implement a nodes/edges parser instead.
        """
        block = cls._extract_json_block(text)
        if block is None:
            return None
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            return None

    @classmethod
    def parse_cot_response(cls, text: str) -> Optional[Any]:
        return cls.parse_json_from_response(text)

    @classmethod
    def parse_self_ask_response(cls, text: str) -> Optional[Any]:
        return cls.parse_json_from_response(text)


_OPENAI_CLIENT: Any = None


def _ensure_openai_client() -> Any:
    """
    Lazy-load and cache the OpenAI client so prompt experiments can hit the API.
    Raises a RuntimeError if the SDK or API key is missing.
    """
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT
    if OpenAI is None:
        raise RuntimeError(
            "openai package is not installed. Please install it to run LLM tests."
        )
    api_key = os.environ.get("PROMPT_FACTORY_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY (or PROMPT_FACTORY_API_KEY) is not set. "
            "Provide an API key for the target OpenAI-compatible endpoint."
        )
    base_url = (
        os.environ.get("PROMPT_FACTORY_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
    )
    organization = os.environ.get("PROMPT_FACTORY_ORGANIZATION") or os.environ.get(
        "OPENAI_ORGANIZATION"
    )
    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url.rstrip("/")
    if organization:
        client_kwargs["organization"] = organization
    extra_headers = os.environ.get("PROMPT_FACTORY_API_HEADERS")
    if extra_headers:
        try:
            client_kwargs["default_headers"] = json.loads(extra_headers)
        except json.JSONDecodeError as exc:  # pragma: no cover
            raise RuntimeError(
                "PROMPT_FACTORY_API_HEADERS must be a valid JSON object string."
            ) from exc

    _OPENAI_CLIENT = OpenAI(**client_kwargs)
    return _OPENAI_CLIENT


def run_prompt_via_openai(prompt: str, model: Optional[str] = None) -> str:
    """
    Execute the given prompt against an OpenAI-compatible chat model.
    """
    client = _ensure_openai_client()
    payload_model = model or os.environ.get("PROMPT_FACTORY_MODEL", "gpt-4o-mini")
    request_args: Dict[str, Any] = {
        "model": payload_model,
        "messages": [{"role": "user", "content": prompt}],
    }
    temperature_value = os.environ.get("PROMPT_FACTORY_TEMPERATURE")
    if temperature_value is not None:
        try:
            request_args["temperature"] = float(temperature_value)
        except ValueError as exc:  # pragma: no cover
            raise ValueError(
                "PROMPT_FACTORY_TEMPERATURE must be a numeric value."
            ) from exc

    response = client.chat.completions.create(**request_args)
    return response.choices[0].message.content.strip()


def parse_nodes_edges_from_response(content: str) -> Tuple[List[str], List[str]]:
    """
    Lightweight parser that extracts nodes and edges sections from a generation.
    """
    nodes: List[str] = []
    edges: List[str] = []
    current = None
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        header = line.lower()
        if header.startswith("nodes:"):
            current = "nodes"
            continue
        if header.startswith("edges:"):
            current = "edges"
            continue
        if "name|label" in header or "relation|from" in header:
            continue
        if current == "nodes":
            nodes.append(line)
        elif current == "edges":
            edges.append(line)
    return nodes, edges


def build_evaluator_prompt(text: str, nodes: List[str], edges: List[str]) -> str:
    """
    Create a grading prompt that asks the LLM to judge completeness/correctness.
    """
    nodes_section = "\n".join(nodes) if nodes else "None"
    edges_section = "\n".join(edges) if edges else "None"
    template = dedent(
        """
        You are an expert evaluator of knowledge graph extractions.
        Given the input passage, the extracted nodes, and edges, assess:
        - completeness: Are important entities/relations missing?
        - correctness: Are the presented nodes/edges faithful to the text?
        - duplication: Are there any duplicated nodes/edges for similar meaning?
        - overall: A number between 0.1 and 1.0 representing quality score of the result. `complete` and `correct` and `no` duplication is 1.0
        - confidence: A number between 0 and 1 representing your certainty.

        Respond strictly in JSON with the following schema:
        {{
          "completeness": "complete" | "partial" | "poor",
          "correctness": "correct" | "mixed" | "incorrect",
          "duplication": "none" | "less than 10%" | "less than 30%" | "more than 30%",
          "overall": float,
          "confidence": float,
          "feedback": "short explanation"
        }}

        Passage:
        \"\"\"{text}\"\"\"

        Nodes:
        {nodes}

        Edges:
        {edges}
        """
    )
    return template.format(text=text, nodes=nodes_section, edges=edges_section)


def evaluate_graph_with_llm(
    text: str, nodes: List[str], edges: List[str], model: Optional[str] = None
) -> str:
    """
    Send the extracted graph back to an LLM to score completeness/correctness.
    """
    prompt = build_evaluator_prompt(text, nodes, edges)
    eval_model = model or os.environ.get(
        "PROMPT_FACTORY_EVAL_MODEL", os.environ.get("PROMPT_FACTORY_MODEL", "gpt-4o-mini")
    )
    return run_prompt_via_openai(prompt, eval_model)


def main() -> None:
    """Manual test harness to inspect prompts for each method."""
    sample_text = (
        "How Smell Works:\n"
        "Smell begins when odor molecules enter the nasal cavity through the nostrils. "
        "These molecules then bind to specialized olfactory receptors located in the "
        "olfactory epithelium, a thin layer of tissue at the back of the nasal cavity. "
        "Each receptor is sensitive to specific odor molecules, allowing us to distinguish "
        "between different smells. When an odor molecule binds to a receptor, it triggers "
        "a series of chemical reactions that send signals to the brain, where they are "
        "processed and interpreted as a particular smell."
    )

    factory = KGPromptFactory()
    run_llm_tests = os.environ.get("PROMPT_FACTORY_RUN_LLM") == "1"
    generation_model = os.environ.get("PROMPT_FACTORY_MODEL", "gpt-4o") # gpt-5-mini, gpt-4o
    evaluator_model = os.environ.get("PROMPT_FACTORY_EVAL_MODEL", generation_model)

    for method in KGPromptFactory.methods:

        prompt = factory.build_prompt(method, sample_text)
        separator = "=" * 20
        print(f"{separator} {method} {separator}")
        #print(prompt)
        #print()

        if not run_llm_tests:
            continue

        try:
            llm_output = run_prompt_via_openai(prompt, generation_model)
        except Exception as exc:  # pragma: no cover
            print(f"[LLM generation error] {exc}")
            continue

        #print("--- LLM Output ---")
        #print(llm_output)

        nodes, edges = parse_nodes_edges_from_response(llm_output)
        if not nodes and not edges:
            print("No nodes/edges parsed; skipping evaluation.\n")
            continue

        try:
            evaluation = evaluate_graph_with_llm(
                sample_text, nodes, edges, evaluator_model
            )
        except Exception as exc:  # pragma: no cover
            print(f"[LLM evaluation error] {exc}\n")
            continue

        print("--- Evaluator Feedback for : " + method + "---\n Prompt: " + prompt + llm_output)
        print(evaluation)

        safe_model = evaluator_model.replace("/", "_")
        safe_method = method.replace("/", "_")
        results_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "results", safe_model)
        )
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(
            results_dir, f"prompt_test_{safe_method}_{safe_model}.txt"
        )
        with open(output_path, "w", encoding="utf-8") as file_obj:
            file_obj.write(f"Method: {method}\n")
            file_obj.write(f"Evaluator Model: {evaluator_model}\n\n")
            file_obj.write("Prompt:\n")
            file_obj.write(prompt)
            file_obj.write("\n\nLLM Output:\n")
            file_obj.write(llm_output)
            file_obj.write("\n\nEvaluator Feedback:\n")
            file_obj.write(evaluation)
            file_obj.write("\n")

        print(f"[Saved evaluator feedback to {output_path}]")
        print()


if __name__ == "__main__":
    main()
