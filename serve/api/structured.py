"""JSON formats use engine grammar; final answers with tools are buffered and validated separately."""
import json
import threading


def check_format(response_format) -> bool:
    """True for a JSON response format (json_object or json_schema), False for plain text. ValueError for a request
    the server cannot take."""
    if response_format is None:
        return False
    if not isinstance(response_format, dict):
        raise ValueError("response_format must be an object")
    kind = response_format.get("type")
    if kind == "text":
        return False
    if kind == "json_object":
        return True
    if kind != "json_schema":
        raise ValueError("response_format.type must be text, json_object or json_schema")
    spec = response_format.get("json_schema")
    if not isinstance(spec, dict) or not isinstance(spec.get("schema"), dict):
        raise ValueError("response_format.json_schema needs a schema object")
    if not isinstance(spec.get("name"), str) or not spec["name"]:
        raise ValueError("response_format.json_schema needs a name")
    if "strict" in spec and not isinstance(spec["strict"], bool):
        raise ValueError("response_format.json_schema.strict must be boolean")
    if spec["schema"].get("type") != "object":
        raise ValueError("response_format schema must have type object at its root")

    def check_refs(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("$ref", "$dynamicRef") and isinstance(value, str) and not value.startswith("#"):
                    raise ValueError("response_format supports only local schema references (#...)")
                check_refs(value)
        elif isinstance(node, list):
            for value in node:
                check_refs(value)
    check_refs(spec["schema"])
    return True


def anthropic_format(req: dict):
    """The chat path's response_format for an Anthropic request's `output_config.format` (or the older top-level
    `output_format`), or None for plain text. ValueError for a request the server cannot take."""
    out = req.get("output_config")
    fmt = out.get("format") if isinstance(out, dict) else None
    where = "output_config.format"
    if fmt is None and req.get("output_format") is not None:
        fmt, where = req["output_format"], "output_format"
    if fmt is None:
        return None
    if not isinstance(fmt, dict) or fmt.get("type") != "json_schema":
        raise ValueError(f"{where}.type must be json_schema")
    response_format = {"type": "json_schema", "json_schema": {"name": "output", "schema": fmt.get("schema")}}
    try:
        check_format(response_format)
    except ValueError as e:
        raise ValueError(str(e).replace("response_format.json_schema", where).replace("response_format", where)) from None
    return response_format


class StructuredOutputError(RuntimeError):
    pass


class _ObjectOnly:
    """The validator for json_object, and for json_schema without jsonschema: one JSON object."""

    def iter_errors(self, value):
        if not isinstance(value, dict):
            yield _ObjectError()


class _ObjectError:
    absolute_path = ()
    message = "the answer is not a JSON object"


_jsonschema = None                     # (validators, SchemaError, Registry, NoSuchResource), False when not installed
_jsonschema_lock = threading.Lock()


def jsonschema_modules():
    """jsonschema, imported on first use; None when it is not installed (said once, in the server window)."""
    global _jsonschema
    with _jsonschema_lock:
        if _jsonschema is None:
            try:
                from jsonschema import validators
                from jsonschema.exceptions import SchemaError
                from referencing import Registry
                from referencing.exceptions import NoSuchResource
                _jsonschema = (validators, SchemaError, Registry, NoSuchResource)
            except ImportError:
                _jsonschema = False
                print('[server] warning: jsonschema not installed, json_schema only checks for one JSON object '
                      '(pip install "jsonschema>=4.23,<5")',
                      flush=True)
        return _jsonschema or None


def prepare_tool_format(response_format, messages):
    """Tool grammar is lazy: constrain the final answer separately from tool arguments."""
    check_format(response_format)
    if response_format["type"] == "json_object":
        schema, modules = {"type": "object"}, None
    else:
        schema, modules = response_format["json_schema"]["schema"], jsonschema_modules()
    if modules is None:
        validator = _ObjectOnly()
    else:
        validators, SchemaError, Registry, NoSuchResource = modules

        def no_remote(uri):
            raise NoSuchResource(ref=uri)

        try:
            cls = validators.validator_for(schema)
            cls.check_schema(schema)
            validator = cls(schema, registry=Registry(retrieve=no_remote))
        except SchemaError as exc:
            raise ValueError(f"invalid response_format schema: {exc.message}") from exc
    directive = ("OUTPUT FORMAT REQUIREMENT: Your final answer must be exactly one JSON object matching the JSON Schema below. "
                 "No Markdown, headings, code fences, commentary, or text outside the JSON. "
                 "Use every required field, correct types, and only allowed fields. "
                 "Put all requested writing inside the appropriate JSON string fields.\nJSON Schema:\n" +
                 json.dumps(schema, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    directive += ("\nYou may call the available tools before your final answer, using their declared arguments. "
                  "This JSON Schema applies only to the final answer, not to tool calls or their arguments.")
    messages = [dict(message) for message in messages]
    if messages and messages[0].get("role") == "system":
        content = messages[0].get("content") or ""
        if isinstance(content, list):
            messages[0]["content"] = content + [{"type": "text", "text": directive}]
        else:
            messages[0]["content"] = content + "\n\n" + directive
    else:
        messages.insert(0, {"role": "system", "content": directive})
    return messages, validator


def validated_json(text, validator, finish):
    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ValueError(f"duplicate JSON key: {key}")
            obj[key] = value
        return obj

    def constant(value):
        raise ValueError(f"invalid JSON constant: {value}")

    if finish != "stop":
        raise StructuredOutputError(f"structured output was incomplete (finish_reason={finish}); increase the output budget")
    try:
        value = json.loads(text or "", object_pairs_hook=pairs, parse_constant=constant)
        canonical = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (ValueError, TypeError) as exc:
        raise StructuredOutputError(f"model did not return valid JSON: {exc}") from exc
    try:
        error = next(validator.iter_errors(value), None)
    except Exception as exc:
        raise StructuredOutputError(f"could not validate structured output: {exc}") from exc
    if error is not None:
        path = "/" + "/".join(str(part) for part in error.absolute_path)
        raise StructuredOutputError(f"model output failed the JSON Schema at {path}: {error.message}")
    return canonical
