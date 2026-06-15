from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import yaml
from scipy.special import expit
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from bayesian_orchestrator.config import output_dir
from bayesian_orchestrator.plotting import configure_matplotlib_cache
from bayesian_orchestrator.report import WorkflowResult, build_metadata, write_report

configure_matplotlib_cache()
import matplotlib.pyplot as plt


ANSWER_LABELS = tuple(chr(ord("A") + idx) for idx in range(26))
ANSWERS = ANSWER_LABELS[:4]
EPS = 1e-6


@dataclass(frozen=True)
class ModelSpec:
    label: str
    model_id: str
    cost_weight: float
    dependence_group: str | None = None
    input_cost_per_1k: float | None = None
    output_cost_per_1k: float | None = None


@dataclass(frozen=True)
class PricingSnapshot:
    catalog_path: str | None
    version: str | None
    as_of: str | None
    currency: str
    prices_per_1m: dict[str, dict[str, float]]


@dataclass(frozen=True)
class MMLUQuestion:
    question_id: str
    subject: str
    question: str
    choices: tuple[str, ...]
    answer: str


@dataclass(frozen=True)
class CallResult:
    question_id: str
    subject: str
    question: str
    choices: tuple[str, ...]
    gold_answer: str
    model_label: str
    model_id: str
    cost_weight: float
    question_length_bucket: int
    answer: str | None
    confidence: float | None
    correct: int
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    json_mode_requested: bool
    json_mode_used: bool
    raw_response: str


@dataclass(frozen=True)
class AdaptivePolicyResult:
    utilities: np.ndarray
    call_counts: np.ndarray
    cumulative_costs: np.ndarray
    stop_after_one: np.ndarray
    accepted_voi_calls: np.ndarray
    selected_rows: list[CallResult]
    called_rows: list[list[CallResult]]
    dependence_caution_triggered: bool
    final_answers: list[str | None]
    first_answers: list[str | None]
    disagreement: np.ndarray
    changed_answers: np.ndarray
    invalid_first_answer: np.ndarray


class GraphState(TypedDict, total=False):
    question: MMLUQuestion
    model: ModelSpec
    prompt: str
    raw_response: str
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    json_mode_requested: bool
    json_mode_used: bool
    parsed_answer: str | None
    confidence: float | None
    result: CallResult


def _answer_labels(choice_count: int) -> tuple[str, ...]:
    if choice_count < 2:
        raise RuntimeError("Multiple-choice questions must have at least two choices.")
    if choice_count > len(ANSWER_LABELS):
        raise RuntimeError(f"Multiple-choice questions with more than {len(ANSWER_LABELS)} choices are unsupported.")
    return ANSWER_LABELS[:choice_count]


def _format_answer_set(labels: tuple[str, ...]) -> str:
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + f", or {labels[-1]}"


def format_mmlu_prompt(question: MMLUQuestion) -> str:
    labels = _answer_labels(len(question.choices))
    choices = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, question.choices))
    return (
        "Answer the multiple-choice question. Return only JSON with keys "
        f'`answer` and `confidence`, where answer is one of {_format_answer_set(labels)} and '
        "confidence is a number from 0 to 1.\n\n"
        f"Subject: {question.subject}\n"
        f"Question: {question.question}\n"
        f"{choices}\n"
    )


def parse_answer_and_confidence(
    text: str,
    valid_answers: tuple[str, ...] = ANSWERS,
) -> tuple[str | None, float | None]:
    text = text.strip()
    valid_answers = tuple(answer.upper() for answer in valid_answers)
    try:
        payload = json.loads(text)
        answer = str(payload.get("answer", "")).strip().upper()
        confidence = payload.get("confidence")
        parsed_confidence = None if confidence is None else float(confidence)
        if parsed_confidence is not None:
            parsed_confidence = float(np.clip(parsed_confidence, 0.0, 1.0))
        return (answer if answer in valid_answers else None, parsed_confidence)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    answer_pattern = "".join(re.escape(answer) for answer in valid_answers)
    answer_match = re.search(rf"\b([{answer_pattern}])\b", text.upper())
    confidence_match = re.search(r"confidence[^0-9]*([01](?:\.\d+)?)", text, flags=re.IGNORECASE)
    confidence = None
    if confidence_match:
        confidence = float(np.clip(float(confidence_match.group(1)), 0.0, 1.0))
    answer = answer_match.group(1) if answer_match else None
    return (answer if answer in valid_answers else None, confidence)


def utility(correct: int | float, cost_weight: float, cost_scale: float) -> float:
    return float(correct) - float(cost_weight) * float(cost_scale)


def _resolve_config_path(config: dict[str, Any], configured_path: str) -> Path:
    path = Path(configured_path).expanduser()
    if path.is_absolute() or path.exists():
        return path.resolve()
    config_path = config.get("_config_path")
    if config_path:
        candidate = Path(config_path).resolve().parent / path
        if candidate.exists():
            return candidate
    return path.resolve()


def _load_pricing_catalog(config: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    pricing_cfg = config.get("pricing", {})
    catalog_value = pricing_cfg.get("catalog")
    if not catalog_value:
        return {}, None
    catalog_path = _resolve_config_path(config, str(catalog_value))
    if not catalog_path.exists():
        raise RuntimeError(f"Pricing catalog does not exist: {catalog_path}")
    with catalog_path.open("r", encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle) or {}
    if catalog.get("unit", "per_1m_tokens") != "per_1m_tokens":
        raise RuntimeError("Pricing catalog unit must be `per_1m_tokens`.")
    if not isinstance(catalog.get("models"), dict):
        raise RuntimeError(f"Pricing catalog must contain a `models` mapping: {catalog_path}")
    return catalog, str(catalog_path)


def _resolve_model_specs(config: dict[str, Any]) -> tuple[list[ModelSpec], PricingSnapshot]:
    catalog, catalog_path = _load_pricing_catalog(config)
    catalog_models = catalog.get("models", {})
    pricing_required = bool(config.get("pricing", {}).get("required", bool(catalog_path)))
    model_specs: list[ModelSpec] = []
    prices_per_1m: dict[str, dict[str, float]] = {}

    for item in config.get("models", []):
        model_id = str(item.get("model_id", item["label"]))
        catalog_price = catalog_models.get(model_id, {})
        input_per_1k = item.get("input_cost_per_1k")
        output_per_1k = item.get("output_cost_per_1k")
        if input_per_1k is None and catalog_price.get("input") is not None:
            input_per_1k = float(catalog_price["input"]) / 1000.0
        if output_per_1k is None and catalog_price.get("output") is not None:
            output_per_1k = float(catalog_price["output"]) / 1000.0
        if pricing_required and (input_per_1k is None or output_per_1k is None):
            raise RuntimeError(
                f"No complete pricing entry for model `{model_id}`. Add it to `{catalog_path}` or set explicit input_cost_per_1k/output_cost_per_1k values."
            )

        input_rate = None if input_per_1k is None else float(input_per_1k)
        output_rate = None if output_per_1k is None else float(output_per_1k)
        if input_rate is not None and output_rate is not None:
            prices_per_1m[model_id] = {
                "input": input_rate * 1000.0,
                "output": output_rate * 1000.0,
            }
        model_specs.append(
            ModelSpec(
                label=str(item["label"]),
                model_id=model_id,
                cost_weight=float(item.get("cost_weight", 0.0)),
                dependence_group=None if item.get("dependence_group") is None else str(item.get("dependence_group")),
                input_cost_per_1k=input_rate,
                output_cost_per_1k=output_rate,
            )
        )

    return model_specs, PricingSnapshot(
        catalog_path=catalog_path,
        version=None if not catalog else str(catalog.get("version", "unknown")),
        as_of=None if not catalog else str(catalog.get("as_of", "unknown")),
        currency=str(catalog.get("currency", "USD")),
        prices_per_1m=prices_per_1m,
    )


def _validate_available_models(provider_cfg: dict[str, Any], models: list[ModelSpec]) -> None:
    if provider_cfg.get("type", "fake") not in {"nebius", "openai_compatible"}:
        return
    if not bool(provider_cfg.get("validate_models", True)):
        return
    api_key_env = str(provider_cfg.get("api_key_env", "NEBIUS_API_KEY"))
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Set `{api_key_env}` before validating provider models.")
    base_url = str(provider_cfg.get("base_url", "https://api.tokenfactory.nebius.com/v1/"))
    models_url = urllib.parse.urljoin(base_url.rstrip("/") + "/", "models?verbose=true")
    request = urllib.request.Request(models_url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(request, timeout=float(provider_cfg.get("model_validation_timeout", 30.0))) as response:
            payload = json.load(response)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to validate provider models using {models_url}: {exc}") from exc
    available_ids = {str(item.get("id")) for item in payload.get("data", []) if item.get("id")}
    missing = sorted(model.model_id for model in models if model.model_id not in available_ids)
    if missing:
        raise RuntimeError(f"Configured models are not available from the provider: {missing}")


def _clip_probs(probs: np.ndarray) -> np.ndarray:
    return np.clip(probs, EPS, 1.0 - EPS)


def _ece(y: np.ndarray, probs: np.ndarray, bins: int) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    score = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probs >= lower) & (probs < upper if upper < 1.0 else probs <= upper)
        if np.any(mask):
            score += float(np.mean(mask) * abs(np.mean(y[mask]) - np.mean(probs[mask])))
    return score


def _metrics(y: np.ndarray, probs: np.ndarray, bins: int) -> dict[str, float]:
    probs = _clip_probs(probs)
    return {
        "accuracy_at_0.5": float(np.mean((probs >= 0.5) == y)),
        "log_score": float(log_loss(y, probs, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, probs)),
        "ece": _ece(y, probs, bins),
    }


def _question_length_bucket(question: str) -> int:
    word_count = len(question.split())
    if word_count < 35:
        return 0
    if word_count < 80:
        return 1
    return 2


def _stable_uniform(seed: int, *parts: str) -> float:
    key = "|".join([str(seed), *parts]).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _fake_questions(subjects: list[str], sample_count: int) -> list[MMLUQuestion]:
    templates = [
        ("Which statement best follows from the definition of a probability distribution?", ("It assigns nonnegative mass that sums to one.", "It always assigns equal mass.", "It cannot model uncertainty.", "It only supports binary events."), "A"),
        ("A model has high variance. Which intervention most directly reduces variance?", ("Use fewer validation examples.", "Increase regularization or add data.", "Remove all features.", "Optimize only training accuracy."), "B"),
        ("In a market with perfect competition, long-run economic profit tends toward what value?", ("Negative infinity", "Zero", "The monopoly price", "The tax rate"), "B"),
        ("Which molecule carries genetic information in most organisms?", ("ATP", "DNA", "Glucose", "Cholesterol"), "B"),
        ("What is the derivative of x squared with respect to x?", ("x", "2x", "x squared", "2"), "B"),
        ("Which protocol is most associated with reliable ordered byte streams?", ("UDP", "IP", "TCP", "ARP"), "C"),
    ]
    questions: list[MMLUQuestion] = []
    for idx in range(sample_count):
        subject = subjects[idx % len(subjects)]
        prompt, choices, answer = templates[idx % len(templates)]
        questions.append(
            MMLUQuestion(
                question_id=f"fake-{idx:04d}",
                subject=subject,
                question=f"{prompt} Scenario id {idx}.",
                choices=choices,
                answer=answer,
            )
        )
    return questions


def _coerce_choices(item: Any) -> tuple[str, ...]:
    raw_choices = item["options"] if "options" in item else item["choices"]
    choices = tuple(str(choice) for choice in raw_choices)
    if len(choices) < 2:
        raise RuntimeError("Dataset row has fewer than two choices.")
    _answer_labels(len(choices))
    return choices


def _coerce_answer_label(item: Any, choices: tuple[str, ...]) -> str:
    labels = _answer_labels(len(choices))
    if "answer_index" in item and item["answer_index"] is not None:
        return labels[int(item["answer_index"])]
    answer = item["answer"]
    if isinstance(answer, int):
        return labels[int(answer)]
    answer_label = str(answer).strip().upper()
    if answer_label not in labels:
        raise RuntimeError(f"Dataset row answer `{answer_label}` is not valid for {len(choices)} choices.")
    return answer_label


def _question_subject(item: Any, fallback_subject: str) -> str:
    for key in ("subject", "category", "src"):
        if key in item and item[key] is not None:
            return str(item[key])
    return fallback_subject


def _question_id(item: Any, fallback_subject: str, index: int) -> str:
    if "question_id" in item and item["question_id"] is not None:
        return f"{fallback_subject}-{item['question_id']}"
    return f"{fallback_subject}-{index}"


def _row_to_question(item: Any, fallback_subject: str, index: int) -> MMLUQuestion:
    choices = _coerce_choices(item)
    subject = _question_subject(item, fallback_subject)
    return MMLUQuestion(
        question_id=_question_id(item, subject, index),
        subject=subject,
        question=str(item["question"]),
        choices=choices,
        answer=_coerce_answer_label(item, choices),
    )


def _is_mmlu_pro_dataset(name: str) -> bool:
    normalized = name.lower().replace("_", "-")
    return "mmlu-pro" in normalized


def _stratified_sample_indices(strata: list[str], sample_count: int, seed: int) -> list[int]:
    if sample_count <= 0 or not strata:
        return []

    groups: dict[str, list[int]] = {}
    for index, stratum in enumerate(strata):
        groups.setdefault(stratum, []).append(index)

    target = min(sample_count, len(strata))
    labels = sorted(groups)
    rng = np.random.default_rng(seed)
    allocations = {label: 0 for label in labels}

    if target < len(labels):
        for label in rng.permutation(labels)[:target]:
            allocations[str(label)] = 1
    else:
        quotas = {label: target * len(groups[label]) / len(strata) for label in labels}
        for label in labels:
            allocations[label] = max(1, min(len(groups[label]), int(np.floor(quotas[label]))))
        while sum(allocations.values()) > target:
            label = max(
                (candidate for candidate in labels if allocations[candidate] > 1),
                key=lambda candidate: (allocations[candidate] - quotas[candidate], allocations[candidate]),
            )
            allocations[label] -= 1
        unallocated = target - sum(allocations.values())
        remainder_order = sorted(
            labels,
            key=lambda label: (quotas[label] - np.floor(quotas[label]), len(groups[label]), label),
            reverse=True,
        )
        for label in remainder_order:
            if unallocated == 0:
                break
            if allocations[label] < len(groups[label]):
                allocations[label] += 1
                unallocated -= 1

    selected: list[int] = []
    for label in labels:
        count = allocations[label]
        if count:
            selected.extend(int(index) for index in rng.permutation(groups[label])[:count])
    rng.shuffle(selected)
    return selected


def _load_hf_questions(config: dict[str, Any], seed: int) -> list[MMLUQuestion]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install `datasets` to load Hugging Face MMLU or MMLU-Pro data.") from exc

    name = config.get("name", "cais/mmlu")
    split = config.get("split", "validation")
    subjects_default = [] if _is_mmlu_pro_dataset(str(name)) else [
        "abstract_algebra",
        "high_school_macroeconomics",
        "professional_medicine",
    ]
    subjects = list(config.get("subjects", subjects_default))
    sample_count = int(config.get("sample_count", 60))

    rows: list[MMLUQuestion] = []
    if _is_mmlu_pro_dataset(str(name)):
        dataset = load_dataset(name, split=split)
        allowed_subjects = set(subjects)
        filtered_indices = [
            idx
            for idx, item in enumerate(dataset)
            if not allowed_subjects or str(item.get("category", item.get("subject", ""))) in allowed_subjects
        ]
        sampling = str(config.get("sampling", "random"))
        if sampling == "stratified":
            strata = [str(dataset[index].get("category", dataset[index].get("subject", "unknown"))) for index in filtered_indices]
            selected_positions = _stratified_sample_indices(strata, sample_count, seed)
            selected_indices = [filtered_indices[position] for position in selected_positions]
        elif sampling == "random":
            selected_indices = [int(index) for index in np.random.default_rng(seed).permutation(filtered_indices)[:sample_count]]
        else:
            raise RuntimeError("MMLU-Pro dataset.sampling must be `random` or `stratified`.")
        for index in selected_indices:
            item = dataset[int(index)]
            rows.append(_row_to_question(item, "mmlu_pro", int(index)))
    else:
        per_subject = max(1, int(np.ceil(sample_count / len(subjects))))
        for subject in subjects:
            dataset = load_dataset(name, subject, split=split)
            indices = np.random.default_rng(seed + len(rows)).permutation(len(dataset))[:per_subject]
            for index in indices:
                item = dataset[int(index)]
                rows.append(_row_to_question(item, subject, int(index)))

    if not rows:
        raise RuntimeError(
            f"No questions loaded from dataset `{name}` split `{split}`. Check dataset.name, dataset.split, and dataset.subjects."
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    return rows[:sample_count]


class FakeProvider:
    def __init__(self, seed: int):
        self.seed = seed

    def call(self, prompt: str, model: ModelSpec, question: MMLUQuestion) -> dict[str, Any]:
        del prompt
        valid_answers = _answer_labels(len(question.choices))
        base = 0.48 + 0.25 * min(model.cost_weight / 0.12, 1.0)
        subject_bonus = 0.12 if model.label.lower().startswith(question.subject[:4].lower()) else 0.0
        p_correct = float(np.clip(base + subject_bonus, 0.2, 0.92))
        draw = _stable_uniform(self.seed, question.question_id, model.label)
        is_correct = draw < p_correct
        if is_correct:
            answer = question.answer
        else:
            wrong = [choice for choice in valid_answers if choice != question.answer]
            answer = wrong[int(draw * 1000) % len(wrong)]
        confidence = p_correct if is_correct else max(0.1, 1.0 - p_correct)
        response = json.dumps({"answer": answer, "confidence": round(confidence, 3)})
        return {
            "text": response,
            "latency_ms": 5.0 + 100.0 * model.cost_weight,
            "prompt_tokens": len(question.question.split()) + 40,
            "completion_tokens": 8,
            "total_tokens": len(question.question.split()) + 48,
            "json_mode_requested": False,
            "json_mode_used": False,
        }


class OpenAICompatibleProvider:
    def __init__(
        self,
        base_url: str,
        api_key_env: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        request_max_attempts: int = 8,
        retry_initial_delay_seconds: float = 2.0,
        retry_max_delay_seconds: float = 60.0,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install `openai` to call Nebius Token Factory.") from exc
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Set `{api_key_env}` before running the live provider workflow.")
        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.json_mode = json_mode
        self.request_max_attempts = max(1, request_max_attempts)
        self.retry_initial_delay_seconds = max(0.0, retry_initial_delay_seconds)
        self.retry_max_delay_seconds = max(self.retry_initial_delay_seconds, retry_max_delay_seconds)

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            return int(status_code) in {429, 500, 502, 503, 504}
        return type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}

    def _create_with_retry(self, request: dict[str, Any]):
        for attempt in range(1, self.request_max_attempts + 1):
            try:
                return self.client.chat.completions.create(**request)
            except Exception as exc:
                if not self._is_retryable(exc) or attempt == self.request_max_attempts:
                    raise
                delay = min(
                    self.retry_initial_delay_seconds * (2 ** (attempt - 1)),
                    self.retry_max_delay_seconds,
                )
                warnings.warn(
                    f"Transient model API failure ({type(exc).__name__}); retrying attempt "
                    f"{attempt + 1}/{self.request_max_attempts} in {delay:g}s.",
                    RuntimeWarning,
                )
                time.sleep(delay)

    def call(self, prompt: str, model: ModelSpec, question: MMLUQuestion) -> dict[str, Any]:
        del question
        started = time.perf_counter()
        request: dict[str, Any] = {
            "model": model.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.json_mode:
            request["response_format"] = {"type": "json_object"}
        json_mode_used = self.json_mode
        try:
            response = self._create_with_retry(request)
        except Exception as exc:
            if not self.json_mode:
                raise
            status_code = getattr(exc, "status_code", None)
            if status_code is not None and int(status_code) != 400:
                raise
            request.pop("response_format", None)
            json_mode_used = False
            response = self._create_with_retry(request)
        latency_ms = (time.perf_counter() - started) * 1000.0
        usage = getattr(response, "usage", None)
        text = response.choices[0].message.content or ""
        return {
            "text": text,
            "latency_ms": latency_ms,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "json_mode_requested": self.json_mode,
            "json_mode_used": json_mode_used,
        }


def _build_langgraph_runner(provider: FakeProvider | OpenAICompatibleProvider):
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise RuntimeError("Install `langgraph` to run the MMLU orchestrator execution graph.") from exc

    def format_node(state: GraphState) -> GraphState:
        return {"prompt": format_mmlu_prompt(state["question"])}

    def call_node(state: GraphState) -> GraphState:
        payload = provider.call(state["prompt"], state["model"], state["question"])
        return {
            "raw_response": payload["text"],
            "latency_ms": payload["latency_ms"],
            "prompt_tokens": payload["prompt_tokens"],
            "completion_tokens": payload["completion_tokens"],
            "total_tokens": payload["total_tokens"],
            "json_mode_requested": payload.get("json_mode_requested", False),
            "json_mode_used": payload.get("json_mode_used", False),
        }

    def parse_node(state: GraphState) -> GraphState:
        valid_answers = _answer_labels(len(state["question"].choices))
        answer, confidence = parse_answer_and_confidence(state["raw_response"], valid_answers)
        return {"parsed_answer": answer, "confidence": confidence}

    def log_node(state: GraphState) -> GraphState:
        question = state["question"]
        model = state["model"]
        answer = state.get("parsed_answer")
        result = CallResult(
            question_id=question.question_id,
            subject=question.subject,
            question=question.question,
            choices=question.choices,
            gold_answer=question.answer,
            model_label=model.label,
            model_id=model.model_id,
            cost_weight=model.cost_weight,
            question_length_bucket=_question_length_bucket(question.question),
            answer=answer,
            confidence=state.get("confidence"),
            correct=int(answer == question.answer),
            latency_ms=float(state["latency_ms"]),
            prompt_tokens=state.get("prompt_tokens"),
            completion_tokens=state.get("completion_tokens"),
            total_tokens=state.get("total_tokens"),
            json_mode_requested=bool(state.get("json_mode_requested", False)),
            json_mode_used=bool(state.get("json_mode_used", False)),
            raw_response=state["raw_response"],
        )
        return {"result": result}

    graph = StateGraph(GraphState)
    graph.add_node("format_prompt", format_node)
    graph.add_node("call_model", call_node)
    graph.add_node("parse_answer", parse_node)
    graph.add_node("log_result", log_node)
    graph.set_entry_point("format_prompt")
    graph.add_edge("format_prompt", "call_model")
    graph.add_edge("call_model", "parse_answer")
    graph.add_edge("parse_answer", "log_result")
    graph.add_edge("log_result", END)
    return graph.compile()


def _read_call_matrix(path: Path, questions: list[MMLUQuestion] | None = None) -> list[CallResult]:
    question_lookup = {} if questions is None else {question.question_id: question for question in questions}
    rows: list[CallResult] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if line.strip():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                remaining_lines = lines[line_number:]
                if not any(remaining.strip() for remaining in remaining_lines):
                    valid_content = "\n".join(lines[: line_number - 1])
                    path.write_text(valid_content + ("\n" if valid_content else ""), encoding="utf-8")
                    break
                raise RuntimeError(
                    f"Call matrix contains invalid JSON before its final record at {path}:{line_number}."
                ) from exc
            question = question_lookup.get(payload.get("question_id"))
            if question is not None:
                payload.setdefault("question", question.question)
                payload.setdefault("choices", question.choices)
                payload.setdefault("gold_answer", question.answer)
            if "choices" in payload:
                payload["choices"] = tuple(payload["choices"])
            payload.setdefault("json_mode_requested", False)
            payload.setdefault("json_mode_used", False)
            rows.append(CallResult(**payload))
    return rows


def _append_call_matrix_row(path: Path, row: CallResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_cached_call_matrix(
    rows: list[CallResult],
    questions: list[MMLUQuestion],
    models: list[ModelSpec],
    cache_path: Path,
    *,
    allow_partial: bool = False,
) -> None:
    expected_pairs = {
        (question.question_id, model.label)
        for question in questions
        for model in models
    }
    pair_counts = Counter((row.question_id, row.model_label) for row in rows)
    cached_pairs = set(pair_counts)
    duplicate_pairs = sorted(pair for pair, count in pair_counts.items() if count > 1)
    expected_ids = {model.label: model.model_id for model in models}
    expected_models = {model.label: model for model in models}
    expected_questions = {question.question_id: question for question in questions}
    stale_ids = sorted(
        {
            row.model_label
            for row in rows
            if row.model_label in expected_ids and row.model_id != expected_ids[row.model_label]
        }
    )
    stale_costs = sorted(
        {
            row.model_label
            for row in rows
            if row.model_label in expected_models
            and row.cost_weight != expected_models[row.model_label].cost_weight
        }
    )
    stale_questions = sorted(
        {
            row.question_id
            for row in rows
            if row.question_id in expected_questions
            and (
                row.subject != expected_questions[row.question_id].subject
                or row.question != expected_questions[row.question_id].question
                or row.choices != expected_questions[row.question_id].choices
                or row.gold_answer != expected_questions[row.question_id].answer
            )
        }
    )
    missing_pairs = expected_pairs - cached_pairs
    extra_pairs = cached_pairs - expected_pairs
    compatible = not extra_pairs and not duplicate_pairs and not stale_ids and not stale_costs and not stale_questions
    if compatible and (allow_partial or not missing_pairs):
        return

    missing = sorted(missing_pairs)[:5]
    extra = sorted(extra_pairs)[:5]
    details = [
        f"Cached call matrix is incompatible with the current config: {cache_path}.",
        f"Expected {len(expected_pairs)} question-model rows, found {len(rows)} rows with {len(cached_pairs)} unique question-model pairs.",
    ]
    if missing:
        details.append(f"Missing examples: {missing}")
    if extra:
        details.append(f"Extra examples: {extra}")
    if duplicate_pairs:
        details.append(f"Duplicate examples: {duplicate_pairs[:5]}")
    if stale_ids:
        details.append(f"Model labels with stale model IDs: {stale_ids}")
    if stale_costs:
        details.append(f"Model labels with stale cost weights: {stale_costs}")
    if stale_questions:
        details.append(f"Question IDs with stale content: {stale_questions[:5]}")
    details.append("Delete the cache file or set provider.reuse_cache: false to rebuild it.")
    raise RuntimeError(" ".join(details))


def _collect_call_matrix(
    questions: list[MMLUQuestion],
    models: list[ModelSpec],
    provider_cfg: dict[str, Any],
    seed: int,
    cache_path: Path,
) -> list[CallResult]:
    reuse_cache = bool(provider_cfg.get("reuse_cache", True))
    rows: list[CallResult] = []
    if reuse_cache and cache_path.exists():
        rows = _read_call_matrix(cache_path, questions)
        _validate_cached_call_matrix(rows, questions, models, cache_path, allow_partial=True)
        if len(rows) == len(questions) * len(models):
            _validate_cached_call_matrix(rows, questions, models, cache_path)
            return rows
    elif cache_path.exists():
        cache_path.write_text("", encoding="utf-8")

    provider_type = provider_cfg.get("type", "fake")
    _validate_available_models(provider_cfg, models)
    if provider_type == "fake":
        provider: FakeProvider | OpenAICompatibleProvider = FakeProvider(seed)
    elif provider_type in {"nebius", "openai_compatible"}:
        provider = OpenAICompatibleProvider(
            base_url=provider_cfg.get("base_url", "https://api.tokenfactory.nebius.com/v1/"),
            api_key_env=provider_cfg.get("api_key_env", "NEBIUS_API_KEY"),
            temperature=float(provider_cfg.get("temperature", 0.0)),
            max_tokens=int(provider_cfg.get("max_tokens", 32)),
            json_mode=bool(provider_cfg.get("json_mode", False)),
            request_max_attempts=int(provider_cfg.get("request_max_attempts", 8)),
            retry_initial_delay_seconds=float(provider_cfg.get("retry_initial_delay_seconds", 2.0)),
            retry_max_delay_seconds=float(provider_cfg.get("retry_max_delay_seconds", 60.0)),
        )
    else:
        raise RuntimeError(f"Unsupported provider type: {provider_type}")

    runner = _build_langgraph_runner(provider)
    completed_pairs = {(row.question_id, row.model_label) for row in rows}
    for question in questions:
        for model in models:
            pair = (question.question_id, model.label)
            if pair in completed_pairs:
                continue
            state = runner.invoke({"question": question, "model": model})
            result = state["result"]
            _append_call_matrix_row(cache_path, result)
            rows.append(result)
            completed_pairs.add(pair)
    _validate_cached_call_matrix(rows, questions, models, cache_path)
    return rows


def _split_question_ids(
    questions: list[MMLUQuestion], seed: int, exploration_fraction: float, stratify: bool = False
):
    ids = np.asarray([question.question_id for question in questions])
    exploration_end = max(1, int(len(ids) * exploration_fraction))
    exploration_end = min(exploration_end, len(ids) - 1)
    if stratify:
        selected = set(
            _stratified_sample_indices([question.subject for question in questions], exploration_end, seed)
        )
        exploration_ids = {question.question_id for index, question in enumerate(questions) if index in selected}
        return exploration_ids, set(ids) - exploration_ids
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    return set(ids[:exploration_end]), set(ids[exploration_end:])


def _model_lookup(model_specs: list[ModelSpec]) -> dict[str, ModelSpec]:
    return {model.label: model for model in model_specs}


def _effective_call_cost(row: CallResult, model_specs: dict[str, ModelSpec]) -> float:
    model = model_specs[row.model_label]
    if (
        model.input_cost_per_1k is not None
        and model.output_cost_per_1k is not None
        and row.prompt_tokens is not None
        and row.completion_tokens is not None
    ):
        return (
            float(row.prompt_tokens) * model.input_cost_per_1k
            + float(row.completion_tokens) * model.output_cost_per_1k
        ) / 1000.0
    return row.cost_weight


def _observed_call_cost_usd(row: CallResult, model_specs: dict[str, ModelSpec]) -> float | None:
    model = model_specs[row.model_label]
    if (
        model.input_cost_per_1k is None
        or model.output_cost_per_1k is None
        or row.prompt_tokens is None
        or row.completion_tokens is None
    ):
        return None
    return (
        float(row.prompt_tokens) * model.input_cost_per_1k
        + float(row.completion_tokens) * model.output_cost_per_1k
    ) / 1000.0


def _observed_rows_cost_usd(rows: list[CallResult], model_specs: dict[str, ModelSpec]) -> float:
    costs = [_observed_call_cost_usd(row, model_specs) for row in rows]
    if any(cost is None for cost in costs):
        return float("nan")
    return float(np.sum(costs))


def _operational_metrics(rows: list[CallResult], model_specs: dict[str, ModelSpec]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    total_cost = 0.0
    for label, model in model_specs.items():
        model_rows = [row for row in rows if row.model_label == label]
        if not model_rows:
            continue
        call_costs = [_effective_call_cost(row, model_specs) for row in model_rows]
        total_cost += float(np.sum(call_costs))
        throughputs = [
            float(row.completion_tokens) / (row.latency_ms / 1000.0)
            for row in model_rows
            if row.completion_tokens is not None and row.completion_tokens > 0 and row.latency_ms > 0
        ]
        metrics[f"{label} model id"] = model.model_id
        metrics[f"{label} input price per 1m tokens"] = (
            None if model.input_cost_per_1k is None else round(model.input_cost_per_1k * 1000.0, 8)
        )
        metrics[f"{label} output price per 1m tokens"] = (
            None if model.output_cost_per_1k is None else round(model.output_cost_per_1k * 1000.0, 8)
        )
        metrics[f"{label} observed prompt tokens"] = int(sum(row.prompt_tokens or 0 for row in model_rows))
        metrics[f"{label} observed completion tokens"] = int(sum(row.completion_tokens or 0 for row in model_rows))
        metrics[f"{label} observed total cost"] = round(float(np.sum(call_costs)), 8)
        metrics[f"{label} effective output tokens per second p50"] = (
            None if not throughputs else round(float(np.quantile(throughputs, 0.5)), 2)
        )
        metrics[f"{label} effective output tokens per second p95"] = (
            None if not throughputs else round(float(np.quantile(throughputs, 0.95)), 2)
        )
    metrics["observed call matrix total cost"] = round(total_cost, 8)
    return metrics


def _safe_roc_auc(y: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, probs))


def _score_predictions(y: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    probs = _clip_probs(probs)
    return {
        "brier_score": float(brier_score_loss(y, probs)),
        "log_score": float(log_loss(y, probs, labels=[0, 1])),
        "auroc": _safe_roc_auc(y, probs),
    }


def _encode_rows(
    rows: list[CallResult],
    model_index: dict[str, int],
    subject_index: dict[str, int],
    model_specs: dict[str, ModelSpec] | None = None,
) -> dict[str, np.ndarray]:
    confidence = np.asarray([0.5 if row.confidence is None else row.confidence for row in rows], dtype=float)
    total_tokens = np.asarray([0 if row.total_tokens is None else row.total_tokens for row in rows], dtype=float)
    latency = np.asarray([row.latency_ms for row in rows], dtype=float)
    costs = np.asarray(
        [
            row.cost_weight if model_specs is None else _effective_call_cost(row, model_specs)
            for row in rows
        ],
        dtype=float,
    )
    return {
        "model": np.asarray([model_index[row.model_label] for row in rows], dtype=int),
        "subject": np.asarray([subject_index[row.subject] for row in rows], dtype=int),
        "length_bucket": np.asarray([row.question_length_bucket for row in rows], dtype=int),
        "cost_weight": np.asarray([row.cost_weight for row in rows], dtype=float),
        "confidence": confidence,
        "latency": np.log1p(latency) / 10.0,
        "total_tokens": np.log1p(total_tokens) / 10.0,
        "effective_cost": costs,
        "correct": np.asarray([row.correct for row in rows], dtype=int),
    }


def _fit_reliability_model(
    rows: list[CallResult],
    model_index: dict[str, int],
    subject_index: dict[str, int],
    model_specs: dict[str, ModelSpec],
    warmup: int,
    samples: int,
    seed: int,
    *,
    include_observed_features: bool,
    use_confidence: bool = True,
) -> dict[str, np.ndarray]:
    try:
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
    except ImportError as exc:
        raise RuntimeError("Install `numpyro`, `jax`, and `jaxlib` to fit the Bayesian router.") from exc

    encoded = _encode_rows(rows, model_index, subject_index, model_specs)

    def model(
        model_id,
        subject_id,
        length_bucket,
        cost_weight,
        confidence,
        latency,
        total_tokens,
        effective_cost,
        correct=None,
    ):
        intercept = numpyro.sample("intercept", dist.Normal(0.0, 1.5))
        model_scale = numpyro.sample("model_scale", dist.HalfNormal(1.0))
        subject_scale = numpyro.sample("subject_scale", dist.HalfNormal(1.0))
        with numpyro.plate("model_plate", len(model_index)):
            model_effect = numpyro.sample("model_effect", dist.Normal(0.0, model_scale))
        with numpyro.plate("subject_plate", len(subject_index)):
            subject_effect = numpyro.sample("subject_effect", dist.Normal(0.0, subject_scale))
        with numpyro.plate("length_plate", 3):
            length_effect = numpyro.sample("length_effect", dist.Normal(0.0, 0.5))
        cost_slope = numpyro.sample("cost_slope", dist.Normal(0.0, 1.0))
        logits = (
            intercept
            + model_effect[model_id]
            + subject_effect[subject_id]
            + length_effect[length_bucket]
            + cost_slope * cost_weight
        )
        if include_observed_features:
            latency_slope = numpyro.sample("latency_slope", dist.Normal(0.0, 1.0))
            token_slope = numpyro.sample("token_slope", dist.Normal(0.0, 1.0))
            effective_cost_slope = numpyro.sample("effective_cost_slope", dist.Normal(0.0, 1.0))
            logits = (
                logits
                + latency_slope * latency
                + token_slope * total_tokens
                + effective_cost_slope * effective_cost
            )
            if use_confidence:
                confidence_slope = numpyro.sample("confidence_slope", dist.Normal(0.0, 1.0))
                logits = logits + confidence_slope * (confidence - 0.5)
        numpyro.sample("correct", dist.Bernoulli(logits=logits), obs=correct)

    mcmc = MCMC(NUTS(model, target_accept_prob=0.85), num_warmup=warmup, num_samples=samples, num_chains=1, progress_bar=False)
    mcmc.run(
        jax.random.PRNGKey(seed),
        jnp.asarray(encoded["model"]),
        jnp.asarray(encoded["subject"]),
        jnp.asarray(encoded["length_bucket"]),
        jnp.asarray(encoded["cost_weight"]),
        jnp.asarray(encoded["confidence"]),
        jnp.asarray(encoded["latency"]),
        jnp.asarray(encoded["total_tokens"]),
        jnp.asarray(encoded["effective_cost"]),
        jnp.asarray(encoded["correct"]),
    )
    return {name: np.asarray(value) for name, value in mcmc.get_samples().items()}


def _fit_bayesian_router(
    rows: list[CallResult],
    model_index: dict[str, int],
    subject_index: dict[str, int],
    warmup: int,
    samples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    fallback_specs = {
        label: ModelSpec(label=label, model_id=label, cost_weight=0.0)
        for label in model_index
    }
    return _fit_reliability_model(
        rows,
        model_index,
        subject_index,
        fallback_specs,
        warmup,
        samples,
        seed,
        include_observed_features=True,
        use_confidence=True,
    )


def _posterior_logits(
    samples: dict[str, np.ndarray],
    rows: list[CallResult],
    model_index: dict[str, int],
    subject_index: dict[str, int],
    model_specs: dict[str, ModelSpec],
    *,
    include_observed_features: bool,
    use_confidence: bool = True,
) -> np.ndarray:
    encoded = _encode_rows(rows, model_index, subject_index, model_specs)
    logits = (
        samples["intercept"][:, None]
        + samples["model_effect"][:, encoded["model"]]
        + samples["subject_effect"][:, encoded["subject"]]
        + samples["length_effect"][:, encoded["length_bucket"]]
        + samples["cost_slope"][:, None] * encoded["cost_weight"][None, :]
    )
    if include_observed_features:
        logits = (
            logits
            + samples["latency_slope"][:, None] * encoded["latency"][None, :]
            + samples["token_slope"][:, None] * encoded["total_tokens"][None, :]
            + samples["effective_cost_slope"][:, None] * encoded["effective_cost"][None, :]
        )
        if use_confidence:
            logits = logits + samples["confidence_slope"][:, None] * (encoded["confidence"][None, :] - 0.5)
    return logits


def _predict_reliability_probs(
    samples: dict[str, np.ndarray],
    rows: list[CallResult],
    model_index: dict[str, int],
    subject_index: dict[str, int],
    model_specs: dict[str, ModelSpec],
    *,
    include_observed_features: bool,
    use_confidence: bool = True,
) -> np.ndarray:
    logits = _posterior_logits(
        samples,
        rows,
        model_index,
        subject_index,
        model_specs,
        include_observed_features=include_observed_features,
        use_confidence=use_confidence,
    )
    return expit(logits).mean(axis=0)


def _predict_router_probs(samples: dict[str, np.ndarray], rows: list[CallResult], model_index: dict[str, int], subject_index: dict[str, int]) -> np.ndarray:
    fallback_specs = {
        label: ModelSpec(label=label, model_id=label, cost_weight=0.0)
        for label in model_index
    }
    return _predict_reliability_probs(
        samples,
        rows,
        model_index,
        subject_index,
        fallback_specs,
        include_observed_features=True,
        use_confidence=True,
    )


def _pointwise_log_likelihood(
    samples: dict[str, np.ndarray],
    rows: list[CallResult],
    model_index: dict[str, int],
    subject_index: dict[str, int],
    model_specs: dict[str, ModelSpec],
    *,
    include_observed_features: bool,
    use_confidence: bool = True,
) -> np.ndarray:
    logits = _posterior_logits(
        samples,
        rows,
        model_index,
        subject_index,
        model_specs,
        include_observed_features=include_observed_features,
        use_confidence=use_confidence,
    )
    y = np.asarray([row.correct for row in rows], dtype=float)[None, :]
    return y * logits - np.logaddexp(0.0, logits)


def _loo_elpd(
    samples: dict[str, np.ndarray],
    rows: list[CallResult],
    model_index: dict[str, int],
    subject_index: dict[str, int],
    model_specs: dict[str, ModelSpec],
    *,
    include_observed_features: bool,
    use_confidence: bool = True,
) -> float:
    log_likelihood = _pointwise_log_likelihood(
        samples,
        rows,
        model_index,
        subject_index,
        model_specs,
        include_observed_features=include_observed_features,
        use_confidence=use_confidence,
    )
    try:
        old_home = os.environ.get("HOME")
        os.environ["HOME"] = tempfile.gettempdir()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import arviz as az

            posterior = {
                name: np.expand_dims(value, axis=0)
                for name, value in samples.items()
            }
            idata = az.from_dict(
                posterior=posterior,
                log_likelihood={"correct": np.expand_dims(log_likelihood, axis=0)},
            )
            return float(az.loo(idata, var_name="correct").elpd_loo)
    except Exception:
        # Fallback to summed log pointwise predictive density when ArviZ cannot import in restricted sandboxes.
        max_ll = np.max(log_likelihood, axis=0)
        lppd = max_ll + np.log(np.mean(np.exp(log_likelihood - max_ll[None, :]), axis=0))
        return float(np.sum(lppd))
    finally:
        if "old_home" in locals():
            if old_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = old_home


def _rows_by_question(rows: list[CallResult]) -> dict[str, list[CallResult]]:
    grouped: dict[str, list[CallResult]] = {}
    for row in rows:
        grouped.setdefault(row.question_id, []).append(row)
    return grouped


def _policy_scores(
    rows: list[CallResult],
    probs: np.ndarray,
    model_labels: list[str],
    cost_scale: float,
    model_specs: dict[str, ModelSpec] | None = None,
) -> dict[str, Any]:
    grouped = _rows_by_question(rows)
    prob_by_key = {(row.question_id, row.model_label): float(prob) for row, prob in zip(rows, probs)}
    bayes_utilities = []
    random_utilities = []
    oracle_utilities = []
    selected_rows: list[CallResult] = []
    always: dict[str, list[float]] = {label: [] for label in model_labels}
    for question_id, candidates in grouped.items():
        cost_by_row = {
            row.model_label: row.cost_weight if model_specs is None else _effective_call_cost(row, model_specs)
            for row in candidates
        }
        selected = max(
            candidates,
            key=lambda row: prob_by_key[(question_id, row.model_label)] - cost_by_row[row.model_label] * cost_scale,
        )
        selected_rows.append(selected)
        bayes_utilities.append(utility(selected.correct, cost_by_row[selected.model_label], cost_scale))
        random_utilities.append(float(np.mean([utility(row.correct, cost_by_row[row.model_label], cost_scale) for row in candidates])))
        oracle_utilities.append(float(np.max([utility(row.correct, cost_by_row[row.model_label], cost_scale) for row in candidates])))
        for row in candidates:
            always[row.model_label].append(utility(row.correct, cost_by_row[row.model_label], cost_scale))
    return {
        "bayesian": np.asarray(bayes_utilities),
        "random": np.asarray(random_utilities),
        "oracle": np.asarray(oracle_utilities),
        "always": {label: np.asarray(values) for label, values in always.items()},
        "selected_rows": selected_rows,
    }


def _group_key(row: CallResult) -> tuple[str, str, int]:
    return (row.model_label, row.subject, row.question_length_bucket)


def _fallback_group_key(row: CallResult) -> tuple[str, str, int]:
    return (row.model_label, "__all__", -1)


def _build_future_value_lookup(
    rows: list[CallResult],
    post_probs: np.ndarray,
    model_specs: dict[str, ModelSpec],
) -> tuple[dict[tuple[str, str, int], np.ndarray], dict[tuple[str, str, int], float]]:
    values: dict[tuple[str, str, int], list[float]] = {}
    costs: dict[tuple[str, str, int], list[float]] = {}
    for row, prob in zip(rows, post_probs):
        for key in (_group_key(row), _fallback_group_key(row)):
            values.setdefault(key, []).append(float(prob))
            costs.setdefault(key, []).append(_effective_call_cost(row, model_specs))
    return (
        {key: np.asarray(group_values, dtype=float) for key, group_values in values.items()},
        {key: float(np.mean(group_costs)) for key, group_costs in costs.items()},
    )


def _expected_future_terminal(
    current_best: float,
    row: CallResult,
    future_values: dict[tuple[str, str, int], np.ndarray],
) -> float:
    values = future_values.get(_group_key(row))
    if values is None:
        values = future_values.get(_fallback_group_key(row))
    if values is None or len(values) == 0:
        return current_best
    return float(np.mean(np.maximum(current_best, values)))


def _expected_call_cost(
    row: CallResult,
    future_costs: dict[tuple[str, str, int], float],
    model_specs: dict[str, ModelSpec],
) -> float:
    return future_costs.get(_group_key(row), future_costs.get(_fallback_group_key(row), _effective_call_cost(row, model_specs)))


def _dependence_caution(called_rows: list[CallResult], model_specs: dict[str, ModelSpec]) -> bool:
    groups = [
        model_specs[row.model_label].dependence_group
        for row in called_rows
        if model_specs[row.model_label].dependence_group
    ]
    return len(groups) != len(set(groups))


def _answer_scores(called_rows: list[CallResult], post_prob_by_key: dict[tuple[str, str], float]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for row in called_rows:
        if row.answer not in _answer_labels(len(row.choices)):
            continue
        prob = post_prob_by_key[(row.question_id, row.model_label)]
        scores.setdefault(row.answer, 1.0)
        scores[row.answer] *= 1.0 - prob
    return {answer: 1.0 - miss_prob for answer, miss_prob in scores.items()}


def _adjudicated_answer(called_rows: list[CallResult], post_prob_by_key: dict[tuple[str, str], float]) -> str | None:
    scores = _answer_scores(called_rows, post_prob_by_key)
    if not scores:
        return None
    return max(scores, key=lambda answer: (scores[answer], answer))


def _row_for_answer(called_rows: list[CallResult], answer: str | None, post_prob_by_key: dict[tuple[str, str], float]) -> CallResult:
    candidates = [row for row in called_rows if row.answer == answer]
    if not candidates:
        candidates = list(called_rows)
    return max(candidates, key=lambda row: post_prob_by_key[(row.question_id, row.model_label)])


def _called_answers_disagree(called_rows: list[CallResult]) -> bool:
    answers = [row.answer for row in called_rows if row.answer in _answer_labels(len(row.choices))]
    return len(set(answers)) > 1


def _adaptive_policy_scores(
    rows: list[CallResult],
    pre_probs: np.ndarray,
    post_probs: np.ndarray,
    future_values: dict[tuple[str, str, int], np.ndarray],
    future_costs: dict[tuple[str, str, int], float],
    model_specs: dict[str, ModelSpec],
    cost_scale: float,
    max_calls: int,
    min_expected_gain: float,
) -> AdaptivePolicyResult:
    grouped = _rows_by_question(rows)
    pre_prob_by_key = {(row.question_id, row.model_label): float(prob) for row, prob in zip(rows, pre_probs)}
    post_prob_by_key = {(row.question_id, row.model_label): float(prob) for row, prob in zip(rows, post_probs)}
    utilities: list[float] = []
    call_counts: list[int] = []
    cumulative_costs: list[float] = []
    stop_after_one: list[bool] = []
    accepted_voi_calls: list[int] = []
    selected_rows: list[CallResult] = []
    all_called_rows: list[list[CallResult]] = []
    final_answers: list[str | None] = []
    first_answers: list[str | None] = []
    disagreement: list[bool] = []
    changed_answers: list[bool] = []
    invalid_first_answers: list[bool] = []
    dependence_triggered = False

    for question_id, candidates in grouped.items():
        remaining = list(candidates)
        called: list[CallResult] = []
        cumulative_cost = 0.0
        accepted_calls = 0

        first = max(
            remaining,
            key=lambda row: pre_prob_by_key[(question_id, row.model_label)] - _effective_call_cost(row, model_specs) * cost_scale,
        )
        called.append(first)
        remaining.remove(first)
        cumulative_cost += _effective_call_cost(first, model_specs)

        while remaining and len(called) < max_calls:
            current_best = max(post_prob_by_key[(question_id, row.model_label)] for row in called)
            best_row: CallResult | None = None
            best_gain = -np.inf
            for row in remaining:
                expected_terminal = _expected_future_terminal(current_best, row, future_values)
                expected_cost = _expected_call_cost(row, future_costs, model_specs) * cost_scale
                gain = expected_terminal - current_best - expected_cost
                if gain > best_gain:
                    best_gain = gain
                    best_row = row
            if best_row is None or best_gain <= min_expected_gain:
                break
            called.append(best_row)
            remaining.remove(best_row)
            cumulative_cost += _effective_call_cost(best_row, model_specs)
            accepted_calls += 1

        final_answer = _adjudicated_answer(called, post_prob_by_key)
        selected = _row_for_answer(called, final_answer, post_prob_by_key)
        utilities.append(float(final_answer == selected.gold_answer) - cumulative_cost * cost_scale)
        call_counts.append(len(called))
        cumulative_costs.append(cumulative_cost)
        stop_after_one.append(len(called) == 1)
        accepted_voi_calls.append(accepted_calls)
        selected_rows.append(selected)
        all_called_rows.append(called)
        final_answers.append(final_answer)
        first_answers.append(called[0].answer)
        disagreement.append(_called_answers_disagree(called))
        changed_answers.append(final_answer != called[0].answer)
        invalid_first_answers.append(called[0].answer not in _answer_labels(len(called[0].choices)))
        dependence_triggered = dependence_triggered or _dependence_caution(called, model_specs)

    return AdaptivePolicyResult(
        utilities=np.asarray(utilities, dtype=float),
        call_counts=np.asarray(call_counts, dtype=float),
        cumulative_costs=np.asarray(cumulative_costs, dtype=float),
        stop_after_one=np.asarray(stop_after_one, dtype=bool),
        accepted_voi_calls=np.asarray(accepted_voi_calls, dtype=float),
        selected_rows=selected_rows,
        called_rows=all_called_rows,
        dependence_caution_triggered=dependence_triggered,
        final_answers=final_answers,
        first_answers=first_answers,
        disagreement=np.asarray(disagreement, dtype=bool),
        changed_answers=np.asarray(changed_answers, dtype=bool),
        invalid_first_answer=np.asarray(invalid_first_answers, dtype=bool),
    )


def _adaptive_adjudication_policy_scores(
    rows: list[CallResult],
    pre_probs: np.ndarray,
    post_probs: np.ndarray,
    model_specs: dict[str, ModelSpec],
    cost_scale: float,
    max_calls: int,
    min_expected_gain: float,
    invalid_answer_triggers_call: bool,
) -> AdaptivePolicyResult:
    grouped = _rows_by_question(rows)
    pre_prob_by_key = {(row.question_id, row.model_label): float(prob) for row, prob in zip(rows, pre_probs)}
    post_prob_by_key = {(row.question_id, row.model_label): float(prob) for row, prob in zip(rows, post_probs)}
    utilities: list[float] = []
    call_counts: list[int] = []
    cumulative_costs: list[float] = []
    stop_after_one: list[bool] = []
    accepted_voi_calls: list[int] = []
    selected_rows: list[CallResult] = []
    all_called_rows: list[list[CallResult]] = []
    final_answers: list[str | None] = []
    first_answers: list[str | None] = []
    disagreement: list[bool] = []
    changed_answers: list[bool] = []
    invalid_first_answers: list[bool] = []
    dependence_triggered = False

    for question_id, candidates in grouped.items():
        remaining = list(candidates)
        called: list[CallResult] = []
        cumulative_cost = 0.0
        accepted_calls = 0

        first = max(
            remaining,
            key=lambda row: pre_prob_by_key[(question_id, row.model_label)] - _effective_call_cost(row, model_specs) * cost_scale,
        )
        called.append(first)
        remaining.remove(first)
        cumulative_cost += _effective_call_cost(first, model_specs)
        first_invalid = first.answer not in _answer_labels(len(first.choices))

        while remaining and len(called) < max_calls:
            current_answer = _adjudicated_answer(called, post_prob_by_key)
            current_answer_probability = 0.0
            if current_answer is not None:
                current_answer_probability = _answer_scores(called, post_prob_by_key)[current_answer]
            best_row: CallResult | None = None
            best_gain = -np.inf
            for row in remaining:
                candidate_prob = pre_prob_by_key[(question_id, row.model_label)]
                expected_cost = _effective_call_cost(row, model_specs) * cost_scale
                gain = (1.0 - current_answer_probability) * candidate_prob - expected_cost
                if gain > best_gain:
                    best_gain = gain
                    best_row = row
            force_call = first_invalid and invalid_answer_triggers_call and len(called) == 1
            if best_row is None or (not force_call and best_gain <= min_expected_gain):
                break
            called.append(best_row)
            remaining.remove(best_row)
            cumulative_cost += _effective_call_cost(best_row, model_specs)
            accepted_calls += 1

        final_answer = _adjudicated_answer(called, post_prob_by_key)
        selected = _row_for_answer(called, final_answer, post_prob_by_key)
        utilities.append(float(final_answer == selected.gold_answer) - cumulative_cost * cost_scale)
        call_counts.append(len(called))
        cumulative_costs.append(cumulative_cost)
        stop_after_one.append(len(called) == 1)
        accepted_voi_calls.append(accepted_calls)
        selected_rows.append(selected)
        all_called_rows.append(called)
        final_answers.append(final_answer)
        first_answers.append(first.answer)
        disagreement.append(_called_answers_disagree(called))
        changed_answers.append(final_answer != first.answer)
        invalid_first_answers.append(first_invalid)
        dependence_triggered = dependence_triggered or _dependence_caution(called, model_specs)

    return AdaptivePolicyResult(
        utilities=np.asarray(utilities, dtype=float),
        call_counts=np.asarray(call_counts, dtype=float),
        cumulative_costs=np.asarray(cumulative_costs, dtype=float),
        stop_after_one=np.asarray(stop_after_one, dtype=bool),
        accepted_voi_calls=np.asarray(accepted_voi_calls, dtype=float),
        selected_rows=selected_rows,
        called_rows=all_called_rows,
        dependence_caution_triggered=dependence_triggered,
        final_answers=final_answers,
        first_answers=first_answers,
        disagreement=np.asarray(disagreement, dtype=bool),
        changed_answers=np.asarray(changed_answers, dtype=bool),
        invalid_first_answer=np.asarray(invalid_first_answers, dtype=bool),
    )


def _called_answer_diversity(called_rows: list[list[CallResult]]) -> float:
    diversities = []
    for rows in called_rows:
        answers = [row.answer for row in rows if row.answer is not None]
        if answers:
            diversities.append(len(set(answers)) / len(answers))
    return float(np.mean(diversities)) if diversities else 0.0


def _bootstrap_diff_ci(bayes: np.ndarray, random: np.ndarray, iterations: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(bayes)
    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        diffs.append(float(np.mean(bayes[idx] - random[idx])))
    return tuple(np.quantile(diffs, [0.025, 0.975]))


def _plot_reliability(path: Path, y: np.ndarray, predictions: dict[str, np.ndarray], bins: int) -> None:
    fig, (ax, support_ax) = plt.subplots(
        2,
        1,
        figsize=(7.5, 6.2),
        layout="constrained",
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08},
    )
    ax.plot([0, 1], [0, 1], "--", color="#555555", label="perfect calibration")
    edges = np.linspace(0.0, 1.0, bins + 1)
    colors = {"posterior": "#31688e"}
    for name, probs in predictions.items():
        xs, ys, counts = [], [], []
        for lower, upper in zip(edges[:-1], edges[1:]):
            mask = (probs >= lower) & (probs < upper if upper < 1.0 else probs <= upper)
            if np.any(mask):
                xs.append(float(np.mean(probs[mask])))
                ys.append(float(np.mean(y[mask])))
                counts.append(int(np.sum(mask)))
        color = colors.get(name)
        ax.plot(xs, ys, linewidth=2, label=name, color=color, zorder=2)
        ax.scatter(xs, ys, s=38 + 12 * np.sqrt(counts), color=color, edgecolor="white", linewidth=0.8, zorder=3)
        support_ax.hist(probs, bins=edges, color=color, alpha=0.78, rwidth=0.88)
    ax.set_title("MMLU Router Calibration", loc="left", fontsize=14, fontweight="bold")
    ax.set_ylabel("Observed correctness rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.18)
    ax.legend()
    support_ax.set_xlabel("Predicted correctness probability")
    support_ax.set_ylabel("Calls")
    support_ax.grid(axis="y", alpha=0.18)
    fig.text(0.99, 0.99, f"n = {len(y)} held-out calls", ha="right", va="top", fontsize=9, color="#555555")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_policy_values(
    path: Path,
    policy_metrics: list[dict[str, float | str]],
) -> None:
    display_names = {
        "random": "Random",
        "single_shot_bayesian": "Bayesian\nsingle shot",
        "adaptive_bayesian": "Adaptive\nBayesian",
        "oracle": "Oracle",
    }
    colors = {
        "random": "#8a8f98",
        "single_shot_bayesian": "#2f6f9f",
        "adaptive_bayesian": "#d95f43",
        "oracle": "#d6a238",
    }
    labels = [str(item["label"]) for item in policy_metrics]
    names = [display_names.get(label, label.removeprefix("always_").replace("_", " ").title()) for label in labels]
    palette = [colors.get(label, "#6f9f76") for label in labels]
    accuracies = np.asarray([float(item["accuracy"]) for item in policy_metrics])
    costs_per_1k = np.asarray([float(item["cost_usd"]) * 1000.0 for item in policy_metrics])

    fig, (accuracy_ax, cost_ax) = plt.subplots(1, 2, figsize=(13.2, 6.4), layout="constrained")
    positions = np.arange(len(labels))
    accuracy_bars = accuracy_ax.barh(positions, accuracies, color=palette, height=0.68)
    accuracy_ax.set_yticks(positions, names)
    accuracy_ax.invert_yaxis()
    accuracy_ax.set_xlim(0, 1)
    accuracy_ax.set_xlabel("Held-out accuracy")
    accuracy_ax.set_title("Accuracy", loc="left", fontweight="bold")
    accuracy_ax.grid(axis="x", alpha=0.18)
    for bar, value in zip(accuracy_bars, accuracies):
        accuracy_ax.text(value + 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.1%}", va="center", fontsize=9)

    cost_ax.set_yticks(positions, names)
    cost_ax.invert_yaxis()
    cost_ax.set_xlabel("Observed model cost per 1,000 questions (USD)")
    cost_ax.set_title("Token Cost", loc="left", fontweight="bold")
    cost_ax.grid(axis="x", alpha=0.18)
    if np.all(np.isfinite(costs_per_1k)):
        cost_bars = cost_ax.barh(positions, costs_per_1k, color=palette, height=0.68)
        padding = max(0.0001, float(np.max(costs_per_1k)) * 0.025)
        for bar, value in zip(cost_bars, costs_per_1k):
            cost_ax.text(value + padding, bar.get_y() + bar.get_height() / 2, f"${value:.4f}", va="center", fontsize=9)
    else:
        cost_ax.text(
            0.5,
            0.5,
            "Observed USD cost unavailable\n(pricing or token counts missing)",
            transform=cost_ax.transAxes,
            ha="center",
            va="center",
            color="#555555",
        )

    fig.suptitle("MMLU Accuracy / Cost Tradeoff", x=0.01, ha="left", fontsize=16, fontweight="bold")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_mmlu_bayesian_orchestrator(config: dict[str, Any]) -> WorkflowResult:
    seed = int(config.get("seed", 24051))
    out_dir = output_dir(config)
    dataset_cfg = config.get("dataset", {})
    provider_cfg = config.get("provider", {})
    eval_cfg = config.get("evaluation", {})
    mcmc_cfg = config.get("mcmc", {})
    bins = int(config.get("diagnostics", {}).get("calibration_bins", 10))

    model_specs, pricing_snapshot = _resolve_model_specs(config)
    if len(model_specs) < 2:
        raise RuntimeError("Configure at least two models for the MMLU orchestrator comparison.")

    subjects = list(dataset_cfg.get("subjects", ["abstract_algebra", "high_school_macroeconomics", "professional_medicine"]))
    sample_count = int(dataset_cfg.get("sample_count", 60))
    if dataset_cfg.get("source", "fake") == "fake":
        questions = _fake_questions(subjects, sample_count)
    else:
        questions = _load_hf_questions(dataset_cfg, seed)

    cache_path = out_dir / str(provider_cfg.get("cache_file", "call_matrix.jsonl"))
    rows = _collect_call_matrix(questions, model_specs, provider_cfg, seed, cache_path)

    exploration_ids, test_ids = _split_question_ids(
        questions,
        seed,
        float(eval_cfg.get("exploration_fraction", 0.5)),
        bool(eval_cfg.get("stratify_split", False)),
    )
    exploration_rows = [row for row in rows if row.question_id in exploration_ids]
    test_rows = [row for row in rows if row.question_id in test_ids]
    if not exploration_rows or not test_rows:
        raise RuntimeError("Dataset split produced an empty exploration or test partition.")

    model_index = {model.label: idx for idx, model in enumerate(model_specs)}
    model_spec_by_label = _model_lookup(model_specs)
    operational_metrics = _operational_metrics(rows, model_spec_by_label)
    subject_index = {subject: idx for idx, subject in enumerate(sorted({row.subject for row in rows}))}
    warmup = int(mcmc_cfg.get("warmup", 120))
    sample_count_mcmc = int(mcmc_cfg.get("samples", 180))
    pre_samples = _fit_reliability_model(
        exploration_rows,
        model_index,
        subject_index,
        model_spec_by_label,
        warmup,
        sample_count_mcmc,
        seed,
        include_observed_features=False,
    )
    post_samples = _fit_reliability_model(
        exploration_rows,
        model_index,
        subject_index,
        model_spec_by_label,
        warmup,
        sample_count_mcmc,
        seed + 1,
        include_observed_features=True,
        use_confidence=True,
    )
    post_no_confidence_samples = _fit_reliability_model(
        exploration_rows,
        model_index,
        subject_index,
        model_spec_by_label,
        warmup,
        sample_count_mcmc,
        seed + 2,
        include_observed_features=True,
        use_confidence=False,
    )
    test_pre_probs = _predict_reliability_probs(
        pre_samples,
        test_rows,
        model_index,
        subject_index,
        model_spec_by_label,
        include_observed_features=False,
    )
    test_posterior_probs = _predict_reliability_probs(
        post_samples,
        test_rows,
        model_index,
        subject_index,
        model_spec_by_label,
        include_observed_features=True,
        use_confidence=True,
    )
    test_no_confidence_probs = _predict_reliability_probs(
        post_no_confidence_samples,
        test_rows,
        model_index,
        subject_index,
        model_spec_by_label,
        include_observed_features=True,
        use_confidence=False,
    )
    y_test = np.asarray([row.correct for row in test_rows])

    cost_scale = float(eval_cfg.get("cost_scale", 1.0))
    max_calls = max(1, int(eval_cfg.get("max_calls", len(model_specs))))
    min_expected_gain = float(eval_cfg.get("min_expected_gain", 0.0))
    adaptive_strategy = str(eval_cfg.get("adaptive_strategy", "myopic_voi"))
    invalid_answer_triggers_call = bool(eval_cfg.get("invalid_answer_triggers_call", True))
    policy = _policy_scores(
        test_rows,
        test_pre_probs,
        [model.label for model in model_specs],
        cost_scale,
        model_spec_by_label,
    )
    exploration_post_probs = _predict_reliability_probs(
        post_samples,
        exploration_rows,
        model_index,
        subject_index,
        model_spec_by_label,
        include_observed_features=True,
        use_confidence=True,
    )
    future_values, future_costs = _build_future_value_lookup(
        exploration_rows,
        exploration_post_probs,
        model_spec_by_label,
    )
    if adaptive_strategy == "adjudication":
        adaptive_policy = _adaptive_adjudication_policy_scores(
            test_rows,
            test_pre_probs,
            test_posterior_probs,
            model_spec_by_label,
            cost_scale,
            max_calls,
            min_expected_gain,
            invalid_answer_triggers_call,
        )
    elif adaptive_strategy == "myopic_voi":
        adaptive_policy = _adaptive_policy_scores(
            test_rows,
            test_pre_probs,
            test_posterior_probs,
            future_values,
            future_costs,
            model_spec_by_label,
            cost_scale,
            max_calls,
            min_expected_gain,
        )
    else:
        raise RuntimeError(f"Unsupported adaptive_strategy: {adaptive_strategy}")
    ci_low, ci_high = _bootstrap_diff_ci(
        adaptive_policy.utilities,
        policy["random"],
        int(eval_cfg.get("bootstrap_iterations", 300)),
        seed + 17,
    )

    posterior_metrics = _metrics(y_test, test_posterior_probs, bins)
    confidence_metrics = _score_predictions(y_test, test_posterior_probs)
    no_confidence_metrics = _score_predictions(y_test, test_no_confidence_probs)
    confidence_delta_brier = no_confidence_metrics["brier_score"] - confidence_metrics["brier_score"]
    confidence_delta_log = no_confidence_metrics["log_score"] - confidence_metrics["log_score"]
    confidence_delta_auroc = confidence_metrics["auroc"] - no_confidence_metrics["auroc"]
    confidence_improves = confidence_delta_brier > 0 and confidence_delta_log > 0
    pre_loo_elpd = _loo_elpd(
        pre_samples,
        exploration_rows,
        model_index,
        subject_index,
        model_spec_by_label,
        include_observed_features=False,
    )
    post_loo_elpd = _loo_elpd(
        post_samples,
        exploration_rows,
        model_index,
        subject_index,
        model_spec_by_label,
        include_observed_features=True,
        use_confidence=True,
    )
    bayesian_utility = float(np.mean(policy["bayesian"]))
    adaptive_utility = float(np.mean(adaptive_policy.utilities))
    random_utility = float(np.mean(policy["random"]))
    oracle_utility = float(np.mean(policy["oracle"]))
    bayesian_accuracy = float(np.mean([row.correct for row in policy["selected_rows"]]))
    adaptive_correct = np.asarray(
        [
            final_answer == selected.gold_answer
            for final_answer, selected in zip(adaptive_policy.final_answers, adaptive_policy.selected_rows)
        ],
        dtype=float,
    )
    adaptive_accuracy = float(np.mean(adaptive_correct))
    bayesian_cost = float(np.mean([_effective_call_cost(row, model_spec_by_label) for row in policy["selected_rows"]]))
    latency_values = np.asarray([row.latency_ms for row in test_rows])
    utility_improvement = adaptive_utility - random_utility
    single_shot_utility_improvement = adaptive_utility - bayesian_utility
    policy_beats_random = adaptive_utility > random_utility
    ci_directional = ci_low > 0.0
    best_always = max(policy["always"], key=lambda label: float(np.mean(policy["always"][label])))
    best_always_utility = float(np.mean(policy["always"][best_always]))
    beats_best_always = adaptive_utility > best_always_utility
    accepted_voi_rate = float(np.mean(adaptive_policy.accepted_voi_calls > 0))
    average_call_count = float(np.mean(adaptive_policy.call_counts))
    answer_diversity = _called_answer_diversity(adaptive_policy.called_rows) if average_call_count > 1.0 else float("nan")
    disagreement_rate = float(np.mean(adaptive_policy.disagreement))
    changed_answer_rate = float(np.mean(adaptive_policy.changed_answers))
    invalid_first_answer_rate = float(np.mean(adaptive_policy.invalid_first_answer))

    warnings: list[str] = []
    if dataset_cfg.get("source", "fake") == "fake":
        warnings.append("Using fake MMLU-shaped data and fake model calls; switch dataset.source to `hf` and provider.type to `nebius` for live evaluation.")
    if not policy_beats_random:
        warnings.append("Bayesian orchestrator did not beat the exact random-routing baseline on held-out utility.")
    if single_shot_utility_improvement <= 0.0:
        warnings.append("Adaptive routing did not improve held-out utility over single-shot Bayesian routing in this run.")
    if not beats_best_always:
        warnings.append(f"Adaptive routing did not beat the best always-use-model baseline ({best_always}).")
    if accepted_voi_rate == 0.0:
        warnings.append(f"Adaptive strategy `{adaptive_strategy}` accepted no additional model calls; this run behaves like single-shot Bayesian routing.")
    if ci_low <= 0.0:
        warnings.append("Bootstrap confidence interval is not strictly positive; increase sample size before claiming policy improvement.")
    if len(test_ids) < 30:
        warnings.append(
            f"Held-out question count is small ({len(test_ids)}). Increase dataset.sample_count or reduce evaluation.exploration_fraction "
            "so the test partition has at least 30 questions for a rough demo, and preferably 60-120 before claiming stable lift."
        )
    if adaptive_policy.dependence_caution_triggered:
        warnings.append("Multiple called models shared a dependence group for at least one question; agreement was treated as descriptive only, not independent evidence.")
    json_mode_requested_rate = float(np.mean([row.json_mode_requested for row in rows]))
    json_mode_used_rate = float(np.mean([row.json_mode_used for row in rows]))
    if json_mode_requested_rate > 0.0 and json_mode_used_rate < json_mode_requested_rate:
        warnings.append(
            "JSON mode was requested but rejected for at least one model call; those calls were retried without response_format. "
            "Inspect json_mode_used in call_matrix.jsonl."
        )
    warnings.append("The adaptive policy is a cost-sensitive myopic heuristic, not full BED-LLM expected information gain.")

    policy_plot = out_dir / "mmlu_accuracy_cost.png"
    grouped_test_rows = _rows_by_question(test_rows)
    random_accuracy = float(np.mean([np.mean([row.correct for row in candidates]) for candidates in grouped_test_rows.values()]))
    random_cost_usd = float(
        np.mean(
            [
                np.mean([_observed_rows_cost_usd([row], model_spec_by_label) for row in candidates])
                for candidates in grouped_test_rows.values()
            ]
        )
    )
    oracle_rows = [
        max(
            candidates,
            key=lambda row: utility(row.correct, _effective_call_cost(row, model_spec_by_label), cost_scale),
        )
        for candidates in grouped_test_rows.values()
    ]
    policy_metrics: list[dict[str, float | str]] = [
        {"label": "random", "accuracy": random_accuracy, "cost_usd": random_cost_usd},
        {
            "label": "single_shot_bayesian",
            "accuracy": bayesian_accuracy,
            "cost_usd": float(
                np.mean([_observed_rows_cost_usd([row], model_spec_by_label) for row in policy["selected_rows"]])
            ),
        },
        {
            "label": "adaptive_bayesian",
            "accuracy": adaptive_accuracy,
            "cost_usd": float(
                np.mean([_observed_rows_cost_usd(called, model_spec_by_label) for called in adaptive_policy.called_rows])
            ),
        },
        {
            "label": "oracle",
            "accuracy": float(np.mean([row.correct for row in oracle_rows])),
            "cost_usd": float(np.mean([_observed_rows_cost_usd([row], model_spec_by_label) for row in oracle_rows])),
        },
    ]
    for label in policy["always"]:
        model_rows = [row for row in test_rows if row.model_label == label]
        policy_metrics.append(
            {
                "label": f"always_{label}",
                "accuracy": float(np.mean([row.correct for row in model_rows])),
                "cost_usd": float(
                    np.mean([_observed_rows_cost_usd([row], model_spec_by_label) for row in model_rows])
                ),
            }
        )
    _plot_policy_values(policy_plot, policy_metrics)

    metrics: dict[str, Any] = {
        "dataset source": dataset_cfg.get("source", "fake"),
        "adaptive strategy": adaptive_strategy,
        "pricing catalog": pricing_snapshot.catalog_path or "not configured",
        "pricing catalog version": pricing_snapshot.version or "not configured",
        "pricing catalog as of": pricing_snapshot.as_of or "not configured",
        "pricing currency": pricing_snapshot.currency,
        "pricing snapshot": json.dumps(pricing_snapshot.prices_per_1m, sort_keys=True),
        "questions": len(questions),
        "models": len(model_specs),
        "call matrix rows": len(rows),
        "exploration questions": len(exploration_ids),
        "test questions": len(test_ids),
        "random baseline utility": round(random_utility, 5),
        "single-shot bayesian utility": round(bayesian_utility, 5),
        "adaptive bayesian utility": round(adaptive_utility, 5),
        "adaptive utility improvement over random": round(utility_improvement, 5),
        "adaptive utility improvement over single-shot bayesian": round(single_shot_utility_improvement, 5),
        "bootstrap utility diff ci low": round(float(ci_low), 5),
        "bootstrap utility diff ci high": round(float(ci_high), 5),
        "oracle utility": round(oracle_utility, 5),
        "single-shot bayesian selected accuracy": round(bayesian_accuracy, 5),
        "adaptive selected accuracy": round(adaptive_accuracy, 5),
        "adjudicated answer accuracy": round(adaptive_accuracy, 5),
        "single-shot bayesian average cost": round(bayesian_cost, 5),
        "average calls per question": round(average_call_count, 5),
        "stop after one call rate": round(float(np.mean(adaptive_policy.stop_after_one)), 5),
        "average cumulative cost": round(float(np.mean(adaptive_policy.cumulative_costs)), 5),
        "myopic voi accepted call rate": round(accepted_voi_rate, 5),
        "second call accepted rate": round(accepted_voi_rate, 5),
        "adjudication disagreement rate": round(disagreement_rate, 5),
        "adjudication changed answer rate": round(changed_answer_rate, 5),
        "invalid first answer call rate": round(invalid_first_answer_rate, 5),
        "called answer diversity": None if np.isnan(answer_diversity) else round(answer_diversity, 5),
        "test latency p50 ms": round(float(np.quantile(latency_values, 0.5)), 2),
        "test latency p95 ms": round(float(np.quantile(latency_values, 0.95)), 2),
        "posterior brier score": round(posterior_metrics["brier_score"], 5),
        "posterior log score": round(posterior_metrics["log_score"], 5),
        "posterior ece": round(posterior_metrics["ece"], 5),
        "posterior auroc": round(confidence_metrics["auroc"], 5),
        "self-confidence delta brier": round(confidence_delta_brier, 5),
        "self-confidence delta log score": round(confidence_delta_log, 5),
        "self-confidence delta auroc": round(confidence_delta_auroc, 5),
        "self-confidence improves correctness prediction": "yes" if confidence_improves else "no",
        "pre-call loo elpd": round(pre_loo_elpd, 5),
        "post-call loo elpd": round(post_loo_elpd, 5),
        "dependence caution triggered": "yes" if adaptive_policy.dependence_caution_triggered else "no",
        "json mode requested call rate": round(json_mode_requested_rate, 5),
        "json mode used call rate": round(json_mode_used_rate, 5),
        "robustness gate utility beats random": "pass" if policy_beats_random else "review",
        "robustness gate bootstrap directional": "pass" if ci_directional else "review",
        "robustness gate policy beats best always": "pass" if beats_best_always else "review",
    }
    metrics.update(operational_metrics)
    for item in policy_metrics:
        label = str(item["label"])
        cost_usd = float(item["cost_usd"])
        metrics[f"{label} observed average cost usd"] = None if np.isnan(cost_usd) else round(cost_usd, 8)
    for label, values in policy["always"].items():
        metrics[f"always_{label} utility"] = round(float(np.mean(values)), 5)

    if dataset_cfg.get("source", "fake") == "fake":
        recommendation = (
            "Treat this as an implementation smoke test only. The fake-provider run exercises posterior expected utility, "
            "adaptive routing, diagnostics, and reporting, but it is not evidence of routing lift."
        )
    elif utility_improvement > 0.0 and ci_directional and single_shot_utility_improvement > 0.0 and beats_best_always:
        recommendation = (
            "Use the Bayesian orchestrator as a candidate policy for a larger live Token Factory run. "
            f"This run improved held-out utility over random routing by {utility_improvement:.4f} and cleared the utility gates."
        )
    else:
        recommendation = (
            "Do not claim adaptive-routing lift from this run. Use the report to debug costs, VoI thresholds, model set, and sample size, "
            "then rerun on a larger live Token Factory evaluation."
        )

    if adaptive_strategy == "adjudication":
        adaptive_description = (
            "A simulated adaptive policy chooses the first call by posterior expected utility, optionally makes a second call "
            "when uncertainty or invalid output justifies the cost, and adjudicates called answers by posterior reliability."
        )
        adaptive_diagnostic = (
            "Selected the first held-out call by posterior expected utility, optionally selected a second call by expected correction value, "
            "and adjudicated called answers by posterior answer scores."
        )
    else:
        adaptive_description = (
            "A simulated adaptive policy chooses calls by posterior expected utility and cost-sensitive myopic decision value-of-information."
        )
        adaptive_diagnostic = "Selected the first held-out call by posterior expected utility and additional calls by myopic decision value-of-information."

    return write_report(
        output_dir=out_dir,
        title="MMLU Bayesian Orchestrator Evaluation",
        context=(
            "Small multiple-choice MMLU routing demo. LangGraph executes and logs model calls for every "
            "question-model pair for offline evaluation, then Bayesian reliability models are fit only on "
            f"exploration rows. {adaptive_description} The policy only sees held-out calls it has chosen."
        ),
        diagnostics=[
            "Loaded MMLU-shaped questions from either Hugging Face or the deterministic fake smoke dataset.",
            "Called every configured model on every selected question through a LangGraph execution graph.",
            "Validated missing-call model IDs against the provider model endpoint and resolved token prices from the configured versioned catalog when enabled.",
            "Split by question id into exploration and test partitions to avoid leakage.",
            "Fit separate pre-call and post-call hierarchical Bayesian reliability models over model, subject, length, cost, confidence, latency, and token features.",
            adaptive_diagnostic,
            "Reported proper-score, LOO ELPD, self-confidence, answer-diversity, and dependence diagnostics without using them as policy objectives.",
            "Compared adaptive Bayesian routing against exact random routing, single-shot Bayesian routing, always-use-model baselines, and an oracle upper bound.",
        ],
        metrics=metrics,
        plots=[
            {"label": "MMLU accuracy and observed USD cost comparison", "path": policy_plot.name},
        ],
        warnings=warnings,
        recommendation=recommendation,
        metadata={
            **build_metadata(config, seed),
            "pricing_snapshot": json.dumps(asdict(pricing_snapshot), sort_keys=True),
        },
    )
