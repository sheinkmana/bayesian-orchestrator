from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jinja2

from bayesian_orchestrator import __version__


REPORT_TEMPLATE = """# {{ title }}

## Model / Context
{{ context }}

## Diagnostics Run
{% for item in diagnostics -%}
- {{ item }}
{% endfor %}

## Results
{% for key, value in metrics.items() -%}
- **{{ key }}:** {{ value | report_value }}
{% endfor %}

## Plots
{% for plot in plots -%}
### {{ plot.label }}
![{{ plot.label }}]({{ plot.path }})

{% endfor %}

## Warnings
{% if warnings -%}
{% for warning in warnings -%}
- {{ warning }}
{% endfor %}
{% else -%}
- No blocking warnings.
{% endif %}

## Recommended Next Step
{{ recommendation }}

## Reproducibility Metadata
{% for key, value in metadata.items() -%}
- **{{ key }}:** {{ value }}
{% endfor %}
"""


def _report_value(value: Any) -> str:
    if value is None:
        return "not applicable"
    return str(value)


@dataclass(frozen=True)
class WorkflowResult:
    report_path: Path
    summary_path: Path


def build_metadata(config: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "workflow_version": __version__,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": config.get("_config_path", "unknown"),
        "seed": seed,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def write_report(
    output_dir: Path,
    title: str,
    context: str,
    diagnostics: list[str],
    metrics: dict[str, Any],
    plots: list[dict[str, str]],
    warnings: list[str],
    recommendation: str,
    metadata: dict[str, Any],
) -> WorkflowResult:
    env = jinja2.Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True)
    env.filters["report_value"] = _report_value
    report = env.from_string(REPORT_TEMPLATE).render(
        title=title,
        context=context,
        diagnostics=diagnostics,
        metrics=metrics,
        plots=plots,
        warnings=warnings,
        recommendation=recommendation,
        metadata=metadata,
    )

    report_path = output_dir / "report.md"
    summary_path = output_dir / "summary.json"
    report_path.write_text(report, encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "title": title,
                "context": context,
                "diagnostics": diagnostics,
                "metrics": metrics,
                "plots": plots,
                "warnings": warnings,
                "recommendation": recommendation,
                "metadata": metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return WorkflowResult(report_path=report_path, summary_path=summary_path)
