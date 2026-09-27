# Synthetic report example

**All inputs are fictional.** `DEMO-ONLY` and ExampleCo do not identify a real
company. The narrative was manually written to illustrate the AI-result fields;
it was not produced by an AI call or supported by source research.

| File | Purpose |
| --- | --- |
| [top_10_report.md](top_10_report.md) | Readable report with one fictional candidate; the default filename is retained |
| [top_20.csv](top_20.csv) | One ranked row with flattened fields, computed scores and a `synthetic_demo` marker |
| [detailed_results.json](detailed_results.json) | Complete nested output, scoring explanations and synthetic-run metadata |
| [preview.svg](preview.svg) | Illustrated summary of this output, not a product UI screenshot |
| [input.json](input.json) | Invented financial metrics and manually simulated narrative inputs |

The generator calls the existing `score_candidate`, `combine_dislocation_score`
and `write_reports` functions. The values in the preview are derived from the
same record as the exported bundle. It does not run universe selection, provider
downloads, financial-statement extraction or AI research.

FCF growth is deliberately missing. It remains null in JSON, blank in the
corresponding CSV field and "not supplied" in the report. Weighted coverage and
the coverage adjustment are calculated by the real scoring implementation.

The report's standard "measured facts" and "AI inference" headings name output
sections. In this fixture both sections contain invented inputs. The narrative
gap and value-trap values are illustrative assumptions, not calibrated estimates.
There are no evidence links because no source research was performed. A top-level
warning is prepended to the Markdown to make this distinction visible on its own.

## Regenerate offline

After installing the project (`python -m pip install -e .`), run from its root:

```bash
python examples/generate_synthetic_report.py
```

No API key or network connection is required. The command overwrites the four
generated example files above. Numeric and narrative content is reproducible;
the reporting module inserts the current UTC generation timestamp.
