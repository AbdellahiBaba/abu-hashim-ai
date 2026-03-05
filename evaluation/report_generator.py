import html
import os
import time
from typing import Any, Dict


class ReportGenerator:
    DARK_NAVY = "#0A1A2F"
    GOLD = "#D4AF37"

    def generate(self, results: Dict[str, Any], output_path: str):
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        html_content = self._build_html(results)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    def _build_html(self, results: Dict[str, Any]) -> str:
        summary = results.get("summary", {})
        items = results.get("results", [])
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        rows = ""
        for item in items:
            fluency_score = item.get("fluency", {}).get("fluency_score", 0)
            quality_score = item.get("quality", {}).get("quality_score", 0)
            latency = item.get("latency_s", 0)
            error = item.get("error", "")

            status = f'<span style="color:#e74c3c">{html.escape(error)}</span>' if error else "✓"

            rows += f"""
            <tr>
                <td>{html.escape(str(item.get('id', '')))}</td>
                <td>{html.escape(str(item.get('category', '')))}</td>
                <td dir="rtl">{html.escape(str(item.get('prompt', '')))}</td>
                <td>{fluency_score:.4f}</td>
                <td>{quality_score:.4f}</td>
                <td>{latency:.3f}s</td>
                <td>{status}</td>
            </tr>"""

        style_info = summary.get("style_consistency", {})
        style_score = style_info.get("style_score", 0) if isinstance(style_info, dict) else 0

        return f"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QalamAI - Evaluation Report</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: {self.DARK_NAVY};
    color: #e0e0e0;
    padding: 2rem;
}}
.header {{
    text-align: center;
    margin-bottom: 2rem;
    padding: 2rem;
    border-bottom: 3px solid {self.GOLD};
}}
.header h1 {{
    color: {self.GOLD};
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}}
.header .subtitle {{
    color: #a0a0a0;
    font-size: 1rem;
}}
.summary-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
}}
.summary-card {{
    background: rgba(255,255,255,0.05);
    border: 1px solid {self.GOLD}33;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}}
.summary-card .value {{
    font-size: 2rem;
    font-weight: bold;
    color: {self.GOLD};
}}
.summary-card .label {{
    color: #a0a0a0;
    font-size: 0.9rem;
    margin-top: 0.5rem;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 1rem;
}}
th {{
    background: {self.GOLD};
    color: {self.DARK_NAVY};
    padding: 0.75rem;
    font-weight: 600;
    text-align: right;
}}
td {{
    padding: 0.75rem;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}}
tr:hover {{
    background: rgba(212,175,55,0.05);
}}
.section-title {{
    color: {self.GOLD};
    font-size: 1.5rem;
    margin: 2rem 0 1rem 0;
    border-right: 4px solid {self.GOLD};
    padding-right: 1rem;
}}
.footer {{
    text-align: center;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(212,175,55,0.3);
    color: #666;
    font-size: 0.85rem;
}}
</style>
</head>
<body>
<div class="header">
    <h1>قلم AI</h1>
    <div class="subtitle">QalamAI Evaluation Report — {timestamp}</div>
</div>

<div class="summary-grid">
    <div class="summary-card">
        <div class="value">{summary.get('total_prompts', 0)}</div>
        <div class="label">Total Prompts</div>
    </div>
    <div class="summary-card">
        <div class="value">{summary.get('successful', 0)}</div>
        <div class="label">Successful</div>
    </div>
    <div class="summary-card">
        <div class="value">{summary.get('avg_fluency_score', 0):.4f}</div>
        <div class="label">Avg Fluency</div>
    </div>
    <div class="summary-card">
        <div class="value">{summary.get('avg_quality_score', 0):.4f}</div>
        <div class="label">Avg Quality</div>
    </div>
    <div class="summary-card">
        <div class="value">{summary.get('avg_latency_s', 0):.3f}s</div>
        <div class="label">Avg Latency</div>
    </div>
    <div class="summary-card">
        <div class="value">{style_score:.4f}</div>
        <div class="label">Style Consistency</div>
    </div>
</div>

<h2 class="section-title">Detailed Results</h2>
<table>
<thead>
<tr>
    <th>ID</th>
    <th>Category</th>
    <th>Prompt</th>
    <th>Fluency</th>
    <th>Quality</th>
    <th>Latency</th>
    <th>Status</th>
</tr>
</thead>
<tbody>
{rows}
</tbody>
</table>

<div class="footer">
    QalamAI Evaluation Suite &mdash; Generated {timestamp}
</div>
</body>
</html>"""

    def generate_comparison_report(
        self, comparison: Dict[str, Any], output_path: str
    ):
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        model_rows = ""
        for model_name, data in comparison.items():
            summary = data.get("summary", {})
            model_rows += f"""
            <tr>
                <td style="font-weight:bold">{html.escape(model_name)}</td>
                <td>{summary.get('avg_fluency_score', 0):.4f}</td>
                <td>{summary.get('avg_quality_score', 0):.4f}</td>
                <td>{summary.get('avg_latency_s', 0):.3f}s</td>
                <td>{summary.get('successful', 0)}/{summary.get('total_prompts', 0)}</td>
            </tr>"""

        content = f"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<title>QalamAI - Model Comparison</title>
<style>
body {{
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: {self.DARK_NAVY};
    color: #e0e0e0;
    padding: 2rem;
}}
h1 {{ color: {self.GOLD}; text-align: center; margin-bottom: 2rem; }}
table {{ width: 100%; border-collapse: collapse; }}
th {{ background: {self.GOLD}; color: {self.DARK_NAVY}; padding: 0.75rem; text-align: right; }}
td {{ padding: 0.75rem; border-bottom: 1px solid rgba(255,255,255,0.1); }}
</style>
</head>
<body>
<h1>QalamAI Model Comparison — {timestamp}</h1>
<table>
<thead>
<tr><th>Model</th><th>Fluency</th><th>Quality</th><th>Latency</th><th>Success</th></tr>
</thead>
<tbody>{model_rows}</tbody>
</table>
</body>
</html>"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
