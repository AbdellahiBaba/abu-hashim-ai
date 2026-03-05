import json
import os
import time
from typing import Any, Callable, Dict, List, Optional

from evaluation.metrics import ArabicFluencyMetric, QualityMetric, StyleConsistencyMetric
from evaluation.benchmarks import BenchmarkRunner
from evaluation.report_generator import ReportGenerator


class EvaluationRunner:
    def __init__(
        self,
        generate_fn: Optional[Callable[[str], str]] = None,
        output_dir: str = "evaluation/results",
    ):
        self.generate_fn = generate_fn
        self.output_dir = output_dir
        self.fluency_metric = ArabicFluencyMetric()
        self.quality_metric = QualityMetric()
        self.style_metric = StyleConsistencyMetric()
        self.benchmark_runner = BenchmarkRunner(generate_fn)
        self.report_generator = ReportGenerator()

    def set_generate_fn(self, fn: Callable[[str], str]):
        self.generate_fn = fn
        self.benchmark_runner.set_generate_fn(fn)

    def evaluate_single(
        self, generated: str, reference: Optional[str] = None
    ) -> Dict[str, Any]:
        fluency = self.fluency_metric.compute(generated)
        quality = self.quality_metric.compute(generated, reference)
        return {
            "fluency": fluency,
            "quality": quality,
        }

    def evaluate_batch(
        self, pairs: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        results = []
        all_generated = []

        for pair in pairs:
            generated = pair.get("generated", "")
            reference = pair.get("reference")
            result = self.evaluate_single(generated, reference)
            result["prompt"] = pair.get("prompt", "")
            result["generated"] = generated
            if reference:
                result["reference"] = reference
            results.append(result)
            all_generated.append(generated)

        style = self.style_metric.compute(all_generated)

        scored = results
        avg_fluency = (
            sum(r["fluency"]["fluency_score"] for r in scored) / len(scored) if scored else 0.0
        )
        avg_quality = (
            sum(r["quality"]["quality_score"] for r in scored) / len(scored) if scored else 0.0
        )

        return {
            "summary": {
                "total_samples": len(pairs),
                "avg_fluency_score": round(avg_fluency, 4),
                "avg_quality_score": round(avg_quality, 4),
                "style_consistency": style,
            },
            "results": results,
        }

    def run_quick_evaluation(
        self,
        generate_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        if generate_fn:
            self.set_generate_fn(generate_fn)

        test_prompts = [
            "ما هو الذكاء الاصطناعي؟",
            "اكتب مقدمة مقال عن التعليم.",
            "ما هي أهمية اللغة العربية؟",
        ]

        results = []
        for prompt in test_prompts:
            try:
                if self.generate_fn:
                    output = self.generate_fn(prompt=prompt)
                    text = output.get("generated_text", "") if isinstance(output, dict) else str(output)
                else:
                    text = f"[Demo] Response to: {prompt}"

                eval_result = self.evaluate_single(text)
                eval_result["prompt"] = prompt
                eval_result["generated"] = text
                results.append(eval_result)
            except Exception as e:
                results.append({"prompt": prompt, "error": str(e)})

        scored = [r for r in results if "fluency" in r]
        avg_fluency = sum(r["fluency"]["fluency_score"] for r in scored) / len(scored) if scored else 0.0

        return {
            "summary": {
                "total_tests": len(test_prompts),
                "completed": len(scored),
                "avg_fluency": round(avg_fluency, 4),
            },
            "results": results,
        }

    def run_full_evaluation(
        self,
        prompts: Optional[List[Dict[str, str]]] = None,
        generate_report: bool = True,
    ) -> Dict[str, Any]:
        if self.generate_fn is None:
            raise RuntimeError("No generate function set. Call set_generate_fn() first.")

        benchmark_results = self.benchmark_runner.run_benchmark(prompts)

        os.makedirs(self.output_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(self.output_dir, f"eval_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_results, f, ensure_ascii=False, indent=2)

        report_path = None
        if generate_report:
            report_path = os.path.join(self.output_dir, f"report_{timestamp}.html")
            self.report_generator.generate(benchmark_results, report_path)

        return {
            "benchmark": benchmark_results,
            "json_output": json_path,
            "report_output": report_path,
        }
