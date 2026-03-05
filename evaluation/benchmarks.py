import json
import time
from typing import Any, Callable, Dict, List, Optional

from evaluation.metrics import ArabicFluencyMetric, QualityMetric, StyleConsistencyMetric


BUILTIN_PROMPTS = [
    {
        "id": "greeting",
        "prompt": "مرحباً، كيف يمكنني مساعدتك اليوم؟",
        "category": "conversation",
        "reference": "أهلاً وسهلاً! أنا هنا لمساعدتك في أي استفسار لديك.",
    },
    {
        "id": "fatwa_simple",
        "prompt": "ما حكم صلاة الجمعة؟",
        "category": "fiqh",
        "reference": "صلاة الجمعة فرض عين على كل مسلم بالغ عاقل ذكر مقيم.",
    },
    {
        "id": "quran_tafsir",
        "prompt": "اشرح معنى قوله تعالى: بسم الله الرحمن الرحيم",
        "category": "tafsir",
        "reference": "البسملة هي استفتاح بذكر اسم الله تعالى تبركاً واستعانة.",
    },
    {
        "id": "hadith_explanation",
        "prompt": "اشرح حديث إنما الأعمال بالنيات",
        "category": "hadith",
        "reference": "هذا الحديث أصل عظيم في الإسلام يدل على أن قبول الأعمال مرتبط بالنية.",
    },
    {
        "id": "aqeedah",
        "prompt": "ما هي أركان الإيمان؟",
        "category": "aqeedah",
        "reference": "أركان الإيمان ستة: الإيمان بالله وملائكته وكتبه ورسله واليوم الآخر والقدر خيره وشره.",
    },
    {
        "id": "general_knowledge",
        "prompt": "ما هي فوائد الصيام؟",
        "category": "general",
        "reference": "للصيام فوائد روحية وصحية واجتماعية عديدة.",
    },
]


class BenchmarkRunner:
    def __init__(self, generate_fn: Optional[Callable[[str], str]] = None):
        self.generate_fn = generate_fn
        self.fluency_metric = ArabicFluencyMetric()
        self.quality_metric = QualityMetric()
        self.style_metric = StyleConsistencyMetric()

    def set_generate_fn(self, fn: Callable[[str], str]):
        self.generate_fn = fn

    def run_benchmark(
        self,
        prompts: Optional[List[Dict[str, str]]] = None,
        categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if self.generate_fn is None:
            raise RuntimeError("No generate function set. Call set_generate_fn() first.")

        test_prompts = prompts or BUILTIN_PROMPTS
        if categories:
            test_prompts = [p for p in test_prompts if p.get("category") in categories]

        results: List[Dict[str, Any]] = []
        all_generated: List[str] = []
        total_time = 0.0

        for item in test_prompts:
            prompt = item["prompt"]
            reference = item.get("reference", "")

            start = time.time()
            try:
                generated = self.generate_fn(prompt)
            except Exception as e:
                generated = ""
                item_result = {
                    "id": item.get("id", ""),
                    "category": item.get("category", ""),
                    "prompt": prompt,
                    "generated": "",
                    "error": str(e),
                    "fluency": {},
                    "quality": {},
                    "latency_s": 0.0,
                }
                results.append(item_result)
                continue
            elapsed = time.time() - start
            total_time += elapsed

            fluency = self.fluency_metric.compute(generated)
            quality = self.quality_metric.compute(generated, reference)

            all_generated.append(generated)

            results.append({
                "id": item.get("id", ""),
                "category": item.get("category", ""),
                "prompt": prompt,
                "generated": generated,
                "reference": reference,
                "fluency": fluency,
                "quality": quality,
                "latency_s": round(elapsed, 4),
            })

        style = self.style_metric.compute(all_generated) if all_generated else {}

        scored = [r for r in results if "error" not in r]
        avg_fluency = (
            sum(r["fluency"]["fluency_score"] for r in scored) / len(scored) if scored else 0.0
        )
        avg_quality = (
            sum(r["quality"]["quality_score"] for r in scored) / len(scored) if scored else 0.0
        )
        avg_latency = total_time / len(test_prompts) if test_prompts else 0.0

        return {
            "summary": {
                "total_prompts": len(test_prompts),
                "successful": len(scored),
                "failed": len(results) - len(scored),
                "avg_fluency_score": round(avg_fluency, 4),
                "avg_quality_score": round(avg_quality, 4),
                "avg_latency_s": round(avg_latency, 4),
                "style_consistency": style,
            },
            "results": results,
        }

    def compare_models(
        self,
        model_fns: Dict[str, Callable[[str], str]],
        prompts: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        comparison: Dict[str, Any] = {}
        for model_name, fn in model_fns.items():
            self.set_generate_fn(fn)
            comparison[model_name] = self.run_benchmark(prompts)
        return comparison

    def export_results(self, results: Dict[str, Any], output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
