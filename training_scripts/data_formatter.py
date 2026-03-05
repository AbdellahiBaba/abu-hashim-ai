import json
from typing import Optional
from pathlib import Path

from training_scripts.text_cleaner import TextCleaner
from training_scripts.pii_remover import PIIRemover


INSTRUCTION_TEMPLATE = (
    "<|system|>\n{system}\n"
    "<|user|>\n{input}\n"
    "<|assistant|>\n{output}"
)

DEFAULT_SYSTEM_PROMPT = (
    "أنت أبو هاشم، مساعد ذكي متخصص في الكتابة العربية والإبداعية. "
    "تتميز بأسلوب أدبي راقٍ ومعرفة واسعة."
)


class DataFormatter:
    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        template: str = INSTRUCTION_TEMPLATE,
        min_input_length: int = 10,
        min_output_length: int = 10,
        max_input_length: int = 4096,
        max_output_length: int = 8192,
        text_cleaner: Optional[TextCleaner] = None,
        pii_remover: Optional[PIIRemover] = None,
    ):
        self.system_prompt = system_prompt
        self.template = template
        self.min_input_length = min_input_length
        self.min_output_length = min_output_length
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.cleaner = text_cleaner or TextCleaner()
        self.pii_remover = pii_remover or PIIRemover()

    def format_pair(
        self,
        input_text: str,
        output_text: str,
        system_prompt: Optional[str] = None,
    ) -> Optional[dict]:
        input_text = self.cleaner.clean(input_text)
        output_text = self.cleaner.clean(output_text)

        input_text = self.pii_remover.remove_pii(input_text)
        output_text = self.pii_remover.remove_pii(output_text)

        if len(input_text) < self.min_input_length:
            return None
        if len(output_text) < self.min_output_length:
            return None

        input_text = input_text[: self.max_input_length]
        output_text = output_text[: self.max_output_length]

        sys_prompt = system_prompt or self.system_prompt

        formatted = self.template.format(
            system=sys_prompt,
            input=input_text,
            output=output_text,
        )

        return {
            "input": input_text,
            "output": output_text,
            "system": sys_prompt,
            "text": formatted,
        }

    def format_plain_text(
        self,
        text: str,
        chunk_size: int = 2048,
        overlap: int = 256,
    ) -> list:
        text = self.cleaner.clean(text)
        text = self.pii_remover.remove_pii(text)

        if len(text) < self.min_input_length:
            return []

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if len(chunk) >= self.min_input_length:
                chunks.append({
                    "text": chunk,
                    "type": "plain_text",
                })
            start += chunk_size - overlap

        return chunks

    def format_conversation(self, turns: list) -> list:
        formatted = []
        for i in range(0, len(turns) - 1, 2):
            if i + 1 < len(turns):
                user_msg = turns[i].get("content", turns[i].get("text", ""))
                assistant_msg = turns[i + 1].get(
                    "content", turns[i + 1].get("text", "")
                )
                result = self.format_pair(user_msg, assistant_msg)
                if result:
                    formatted.append(result)
        return formatted

    def format_batch(self, records: list) -> list:
        formatted = []
        for record in records:
            input_text = record.get("input", record.get("prompt", ""))
            output_text = record.get("output", record.get("response", record.get("completion", "")))
            system_prompt = record.get("system", record.get("system_prompt", None))

            result = self.format_pair(input_text, output_text, system_prompt)
            if result:
                formatted.append(result)

        return formatted

    def save_jsonl(self, records: list, output_path: str) -> int:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        return count
