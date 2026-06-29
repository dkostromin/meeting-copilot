"""OpenAI-compatible API клиент."""

import json
import time
from typing import Optional
import requests
from config import config


class LlmClient:
    """Клиент для любого OpenAI-compatible API."""

    def __init__(self):
        cfg = config.llm
        self.api_key = cfg.api_key
        self.base_url = cfg.base_url.rstrip("/")
        self.model = cfg.model
        self.max_tokens = cfg.max_tokens
        self.temperature = cfg.temperature
        self.timeout = cfg.timeout
        self.system_prompt = cfg.system_prompt
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    @property
    def _endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def ask(
        self,
        user_message: str,
        context: Optional[list[dict]] = None,
        system_override: Optional[str] = None,
    ) -> str:
        """Отправить запрос в LLM."""
        messages = [{"role": "system", "content": system_override or self.system_prompt}]

        if context:
            recent = context[-20:]
            messages.extend(recent)

        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }

        t0 = time.time()
        try:
            resp = self._session.post(self._endpoint, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            elapsed = time.time() - t0

            choice = data["choices"][0]
            msg = choice.get("message", {})

            # Пробуем разные поля (DeepSeek кладёт ответ в reasoning_content)
            content = msg.get("content", "")
            if not content:
                content = msg.get("reasoning_content", "") or msg.get("reasoning", "") or msg.get("text", "")

            usage = data.get("usage", {})
            if config.debug:
                print(f"[LLM] {elapsed:.1f}s | in={usage.get('prompt_tokens',0)} out={usage.get('completion_tokens',0)})")
                # Если ответ пустой — покажем весь ответ
                if not content or not content.strip():
                    print(f"[LLM] ⚠️ Пустой ответ. Сырой: {json.dumps(data, indent=2)[:500]}")
                    # Попробуем альтернативный путь — choice.finish_reason + model
                    finish = choice.get("finish_reason", "?")
                    print(f"[LLM] finish_reason={finish}, model={data.get('model','?')}")

            if not content or not content.strip():
                return "[LLM не дал ответа — попробуй другую модель]"
            return content.strip()

        except requests.exceptions.Timeout:
            return "[LLM превысила таймаут]"
        except requests.exceptions.ConnectionError:
            return "[LLM недоступна — проверь API ключ и URL]"
        except requests.exceptions.RequestException as e:
            return f"[Ошибка API] {e}"
        except (KeyError, json.JSONDecodeError) as e:
            return f"[Ошибка парсинга] {e}"

    def ask_stream(
        self,
        user_message: str,
        context: Optional[list[dict]] = None,
    ):
        """Streaming версия — для посимвольного вывода."""
        messages = [{"role": "system", "content": self.system_prompt}]
        if context:
            messages.extend(context[-20:])
        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }

        try:
            resp = self._session.post(self._endpoint, json=payload, stream=True, timeout=self.timeout)
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    line = line.decode().strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            data = json.loads(line[6:])
                            delta = data["choices"][0].get("delta", {}).get("content", "")
                            if delta:
                                yield delta
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
        except Exception as e:
            yield f"[Ошибка] {e}"
