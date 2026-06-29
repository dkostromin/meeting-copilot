"""OpenCode / OpenAI-compatible API клиент."""

import json
import time
from typing import Optional
import requests
from config import config

class LlmClient:
    """Клиент для OpenCode (DeepSeek) / любого OpenAI-compatible API."""

    def __init__(self):
        cfg = config.llm
        self.api_key = cfg.api_key
        self.base_url = cfg.base_url.rstrip("/")
        self.model = cfg.model
        self.max_tokens = cfg.max_tokens
        self.temperature = cfg.temperature
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
        rag_context: Optional[str] = None,
        system_override: Optional[str] = None,
    ) -> str:
        """
        Отправить запрос в LLM.

        Args:
            user_message: Текущий вопрос/запрос
            context: История последних фраз [{"role": "user"/"assistant", "content": "..."}]
            rag_context: Результаты RAG поиска
            system_override: Переопределить system prompt
        """
        messages = [{"role": "system", "content": system_override or self.system_prompt}]

        # RAG контекст (документация)
        if rag_context:
            messages.append({
                "role": "system",
                "content": f"Вот релевантная документация из базы знаний:\n\n{rag_context[:4000]}"
            })

        # Контекст разговора
        if context:
            # Берем последние N сообщений
            max_context = 20
            recent = context[-max_context:]
            messages.extend(recent)

        # Текущий запрос
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
            resp = self._session.post(self._endpoint, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            elapsed = time.time() - t0

            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            if config.debug:
                print(f"[LLM] {elapsed:.1f}s | in={usage.get('prompt_tokens',0)} out={usage.get('completion_tokens',0)}")

            return content.strip()

        except requests.exceptions.RequestException as e:
            return f"[Ошибка API] {e}"
        except (KeyError, json.JSONDecodeError) as e:
            return f"[Ошибка парсинга ответа] {e}"

    def ask_stream(
        self,
        user_message: str,
        context: Optional[list[dict]] = None,
        rag_context: Optional[str] = None,
    ):
        """Streaming версия — для посимвольного вывода."""
        messages = [{"role": "system", "content": self.system_prompt}]
        if rag_context:
            messages.append({"role": "system", "content": f"Контекст из базы:\n{rag_context[:3000]}"})
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
            resp = self._session.post(self._endpoint, json=payload, stream=True, timeout=60)
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    line = line.decode().strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        data = json.loads(line[6:])
                        delta = data["choices"][0].get("delta", {}).get("content", "")
                        if delta:
                            yield delta
        except Exception as e:
            yield f"[Ошибка] {e}"
