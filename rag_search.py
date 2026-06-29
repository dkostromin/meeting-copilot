"""RAG поиск по GBrain + Obsidian vault."""

import json
import urllib.parse
from typing import Optional
import requests

class RagSearch:
    """Поиск релевантного контекста по базам знаний."""

    def __init__(self):
        import os
        home = os.path.expanduser("~")
        # macOS / Linux автодетект
        self.gbrain_url = os.environ.get("GBRAIN_API_URL", "http://localhost:1865")
        self.obsidian_path = os.environ.get(
            "OBSIDIAN_VAULT_PATH",
            f"{home}/obsidian" if os.path.exists(f"{home}/obsidian") else None
        )
        self.gbrain_enabled = self._check_gbrain()
        self.obsidian_enabled = self.obsidian_path is not None

    def _check_gbrain(self) -> bool:
        """Проверить доступность GBrain API."""
        try:
            resp = requests.get(f"{self.gbrain_url}/health", timeout=3)
            return resp.ok
        except requests.exceptions.RequestException:
            return False

    def search_gbrain(self, query: str, limit: int = 5) -> Optional[list[dict]]:
        """Поиск по GBrain через HTTP API."""
        if not self.gbrain_enabled:
            return None
        try:
            params = {
                "q": query,
                "limit": min(limit, 10),
                "detail": "low",
            }
            # GBrain HTTP query endpoint
            resp = requests.get(
                f"{self.gbrain_url}/api/query",
                params=params,
                timeout=10,
            )
            if resp.ok:
                data = resp.json()
                results = data.get("results", [])
                if results:
                    return results
        except requests.exceptions.RequestException:
            pass
        except (json.JSONDecodeError, KeyError):
            pass
        return None

    def search_obsidian(self, query: str, limit: int = 5) -> Optional[list[str]]:
        """Поиск по Obsidian vault через grep."""
        if not self.obsidian_enabled or not self.obsidian_path:
            return None
        import subprocess
        try:
            result = subprocess.run(
                ["grep", "-ril", query, self.obsidian_path, "--include=*.md"],
                capture_output=True, text=True, timeout=5,
                env={"PATH": "/usr/bin:/bin"},
            )
            if result.returncode == 0:
                files = result.stdout.strip().split("\n")[:limit]
                excerpts = []
                for f in files:
                    with open(f, "r", errors="ignore") as fh:
                        lines = fh.readlines()[:30]
                        excerpts.append(f"--- {f} ---\n" + "".join(lines))
                return excerpts
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None

    def search(self, query: str, limit: int = 5) -> str:
        """
        Поиск по всем источникам. Возвращает текстовый контекст.
        """
        context_parts = []

        # 1. GBrain
        gbrain_results = self.search_gbrain(query, limit)
        if gbrain_results:
            texts = []
            for r in gbrain_results:
                title = r.get("title", "")
                chunk = r.get("chunk_text", "")
                if chunk:
                    texts.append(f"## {title}\n{chunk[:500]}")
            if texts:
                context_parts.append("=== GBrain ===")
                context_parts.extend(texts)

        # 2. Obsidian
        obsidian_results = self.search_obsidian(query, limit)
        if obsidian_results:
            context_parts.append("=== Obsidian ===")
            context_parts.extend(obsidian_results)

        return "\n\n".join(context_parts) if context_parts else ""
