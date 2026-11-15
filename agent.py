#!/usr/bin/env python3
"""
Ядро агента (agent.py).

Архитектура:
  3.1 Конфиг        — читается из config.json (llama_url, model, workspace, max_tool_rounds)
  3.2 Инструменты   — только чтение: read_file(path), grep(pattern), list_files()
  3.3 Граница       — safe_path(): любой путь проверяется на принадлежность workspace,
                      иначе — отказ (не «спросить»)
  3.4 Цикл          — сообщение -> история -> llama-server (tools, tool_choice:"auto")
                      -> tool_calls: выполнить, добавить результат, повторить (до max_tool_rounds)
                      -> текст: показать пользователю, выйти из цикла
  3.5 Память сессии — история сообщений в памяти процесса; при перезапуске сбрасывается.
                      Долгая память — не в этой версии.

Запуск:  python3 agent.py            (интерактивный REPL)
Тесты:   python3 test_agent.py       (unit-тесты без реальной LLM)
"""

import json
import os
import sys
import urllib.request
import urllib.error

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


# ----------------------------------------------------------------------------
# 3.1 Конфиг
# ----------------------------------------------------------------------------

def load_config(path=CONFIG_PATH):
    """Читает config.json и возвращает словарь настроек с проверкой полей."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    for key in ("llama_url", "model", "workspace", "max_tool_rounds"):
        if key not in cfg:
            raise ValueError(f"config.json: отсутствует обязательное поле '{key}'")

    cfg["workspace"] = os.path.abspath(cfg["workspace"])
    if not os.path.isdir(cfg["workspace"]):
        raise ValueError(f"config.json: workspace не существует: {cfg['workspace']}")

    cfg["max_tool_rounds"] = int(cfg["max_tool_rounds"])
    if cfg["max_tool_rounds"] < 1:
        raise ValueError("config.json: max_tool_rounds должен быть >= 1")
    return cfg


# ----------------------------------------------------------------------------
# 3.3 Граница (safe_path)
# ----------------------------------------------------------------------------

def safe_path(workspace, path):
    """
    Проверяет, что `path` (относительный или абсолютный) находится внутри `workspace`.

    os.path.abspath(os.path.join(workspace, path)) должен начинаться с workspace.
    Иначе —raise ValueError (отказ, не «спросить»).
    Возвращает нормализованный абсолютный путь.
    """
    workspace = os.path.abspath(workspace)
    candidate = os.path.abspath(os.path.join(workspace, path))
    # граница: сам workspace или путь внутри него (с разделителем, чтобы
    # /workspace-evil не проходил как префикс /workspace)
    if candidate == workspace or candidate.startswith(workspace + os.sep):
        return candidate
    raise ValueError(f"Отказано: путь вне workspace: {path!r}")


# ----------------------------------------------------------------------------
# 3.2 Инструменты (только чтение)
# ----------------------------------------------------------------------------

MAX_OUTPUT_BYTES = 60_000  # защита от гигантских файлов в контексте


def _clip(text):
    if len(text.encode("utf-8")) <= MAX_OUTPUT_BYTES:
        return text
    return text[:MAX_OUTPUT_BYTES] + "\n...[обрезано: слишком длинный вывод]"


def tool_read_file(workspace, path):
    """Прочитать файл, только внутри workspace."""
    full = safe_path(workspace, path)
    if not os.path.isfile(full):
        return f"ОШИБКА: файл не найден: {path}"
    try:
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            return _clip(f.read())
    except OSError as e:
        return f"ОШИБКА чтения: {e}"


def _iter_files(workspace):
    """Обход всех файлов workspace (пропускаем бинарные и скрытые каталоги типа .git)."""
    for root, dirs, files in os.walk(workspace):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "node_modules")]
        for name in files:
            yield os.path.join(root, name)


def tool_grep(workspace, pattern):
    """Найти строки по паттерну (подстрока, регистронезависимо) во всех файлах workspace."""
    matches = []
    for fpath in _iter_files(workspace):
        rel = os.path.relpath(fpath, workspace)
        try:
            with open(fpath, "r", encoding="utf-8", errors="strict") as f:
                for lineno, line in enumerate(f, 1):
                    if pattern.lower() in line.lower():
                        matches.append(f"{rel}:{lineno}: {line.rstrip()}")
        except (UnicodeDecodeError, OSError):
            continue  # бинарные/нечитаемые пропускаем
    if not matches:
        return f"Совпадений нет: {pattern!r}"
    return _clip("\n".join(matches))


def tool_list_files(workspace):
    """Список файлов в workspace (относительные пути)."""
    files = sorted(
        os.path.relpath(p, workspace) for p in _iter_files(workspace)
    )
    if not files:
        return "Workspace пуст."
    return _clip("\n".join(files))


# JSON Schema инструментов для OpenAI-совместимого API llama.cpp (llama-server)

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Прочитать содержимое файла. Путь должен быть внутри workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Путь к файлу (относительно workspace)"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Искать подстроку во всех текстовых файлах workspace. "
                           "Возвращает 'файл:строка: содержимое'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Искомая подстрока"}
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Список всех файлов в workspace.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

SYSTEM_PROMPT = (
    "Ты — агент-ассистент по коду. Тебе доступен только workspace (каталог проекта).\n"
    "У тебя есть инструменты ТОЛЬКО ДЛЯ ЧТЕНИЯ: read_file, grep, list_files.\n"
    "Ты не можешь изменять файлы. Отвечай по-русски. Если нужен факт о коде — сначала "
    "используй инструмент, потом отвечай."
)


# ----------------------------------------------------------------------------
# 3.5 Память сессии + 3.4 Цикл
# ----------------------------------------------------------------------------

class Agent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.workspace = cfg["workspace"]
        # Память сессии: история в памяти процесса; при перезапуске сбрасывается.
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    # --- вызов llama-server -------------------------------------------------
    def _call_llm(self):
        payload = {
            "model": self.cfg["model"],
            "messages": self.history,
            "tools": TOOL_SCHEMAS,
            "tool_choice": "auto",
            "temperature": 0.2,
            "stream": False,
        }
        req = urllib.request.Request(
            self.cfg["llama_url"],
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]

    # --- выполнение инструмента ---------------------------------------------
    def _execute_tool(self, name, args):
        try:
            if name == "read_file":
                return tool_read_file(self.workspace, args.get("path", ""))
            if name == "grep":
                return tool_grep(self.workspace, args.get("pattern", ""))
            if name == "list_files":
                return tool_list_files(self.workspace)
            return f"ОШИБКА: неизвестный инструмент: {name}"
        except ValueError as e:
            # safe_path поднял отказ — это штатный ответ модели, не падение
            return f"ОТКАЗ: {e}"

    # --- основной цикл --------------------------------------------------------
    def chat(self, user_message, echo=True):
        """
        1. Принять сообщение пользователя.
        2. Добавить в историю.
        3. Отправить в llama-server с tools и tool_choice:"auto".
        4. Если tool_calls — выполнить, добавить результат, повторить (до max_tool_rounds).
        5. Если текст — показать пользователю, выйти из цикла.
        """
        self.history.append({"role": "user", "content": user_message})

        for round_no in range(1, self.cfg["max_tool_rounds"] + 1):
            msg = self._call_llm()

            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                text = msg.get("content") or "(пустой ответ)"
                self.history.append({"role": "assistant", "content": text})
                if echo:
                    print(f"\n{text}\n")
                return text

            # записываем assistant-сообщение с вызовами инструментов
            self.history.append(
                {
                    "role": "assistant",
                    "content": msg.get("content"),
                    "tool_calls": tool_calls,
                }
            )

            for tc in tool_calls:
                fn = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                if echo:
                    print(f"  ⚙ {fn}({args})")
                result = self._execute_tool(fn, args)
                self.history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", fn),
                        "content": result,
                    }
                )

        # исчерпали лимит раундов
        note = (
            f"Достигнут лимит обращений к инструментам "
            f"(max_tool_rounds={self.cfg['max_tool_rounds']}). Уточните запрос."
        )
        self.history.append({"role": "assistant", "content": note})
        if echo:
            print(f"\n{note}\n")
        return note


# ----------------------------------------------------------------------------
# REPL
# ----------------------------------------------------------------------------

def main():
    try:
        cfg = load_config()
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"Не удалось загрузить config.json: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Модель : {cfg['model']}")
    print(f"Сервер : {cfg['llama_url']}")
    print(f"Граница: {cfg['workspace']} (только чтение)")
    print("Команды: /reset — очистить историю, /exit — выход\n")

    agent = Agent(cfg)
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/exit", "/quit"):
            break
        if user == "/reset":
            agent.history = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("(история очищена)")
            continue
        try:
            agent.chat(user)
        except urllib.error.URLError as e:
            print(f"Сервер недоступен ({cfg['llama_url']}): {e}\n"
                  f"Проверьте, запущен ли llama-server.", file=sys.stderr)
        except Exception as e:
            print(f"Ошибка: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
