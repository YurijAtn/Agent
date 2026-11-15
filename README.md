# Agent

Ядро агента: `agent.py` + `config.json`. Инструменты — только чтение, граница — workspace.

## Быстрый старт

1. Запустите llama-server (OpenAI-совместимый API), например:
   `llama-server -m модель.gguf --port 8080`
   Имя модели возьмите из `curl http://127.0.0.1:8080/v1/models`.
2. Отредактируйте `config.json`:
   - `llama_url` — `http://127.0.0.1:8080/v1/chat/completions`
   - `model` — имя модели из `/v1/models`
   - `workspace` — путь к папке с кодом (граница)
   - `max_tool_rounds` — сколько раз агент может вызвать инструмент за запрос (напр. 5)
3. `python3 agent.py` — REPL (`/reset` — очистить историю, `/exit` — выход).
4. Тесты без LLM: `python3 test_agent.py`.

## Архитектура (соответствие п.3)

- **3.1 Конфиг** — `load_config()` читает и валидирует `config.json`.
- **3.2 Инструменты (только чтение)** — `read_file(path)`, `grep(pattern)`, `list_files()`;
  схемы для API — в `TOOL_SCHEMAS`.
- **3.3 Граница** — `safe_path()`: `os.path.abspath(os.path.join(workspace, path))`
  обязан начинаться с `workspace`, иначе — отказ (не «спросить»); учтена ловушка
  префикса (`/workspace-evil`).
- **3.4 Цикл** — `Agent.chat()`: сообщение → история → llama-server (`tools`,
  `tool_choice:"auto"`) → при `tool_calls` выполнить инструмент, добавить результат,
  повторить (до `max_tool_rounds`) → при тексте показать и выйти.
- **3.5 Память сессии** — `Agent.history` живёт в памяти процесса; перезапуск сбрасывает.
  Долгая память — не в этой версии.