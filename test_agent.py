#!/usr/bin/env python3
"""
Тесты ядра агента без реальной LLM: _call_llm подменяется заготовленными
ответами (tool_calls -> текст), проверяются граница, инструменты, цикл и память.

Запуск:  python3 test_agent.py
"""

import json
import os
import tempfile
import unittest

import agent


class TestSafePath(unittest.TestCase):
    def setUp(self):
        self.ws = os.path.abspath("/tmp/ws_test")

    def test_ok_relative(self):
        self.assertEqual(agent.safe_path(self.ws, "a/b.txt"),
                         os.path.join(self.ws, "a/b.txt"))

    def test_ok_dot(self):
        self.assertEqual(agent.safe_path(self.ws, "."), self.ws)

    def test_reject_traversal(self):
        with self.assertRaises(ValueError):
            agent.safe_path(self.ws, "../etc/passwd")

    def test_reject_absolute_outside(self):
        with self.assertRaises(ValueError):
            agent.safe_path(self.ws, "/etc/passwd")

    def test_reject_prefix_trap(self):
        # /tmp/ws_test_evil начинается со строки /tmp/ws_test, но это НЕ внутри
        with self.assertRaises(ValueError):
            agent.safe_path(self.ws, "/tmp/ws_test_evile/x")


class TestTools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        with open(os.path.join(self.tmp, "hello.txt"), "w") as f:
            f.write("line one\nmagic word here\n")
        os.makedirs(os.path.join(self.tmp, "sub"))
        with open(os.path.join(self.tmp, "sub", "code.py"), "w") as f:
            f.write("def magic():\n    return 42\n")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_read_file(self):
        out = agent.tool_read_file(self.tmp, "hello.txt")
        self.assertIn("magic word", out)

    def test_read_missing(self):
        self.assertIn("ОШИБКА", agent.tool_read_file(self.tmp, "nope.txt"))

    def test_read_escape_denied(self):
        # попытка выйти за границу — отказ (ValueError), без «спросить»
        with self.assertRaises(ValueError):
            agent.safe_path(self.tmp, "../../etc/passwd")
        # а через Agent-обёртку отказ превращается в штатный текст ОТКАЗ
        a = agent.Agent({"workspace": self.tmp})
        self.assertTrue(
            a._execute_tool("read_file", {"path": "/etc/passwd"}).startswith("ОТКАЗ")
        )

    def test_grep(self):
        out = agent.tool_grep(self.tmp, "MAGIC")  # регистронезависимо
        self.assertIn("hello.txt:2:", out)
        self.assertIn(os.path.join("sub", "code.py") + ":1:", out)

    def test_grep_no_match(self):
        self.assertIn("Совпадений нет", agent.tool_grep(self.tmp, "zzz_qq"))

    def test_list_files(self):
        out = agent.tool_list_files(self.tmp)
        self.assertIn("hello.txt", out)
        self.assertIn(os.path.join("sub", "code.py"), out)


class TestAgentLoop(unittest.TestCase):
    """Цикл 3.4 с мок-LLM: tool_calls -> инструмент -> результат -> текст."""

    def make_agent(self, replies, max_rounds=5):
        cfg = {
            "llama_url": "http://mock/v1/chat/completions",
            "model": "mock",
            "workspace": os.path.abspath(os.path.dirname(__file__)),
            "max_tool_rounds": max_rounds,
        }
        a = agent.Agent(cfg)
        a._replies = list(replies)
        a._call_llm = lambda: a._replies.pop(0)
        return a

    def test_tool_then_answer(self):
        replies = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "read_file",
                              "arguments": json.dumps({"path": "config.json"})}}
            ]},
            {"role": "assistant", "content": "Готово: конфиг прочитан."},
        ]
        a = self.make_agent(replies)
        out = a.chat("прочитай config.json", echo=False)
        self.assertEqual(out, "Готово: конфиг прочитан.")
        # история: system, user, assistant(tool_calls), tool, assistant(text)
        roles = [m["role"] for m in a.history]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "assistant"])
        self.assertIn("llama_url", a.history[3]["content"])

    def test_escape_attempt_denied_not_crash(self):
        replies = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "read_file",
                              "arguments": json.dumps({"path": "../../../etc/passwd"})}}
            ]},
            {"role": "assistant", "content": "Файл вне workspace — отказ."},
        ]
        a = self.make_agent(replies)
        a.chat("прочитай /etc/passwd", echo=False)
        tool_msg = [m for m in a.history if m["role"] == "tool"][0]
        self.assertTrue(tool_msg["content"].startswith("ОТКАЗ"))

    def test_max_tool_rounds_limit(self):
        # модель зациклилась на инструментах
        looping = {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c", "type": "function",
             "function": {"name": "list_files", "arguments": "{}"}}
        ]}
        a = self.make_agent([looping] * 3, max_rounds=3)
        out = a.chat("цикль", echo=False)
        self.assertIn("лимит", out)

    def test_session_memory_in_process(self):
        replies = [
            {"role": "assistant", "content": "Привет!"},
            {"role": "assistant", "content": "Помню."},
        ]
        a = self.make_agent(replies)
        a.chat("привет", echo=False)
        a.chat("помнишь?", echo=False)
        users = [m for m in a.history if m["role"] == "user"]
        self.assertEqual(len(users), 2)  # накопление в памяти процесса


if __name__ == "__main__":
    unittest.main(verbosity=2)
