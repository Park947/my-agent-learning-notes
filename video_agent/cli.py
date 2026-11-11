"""命令行入口。"""

from __future__ import annotations

import sys

from video_agent.graph.builder import build_graph
from video_agent.graph.utils import (
    RECENT_HISTORY_MESSAGES,
    clean_model_text,
    format_final_params,
)
from video_agent.models import ChatTurn, CreativeParams, CreativeState, FrontendParams, ToolCallRequest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run_cli() -> None:
    """命令行多轮对话入口。"""

    app = build_graph()
    frontend_params = FrontendParams()
    state: CreativeState = {
        "user_input": "",
        "params": CreativeParams(),
        "frontend_params": frontend_params,
        "chat_history": [],
        "long_term_memory": "",
        "should_retrieve": False,
        "retrieval_query": "",
        "rag_context": "",
        "rag_sources": [],
        "resolved_image_url": "",
        "route": "chat",
        "reply": "",
        "is_ready": False,
        "awaiting_confirmation": False,
        "tool_call": ToolCallRequest(),
        "reflection_passed": True,
        "reflection_issues": [],
        "should_optimize_prompt": False,
    }

    print("视频创作助手已启动，输入 exit 或 quit 退出。")

    while not state["is_ready"]:
        user_input = input("User: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("已退出。")
            return
        if not user_input:
            print("Agent: 可以告诉我你想做什么类型的视频，或者问我能帮你做什么。")
            continue

        state = {
            **state,
            "user_input": user_input,
            "reply": "",
            "long_term_memory": "",
            "should_retrieve": False,
            "retrieval_query": "",
            "rag_context": "",
            "rag_sources": [],
            "resolved_image_url": "",
            "is_ready": False,
            "tool_call": ToolCallRequest(),
            "reflection_passed": True,
            "reflection_issues": [],
            "should_optimize_prompt": False,
            "chat_history": [
                turn
                for turn in state["chat_history"][-RECENT_HISTORY_MESSAGES:]
                if clean_model_text(turn.content)
            ],
        }
        try:
            state = app.invoke(
                state,
                config={
                    "run_name": "video_creation_agent",
                    "tags": ["cli", "langgraph", "litmedia"],
                    "metadata": {
                        "route": state.get("route"),
                        "awaiting_confirmation": state.get("awaiting_confirmation"),
                        "has_prompt": bool(state["params"].prompt),
                        "has_long_term_memory": bool(state.get("long_term_memory")),
                        "should_retrieve": state.get("should_retrieve"),
                        "retrieval_query": state.get("retrieval_query"),
                        "rag_sources": state.get("rag_sources", []),
                        "has_source_image": bool(state["params"].source_image_url),
                    },
                },
            )
        except Exception as exc:
            state = {
                **state,
                "reply": (
                    "网络请求超时或接口异常，本次没有成功提交任务："
                    f"{exc}\n请稍后再次表达提交意图重试，或继续修改需求。"
                ),
                "awaiting_confirmation": bool(state["params"].prompt),
                "is_ready": False,
                "tool_call": ToolCallRequest(),
            }

        reply = clean_model_text(state.get("reply"))
        if not reply:
            reply = "我已记录当前信息，但本轮没有拿到可展示回复。你可以继续补充需求。"
            state["reply"] = reply

        print(f"Agent: {reply}")
        state["chat_history"].extend(
            [
                ChatTurn(role="user", content=user_input),
                ChatTurn(role="assistant", content=reply),
            ]
        )
        state["chat_history"] = state["chat_history"][-RECENT_HISTORY_MESSAGES:]

    print(f"Agent: {format_final_params(state['params'], state['frontend_params'])}")
