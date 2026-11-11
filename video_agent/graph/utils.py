"""图内通用小函数。"""

from __future__ import annotations

from video_agent.models import ChatTurn, CreativeParams, FrontendParams

RECENT_HISTORY_ROUNDS = 100
RECENT_HISTORY_MESSAGES = RECENT_HISTORY_ROUNDS * 2


def clean_model_text(value: object) -> str:
    """Treat literal null-like model text as empty."""

    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"null", "none", "nil"}:
        return ""
    return text


def history_to_text(history: list[ChatTurn], limit: int = RECENT_HISTORY_ROUNDS) -> str:
    """把最近对话转成提示词上下文。"""

    message_limit = limit * 2
    lines = []
    for turn in history[-message_limit:]:
        content = clean_model_text(turn.content)
        if content:
            lines.append(f"{turn.role}: {content}")
    return "\n".join(lines)


def format_final_params(
    params: CreativeParams,
    frontend_params: FrontendParams | None = None,
) -> str:
    """输出最终内容参数。"""

    content = params.model_dump_json(ensure_ascii=False, exclude_none=True, indent=2)
    if frontend_params is None:
        return f"内容已确认，准备合并前端 UI 参数后提交：\n{content}"

    ui_params = frontend_params.model_dump_json(
        ensure_ascii=False,
        exclude_none=True,
        indent=2,
    )
    return (
        "内容已确认，已合并前端 UI 参数提交：\n"
        f"内容参数：\n{content}\n"
        f"前端 UI 参数：\n{ui_params}"
    )
