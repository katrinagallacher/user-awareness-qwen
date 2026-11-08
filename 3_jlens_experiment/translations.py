"""English translations for the non-English tokens that show up in Qwen's J-space.

Qwen often "thinks" in Chinese, so many of the concepts we track are Chinese words.
Every non-English token on a figure or in a table must carry a translation, so
3_analyze.py looks tokens up here. If a re-run discovers a new non-English token, the
analysis stops and asks you to add it below.
"""

TRANSLATIONS = {
    # --- the user / the question ---
    "用户": "user",
    "用戶": "user, traditional script",
    "用户的": "the user's",
    "的用户": "the user, after a modifier",
    "和用户": "and the user",
    "为用户": "for the user",
    "用户在": "the user is / the user at",
    "用户提供": "provided by the user",
    "ユーザー": "user, Japanese",
    "ผู้ใช้": "user, Thai",
    "пользователь": "user, Russian",
    "客户": "client",
    "患者": "patient",
    "提问": "to ask a question",
    "提问者": "questioner",
    "询问": "to inquire",
    "问": "to ask",
    "问我": "ask me",
    "问题": "question / problem",
    "问题分析": "problem analysis",
    "问答": "Q&A",
    "请求": "request",
    "質問": "question, Japanese",
    "질문": "question, Korean",
    "pertanyaan": "question, Indonesian",
    "pergunta": "question, Portuguese",
    "Anfrage": "request, German",
    # --- answering ---
    "答": "answer",
    "答案": "answer",
    "回答": "answer / reply",
    "的回答": "the answer, after a modifier",
    "回应": "response",
    # --- discussion framing ---
    "讨论": "discussion",
    "探讨": "to explore / discuss",
    "对话": "dialogue",
    # --- ethics, law, safety ---
    "伦理": "ethics",
    "倫": "ethics, first character of 倫理",
    "ética": "ethics, Spanish / Portuguese",
    "道德": "morality",
    "的道德": "the morality, after a modifier",
    "违法": "illegal",
    "安全": "safety / security",
    "安全问题": "safety issue",
    "隐私": "privacy",
    "敏感": "sensitive",
    "警示": "warning",
    "质疑": "to question / challenge",
    "建议": "advice / suggestion",
    "不建议": "not advised",
    # --- deliberation ---
    "思考": "to think",
    "分析": "analysis",
    "分析问题": "to analyse the problem",
    "什么": "what",
    "怎么": "how",
    # --- interjections ---
    "嗯": "hmm",
    "啊": "ah",
    "哦": "oh",
    "噢": "oh",
    "哇": "wow",
    # --- function words ---
    "我": "I / me",
    "的": "possessive particle: 's / of",
    "一": "one / a",
    "在": "at / in; is doing",
    "多": "many / much",
    # --- other ---
    "邮件": "email",
    "邮箱": "email address / mailbox",
    "人工智能": "artificial intelligence",
    "机器人": "robot",
    "墨西哥": "Mexico",
    "西班牙语": "Spanish language",
    "非遗": "intangible cultural heritage",
}


def translation(token):
    """The English translation of a token, or "" if it is English (or not in the list)."""
    return TRANSLATIONS.get(token.strip(), "")


def needs_translation(token):
    """True for tokens with non-ASCII letters (Chinese, Japanese, Cyrillic, ...)."""
    return not token.strip().isascii()


def label(token):
    """Panel title: '"违法"  (illegal)' for non-English tokens, '"moral"' for English ones."""
    token = token.strip()
    if needs_translation(token) and not translation(token):
        raise KeyError(f'No English translation for "{token}". '
                       f"Please add it to TRANSLATIONS in 3_jlens_experiment/translations.py.")
    if translation(token):
        return f'"{token}"  ({translation(token)})'
    return f'"{token}"'
