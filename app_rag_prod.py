# -*- coding: utf-8 -*-
import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


# =========================================================
# Config
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "out"

ENV_CANDIDATES = [
    BASE_DIR / ".env",
    BASE_DIR.parent / ".env",
    Path.cwd() / ".env",
]

ROUTER_MODEL = "gpt-4o-mini"
L1_MODEL = "gpt-4o-mini"
L2_MODEL = "gpt-4o-mini"
L4_MODEL = "gpt-4o-mini"
L5_MODEL = "gpt-4o-mini"
MARKETING_MODEL = "gpt-4o-mini"
STRATEGY_MODEL = "gpt-4o-mini"

EMB_MODEL = "intfloat/multilingual-e5-small"

TOP_K_RAG = 5

CN_MAP = {
    "bee": "蜜蜂",
    "tiger": "老虎",
    "dolphin": "海豚",
    "octopus": "章魚",
    "penguin": "企鵝",
    "none": "none",
}

INDEX_META_MAP = {
    "all": {
        "faiss": OUT_DIR / "index_all.faiss",
        "meta": OUT_DIR / "metadata_all.json",
    },
    "bee": {
        "faiss": OUT_DIR / "index_bee.faiss",
        "meta": OUT_DIR / "metadata_bee.json",
    },
    "tiger": {
        "faiss": OUT_DIR / "index_tiger.faiss",
        "meta": OUT_DIR / "metadata_tiger.json",
    },
    "dolphin": {
        "faiss": OUT_DIR / "index_dolphin.faiss",
        "meta": OUT_DIR / "metadata_dolphin.json",
    },
    "octopus": {
        "faiss": OUT_DIR / "index_octopus.faiss",
        "meta": OUT_DIR / "metadata_octopus.json",
    },
    "penguin": {
        "faiss": OUT_DIR / "index_penguin.faiss",
        "meta": OUT_DIR / "metadata_penguin.json",
    },
}


# =========================================================
# Persona Definitions
# =========================================================

PERSONA_DEFINITIONS = """
[bee]
核心特徵：
- 重視規則、流程、承諾、責任、準時、標準、公平、角色責任
- 在意事情是否照講好的方式進行
- 對流程混亂、標準不清、說好不做容易感到不舒服
- 決策邏輯偏向：應該這樣做、原則上這樣做、照規則怎麼做
- 常見雷點：沒照流程、沒照規則、模糊承諾、標準不一致、偏離既定行程

[tiger]
核心特徵：
- 重視行動力、推進、掌控、結果、速度、主導權
- 不喜歡拖延、猶豫、低效率、節奏過慢
- 傾向直接主導、快速決策、推著事情往前走
- 決策邏輯偏向：怎麼最快、怎麼最有效率、怎麼直接往前推
- 常見雷點：拖、慢、沒效率、一直討論不決定

[dolphin]
核心特徵：
- 重視感受、氣氛、互動、情緒交流、共鳴、分享、彈性
- 在意關係是不是舒服、有沒有回應、有沒有熱度
- 喜歡熱烈互動、互相給反應、想法來了就行動、比較隨性
- 決策邏輯偏向：這樣感覺如何、氣氛如何、當下舒服不舒服
- 常見雷點：氣氛尷尬、互動很冷、被過度限制、太拘束、太無聊

[octopus]
核心特徵：
- 重視分析、邏輯、風險、全盤思考、方案比較、資訊完整
- 習慣先蒐集資訊、列出選項、比較利弊，再做決定
- 不喜歡在資訊不足時太快下結論
- 決策邏輯偏向：為什麼這樣做、風險是什麼、資訊夠不夠、先想清楚
- 常見雷點：資訊不完整、沒想清楚、太衝動、沒評估風險

[penguin]
核心特徵：
- 重視穩定、安全感、可預測性、信任、長期穩固、安心
- 不喜歡過度突然的改變、不確定、失去依靠
- 傾向先建立穩定基礎，再慢慢往前走
- 決策邏輯偏向：這樣穩不穩、安不安心、能不能長久
- 常見雷點：不穩定、變來變去、沒有安全感、關係不可靠

[none]
以下情況應判 none：
- 生理狀態
- 單一事件
- 工具查詢
- 技術問題
- 產品問題
- 純情緒
- 純場景描述
- 單純願望、理想、期待、社交目標
- 過度模糊、沒有穩定人格證據
""".strip()


# =========================================================
# Prompts
# =========================================================

TASK_ROUTER_PROMPT = """
你是一個任務分類器。

請判斷使用者輸入屬於哪一種任務：

1. interpersonal_chat
- 人際關係
- 溝通問題
- 情緒困擾
- 衝突
- 人格分析
- 人與人互動建議

2. marketing_copy
- 行銷內容生成
- 廣告語
- slogan
- CTA
- 幫我寫文案
- 幫我寫廣告

3. strategy_advice
- 行銷策略
- 活動企劃
- 商業建議
- 品牌策略
- 溝通策略
- 方案規劃
- 怎麼設計一個更有效的做法
- 怎麼讓某類人產生信任 / 好感 / 參與
- 不只是寫內容，而是要想整體方向

4. other
- 其他無關問題

判斷規則：
- 如果在問「我跟某人怎麼相處 / 為什麼會這樣 / 我該怎麼辦」→ interpersonal_chat
- 如果在問「幫我寫廣告 / 文案 / slogan / CTA」→ marketing_copy
- 如果在問「幫我想策略 / 怎麼做 / 怎麼設計 / 怎麼讓某類人」→ strategy_advice
- 其他 → other

只輸出以下其中一個字：
interpersonal_chat
marketing_copy
strategy_advice
other
""".strip()


L1_SYSTEM_PROMPT = """
你是一個多人情境解析器。

你的任務不是分析人格，也不是給建議。
你的任務是找出文本中真正被描述到的「人物或群體單位」，供後續模組處理。

【目標】
請輸出：
1. key_profiles：文本中真正被描述到的人物 / 群體單位
2. ignored_others：可忽略的人物

【重要分工】
- L1 只負責抽單位，不負責判斷這段文字是否具有人格分析價值
- 是否值得做人格分析，是 L2 的工作
- 所以只要文本中明確描述到某人物 / 群體的想法、感受、行為、偏好、傾向、困擾，就可以建立 profile
- 不要因為內容看起來不像人格就省略 profile

【規則】
1. 如果句子明確在描述某人物，就建立 profile
2. 不要強制建立 speaker；只有當文本真的在描述「我」時才建立
3. 如果文本描述的是某人的穩定傾向、互動風格、決策方式、做事偏好，就可建立 profile
4. 如果句子包含「但 / 因為 / 所以 / 其實」等，且前後是在補充同一個人的完整意思，應保留為同一個 profile，不要拆碎
5. 如果文本描述的是「我們 / 我跟A / 我和朋友」的共同互動模式，且可視為一個共享單位，允許建立群體 profile
6. 不要把 speaker 的內容混進 other
7. 不要把 other 的內容混進 speaker
8. ignored_others 只放背景角色、路人、明顯與主問題無關的雜訊人物
9. 不要做人格分類
10. 不要補充原文沒有的資訊

【輸出格式】
只輸出 JSON：
{
  "key_profiles": [
    {
      "name": "我",
      "text": "..."
    },
    {
      "name": "朋友A",
      "text": "..."
    },
    {
      "name": "我們",
      "text": "..."
    }
  ],
  "ignored_others": []
}
""".strip()


L2_SYSTEM_PROMPT = """
你是一個 persona relevance gate。

你的任務不是判斷五種人格，而是判斷：
這段文字是否值得進行「人格特徵分析」。

只允許輸出 JSON：
{
  "label": "persona_relevant" | "not_persona" | "uncertain",
  "reason": "一句簡短理由"
}

【可判 persona_relevant 的情況】
文本中明確呈現至少一種：
1. 穩定的價值觀
2. 穩定的處事風格
3. 穩定的決策偏好
4. 穩定的人際互動模式
5. 穩定的抽象人格傾向詞（例如：隨心所欲、很有原則、很隨性、很重感覺）

【必須判 not_persona 的情況】
- 生理狀態或醫療問題
- 純工具查詢
- 產品/技術/保固/故障問題
- 單純事實查詢
- 無意義閒聊
- 純情緒宣洩，沒有穩定人格證據
- 單純願望、理想、期待、社交目標，沒有穩定行為或互動模式
- 太短、太空泛、沒有穩定人格證據

【判 uncertain 的情況】
- 有一些人格線索，但證據不足
- 有抽象人格傾向，但缺乏具體行為或穩定模式支撐
- 很難確定是否為穩定人格，而不是暫時反應

【重要規則】
- 不要猜五種人格
- 不要延伸腦補
- 沒有明確證據就不要放行
""".strip()


L4_SYSTEM_PROMPT = f"""
你是一個嚴謹的人格判斷系統，負責將「單一人物文本」分類到五種人格之一，或判定為 none。

【任務】
根據提供的文本，判斷該人物的穩定行為傾向或決策風格，輸出：
- persona
- confidence
- reason
- evidence
- behavior
- motive
- pain_point
- comparisons

【固定人格定義】
{PERSONA_DEFINITIONS}

【核心原則】
人格 = 一個人如何做選擇、如何行動、為什麼這樣做、最不能接受什麼
不是情緒
不是單次事件
不是純場景描述
不是單純願望、理想、期待

【你必須用三層分析】

1. 行為（Behavior）
- 這個人表面上怎麼做？

2. 動機（Motive / Decision Logic）
- 他為什麼這樣做？
- 優先判斷：
  - bee：因為規則、流程、承諾、應該這樣做
  - tiger：因為效率、推進、掌控、不要拖
  - dolphin：因為感受、氣氛、互動、彈性、當下
  - octopus：因為分析、推理、風險、資訊夠不夠
  - penguin：因為穩定、安全感、信任、可預測性

3. 雷點（Pain Point / Aversion）
- 他最不能接受什麼？

【判斷優先順序】
最終人格判定時，請依照：
1. 動機（最優先）
2. 雷點（第二）
3. 行為（第三）

【證據強度規則】
1. 行為型證據明確、動機清楚 → 可以 high
2. 只有抽象人格傾向詞，沒有具體行為 → 最多 medium
3. 只有情緒、場景、模糊反應 → none
4. 若文本只表達願望、理想、期待、社交目標，而沒有穩定行為、決策方式或互動模式 → none

【重要限制】
1. 不可因為有情緒就推人格
2. 不可因為場景就推人格
3. 不可因為「想要成為怎樣的人」就直接推人格
4. 不可過度腦補
5. 不確定時，選擇 none
6. 若輸入只有抽象人格詞，仍可分類，但 confidence 不可高於 medium

【輸出格式】
只輸出 JSON：
{{
  "persona": "bee | tiger | dolphin | octopus | penguin | none",
  "confidence": "high | medium | low",
  "behavior": "一句話描述表面行為",
  "motive": "一句話描述核心動機",
  "pain_point": "一句話描述最可能的雷點",
  "reason": "簡短總結說明",
  "evidence": ["..."],
  "comparisons": {{
    "bee": "high | medium | low",
    "tiger": "high | medium | low",
    "dolphin": "high | medium | low",
    "octopus": "high | medium | low",
    "penguin": "high | medium | low"
  }}
}}
""".strip()


L5_INTERPERSONAL_PROMPT = """
你是一個「人格 × 人際關係分析助理」。

你的目標不是安慰，而是：
找出真正的衝突核心
解釋關係中的錯位
提供可以直接使用的行為策略

【回答要求】
1. 核心問題
- 不要只描述表面
- 必須指出底層衝突
  例如：規則 vs 氣氛 / 效率 vs 感受 / 節奏 vs 安全感

2. 關係解析
- 說清楚雙方各自怎麼看這件事
- 指出：
  A 以為自己在做什麼
  B 實際感受到什麼

3. 建議
- 一定要具體
- 盡量給可直接講的句子或具體做法
- 不要只說多溝通、多理解

4. persona 使用規則
- persona 是用來幫助理解與調整策略
- 不要變成人格分析報告
- 如果某角色沒有明確 persona，不要替他硬貼人格標籤
- 可以用「比較重視彈性 / 比較重視感受 / 比較重視秩序」這種行為語言描述

【輸出格式】
請使用：

【核心問題】
...

【關係解析】
...

【建議】
1.
2.
""".strip()


L5_GENERAL_PROMPT = """
你是一個人際問題解決助手。

你的目標不是安慰，而是：
找出真正的問題
提供可以直接使用的建議

【回答要求】
1. 不要只重述問題
2. 要指出核心困擾
3. 建議要具體、可執行
4. 不要使用人格分類
5. 不要講空話

【輸出格式】
請使用：

【核心問題】
...

【建議】
1.
2.
""".strip()


MARKETING_PARSER_PROMPT = """
你是一個「人格傾向推測器」。

任務：
從使用者的行銷需求中推測：
1. product（產品）
2. persona（人格傾向）
3. tone（語氣）

【人格定義（語意理解）】

dolphin（海豚）
→ 開心、互動、氣氛、陪伴、分享、情緒連結、熱鬧、有人味、有感覺

tiger（老虎）
→ 效率、結果、目標、速度、成就、掌控、直接、快速、衝刺

bee（蜜蜂）
→ 規則、準時、責任、穩定、流程、可靠、秩序、清楚、規劃

penguin（企鵝）
→ 安全感、信任、穩定、被照顧、安心感、溫暖、陪伴

octopus（章魚）
→ 自由、創意、隨性、多變、探索、彈性、可能性

【判斷原則】
- 不需要明確提到人格
- 允許推測
- 只要語意接近，就選最可能的一個
- 不要過度保守
- 除非完全看不出來，否則不要輸出 general

【語氣推測】
- 開心、互動、熱鬧 → 活潑
- 安全感、溫暖、陪伴 → 溫暖
- 效率、結果、速度 → 直接 / 專業
- 自由、創意、探索 → 輕鬆 / 有趣
- 規則、清楚、可靠 → 穩重 / 清楚

只輸出 JSON：
{
  "product": "...",
  "persona": "dolphin / tiger / bee / penguin / octopus / general",
  "tone": "..."
}
""".strip()


STRATEGY_PARSER_PROMPT = """
你是一個「人格策略需求解析器」。

你的任務是從使用者輸入中解析出：
1. target_object（產品 / 活動 / 服務 / 品牌 / 方案）
2. target_persona（bee / tiger / dolphin / octopus / penguin / general）
3. strategy_goal（想達成什麼）
4. tone（語氣 / 策略風格）

【人格定義（語意理解）】
bee（蜜蜂）
→ 規則、可靠、清楚、秩序、制度、標準、流程、可預期

tiger（老虎）
→ 效率、結果、速度、推進、掌控、行動、直接、目標導向

dolphin（海豚）
→ 感受、互動、氣氛、共鳴、熱度、陪伴、分享、連結

octopus（章魚）
→ 分析、選項、自由、彈性、探索、創意、可能性、思考

penguin（企鵝）
→ 安全感、穩定、信任、陪伴、溫和、長期、安心、可依靠

【判斷原則】
- 如果使用者已明確指定人格，直接採用
- 如果沒有明確指定，就依語意推測最可能人格
- target_object 盡量抽象成一個清楚名詞
- strategy_goal 要整理成一句簡潔目標
- 不要過度保守，除非真的無法判斷才用 general

只輸出 JSON：
{
  "target_object": "...",
  "target_persona": "bee / tiger / dolphin / octopus / penguin / general",
  "strategy_goal": "...",
  "tone": "..."
}
""".strip()


PERSONA_MARKETING_MAP = {
    "bee": """
你要對「重視規則、秩序、可靠性」的人說話。

文案規則：
- 強調清楚、安心、可預期、穩定
- 強調制度、品質、規劃、保障
- 語氣要清楚、乾淨、不要浮誇
- 避免太情緒化、太空泛、太夢幻
- 讓人感覺：這是可信、能放心選的
""".strip(),

    "tiger": """
你要對「重視效率、結果、推進感」的人說話。

文案規則：
- 用直接、有力、結果導向的語言
- 多用強動詞：突破、提升、加速、做到
- CTA 要明確、果斷
- 不要拖泥帶水，不要過度鋪陳
- 讓人感覺：現在就行動，會有成果
""".strip(),

    "dolphin": """
你要對「重視感受、互動、氣氛、共鳴」的人說話。

文案規則：
- 一定要有情境感、畫面感
- 盡量用第二人稱（你）直接對話
- 強調陪伴、感受、享受、一起、互動
- 像朋友邀請，不像品牌口號
- 避免 generic 正能量廣告句，例如：
  「成為更好的自己」「燃燒你的熱情」「開啟你的旅程」
- 要讓人感覺：這跟我有關、我會想靠近
""".strip(),

    "octopus": """
你要對「重視自由、創意、可能性、彈性」的人說話。

文案規則：
- 強調選擇、可能性、探索、彈性
- 讓人感覺沒有被綁住
- 語言可以比較靈活、有空間感
- 不要太制式、太硬
- 讓人感覺：我可以用自己的方式參與
""".strip(),

    "penguin": """
你要對「重視溫暖、安全感、陪伴、長期穩定」的人說話。

文案規則：
- 語氣溫和、安心、被照顧
- 強調陪伴、支持、慢慢來、放心
- 不要太刺激、太壓迫
- 讓人感覺：這裡是安全的，可以信任
""".strip(),

    "general": """
你要對一般大眾說話。

文案規則：
- 清楚、自然、不要 generic
- 要有情境感，但不要太浮誇
- 保持吸引力與可讀性
""".strip(),
}


PERSONA_STRATEGY_MAP = {
    "bee": """
你要對「重視規則、秩序、可靠性」的人設計策略。

策略規則：
- 強調制度、流程、清楚規範、可信度、穩定品質
- 設計上要有明確步驟、標準與可預期結果
- 避免太飄、太感覺派、太臨場 improvisation
- 讓對方感覺：這是有規劃、能放心執行的方案
""".strip(),

    "tiger": """
你要對「重視效率、結果、推進感」的人設計策略。

策略規則：
- 強調快速決策、成效、轉換、成果、行動推進
- 設計上要直接、有力、可執行
- 避免太多鋪陳、太慢熱、太重情緒氛圍
- 讓對方感覺：這方案能有效推進、很快看到成果
""".strip(),

    "dolphin": """
你要對「重視感受、互動、氣氛、共鳴」的人設計策略。

策略規則：
- 強調體驗感、參與感、互動感、分享感、情緒連結
- 設計上要有人味、有情境、有社交性
- 避免太生硬、太制度化、太冷
- 讓對方感覺：這方案有溫度，會想參與、想靠近、想分享
""".strip(),

    "octopus": """
你要對「重視分析、彈性、探索、可能性」的人設計策略。

策略規則：
- 強調選擇、比較、資訊、創意空間、自由度
- 設計上要給思考空間與不同玩法
- 避免太單線、太封閉、太硬性規定
- 讓對方感覺：這方案有腦、有彈性、可以用自己的方式參與
""".strip(),

    "penguin": """
你要對「重視安全感、穩定、信任、長期陪伴」的人設計策略。

策略規則：
- 強調陪伴、穩定、安心、低風險、可持續
- 設計上要溫和、可靠、不刺激
- 避免太競爭、太壓迫、太高風險
- 讓對方感覺：這方案是可信賴、能長期走下去的
""".strip(),

    "general": """
你要對一般大眾設計策略。

策略規則：
- 清楚、具體、可執行
- 兼顧吸引力與落地性
- 不要太空泛
""".strip(),
}


L5_MARKETING_PROMPT = """
你是一個人格導向的行銷策略顧問與文案專家。

你的任務是：
先用顧問語氣說明「怎麼打動這種人」，再自然銜接到文案。

【產品】
{product}

【目標人格】
{persona}

【人格約束模板】
{persona_style}

【語氣】
{tone}

【核心要求】
- 文案必須符合該人格的核心驅動與決策邏輯
- 要讓人理解「這段文案為什麼這樣寫」
- 避免空泛與 generic 廣告語
- 不得混入其他人格語氣

--------------------------------

【人格表達方式（關鍵）】

文案的呈現方式必須依照人格調整：

dolphin（情感 / 互動）
→ 強調人與人互動、回應、共鳴
→ 讓人感覺「我在裡面」

tiger（結果 / 行動）
→ 強調成果、效率、優勢、推進感
→ 語氣直接、有力，少鋪陳

bee（規則 / 信任）
→ 強調流程、可靠性、清楚資訊
→ 語氣穩定、有邏輯

penguin（穩定 / 陪伴）
→ 強調安心感、長期關係
→ 有互動但不能強烈或壓迫

octopus（彈性 / 多元）
→ 呈現多選擇、多觀點
→ 避免單一強推

不要使用同一種文案套路套所有人格

--------------------------------

【輸出方式】

整體請寫成「一段自然流動的內容」，不要顯示段落名稱或結構提示。

內容順序如下（但不要標示出來）：

1. 開頭用顧問語氣說明：
   - 如果想打動這種人，應該怎麼做
   - 可以自然補一句「不要怎麼做」（避免踩雷）

   使用這種句型：
   - 如果你想打動這種人，你要…
   - 想讓他們願意行動，關鍵在於…
   - 重點不是…而是…

   注意：
   - 不要解釋人格
   - 不要說「某某人格的人」

2. 接一個自然轉場句，銜接到文案，例如：
   - 我幫你整理成一段可以直接用的文案
   - 接下來這段可以直接使用

   （保持簡單，不要搶戲）

3. 接續完整行銷文案：

   文案需包含：
   - 一個有力的開頭（hook）
   - 中段推進（情境 / 成果 / 價值）
   - 自然收斂（CTA）

   長度要求：
   - 根據使用者需求自動調整
   - 可以是短文案，也可以是約200字貼文
   - 不要被固定段落限制

--------------------------------

【畫面與內容要求】

文案中必須呈現「符合人格的畫面感」：

- tiger → 突破、成果、成長
- dolphin → 互動、情緒、群體
- penguin → 陪伴、穩定、安心
- bee → 流程、系統、清楚步驟
- octopus → 選擇、多路徑

但不要強制所有人格用同一種畫面

--------------------------------

【限制】

不要使用：
- 【人格分析】【策略說明】這類標題
- 條列式文案
- 教科書語氣
- 過度模板化句型

整體要像真實品牌文案，
讓人看完會想行動
"""


L5_STRATEGY_EXEC_PROMPT = """
你是一個人格導向策略顧問。

你的任務是根據目標人格、目標對象與策略目標，
提供自然、好讀、可以直接使用的策略建議。

【目標對象】
{target_object}

【目標人格】
{target_persona}

【策略目標】
{strategy_goal}

【人格約束模板】
{persona_style}

【風格】
{tone}

【核心要求】
- 必須以人格的核心驅動、抗拒點、說服切入點為基礎
- 不得變成人格教科書解釋
- 必須讓人理解「為什麼這樣設計有效」
- 必須提供可執行做法

【回答格式（必須遵守順序）】

請使用 Markdown 標題（###）來組織內容

順序必須是：

1️⃣ 人格洞察（人會怎麼想）
2️⃣ 策略推導（應該怎麼做） ← 🔥 必須出現在這裡
3️⃣ 執行注意事項
4️⃣ 避雷
5️⃣ 具體做法 / 活動
6️⃣ 對外說法

--------------------------------

【每一段要怎麼寫】

### 人格洞察（第一段）
- 不要解釋人格
- 要像在講一種「人的反應」

使用這種語氣：
- 這種人其實更在意的是…
- 對他們來說，關鍵不是…而是…
- 他們通常會被…吸引

--------------------------------

### 策略推導（第二段，🔥關鍵）
這一段一定要做：

👉 從人格 → 推導出「整體應該怎麼設計」

必須做到：
- 把人性轉成策略方向
- 說清楚「你應該用什麼方式做整體設計」
- 不要講細節（那是後面才講）

語氣像：
- 既然他們在意的是…那整體設計就要…
- 換句話說，你不是在做…而是在做…
- 所以整場應該走的是…而不是…

--------------------------------

### 執行注意事項
- 說「過程中要注意什麼」
- 偏控制節奏 / 感受 / 流程

--------------------------------

### 避雷
- 說「什麼會讓他們直接退出」
- 要具體、有畫面

--------------------------------

### 具體做法 / 活動
- 3~5個
- 一看就懂（桌遊、聚餐、說明會形式等）
- 這一段一定在策略推導後面

--------------------------------

### 對外說法
- 2~4句
- 可直接拿去用（像宣傳文案）

--------------------------------

【風格限制】

不要使用：
- 【心理分析】【策略重點】這種標題
- 「某某人格的人通常會」
- 「首先、其次、再者」
- 分隔線
- 教科書語氣

可以：
- 用自然標題（像人在整理）
- 有少量條列（但不要整篇條列）

--------------------------------

【最重要】

👉 結構一定是：

人格洞察
→ 策略推導（🔥一定要有）
→ 注意
→ 避雷
→ 做法
→ 說法

順序不能錯

整體要像一個有經驗的人在幫你「想怎麼做」
而不是在分析
"""


# =========================================================
# Utilities
# =========================================================

def load_env() -> None:
    for p in ENV_CANDIDATES:
        if p.exists():
            load_dotenv(dotenv_path=p, override=False)
            return
    load_dotenv(override=False)


def get_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("找不到 OPENAI_API_KEY，請確認 .env 已設定。")
    return api_key


def safe_json_parse(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))

    raise ValueError(f"無法解析 JSON：\n{text}")


def normalize_l1_output(data: Dict[str, Any]) -> Dict[str, Any]:
    key_profiles = []
    raw_profiles = data.get("key_profiles", [])

    if isinstance(raw_profiles, list):
        for item in raw_profiles:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            text = str(item.get("text", "")).strip()
            if name and text:
                key_profiles.append({"name": name, "text": text})

    ignored = data.get("ignored_others", [])
    if not isinstance(ignored, list):
        ignored = []

    clean_ignored = []
    for x in ignored:
        if isinstance(x, dict):
            clean_ignored.append(json.dumps(x, ensure_ascii=False))
        else:
            sx = str(x).strip()
            if sx:
                clean_ignored.append(sx)

    return {"key_profiles": key_profiles, "ignored_others": clean_ignored}


def normalize_chunk(chunk: Any, idx: int = -1) -> Dict[str, Any]:
    if isinstance(chunk, dict):
        source_file = str(
            chunk.get("source_file")
            or chunk.get("file")
            or chunk.get("filename")
            or chunk.get("source")
            or chunk.get("persona")
            or "unknown"
        ).strip()

        h2_title = str(
            chunk.get("h2_title")
            or chunk.get("title")
            or chunk.get("section")
            or chunk.get("section_title")
            or chunk.get("topic")
            or ""
        ).strip()

        text_value = (
            chunk.get("text")
            or chunk.get("content")
            or chunk.get("chunk")
            or chunk.get("chunk_text")
            or chunk.get("body")
            or ""
        )
        text_str = str(text_value).strip()

        return {
            "id": chunk.get("id", f"chunk_{idx}" if idx >= 0 else "chunk"),
            "persona": str(chunk.get("persona", "")).strip(),
            "source_file": source_file,
            "h2_title": h2_title,
            "text": text_str,
            "raw": chunk,
        }

    if isinstance(chunk, str):
        return {
            "id": f"chunk_{idx}" if idx >= 0 else "chunk",
            "persona": "",
            "source_file": "unknown",
            "h2_title": "",
            "text": chunk.strip(),
            "raw": chunk,
        }

    return {
        "id": f"chunk_{idx}" if idx >= 0 else "chunk",
        "persona": "",
        "source_file": "unknown",
        "h2_title": "",
        "text": str(chunk).strip(),
        "raw": chunk,
    }


def get_chunk_text(chunk: Any) -> str:
    return normalize_chunk(chunk).get("text", "")


def get_chunk_title(chunk: Any) -> str:
    c = normalize_chunk(chunk)
    source_file = c.get("source_file", "unknown")
    h2_title = c.get("h2_title", "")
    if h2_title:
        return f"{source_file} | {h2_title}"
    return source_file


# =========================================================
# RAG
# =========================================================

def load_index_and_chunks(index_key: str = "all"):
    if index_key not in INDEX_META_MAP:
        raise ValueError(f"未知的 index_key: {index_key}")

    faiss_file = INDEX_META_MAP[index_key]["faiss"]
    meta_file = INDEX_META_MAP[index_key]["meta"]

    if not faiss_file.exists() or not meta_file.exists():
        raise FileNotFoundError(
            f"找不到 RAG 檔案：\n{faiss_file}\n{meta_file}"
        )

    index = faiss.read_index(str(faiss_file))

    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if not isinstance(metadata, list):
        raise ValueError(f"{meta_file.name} 格式錯誤：必須是 list")

    chunks = []
    for i, item in enumerate(metadata):
        chunks.append({
            "id": item.get("id", f"chunk_{i}"),
            "persona": str(item.get("persona", "")).strip(),
            "source_file": str(item.get("source", "")).strip(),
            "h2_title": str(item.get("section_title", "")).strip(),
            "text": str(item.get("text", "")).strip(),
            "raw": item,
        })

    if index.ntotal != len(chunks):
        raise ValueError(
            f"FAISS 向量數量 ({index.ntotal}) 與 metadata 數量 ({len(chunks)}) 不一致，請重新 build_persona_indexes.py"
        )

    return index, chunks


def load_all_indexes():
    stores = {}
    for key in INDEX_META_MAP.keys():
        stores[key] = load_index_and_chunks(key)
    return stores


def retrieve_rag_context(
    query_text: str,
    persona_summaries: List[Dict[str, Any]],
    index,
    chunks,
    emb_model,
    top_k: int = TOP_K_RAG,
    target_persona: str = ""
) -> List[Dict[str, Any]]:
    persona_hint_parts = []
    for p in persona_summaries:
        persona_hint_parts.append(
            f"{p['name']} persona={p['persona']} motive={p['motive']} pain={p['pain_point']}"
        )

    retrieval_query = (
        f"人際衝突 溝通 協調 建議 做法 行銷文案 廣告 行動建議 策略 受眾 {query_text} "
        + " ".join(persona_hint_parts)
    )

    q_emb = emb_model.encode(
        [f"query: {retrieval_query}"],
        normalize_embeddings=True
    ).astype("float32")

    D, I = index.search(q_emb, top_k)

    results = []
    for rank, idx in enumerate(I[0], start=1):
        if idx < 0 or idx >= len(chunks):
            continue

        c = chunks[idx]
        results.append({
            "rank": rank,
            "score": float(D[0][rank - 1]),
            "source": get_chunk_title(c),
            "text": get_chunk_text(c),
        })

    return results


def build_rag_text(retrieved: List[Dict[str, Any]]) -> str:
    if not retrieved:
        return "(no rag context)"
    blocks = []
    for r in retrieved:
        blocks.append(
            f"[{r['rank']}] {r['source']} | score={r['score']:.4f}\n{r['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def build_strategy_focused_rag_text(retrieved: List[Dict[str, Any]]) -> str:
    if not retrieved:
        return "(no rag context)"
    prefix = (
        "請優先從以下人格知識中整理：\n"
        "1. 核心驅動動機\n"
        "2. 抗拒點\n"
        "3. 有效說服策略\n\n"
    )
    return prefix + build_rag_text(retrieved)


# =========================================================
# Router / Marketing Parser / Strategy Parser
# =========================================================

def route_task(client: OpenAI, user_input: str) -> str:
    resp = client.chat.completions.create(
        model=ROUTER_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": TASK_ROUTER_PROMPT},
            {"role": "user", "content": user_input},
        ],
    )
    result = resp.choices[0].message.content.strip().lower()

    if "interpersonal_chat" in result:
        return "interpersonal_chat"
    if "marketing_copy" in result:
        return "marketing_copy"
    if "strategy_advice" in result:
        return "strategy_advice"
    return "other"


def parse_marketing_input(client: OpenAI, user_input: str) -> Dict[str, Any]:
    user_prompt = f"{MARKETING_PARSER_PROMPT}\n\n使用者輸入：{user_input}"

    resp = client.chat.completions.create(
        model=MARKETING_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": "你是人格傾向推測器。"},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = resp.choices[0].message.content

    try:
        parsed = safe_json_parse(raw)
        persona = str(parsed.get("persona", "general")).strip().lower()
        if persona not in {"dolphin", "tiger", "bee", "penguin", "octopus", "general"}:
            persona = "general"

        return {
            "product": parsed.get("product", user_input),
            "persona": persona,
            "tone": parsed.get("tone", "normal"),
        }
    except Exception:
        return {
            "product": user_input,
            "persona": "general",
            "tone": "normal",
        }


def parse_strategy_input(client: OpenAI, user_input: str) -> Dict[str, Any]:
    user_prompt = f"{STRATEGY_PARSER_PROMPT}\n\n使用者輸入：{user_input}"

    resp = client.chat.completions.create(
        model=STRATEGY_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": "你是人格策略需求解析器。"},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = resp.choices[0].message.content

    try:
        parsed = safe_json_parse(raw)

        target_persona = str(parsed.get("target_persona", "general")).strip().lower()
        if target_persona not in {"bee", "tiger", "dolphin", "octopus", "penguin", "general"}:
            target_persona = "general"

        return {
            "target_object": str(parsed.get("target_object", user_input)).strip(),
            "target_persona": target_persona,
            "strategy_goal": str(parsed.get("strategy_goal", user_input)).strip(),
            "tone": str(parsed.get("tone", "normal")).strip(),
        }
    except Exception:
        return {
            "target_object": user_input,
            "target_persona": "general",
            "strategy_goal": user_input,
            "tone": "normal",
        }


# =========================================================
# LLM Calls
# =========================================================

def run_l1(client: OpenAI, user_input: str) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=L1_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": L1_SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
    )
    raw = resp.choices[0].message.content
    data = safe_json_parse(raw)
    norm = normalize_l1_output(data)
    norm["_raw"] = raw
    return norm


def run_l2(client: OpenAI, profile_text: str) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=L2_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": L2_SYSTEM_PROMPT},
            {"role": "user", "content": profile_text},
        ],
    )
    raw = resp.choices[0].message.content
    data = safe_json_parse(raw)

    label = str(data.get("label", "")).strip()
    reason = str(data.get("reason", "")).strip()

    if label not in {"persona_relevant", "not_persona", "uncertain"}:
        label = "uncertain"
        if not reason:
            reason = "invalid label returned"

    return {"label": label, "reason": reason, "_raw": raw}


def run_l4(client: OpenAI, profile_name: str, profile_text: str) -> Dict[str, Any]:
    user_prompt = f"人物名稱：{profile_name}\n人物文本：{profile_text}\n"

    resp = client.chat.completions.create(
        model=L4_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": L4_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = resp.choices[0].message.content
    data = safe_json_parse(raw)

    persona = str(data.get("persona", "")).strip()
    confidence = str(data.get("confidence", "")).strip()
    behavior = str(data.get("behavior", "")).strip()
    motive = str(data.get("motive", "")).strip()
    pain_point = str(data.get("pain_point", "")).strip()
    reason = str(data.get("reason", "")).strip()
    evidence = data.get("evidence", [])
    comparisons = data.get("comparisons", {})

    if persona not in {"bee", "tiger", "dolphin", "octopus", "penguin", "none"}:
        persona = "none"
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    if not isinstance(evidence, list):
        evidence = []
    if not isinstance(comparisons, dict):
        comparisons = {}

    clean_comparisons = {}
    for k in ["bee", "tiger", "dolphin", "octopus", "penguin"]:
        v = str(comparisons.get(k, "low")).strip().lower()
        if v not in {"high", "medium", "low"}:
            v = "low"
        clean_comparisons[k] = v

    return {
        "persona": persona,
        "confidence": confidence,
        "behavior": behavior,
        "motive": motive,
        "pain_point": pain_point,
        "reason": reason,
        "evidence": [str(x) for x in evidence],
        "comparisons": clean_comparisons,
        "_raw": raw,
    }


def build_persona_summary(l4_profiles: List[Dict[str, Any]]) -> str:
    if not l4_profiles:
        return "(none)"
    lines = []
    for p in l4_profiles:
        lines.append(
            f"{p['name']}：{p['persona']}，信心={p['confidence']}，行為={p['behavior']}，動機={p['motive']}，雷點={p['pain_point']}"
        )
    return "\n".join(lines)


def run_l5_interpersonal(
    client: OpenAI,
    user_input: str,
    l4_profiles: List[Dict[str, Any]],
    rag_text: str
) -> str:
    persona_summary = build_persona_summary(l4_profiles)

    user_prompt = f"""
使用者問題：
{user_input}

--------------------------------
人格資訊：
{persona_summary}

--------------------------------
參考資料（RAG）：
{rag_text}

--------------------------------
請結合人格與情境進行分析。
""".strip()

    resp = client.chat.completions.create(
        model=L5_MODEL,
        temperature=0.5,
        messages=[
            {"role": "system", "content": L5_INTERPERSONAL_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


def run_l5_general(client: OpenAI, user_input: str, rag_text: str) -> str:
    user_prompt = f"""
使用者問題：
{user_input}

--------------------------------
參考資料（RAG）：
{rag_text}

--------------------------------
請以一般人際分析方式回答（無人格）。
""".strip()

    resp = client.chat.completions.create(
        model=L5_MODEL,
        temperature=0.5,
        messages=[
            {"role": "system", "content": L5_GENERAL_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


def run_l5_marketing(client: OpenAI, parsed: Dict[str, Any], rag_text: str = "") -> str:
    persona = parsed.get("persona", "general")
    persona_style = PERSONA_MARKETING_MAP.get(persona, PERSONA_MARKETING_MAP["general"])

    user_prompt = L5_MARKETING_PROMPT.format(
        product=parsed.get("product", ""),
        persona=persona,
        persona_style=persona_style,
        tone=parsed.get("tone", "normal"),
    )

    full_prompt = f"""
{user_prompt}

--------------------------------
參考資料（RAG）：
{rag_text}
""".strip()

    resp = client.chat.completions.create(
        model=MARKETING_MODEL,
        temperature=0.7,
        messages=[
            {"role": "system", "content": "你是頂級行銷文案專家。"},
            {"role": "user", "content": full_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


def run_l5_strategy(client: OpenAI, parsed: Dict[str, Any], rag_text: str) -> str:
    persona = parsed.get("target_persona", "general")
    persona_style = PERSONA_STRATEGY_MAP.get(persona, PERSONA_STRATEGY_MAP["general"])

    user_prompt = L5_STRATEGY_EXEC_PROMPT.format(
        target_object=parsed.get("target_object", ""),
        target_persona=persona,
        strategy_goal=parsed.get("strategy_goal", ""),
        persona_style=persona_style,
        tone=parsed.get("tone", "normal"),
    )

    full_prompt = f"""
{user_prompt}

--------------------------------
參考資料（RAG）：
{rag_text}
""".strip()

    resp = client.chat.completions.create(
        model=L5_MODEL,
        temperature=0.5,
        messages=[
            {"role": "system", "content": "你是人格導向策略顧問。"},
            {"role": "user", "content": full_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


# =========================================================
# Pipelines
# =========================================================

def run_interpersonal_pipeline(client: OpenAI, emb_model, index, chunks, user_input: str) -> Dict[str, Any]:
    result = {
        "mode": "interpersonal_chat",
        "input": user_input,
        "l1": None,
        "profiles": [],
        "rag": [],
        "l5_answer": None,
    }

    l1 = run_l1(client, user_input)
    result["l1"] = l1

    l4_profiles_for_l5 = []

    for profile in l1["key_profiles"]:
        name = profile["name"]
        text = profile["text"]

        profile_result = {
            "name": name,
            "text": text,
            "l2": None,
            "l4": None
        }

        l2 = run_l2(client, text)
        profile_result["l2"] = l2

        if l2["label"] in {"persona_relevant", "uncertain"}:
            l4 = run_l4(client, name, text)
            profile_result["l4"] = l4

            if l4["persona"] != "none":
                l4_profiles_for_l5.append({
                    "name": name,
                    "text": text,
                    "persona": l4["persona"],
                    "confidence": l4["confidence"],
                    "behavior": l4["behavior"],
                    "motive": l4["motive"],
                    "pain_point": l4["pain_point"],
                    "reason": l4["reason"],
                    "evidence": l4["evidence"],
                })

        result["profiles"].append(profile_result)

    has_persona = len(l4_profiles_for_l5) > 0

    rag_results = retrieve_rag_context(
        query_text=user_input,
        persona_summaries=l4_profiles_for_l5 if has_persona else [],
        index=index,
        chunks=chunks,
        emb_model=emb_model,
        top_k=TOP_K_RAG
    )
    result["rag"] = rag_results
    rag_text = build_rag_text(rag_results)

    if has_persona:
        l5_answer = run_l5_interpersonal(
            client=client,
            user_input=user_input,
            l4_profiles=l4_profiles_for_l5,
            rag_text=rag_text
        )
    else:
        l5_answer = run_l5_general(
            client=client,
            user_input=user_input,
            rag_text=rag_text
        )

    result["l5_answer"] = l5_answer
    return result


def run_marketing_pipeline(client: OpenAI, emb_model, rag_stores, user_input: str) -> Dict[str, Any]:
    result = {
        "mode": "marketing_copy",
        "input": user_input,
        "parsed_marketing": None,
        "rag": [],
        "l5_answer": None,
    }

    parsed = parse_marketing_input(client, user_input)
    result["parsed_marketing"] = parsed

    persona = parsed.get("persona", "general")
    product = parsed.get("product", "")
    tone = parsed.get("tone", "")

    persona_summaries = []
    if persona != "general":
        persona_summaries.append({
            "name": "target_audience",
            "persona": persona,
            "motive": f"偏好{persona}型的語言與訴求",
            "pain_point": f"不符合{persona}型偏好的文案風格",
        })

    retrieval_query = f"{product} {tone} {persona}"

    store_key = persona if persona in {"bee", "tiger", "dolphin", "octopus", "penguin"} else "all"
    index, chunks = rag_stores[store_key]

    rag_results = retrieve_rag_context(
        query_text=retrieval_query,
        persona_summaries=persona_summaries,
        index=index,
        chunks=chunks,
        emb_model=emb_model,
        top_k=TOP_K_RAG,
    )
    result["rag"] = rag_results
    rag_text = build_strategy_focused_rag_text(rag_results)

    answer = run_l5_marketing(client, parsed, rag_text)
    result["l5_answer"] = answer
    return result


def run_strategy_pipeline(client: OpenAI, emb_model, rag_stores, user_input: str) -> Dict[str, Any]:
    result = {
        "mode": "strategy_advice",
        "input": user_input,
        "parsed_strategy": None,
        "rag": [],
        "l5_answer": None,
    }

    parsed = parse_strategy_input(client, user_input)
    result["parsed_strategy"] = parsed

    persona = parsed.get("target_persona", "general")
    target_object = parsed.get("target_object", "")
    strategy_goal = parsed.get("strategy_goal", "")

    persona_summaries = []
    if persona != "general":
        persona_summaries.append({
            "name": "target_audience",
            "persona": persona,
            "motive": strategy_goal,
            "pain_point": f"不符合{persona}型偏好的設計",
        })

    retrieval_query = f"{target_object} {strategy_goal} {persona}"

    store_key = persona if persona in {"bee", "tiger", "dolphin", "octopus", "penguin"} else "all"
    index, chunks = rag_stores[store_key]

    rag_results = retrieve_rag_context(
        query_text=retrieval_query,
        persona_summaries=persona_summaries,
        index=index,
        chunks=chunks,
        emb_model=emb_model,
        top_k=TOP_K_RAG,
    )
    result["rag"] = rag_results
    rag_text = build_strategy_focused_rag_text(rag_results)

    answer = run_l5_strategy(client, parsed, rag_text)
    result["l5_answer"] = answer
    return result


def process_query(client: OpenAI, emb_model, rag_stores, user_input: str) -> Dict[str, Any]:
    task = route_task(client, user_input)

    if task == "interpersonal_chat":
        index, chunks = rag_stores["all"]
        return run_interpersonal_pipeline(client, emb_model, index, chunks, user_input)

    if task == "marketing_copy":
        return run_marketing_pipeline(client, emb_model, rag_stores, user_input)

    if task == "strategy_advice":
        return run_strategy_pipeline(client, emb_model, rag_stores, user_input)

    return {
        "mode": "other",
        "input": user_input,
        "l5_answer": "這個問題不在目前系統的處理範圍內。"
    }


# =========================================================
# Debug Print
# =========================================================

def print_debug(result: Dict[str, Any]) -> None:
    print("\n==============================")
    print("=== PERSONA ENGINE ===")
    print("==============================")
    print(f"MODE : {result['mode']}")
    print(f"INPUT: {result['input']}")

    if result["mode"] == "interpersonal_chat":
        l1 = result["l1"]

        print("\n--- L1: KEY PROFILES ---")
        if not l1["key_profiles"]:
            print("(none)")
        else:
            for idx, p in enumerate(l1["key_profiles"], start=1):
                print(f"[{idx}] name : {p['name']}")
                print(f"    text : {p['text']}")

        print(f"\nIGNORED_OTHERS: {l1['ignored_others']}")

        print("\n--- PROFILE ANALYSIS ---")
        if not result["profiles"]:
            print("(none)")
        else:
            for idx, p in enumerate(result["profiles"], start=1):
                print(f"\n[{idx}] PROFILE")
                print(f"name                : {p['name']}")
                print(f"text                : {p['text']}")
                if p["l2"] is not None:
                    print(f"L2 label            : {p['l2']['label']}")
                    print(f"L2 reason           : {p['l2']['reason']}")
                if p["l4"] is None:
                    print("L4 result           : skipped")
                else:
                    print(f"L4 persona          : {p['l4']['persona']} ({CN_MAP.get(p['l4']['persona'], p['l4']['persona'])})")
                    print(f"L4 confidence       : {p['l4']['confidence']}")
                    print(f"L4 behavior         : {p['l4']['behavior']}")
                    print(f"L4 motive           : {p['l4']['motive']}")
                    print(f"L4 pain_point       : {p['l4']['pain_point']}")
                    print(f"L4 reason           : {p['l4']['reason']}")
                    print(f"L4 evidence         : {p['l4']['evidence']}")
                    print("L4 comparisons      :")
                    for k in ["bee", "tiger", "dolphin", "octopus", "penguin"]:
                        print(f"  - {k:8s}: {p['l4']['comparisons'].get(k, 'low')}")

        print("\n--- RAG RETRIEVAL ---")
        if not result["rag"]:
            print("(none)")
        else:
            for r in result["rag"]:
                print(f"[{r['rank']}] {r['source']} | score={r['score']:.4f}")

        print("\n--- L5 ANSWER ---")
        print(result["l5_answer"])
        return

    if result["mode"] == "marketing_copy":
        print("\n--- MARKETING PARSER ---")
        print(json.dumps(result["parsed_marketing"], ensure_ascii=False, indent=2))

        print("\n--- RAG RETRIEVAL ---")
        if not result.get("rag"):
            print("(none)")
        else:
            for r in result["rag"]:
                print(f"[{r['rank']}] {r['source']} | score={r['score']:.4f}")

        print("\n--- L5 ANSWER ---")
        print(result["l5_answer"])
        return

    if result["mode"] == "strategy_advice":
        print("\n--- STRATEGY PARSER ---")
        print(json.dumps(result["parsed_strategy"], ensure_ascii=False, indent=2))

        print("\n--- RAG RETRIEVAL ---")
        if not result["rag"]:
            print("(none)")
        else:
            for r in result["rag"]:
                print(f"[{r['rank']}] {r['source']} | score={r['score']:.4f}")

        print("\n--- L5 ANSWER ---")
        print(result["l5_answer"])
        return

    print("\n--- OUTPUT ---")
    print(result["l5_answer"])


# =========================================================
# Main
# =========================================================

def main():
    load_env()
    api_key = get_api_key()

    client = OpenAI(api_key=api_key)
    emb_model = SentenceTransformer(EMB_MODEL)
    rag_stores = load_all_indexes()

    print("Persona Engine Final v3 ready.")
    print("Modes: interpersonal_chat / marketing_copy / strategy_advice / other")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("\nUser> ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        try:
            result = process_query(client, emb_model, rag_stores, user_input)
            print_debug(result)
        except Exception as e:
            print(f"\n[ERROR] {e}\n")


if __name__ == "__main__":
    main()
