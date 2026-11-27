from __future__ import annotations

import random
import re
from typing import Optional

from backend.models import ChatMessage, GameState, Gender, Phase, Player, Role

# Shared system prompt (cached in Ollama KV-cache across turns)
STATIC_SYSTEM_PROMPT = (
    "Настольная игра «Мафия» в текстовом чате (7 игроков: 5 мирных, 2 мафии; первая ночь без убийств). "
    "ИГРА В ЗАКРЫТУЮ: роли выбывших НЕ раскрываются до финала. "
    "Железные правила логики и речи:\n"
    "- За столом строгая очерёдность ходов: КАЖДЫЙ живой игрок обязательно высказывается в свою очередь! СТРОГО ЗАПРЕЩЕНО обвинять кого-либо в «молчании», «молчаливости», «отмалчивании» или в том, что он «мало говорит».\n"
    "- Общение идёт ТОЛЬКО текстом: не пиши про чужие лица, взгляды или «нервозность» — суди строго по смыслу сказанных фраз (противоречия, уход от вопросов, перевод стрелок) и прошлым голосам.\n"
    "- Игроков, до которых очередь в круге ещё не дошла, нельзя оценивать — им можно только задать вопрос.\n"
    "- Мёртвых игроков НЕЛЬЗЯ подозревать или спрашивать.\n"
    "- Не поддакивай стадно чужим обвинениям без фактов. Говори простым русским языком на «ты», строго от первого лица («я», «мне»), без своего имени в 3-м лице."
)

_TACTICAL_ANGLES = (
    "Твой фокус: проверяй, есть ли логика за чужими обвинениями; кто кидает пустые обвинения без аргументов — сам подозрителен.",
    "Твой фокус: следи за связками — кто с кем слишком охотно соглашается и выгораживает друг друга.",
    "Твой фокус: цепляйся к нестыковкам и уклончивым формулировкам уже высказавшихся игроков.",
    "Твой фокус: думай независимо от толпы — предлагай свой взгляд на сказанные выше реплики.",
)


def _get_mafia_partner(player: Player, all_players: list[Player]) -> Optional[Player]:
    if player.role != Role.MAFIA:
        return None
    for p in all_players:
        if p.role == Role.MAFIA and p.seat != player.seat:
            return p
    return None


def build_name_mention_regex(name: str) -> re.Pattern:
    """Build case-insensitive regex matching Russian declensions of a first name."""
    low = name.strip().lower().replace("ё", "[её]")
    if low == "павел":
        stem = "пав(?:ел|л)"
    elif low == "лев":
        stem = "л(?:ев|ьв)"
    elif len(low) >= 4 and low[-1] in ("а", "я", "й", "ь"):
        stem = re.escape(low[:-1]).replace(r"\[её\]", "[её]")
    else:
        stem = re.escape(low).replace(r"\[её\]", "[её]")
    return re.compile(rf"(?<![а-яё]){stem}[а-яё]{{0,3}}(?![а-яё])", re.IGNORECASE)


def find_direct_mentions_for_player(player: Player, state: GameState) -> list[ChatMessage]:
    """Find public discussion messages today (since `player`'s last speech) from ALIVE players that mention `player`."""
    alive_seats = {p.seat for p in state.alive_players()}
    today_public = [
        m
        for m in state.messages
        if m.day == state.day
        and m.visibility == "public"
        and m.phase == Phase.DAY_DISCUSSION
    ]
    last_own_idx = -1
    for idx, m in enumerate(today_public):
        if m.sender_seat == player.seat:
            last_own_idx = idx
    candidates = today_public[last_own_idx + 1 :]

    pattern = build_name_mention_regex(player.name)
    return [
        m
        for m in candidates
        if m.sender_seat != player.seat
        and m.sender_seat in alive_seats
        and pattern.search(m.text)
    ]


def build_system_prompt(_player: Player, _state: GameState) -> str:
    return STATIC_SYSTEM_PROMPT


def _shuffled_names(names: list[str], seed_key: int) -> list[str]:
    if len(names) <= 1:
        return list(names)
    rng = random.Random(seed_key)
    copied = list(names)
    rng.shuffle(copied)
    return copied


def build_table_state_block(player: Player, state: GameState) -> str:
    rivals_alive = []
    dead_items = []
    for p in state.players:
        g = "М" if p.gender == Gender.MALE else "Ж"
        if p.is_alive:
            if p.seat != player.seat:
                rivals_alive.append(f"{p.name}({g})")
        else:
            if p.seat != player.seat:
                how = "убит мафией ночью" if p.death_reason == "night" else "казнён городом днём"
                dead_items.append(f"{p.name} ({how} {p.death_day})")

    lines: list[str] = []

    today_rec = next((r for r in state.history if r.day == state.day), None)
    if state.day == 1:
        lines.append("День 1 (первая ночь прошла без убийств, все 7 игроков живы, улик ещё нет).")
    elif today_rec and today_rec.night_victim:
        lines.append(
            f"День {state.day}: этой ночью мафия убила игрока {today_rec.night_victim} ({today_rec.night_victim} мёртв и выбыл)."
        )

    if dead_items:
        lines.append(
            f"МЁРТВЫЕ (выбыли навсегда — НЕ подозревать их и НЕ обращаться к ним!): {', '.join(dead_items)}."
        )

    past_records = [r for r in state.history if r.day < state.day]
    for rec in past_records[-2:]:
        parts = [f"Архив Д{rec.day}:"]
        if rec.night_victim:
            parts.append(f"ночью убит {rec.night_victim};")
        if rec.votes:
            v_str = ", ".join(f"{v}->{t}" for v, t in rec.votes.items())
            parts.append(f"голоса [{v_str}];")
        if rec.executed:
            parts.append(f"казнён {rec.executed}.")
        elif rec.tie_no_execution:
            parts.append("ничья.")
        if len(parts) > 1:
            lines.append(" ".join(parts))

    seat_gender = {p.seat: ("М" if p.gender == Gender.MALE else "Ж") for p in state.players}
    today_discussion = [
        m
        for m in state.messages
        if m.day == state.day
        and m.visibility == "public"
        and m.phase == Phase.DAY_DISCUSSION
    ]
    if today_discussion:
        lines.append("Реплики за столом сегодня:")
        for m in today_discussion[-14:]:
            g = seat_gender.get(m.sender_seat or -1, "")
            if m.sender_seat == player.seat:
                author = f"Я ({m.sender_name})"
            elif "Последнее слово" in m.sender_name:
                author = f"{m.sender_name} [уже выбыл]"
            else:
                author = f"{m.sender_name}({g})" if g else m.sender_name
            lines.append(f"- {author}: {m.text}")

    if state.phase == Phase.DAY_VOTING and state.current_votes:
        cast_str = ", ".join(f"{voter} -> против {target}" for voter, target in state.current_votes.items())
        lines.append(f"Уже проголосовали сейчас: {cast_str}.")

    gender_hint = (
        "мужской род: я сказал, я уверен"
        if player.gender == Gender.MALE
        else "женский род: я сказала, я уверена"
    )
    shuffled_rivals = _shuffled_names(rivals_alive, seed_key=player.seat * 31 + state.day * 7 + state.round_num)
    angle = _TACTICAL_ANGLES[(player.seat + state.day) % len(_TACTICAL_ANGLES)]

    lines.append(
        f"\nТвоя карточка: ТЫ — {player.name} ({gender_hint}; пиши о себе ТОЛЬКО «я/мне/меня», без имени «{player.name}»!)."
    )
    if shuffled_rivals:
        lines.append(f"Живые соперники: {', '.join(shuffled_rivals)}.")
    lines.append(angle)

    if player.role == Role.CITIZEN:
        lines.append("ТВОЯ СЕКРЕТНАЯ РОЛЬ: МИРНЫЙ ЖИТЕЛЬ. Чужие роли неизвестны, вычисляй двух мафиози по смыслу их слов и голосам.")
    else:
        partner = _get_mafia_partner(player, state.players)
        if partner:
            p_state = "ЖИВ — он за столом" if partner.is_alive else "МЁРТВ — уже выбыл"
            partner_str = f"Твой напарник по мафии: {partner.name} ({p_state})."
        else:
            partner_str = "Ты один в мафии."
        lines.append(
            f"ТВОЯ СЕКРЕТНАЯ РОЛЬ: МАФИЯ. {partner_str} Днём притворяйся рассудительным мирным, не выдавай себя и напарника."
        )
        mafia_msgs = [m for m in state.messages if m.visibility == "mafia"]
        if mafia_msgs:
            lines.append(
                "СЕКРЕТНЫЙ НОЧНОЙ ЧАТ МАФИИ: "
                + " | ".join(f"{m.sender_name}: {m.text}" for m in mafia_msgs[-2:])
            )

    return "\n".join(lines)


def build_discussion_task_prompt(player: Player, state: GameState) -> str:
    context = build_table_state_block(player, state)

    alive_seats = {p.seat for p in state.alive_players()}
    today_discussion = [
        m
        for m in state.messages
        if m.day == state.day
        and m.visibility == "public"
        and m.phase == Phase.DAY_DISCUSSION
        and m.sender_seat in alive_seats
    ]
    spoken_seats = {m.sender_seat for m in today_discussion if m.sender_seat is not None}
    other_alive = [p for p in state.alive_players() if p.seat != player.seat]
    seed = player.seat * 17 + state.day * 13 + state.round_num

    already_spoke = _shuffled_names([p.name for p in other_alive if p.seat in spoken_seats], seed)
    not_yet_turn = _shuffled_names([p.name for p in other_alive if p.seat not in spoken_seats], seed + 1)
    rivals_shuffled = _shuffled_names([p.name for p in other_alive], seed + 2)

    mentions = find_direct_mentions_for_player(player, state)

    if mentions:
        last_m = mentions[-1]
        return (
            f"{context}\n"
            f"ТЕБЯ УПОМЯНУЛ ИГРОК {last_m.sender_name}: «{last_m.text}»\n"
            f"Ответь игроку {last_m.sender_name} строго от ПЕРВОГО лица («я», «мне»): если к тебе вопрос или наезд — ответь по делу ИЛИ прямо откажись отвечать на провокацию; если тебя поддержали — коротко отреагируй (не обвиняй никого в молчании!, 1-2 предложения):"
        )

    if state.day == 1 and state.round_num == 1 and len(today_discussion) == 0:
        rivals_str = ", ".join(rivals_shuffled)
        return (
            f"{context}\n"
            f"Ты открываешь День 1 первым (очередь других ещё впереди!). Поздоровайся по-простому и задай вопрос на «ты» одному игроку из ({rivals_str}), с чего начнём искать мафию (1-2 коротких предложения):"
        )

    if not_yet_turn:
        spoke_block = (
            f"Уже сходили в этом круге (комментировать слова можно ТОЛЬКО у них): {', '.join(already_spoke)}.\n"
            f"Очередь хода ещё впереди (они выступят позже, НЕ оценивай их и НЕ говори что они молчат!): {', '.join(not_yet_turn)}."
        )
    else:
        spoke_block = f"Все живые соперники уже высказались ({', '.join(already_spoke)})."

    if state.round_num == 1:
        if state.day == 1:
            ask_target_pool = ", ".join(not_yet_turn if not_yet_turn else already_spoke)
            spoke_str = ", ".join(already_spoke)
            task = (
                f"Начало 1-го дня — не обвиняй никого в молчании! "
                f"Отреагируй по смыслу на реплику кого-то из ({spoke_str}) ИЛИ задай прямой вопрос на «ты» игроку из ({ask_target_pool})."
            )
        else:
            rivals_str = ", ".join(rivals_shuffled)
            task = (
                f"Обсуди, кому из ЖИВЫХ ({rivals_str}) было выгодно ночное убийство судя по прошлым голосам и их словам (не обвиняй в молчании!), "
                f"и предъяви аргумент кому-то из ({rivals_str})."
            )
    else:
        rivals_str = ", ".join(rivals_shuffled)
        task = (
            f"Второй круг перед голосованием: назови от лица «я», чьи СЛОВА или АРГУМЕНТЫ из ({rivals_str}) кажутся тебе самыми неискренними или подозрительными и почему (без обвинений в молчании!)."
        )

    return f"{context}\n{spoke_block}\n{task} 1-2 коротких предложения:"


def build_last_word_prompt(player: Player, state: GameState, reason: str) -> str:
    context = build_table_state_block(player, state)
    alive_names = ", ".join(p.name for p in state.alive_players())
    how_text = (
        "Тебя только что убили ночью."
        if reason == "night"
        else "Город только что проголосовал за твою казнь."
    )
    role_goal = (
        f"Оставь честное завещание мирным: скажи в 1-2 предложениях от лица «я», чьи СЛОВА или ГОЛОСА среди живых ({alive_names}) кажутся тебе самыми подозрительными и почему (не обвиняй в молчании!)."
        if player.role == Role.CITIZEN
        else f"Ты мафия, но игра в закрытую (твою роль НЕ знают!): притворись убранным мирным и в 1-2 предложениях тонко наведи подозрение на кого-то из живых мирных ({alive_names}) по их словам, не выдавая напарника."
    )
    return f"{context}\nПОСЛЕДНЕЕ СЛОВО ВЫБЫВШЕГО: {how_text} {role_goal} Только сама реплика:"


def build_voting_task_prompt(player: Player, state: GameState, valid_targets: list[str]) -> str:
    context = build_table_state_block(player, state)
    shuffled_targets = _shuffled_names(valid_targets, seed_key=player.seat * 23 + state.day * 11)
    targets_str = ", ".join(shuffled_targets)
    return (
        f"{context}\n"
        f"Голосование на выбывание. Выбери ОДНОГО самого подозрительного тебе ЖИВОГО соперника из [{targets_str}] по его словам и позиции (запрещено голосовать с причиной «молчит / мало говорит»!).\n"
        "Ответь строго в 2 строки, называя в обеих строках ОДНО И ТО ЖЕ имя подозреваемого:\n"
        "ПРИЧИНА: Я голосую против <Имя>, потому что <короткая причина по его словам/аргументам>\n"
        "ГОЛОС: <То же самое Имя>"
    )


def build_night_suggest_prompt(player: Player, partner: Player, state: GameState, valid_targets: list[str]) -> str:
    context = build_table_state_block(player, state)
    targets_str = ", ".join(valid_targets)
    return (
        f"{context}\n"
        f"Ночь: предложи напарнику {partner.name} одну жертву из живых мирных [{targets_str}]. Строго 2 строки:\n"
        "ПРИЧИНА: <1 короткая фраза напарнику с именем жертвы>\n"
        "ЦЕЛЬ: <Точное имя этой жертвы>"
    )


def build_night_kill_prompt(player: Player, state: GameState, valid_targets: list[str]) -> str:
    context = build_table_state_block(player, state)
    targets_str = ", ".join(valid_targets)
    return (
        f"{context}\n"
        f"Ночной выстрел мафии по одному из живых мирных [{targets_str}]. Строго 2 строки:\n"
        "ПРИЧИНА: <1 короткая фраза почему убираем этого игрока>\n"
        "ЦЕЛЬ: <Точное имя этой жертвы>"
    )
