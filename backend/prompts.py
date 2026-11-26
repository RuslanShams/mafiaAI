from __future__ import annotations

from typing import Optional

from backend.models import GameState, Gender, Phase, Player, Role


def _get_mafia_partner(player: Player, all_players: list[Player]) -> Optional[Player]:
    if player.role != Role.MAFIA:
        return None
    for p in all_players:
        if p.role == Role.MAFIA and p.seat != player.seat:
            return p
    return None


def build_system_prompt(player: Player, state: GameState) -> str:
    """Build strictly isolated system prompt for a single AI player."""
    gender_instruction = (
        "Твой пол — МУЖСКОЙ. Всегда говори о себе строго в мужском роде (я подумал, я заметил, я уверен, согласен)."
        if player.gender == Gender.MALE
        else "Твой пол — ЖЕНСКИЙ. Всегда говори о себе строго в женском роде (я подумала, я заметила, я уверена, согласна)."
    )

    persona_title = player.persona.title if player.persona else "Обычный игрок"
    persona_style = player.persona.prompt_style if player.persona else "Говори естественно и по делу."

    if player.role == Role.CITIZEN:
        role_block = (
            "ТВОЯ СЕКРЕТНАЯ РОЛЬ: МИРНЫЙ ЖИТЕЛЬ.\n"
            "- Ты НЕ знаешь, кто из остальных 6 игроков мафия, а кто мирный.\n"
            "- Твоя цель — вычислить двух игроков мафии по нестыковкам в их речах, поведению и голосованиям, "
            "и убедить стол проголосовать против них.\n"
            "- Защищай себя, если тебя несправедливо подозревают, но не верь никому слепо."
        )
    else:
        partner = _get_mafia_partner(player, state.players)
        if partner:
            partner_status = "сейчас ЖИВ(А)" if partner.is_alive else "УЖЕ ВЫБЫЛ(А) из игры"
            partner_line = f"Твой единственный напарник по мафии: {partner.name} ({partner_status})."
        else:
            partner_line = "Ты единственный представитель мафии."
        role_block = (
            "ТВОЯ СЕКРЕТНАЯ РОЛЬ: МАФИЯ.\n"
            f"- {partner_line} Все остальные игроки за столом — мирные жители.\n"
            "- СТРОЖАЙШИЙ ЗАПРЕТ ДНЁМ: никогда не признавайся за общим столом, что ты или твой напарник — мафия! "
            "Днём ты обязан(а) убедительно отыгрывать роль мирного жителя, который якобы ищет мафию.\n"
            "- Тактика мафии днём: переводи подозрения на убедительных кандидатов среди мирных, подхватывай чужие ошибки, "
            "не защищай напарника слишком очевидно, чтобы вас не вычислили парой. Если напарника прижали фактами — "
            "действуй осторожно, чтобы не утонуть вместе с ним.\n"
            "- Победа мафии наступает, когда число живых мафиози сравняется с числом живых мирных."
        )

    return (
        f"Ты — живой человек по имени {player.name}, играющий за столом в классическую настольную игру «Мафия».\n"
        f"{gender_instruction}\n\n"
        f"ХАРАКТЕР И МАНЕРА РЕЧИ ({persona_title}):\n"
        f"{persona_style}\n\n"
        "ПРАВИЛА ЭТОЙ ПАРТИИ:\n"
        "- Всего в начале игры было 7 игроков: 2 Мафии и 5 Мирных жителей (комиссара и доктора нет).\n"
        "- ИГРА В ЗАКРЫТУЮ: когда игрока убивают ночью или казнят днём на голосовании, ведущий НЕ объявляет его роль! "
        "Роли выбывших игроков остаются тайной до самого конца игры. Не утверждай как доказанный факт роль выбывшего, "
        "если ты её не знаешь наверняка.\n\n"
        f"{role_block}\n\n"
        "ЖЕСТКИЕ ТРЕБОВАНИЯ К ЯЗЫКУ И ФОРМАТУ:\n"
        "1. Пиши ИСКЛЮЧИТЕЛЬНО на чистом, живом русском языке. Никаких китайских иероглифов, английских слов (кроме ключей JSON) и эмодзи.\n"
        "2. Говори ТОЛЬКО в контексте партии за игровым столом. Запрещено упоминать ИИ, нейросети, языковые модели, промпты или правила генерации.\n"
        "3. Не выдумывай реплики или голосования, которых не было в логе игры. Обращайся только к реально существующим за столом игрокам.\n"
        "4. Возвращай ответ СТРОГО в формате валидного JSON-объекта без обёрток и комментариев вне JSON."
    )


def build_table_state_block(player: Player, state: GameState) -> str:
    """Build factual game summary with zero leakage of hidden roles."""
    alive_list = []
    dead_list = []
    for p in state.players:
        g = "М" if p.gender == Gender.MALE else "Ж"
        self_mark = " (ЭТО ТЫ)" if p.seat == player.seat else ""
        if p.is_alive:
            alive_list.append(f"- [{p.seat}] {p.name} ({g}){self_mark}")
        else:
            how = "убит(а) ночью" if p.death_reason == "night" else "казнён(а) на голосовании"
            dead_list.append(f"- [{p.seat}] {p.name} ({g}) — ВЫБЫЛ(А): {how} (день {p.death_day}, роль не раскрыта)")

    lines = [
        f"ТЕКУЩИЙ МОМЕНТ: День {state.day}, фаза: {state.phase.label_ru}.",
        f"В живых осталось: {len(state.alive_players())} чел.",
        "ЖИВЫЕ ИГРОКИ ЗА СТОЛОМ:",
        *alive_list,
    ]
    if dead_list:
        lines.append("\nВЫБЫВШИЕ ИГРОКИ (не могут говорить, за них нельзя голосовать):")
        lines.extend(dead_list)

    # Past days factual archive
    past_records = [r for r in state.history if r.day < state.day or (r.day == state.day and r.night_victim)]
    if past_records:
        lines.append("\nФАКТЫ ПРОШЛЫХ РАУНДОВ:")
        for rec in past_records:
            parts = [f"День {rec.day}:"]
            if rec.night_victim:
                parts.append(f"ночью убит(а) {rec.night_victim}.")
            if rec.votes:
                votes_str = ", ".join(f"{voter} -> {target}" for voter, target in rec.votes.items())
                parts.append(f"Голоса днём: [{votes_str}].")
            if rec.executed:
                parts.append(f"Казнён(а): {rec.executed} (роль неизвестна).")
            elif rec.tie_no_execution:
                parts.append("Ничья на голосовании — никто не казнён.")
            lines.append(" ".join(parts))

    # Private mafia night log ONLY for mafia players
    if player.role == Role.MAFIA:
        mafia_msgs = [m for m in state.messages if m.visibility == "mafia"]
        if mafia_msgs:
            lines.append("\nСЕКРЕТНЫЙ НОЧНОЙ ЧАТ МАФИИ (известен только мафии):")
            for m in mafia_msgs[-8:]:
                lines.append(f"[Ночь {m.day}] {m.sender_name}: {m.text}")

    # Public speeches of the current day (or Day 0 intro if Day 1)
    if state.day == 0:
        recent_public = [m for m in state.messages if m.visibility in ("public", "system")]
    else:
        day0_msgs = [m for m in state.messages if m.day == 0 and m.visibility == "public"]
        today_msgs = [m for m in state.messages if m.day == state.day and m.visibility in ("public", "system")]
        # Include Day 0 intro on Day 1 so players remember initial impressions
        recent_public = (day0_msgs + today_msgs) if state.day == 1 else today_msgs

    if recent_public:
        lines.append("\nСТЕНОГРАММА ТЕКУЩЕГО ОБСУЖДЕНИЯ ЗА СТОЛОМ:")
        for m in recent_public[-24:]:
            if m.visibility == "system":
                lines.append(f"[Ведущий]: {m.text}")
            else:
                vote_suffix = f" [ГОЛОС ПРОТИВ: {m.vote_target}]" if m.vote_target else ""
                lines.append(f"{m.sender_name}: «{m.text}»{vote_suffix}")

    return "\n".join(lines)


def build_intro_task_prompt(player: Player, state: GameState) -> str:
    context = build_table_state_block(player, state)
    return (
        f"{context}\n\n"
        "ЗАДАНИЕ (Знакомство перед первой ночью):\n"
        f"Сейчас нулевой круг знакомства. Скажи от лица игрока {player.name} короткую вступительную реплику (1-2 предложения) "
        "в своём характере: поприветствуй стол, покажи свой настрой на игру. Ночи и убийств ещё не было, поэтому никого предметно обвинять рано.\n"
        "Верни СТРОГО JSON:\n"
        '{"thought": "твоя короткая скрытая мысль в 1 предложении", "speech": "твоя реплика вслух на русском языке (1-2 предложения)"}'
    )


def build_discussion_task_prompt(player: Player, state: GameState) -> str:
    context = build_table_state_block(player, state)
    other_alive = [p.name for p in state.alive_players() if p.seat != player.seat]
    others_str = ", ".join(other_alive)

    if state.round_num == 1:
        round_hint = (
            "Сейчас 1-й круг обсуждения этого дня. Проанализируй ночное убийство и прошлые события, "
            f"выскажи свои подозрения или наблюдения относительно живых игроков ({others_str}) и при желании задай прямой вопрос кому-то из них."
        )
    else:
        round_hint = (
            "Сейчас 2-й (заключительный перед голосованием) круг обсуждения. "
            "Обязательно отреагируй на реплики из 1-го круга (особенно если кто-то упомянул или обвинил тебя!), "
            f"и чётко обозначь, кого из живых ({others_str}) ты сейчас считаешь главным кандидатом на голосование и почему."
        )

    return (
        f"{context}\n\n"
        f"ЗАДАНИЕ (Дневная речь игрока {player.name}, круг {state.round_num} из 2):\n"
        f"{round_hint}\n"
        "Длина реплики `speech`: 2-3 живых, естественных предложения на русском языке без повторения чужих слов слово в слово.\n"
        "Верни СТРОГО JSON:\n"
        '{"thought": "скрытая тактическая мысль (1-2 предложения, её никто не услышит)", '
        '"speech": "твоя речь вслух за столом на русском языке (2-3 предложения)"}'
    )


def build_voting_task_prompt(player: Player, state: GameState, valid_targets: list[str]) -> str:
    context = build_table_state_block(player, state)
    targets_str = ", ".join(valid_targets)
    return (
        f"{context}\n\n"
        f"ЗАДАНИЕ (Голосование игрока {player.name}):\n"
        "Обсуждение окончено, началось открытое голосование на выбывание.\n"
        f"Ты ОБЯЗАН(А) проголосовать ровно против ОДНОГО живого игрока из этого списка: [{targets_str}].\n"
        "Голосовать против себя или выбывших нельзя.\n"
        "Верни СТРОГО JSON:\n"
        '{"thought": "почему мне выгодно проголосовать именно так", '
        f'"target": "Точное_Имя_из_списка ({targets_str})", '
        '"speech": "короткое обоснование голоса вслух (1 предложение на русском языке)"}'
    )


def build_night_suggest_prompt(player: Player, partner: Player, state: GameState, valid_targets: list[str]) -> str:
    context = build_table_state_block(player, state)
    targets_str = ", ".join(valid_targets)
    return (
        f"{context}\n\n"
        f"ЗАДАНИЕ (Ночное совещание мафии — {player.name} шепчет напарнику {partner.name}):\n"
        "Город спит. Тебе нужно предложить напарнику одну жертву на эту ночь из списка живых мирных жителей: "
        f"[{targets_str}]. Выбирай того, кто опасен для мафии или чья смерть запутает стол днём.\n"
        "Верни СТРОГО JSON:\n"
        '{"thought": "тактический расчёт выбора жертвы", '
        f'"target": "Точное_Имя_из_списка ({targets_str})", '
        '"speech": "короткая фраза шёпотом напарнику на русском языке (1 предложение: кого убираем и почему)"}'
    )


def build_night_kill_prompt(player: Player, state: GameState, valid_targets: list[str]) -> str:
    context = build_table_state_block(player, state)
    targets_str = ", ".join(valid_targets)
    return (
        f"{context}\n\n"
        f"ЗАДАНИЕ (Ночной выстрел мафии — решение принимает {player.name}):\n"
        f"Город спит. Укажи финальную жертву ночного убийства мафии строго из списка живых мирных: [{targets_str}].\n"
        "Учти предложение напарника, если оно было в ночном чате.\n"
        "Верни СТРОГО JSON:\n"
        '{"thought": "почему убираем именно этого игрока", '
        f'"target": "Точное_Имя_из_списка ({targets_str})", '
        '"speech": "короткая реплика шёпотом в ночной чат на русском языке (1 предложение)"}'
    )

