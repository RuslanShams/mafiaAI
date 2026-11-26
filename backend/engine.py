from __future__ import annotations

import asyncio
import logging
import random
import uuid
from collections import Counter
from typing import Awaitable, Callable, Optional

from backend.config import DISCUSSION_ROUNDS, MAFIA_COUNT, TOTAL_PLAYERS
from backend.llm_client import OllamaClient
from backend.models import (
    ChatMessage,
    DayRecord,
    GameState,
    Gender,
    Phase,
    Player,
    Role,
)
from backend.personas import generate_ai_profiles
from backend.prompts import (
    build_discussion_task_prompt,
    build_intro_task_prompt,
    build_night_kill_prompt,
    build_night_suggest_prompt,
    build_system_prompt,
    build_voting_task_prompt,
)

logger = logging.getLogger("mafia.engine")

BroadcastFn = Callable[[dict], Awaitable[None]]


class GameEngine:
    """Manages a 7-player closed tabletop Mafia game (1 Human + 6 qwen2.5:7b AIs)."""

    def __init__(self, llm: OllamaClient, broadcast: BroadcastFn) -> None:
        self.llm = llm
        self.broadcast = broadcast
        self.state = GameState()
        self._loop_task: Optional[asyncio.Task] = None
        self._human_event = asyncio.Event()
        self._human_payload: dict = {}
        self._start_seat_offset: int = 0

    async def stop_current_game(self) -> None:
        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        self._loop_task = None
        self.state.waiting_for_human = None
        self.state.active_speaker_seat = None

    async def reset_to_lobby(self) -> None:
        await self.stop_current_game()
        self.state = GameState(phase=Phase.LOBBY, ai_model=self.llm.model)
        await self.push_state()

    def _all_player_names(self) -> list[str]:
        return [p.name for p in self.state.players]

    async def _ai_generate(
        self,
        player: Player,
        system_prompt: str,
        user_prompt: str,
        valid_targets: Optional[list[str]] = None,
        is_public_day: bool = True,
        fallback_kind: str = "discussion",
    ) -> dict[str, str]:
        return await self.llm.generate_turn(
            player=player,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            valid_targets=valid_targets,
            is_public_day=is_public_day,
            fallback_kind=fallback_kind,
            allowed_names=self._all_player_names(),
        )

    def serialize_for_human(self) -> dict:
        human = self.state.human_player()
        reveal_all = self.state.phase == Phase.GAME_OVER

        visible_messages = []
        for msg in self.state.messages:
            if reveal_all:
                visible_messages.append(msg.to_client_dict(include_thought=True))
            elif msg.visibility in ("public", "system"):
                visible_messages.append(msg.to_client_dict(include_thought=False))
            elif msg.visibility == "mafia" and human and human.role == Role.MAFIA:
                visible_messages.append(msg.to_client_dict(include_thought=False))

        return {
            "phase": self.state.phase.value,
            "phase_ru": self.state.phase.label_ru,
            "day": self.state.day,
            "round_num": self.state.round_num,
            "ai_model": self.state.ai_model,
            "players": [p.to_public_dict(viewer=human, reveal_all=reveal_all) for p in self.state.players],
            "messages": visible_messages,
            "winner": self.state.winner.value if self.state.winner else None,
            "winner_ru": self.state.winner.label_ru if self.state.winner else None,
            "win_reason": self.state.win_reason,
            "active_speaker_seat": self.state.active_speaker_seat,
            "waiting_for_human": self.state.waiting_for_human,
            "valid_targets": self.state.valid_targets,
            "mafia_suggestion": self.state.mafia_suggestion,
            "current_votes": self.state.current_votes,
        }

    async def push_state(self) -> None:
        await self.broadcast({"type": "state", "state": self.serialize_for_human()})

    async def start_new_game(
        self,
        human_name: str,
        human_gender: str,
        ai_model: Optional[str] = None,
    ) -> None:
        await self.stop_current_game()

        if ai_model and ai_model.strip():
            self.llm.set_model(ai_model.strip())

        clean_name = (human_name or "Игрок").strip()[:20] or "Игрок"
        h_gender = Gender.FEMALE if human_gender == "F" else Gender.MALE

        # Randomize roles: 2 Mafia, 5 Citizens across 7 seats
        roles = [Role.MAFIA] * MAFIA_COUNT + [Role.CITIZEN] * (TOTAL_PLAYERS - MAFIA_COUNT)
        random.shuffle(roles)

        # Randomize which seat (1..7) the human occupies so turn order varies naturally
        human_seat = random.randint(1, TOTAL_PLAYERS)
        ai_profiles = iter(generate_ai_profiles(count=TOTAL_PLAYERS - 1, forbidden_name=clean_name))

        players: list[Player] = []
        for seat in range(1, TOTAL_PLAYERS + 1):
            role = roles[seat - 1]
            if seat == human_seat:
                players.append(
                    Player(
                        seat=seat,
                        name=clean_name,
                        gender=h_gender,
                        is_human=True,
                        role=role,
                        persona=None,
                    )
                )
            else:
                ai_name, ai_gender, ai_persona = next(ai_profiles)
                players.append(
                    Player(
                        seat=seat,
                        name=ai_name,
                        gender=ai_gender,
                        is_human=False,
                        role=role,
                        persona=ai_persona,
                    )
                )

        self.state = GameState(
            phase=Phase.DAY_INTRO,
            day=0,
            round_num=1,
            ai_model=self.llm.model,
            players=players,
        )
        self._start_seat_offset = random.randint(0, TOTAL_PLAYERS - 1)

        human = self.state.human_player()
        partner_info = ""
        if human and human.role == Role.MAFIA:
            partners = [p.name for p in players if p.role == Role.MAFIA and not p.is_human]
            if partners:
                partner_info = f" Ваш напарник по мафии — {partners[0]}."

        self._add_system_message(
            f"Партия началась! За столом 7 игроков: 5 мирных жителей и 2 мафии (модель ИИ: {self.llm.model}). "
            "Игра идёт в закрытую — роли выбывших не раскрываются до конца партии."
        )
        if human:
            self._add_Message_direct(
                sender_seat=None,
                sender_name="Система",
                text=f"Ваше место №{human.seat}. Ваша роль: {human.role.label_ru}.{partner_info}",
                visibility="system",
            )

        await self.push_state()
        self._loop_task = asyncio.create_task(self._run_game_loop())

    def submit_human_input(self, action_type: str, text: str = "", target: str = "") -> bool:
        if self.state.waiting_for_human != action_type:
            return False
        if action_type == "speak":
            cleaned = text.strip()
            if not cleaned:
                return False
            self._human_payload = {"text": cleaned[:500]}
        elif action_type in ("vote", "night_kill"):
            if target not in self.state.valid_targets:
                return False
            self._human_payload = {"target": target, "text": text.strip()[:250]}
        else:
            return False

        self._human_event.set()
        return True

    def _ordered_alive_players(self) -> list[Player]:
        alive = self.state.alive_players()
        if not alive:
            return []
        shift = (self._start_seat_offset + self.state.day) % len(alive)
        return alive[shift:] + alive[:shift]

    def _add_Message_direct(
        self,
        sender_seat: Optional[int],
        sender_name: str,
        text: str,
        visibility: str = "public",
        thought: Optional[str] = None,
        vote_target: Optional[str] = None,
    ) -> ChatMessage:
        msg = ChatMessage(
            id=uuid.uuid4().hex[:10],
            day=self.state.day,
            phase=self.state.phase,
            round_num=self.state.round_num,
            sender_seat=sender_seat,
            sender_name=sender_name,
            text=text,
            visibility=visibility,
            thought=thought,
            vote_target=vote_target,
        )
        self.state.messages.append(msg)
        return msg

    def _add_system_message(self, text: str) -> None:
        self._add_Message_direct(sender_seat=None, sender_name="Ведущий", text=text, visibility="system")

    async def _wait_for_human(self, mode: str, valid_targets: Optional[list[str]] = None) -> dict:
        self._human_event.clear()
        self._human_payload = {}
        self.state.waiting_for_human = mode
        self.state.valid_targets = valid_targets or []
        await self.push_state()
        await self._human_event.wait()
        self.state.waiting_for_human = None
        self.state.valid_targets = []
        return dict(self._human_payload)

    def _check_victory(self) -> bool:
        mafia_alive = len(self.state.alive_mafia())
        citizens_alive = len(self.state.alive_citizens())

        if mafia_alive == 0:
            self.state.phase = Phase.GAME_OVER
            self.state.winner = Role.CITIZEN
            self.state.win_reason = "Оба игрока мафии вычислены и устранены. Мирный город победил!"
            self._add_system_message(f"ИГРА ОКОНЧЕНА! {self.state.win_reason}")
            return True

        if mafia_alive >= citizens_alive:
            self.state.phase = Phase.GAME_OVER
            self.state.winner = Role.MAFIA
            self.state.win_reason = (
                f"Мафия сравнялась по численности с мирными ({mafia_alive} против {citizens_alive}) "
                "и захватила контроль над столом. Победа Мафии!"
            )
            self._add_system_message(f"ИГРА ОКОНЧЕНА! {self.state.win_reason}")
            return True

        return False

    async def _pace_if_spectating(self) -> None:
        human = self.state.human_player()
        if human and not human.is_alive:
            await asyncio.sleep(0.8)

    async def _run_game_loop(self) -> None:
        try:
            # 1. Day 0: Short acquaintance round
            await self._run_intro_round()

            # 2. Main Night -> Day Discussion -> Day Voting cycle
            while self.state.phase != Phase.GAME_OVER:
                self.state.day += 1
                day_rec = DayRecord(day=self.state.day)
                self.state.history.append(day_rec)

                # Night Phase
                await self._run_night_phase(day_rec)
                if self._check_victory():
                    self.state.active_speaker_seat = None
                    await self.push_state()
                    break

                # Day Discussion Phase (2 rounds)
                await self._run_day_discussion()

                # Day Voting Phase
                await self._run_day_voting(day_rec)
                if self._check_victory():
                    self.state.active_speaker_seat = None
                    await self.push_state()
                    break

        except asyncio.CancelledError:
            logger.info("Game loop cancelled.")
            raise
        except Exception as exc:
            logger.exception("Unexpected error in game loop: %s", exc)
            self._add_system_message(f"Произошла техническая ошибка в движке партии: {exc}")
            await self.push_state()

    async def _run_intro_round(self) -> None:
        self.state.phase = Phase.DAY_INTRO
        self.state.day = 0
        self.state.round_num = 1
        self._add_system_message("Нулевой круг — короткое знакомство за столом перед первой ночью.")
        await self.push_state()

        for player in self._ordered_alive_players():
            self.state.active_speaker_seat = player.seat
            await self.push_state()

            if player.is_human:
                payload = await self._wait_for_human("speak")
                self._add_Message_direct(
                    sender_seat=player.seat,
                    sender_name=player.name,
                    text=payload["text"],
                    visibility="public",
                )
            else:
                sys_prompt = build_system_prompt(player, self.state)
                user_prompt = build_intro_task_prompt(player, self.state)
                res = await self._ai_generate(
                    player=player,
                    system_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    is_public_day=True,
                    fallback_kind="intro",
                )
                self._add_Message_direct(
                    sender_seat=player.seat,
                    sender_name=player.name,
                    text=res["speech"],
                    visibility="public",
                    thought=res["thought"],
                )
            await self.push_state()

        self.state.active_speaker_seat = None
        await self.push_state()

    async def _run_night_phase(self, day_rec: DayRecord) -> None:
        self.state.phase = Phase.NIGHT
        self.state.round_num = 1
        self.state.mafia_suggestion = None
        self.state.current_votes = {}
        self._add_system_message(f"Наступает Ночь {self.state.day}. Город засыпает, просыпается Мафия...")
        await self.push_state()

        alive_mafia = self.state.alive_mafia()
        valid_targets = [p.name for p in self.state.alive_citizens()]
        if not alive_mafia or not valid_targets:
            return

        human_mafia = next((m for m in alive_mafia if m.is_human), None)
        ai_mafia = [m for m in alive_mafia if not m.is_human]

        chosen_target_name: Optional[str] = None

        if human_mafia:
            # If AI mafia partner is alive, let AI partner whisper a suggestion first
            if ai_mafia:
                partner = ai_mafia[0]
                self.state.active_speaker_seat = partner.seat
                await self.push_state()
                sys_prompt = build_system_prompt(partner, self.state)
                user_prompt = build_night_suggest_prompt(partner, human_mafia, self.state, valid_targets)
                res = await self._ai_generate(
                    player=partner,
                    system_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    valid_targets=valid_targets,
                    is_public_day=False,
                    fallback_kind="night",
                )
                self.state.mafia_suggestion = res["target"]
                self._add_Message_direct(
                    sender_seat=partner.seat,
                    sender_name=f"{partner.name} (Мафия)",
                    text=f"{res['speech']} [Предлагает цель: {res['target']}]",
                    visibility="mafia",
                    thought=res["thought"],
                )

            self.state.active_speaker_seat = human_mafia.seat
            payload = await self._wait_for_human("night_kill", valid_targets=valid_targets)
            chosen_target_name = payload["target"]
            note = payload.get("text") or f"Убираем игрока {chosen_target_name}."
            self._add_Message_direct(
                sender_seat=human_mafia.seat,
                sender_name=f"{human_mafia.name} (Мафия)",
                text=f"{note} [Выстрел в: {chosen_target_name}]",
                visibility="mafia",
            )
        else:
            # AI-only Mafia night
            if len(ai_mafia) == 2:
                m1, m2 = ai_mafia[0], ai_mafia[1]
                self.state.active_speaker_seat = m1.seat
                await self.push_state()
                res1 = await self._ai_generate(
                    player=m1,
                    system_prompt=build_system_prompt(m1, self.state),
                    user_prompt=build_night_suggest_prompt(m1, m2, self.state, valid_targets),
                    valid_targets=valid_targets,
                    is_public_day=False,
                    fallback_kind="night",
                )
                self._add_Message_direct(
                    sender_seat=m1.seat,
                    sender_name=f"{m1.name} (Мафия)",
                    text=f"{res1['speech']} [Предлагает: {res1['target']}]",
                    visibility="mafia",
                    thought=res1["thought"],
                )

                self.state.active_speaker_seat = m2.seat
                await self.push_state()
                res2 = await self._ai_generate(
                    player=m2,
                    system_prompt=build_system_prompt(m2, self.state),
                    user_prompt=build_night_kill_prompt(m2, self.state, valid_targets),
                    valid_targets=valid_targets,
                    is_public_day=False,
                    fallback_kind="night",
                )
                chosen_target_name = res2["target"]
                self._add_Message_direct(
                    sender_seat=m2.seat,
                    sender_name=f"{m2.name} (Мафия)",
                    text=f"{res2['speech']} [Решено убрать: {chosen_target_name}]",
                    visibility="mafia",
                    thought=res2["thought"],
                )
            else:
                sole = ai_mafia[0]
                self.state.active_speaker_seat = sole.seat
                await self.push_state()
                res = await self._ai_generate(
                    player=sole,
                    system_prompt=build_system_prompt(sole, self.state),
                    user_prompt=build_night_kill_prompt(sole, self.state, valid_targets),
                    valid_targets=valid_targets,
                    is_public_day=False,
                    fallback_kind="night",
                )
                chosen_target_name = res["target"]
                self._add_Message_direct(
                    sender_seat=sole.seat,
                    sender_name=f"{sole.name} (Мафия)",
                    text=f"{res['speech']} [Выстрел в: {chosen_target_name}]",
                    visibility="mafia",
                    thought=res["thought"],
                )

        self.state.active_speaker_seat = None
        self.state.mafia_suggestion = None

        victim = self.state.find_by_name(chosen_target_name or "")
        if victim and victim.is_alive:
            victim.is_alive = False
            victim.death_reason = "night"
            victim.death_day = self.state.day
            day_rec.night_victim = victim.name
            killed_verb = "убит" if victim.gender == Gender.MALE else "убита"
            self._add_system_message(
                f"Наступает утро Дня {self.state.day}. Этой ночью {killed_verb} {victim.name}. "
                "По правилам закрытой игры роль не раскрывается."
            )
        await self.push_state()

    async def _run_day_discussion(self) -> None:
        self.state.phase = Phase.DAY_DISCUSSION
        speakers = self._ordered_alive_players()

        for r in range(1, DISCUSSION_ROUNDS + 1):
            self.state.round_num = r
            round_label = (
                f"День {self.state.day} — Круг обсуждения 1 из {DISCUSSION_ROUNDS} (основные версии)."
                if r == 1
                else f"День {self.state.day} — Круг обсуждения 2 из {DISCUSSION_ROUNDS} (ответы и прения перед голосованием)."
            )
            self._add_system_message(round_label)
            await self.push_state()

            for player in speakers:
                if not player.is_alive:
                    continue
                self.state.active_speaker_seat = player.seat
                await self.push_state()

                if player.is_human:
                    payload = await self._wait_for_human("speak")
                    self._add_Message_direct(
                        sender_seat=player.seat,
                        sender_name=player.name,
                        text=payload["text"],
                        visibility="public",
                    )
                else:
                    sys_prompt = build_system_prompt(player, self.state)
                    user_prompt = build_discussion_task_prompt(player, self.state)
                    res = await self._ai_generate(
                        player=player,
                        system_prompt=sys_prompt,
                        user_prompt=user_prompt,
                        is_public_day=True,
                        fallback_kind="discussion",
                    )
                    self._add_Message_direct(
                        sender_seat=player.seat,
                        sender_name=player.name,
                        text=res["speech"],
                        visibility="public",
                        thought=res["thought"],
                    )
                    await self._pace_if_spectating()

                await self.push_state()

        self.state.active_speaker_seat = None
        await self.push_state()

    async def _run_day_voting(self, day_rec: DayRecord) -> None:
        self.state.phase = Phase.DAY_VOTING
        self.state.round_num = 1
        self.state.current_votes = {}
        self._add_system_message(
            f"Обсуждение Дня {self.state.day} завершено! Начинается открытое голосование на выбывание."
        )
        await self.push_state()

        voters = self._ordered_alive_players()
        for player in voters:
            if not player.is_alive:
                continue
            valid_targets = [p.name for p in self.state.alive_players() if p.seat != player.seat]
            self.state.active_speaker_seat = player.seat
            await self.push_state()

            if player.is_human:
                payload = await self._wait_for_human("vote", valid_targets=valid_targets)
                target_name = payload["target"]
                reason_text = payload.get("text") or f"Голосую против {target_name}."
                self._add_Message_direct(
                    sender_seat=player.seat,
                    sender_name=player.name,
                    text=reason_text,
                    visibility="public",
                    vote_target=target_name,
                )
            else:
                sys_prompt = build_system_prompt(player, self.state)
                user_prompt = build_voting_task_prompt(player, self.state, valid_targets)
                res = await self._ai_generate(
                    player=player,
                    system_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    valid_targets=valid_targets,
                    is_public_day=True,
                    fallback_kind="vote",
                )
                target_name = res["target"]
                self._add_Message_direct(
                    sender_seat=player.seat,
                    sender_name=player.name,
                    text=res["speech"],
                    visibility="public",
                    thought=res["thought"],
                    vote_target=target_name,
                )
                await self._pace_if_spectating()

            self.state.current_votes[player.name] = target_name
            day_rec.votes[player.name] = target_name
            await self.push_state()

        self.state.active_speaker_seat = None

        # Tally votes
        tally = Counter(day_rec.votes.values())
        if not tally:
            return
        max_votes = max(tally.values())
        leaders = [name for name, cnt in tally.items() if cnt == max_votes]

        tally_summary = ", ".join(f"{name}: {cnt}" for name, cnt in tally.most_common())

        if len(leaders) > 1:
            day_rec.tie_no_execution = True
            tied_str = ", ".join(leaders)
            self._add_system_message(
                f"Итоги голосования ({tally_summary}). Равенство голосов между: {tied_str}. "
                "Город не пришёл к большинству — сегодня никто не казнён!"
            )
        else:
            executed_name = leaders[0]
            executed_player = self.state.find_by_name(executed_name)
            if executed_player:
                executed_player.is_alive = False
                executed_player.death_reason = "vote"
                executed_player.death_day = self.state.day
                day_rec.executed = executed_player.name
                left_verb = "покинул" if executed_player.gender == Gender.MALE else "покинула"
                self._add_system_message(
                    f"Итоги голосования ({tally_summary}). Большинством голосов стол {left_verb} {executed_player.name}. "
                    "В закрытой игре истинная роль остаётся в тайне."
                )

        await self.push_state()
