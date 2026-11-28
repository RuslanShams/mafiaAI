from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from collections import Counter
from typing import Awaitable, Callable, Optional

from backend.config import (
    DISCUSSION_ROUNDS,
    MAFIA_COUNT,
    SPEECH_CHARS_PER_SEC,
    SPEECH_READ_HOLD_SEC,
    TOTAL_PLAYERS,
)
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
    _find_public_commissar_claims,
    build_commissar_check_prompt,
    build_discussion_task_prompt,
    build_doctor_heal_prompt,
    build_last_word_prompt,
    build_night_kill_prompt,
    build_night_suggest_prompt,
    build_system_prompt,
    build_voting_task_prompt,
)

logger = logging.getLogger("mafia.engine")

BroadcastFn = Callable[[dict], Awaitable[None]]


class GameEngine:
    """Manages a 7-player closed tabletop Mafia game (1 Human + 6 qwen2.5:7b AIs)."""

    def __init__(
        self,
        llm: OllamaClient,
        broadcast: BroadcastFn,
        spectator_delay: float = 0.8,
        pace_speech: Optional[bool] = None,
    ) -> None:
        self.llm = llm
        self.broadcast = broadcast
        self.spectator_delay = spectator_delay
        self.pace_speech = (spectator_delay > 0) if pace_speech is None else pace_speech
        self.state = GameState()
        self._loop_task: Optional[asyncio.Task] = None
        self._human_event = asyncio.Event()
        self._human_payload: dict = {}
        self._start_seat_offset: int = 0
        self._last_stream_started_at: Optional[float] = None

    async def stop_current_game(self) -> None:
        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        self._loop_task = None
        self.state.waiting_for_human = None
        self.state.awaiting_last_word_seat = None
        self.state.active_speaker_seat = None

    async def reset_to_lobby(self) -> None:
        prev_model = self.state.ai_model or self.llm.model
        await self.stop_current_game()
        self.state = GameState(phase=Phase.LOBBY, ai_model=prev_model)
        await self.push_state()

    def _all_player_names(self) -> list[str]:
        return [p.name for p in self.state.players]

    def _dead_player_names(self, exclude_seat: Optional[int] = None) -> list[str]:
        return [
            p.name
            for p in self.state.players
            if not p.is_alive and p.seat != exclude_seat
        ]

    def _unspoken_alive_names_today(self, exclude_seat: Optional[int] = None) -> list[str]:
        if self.state.phase != Phase.DAY_DISCUSSION:
            return []
        spoken_seats = {
            m.sender_seat
            for m in self.state.messages
            if m.day == self.state.day
            and m.visibility == "public"
            and m.phase == Phase.DAY_DISCUSSION
            and m.sender_seat is not None
        }
        return [
            p.name
            for p in self.state.alive_players()
            if p.seat != exclude_seat and p.seat not in spoken_seats
        ]

    async def _ai_generate(
        self,
        player: Player,
        system_prompt: str,
        user_prompt: str,
        valid_targets: Optional[list[str]] = None,
        is_public_day: bool = True,
        fallback_kind: str = "discussion",
    ) -> dict[str, str]:
        self._last_stream_started_at = None
        last_stream_len = 0

        async def _stream_partial(partial_text: str) -> None:
            nonlocal last_stream_len
            if self._last_stream_started_at is None or len(partial_text) < last_stream_len:
                self._last_stream_started_at = time.monotonic()
            last_stream_len = len(partial_text)
            await self.broadcast(
                {
                    "type": "stream_chunk",
                    "seat": player.seat,
                    "sender_name": player.name,
                    "day": self.state.day,
                    "text": partial_text,
                }
            )

        return await self.llm.generate_turn(
            player=player,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            valid_targets=valid_targets,
            is_public_day=is_public_day,
            fallback_kind=fallback_kind,
            allowed_names=self._all_player_names(),
            dead_names=self._dead_player_names(exclude_seat=player.seat),
            unspoken_names=self._unspoken_alive_names_today(exclude_seat=player.seat)
            if fallback_kind == "discussion"
            else None,
            on_stream_text=_stream_partial if (is_public_day and valid_targets is None) else None,
            model=self.state.ai_model,
        )

    def serialize_for_human(self) -> dict:
        human = self.state.human_player()
        # Reveal all roles either when game ends OR once human is eliminated AND done with their Last Word!
        human_is_spectator = (
            human is not None
            and not human.is_alive
            and self.state.awaiting_last_word_seat != human.seat
            and self.state.waiting_for_human != "last_word"
        )
        reveal_all = self.state.phase == Phase.GAME_OVER or human_is_spectator
        # Never show "Думает..." on any AI player during Night so Mafia is never exposed
        visible_active_seat = (
            self.state.active_speaker_seat
            if (
                self.state.phase != Phase.NIGHT
                or (human is not None and self.state.active_speaker_seat == human.seat)
            )
            else None
        )

        visible_messages = []
        for msg in self.state.messages:
            if self.state.phase == Phase.GAME_OVER:
                visible_messages.append(msg.to_client_dict(include_thought=True))
            elif reveal_all:
                visible_messages.append(msg.to_client_dict(include_thought=False))
            elif msg.visibility in ("public", "system"):
                visible_messages.append(msg.to_client_dict(include_thought=False))
            elif msg.visibility == "mafia" and human and human.role == Role.MAFIA:
                visible_messages.append(msg.to_client_dict(include_thought=False))
            elif msg.visibility == "commissar" and human and human.role == Role.COMMISSAR:
                visible_messages.append(msg.to_client_dict(include_thought=False))
            elif msg.visibility == "doctor" and human and human.role == Role.DOCTOR:
                visible_messages.append(msg.to_client_dict(include_thought=False))

        return {
            "phase": self.state.phase.value,
            "phase_ru": self.state.phase.label_ru,
            "day": self.state.day,
            "round_num": self.state.round_num,
            "ai_model": self.state.ai_model,
            "spectator_reveal": human_is_spectator,
            "players": [p.to_public_dict(viewer=human, reveal_all=reveal_all) for p in self.state.players],
            "messages": visible_messages,
            "winner": self.state.winner.value if self.state.winner else None,
            "winner_ru": self.state.winner.label_ru if self.state.winner else None,
            "win_reason": self.state.win_reason,
            "active_speaker_seat": visible_active_seat,
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

        chosen_model = (ai_model or self.state.ai_model or self.llm.model).strip() or self.llm.model
        clean_name = (human_name or "Игрок").strip()[:20] or "Игрок"
        h_gender = Gender.FEMALE if human_gender == "F" else Gender.MALE

        # Randomize roles: 2 Mafia, 1 Commissar, 1 Doctor, 3 Citizens across 7 seats
        roles = (
            [Role.MAFIA] * MAFIA_COUNT
            + [Role.COMMISSAR, Role.DOCTOR]
            + [Role.CITIZEN] * (TOTAL_PLAYERS - MAFIA_COUNT - 2)
        )
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
                ai_name, ai_gender = next(ai_profiles)
                players.append(
                    Player(
                        seat=seat,
                        name=ai_name,
                        gender=ai_gender,
                        is_human=False,
                        role=role,
                        persona=None,
                    )
                )

        self.state = GameState(
            phase=Phase.NIGHT,
            day=1,
            round_num=1,
            ai_model=chosen_model,
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
            f"Партия началась! За столом 7 игроков: 2 мафии, 1 комиссар, 1 доктор и 3 мирных (модель ИИ: {chosen_model}). "
            "Игра идёт в закрытую, а в первую ночь мафия никого не убивает."
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
        elif action_type == "last_word":
            # Empty text is allowed if human clicks "Skip last word"
            self._human_payload = {"text": text.strip()[:500]}
        elif action_type in ("vote", "night_kill", "night_heal", "night_check"):
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

    def _is_victory_reached(self) -> bool:
        mafia_alive = len(self.state.alive_mafia())
        citizens_alive = len(self.state.alive_citizens())
        return mafia_alive == 0 or mafia_alive >= citizens_alive

    async def _run_last_word(self, player: Player, reason: str) -> None:
        """Give an eliminated player (night victim or day execution) their Last Word if game continues."""
        if self._is_victory_reached():
            self.state.awaiting_last_word_seat = None
            return

        prev_phase = self.state.phase
        if reason == "night":
            self.state.phase = Phase.DAY_DISCUSSION
        self.state.awaiting_last_word_seat = player.seat
        self.state.active_speaker_seat = player.seat
        self._add_system_message(f"Последнее слово выбывающего игрока {player.name}...")

        if player.is_human:
            # _wait_for_human sets waiting_for_human="last_word" BEFORE pushing state so input bar opens immediately!
            payload = await self._wait_for_human("last_word")
            text = payload.get("text", "").strip()
            if text:
                self._add_Message_direct(
                    sender_seat=player.seat,
                    sender_name=f"{player.name} (Последнее слово)",
                    text=text,
                    visibility="public",
                )
            else:
                self._add_system_message(f"{player.name} покидает стол без последнего слова.")
        else:
            await self.push_state()
            sys_prompt = build_system_prompt(player, self.state)
            user_prompt = build_last_word_prompt(player, self.state, reason=reason)
            res = await self._ai_generate(
                player=player,
                system_prompt=sys_prompt,
                user_prompt=user_prompt,
                is_public_day=True,
                fallback_kind="last_word",
            )
            self._add_Message_direct(
                sender_seat=player.seat,
                sender_name=f"{player.name} (Последнее слово)",
                text=res["speech"],
                visibility="public",
                thought=res["thought"],
            )
            self.state.active_speaker_seat = None
            await self.push_state()
            await self._pace_public_ai_speech(res["speech"], was_streamed=True)

        self.state.awaiting_last_word_seat = None
        self.state.active_speaker_seat = None
        self.state.phase = prev_phase
        await self.push_state()

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

    def _compute_speech_pace_delay(
        self,
        speech: str,
        stream_started_at: Optional[float] = None,
        now: Optional[float] = None,
    ) -> float:
        """
        Calculate how long to keep an AI speech bubble on screen so the frontend
        typewriter (~22 chars/sec) finishes typing and the human has time to read it.
        """
        if not self.pace_speech:
            return 0.0
        cleaned = speech.strip()
        if not cleaned:
            return 0.0
        total_type_sec = len(cleaned) / max(5.0, SPEECH_CHARS_PER_SEC)
        already_elapsed = 0.0
        if stream_started_at is not None:
            ref_now = time.monotonic() if now is None else now
            already_elapsed = max(0.0, ref_now - stream_started_at)
        remaining_type_sec = max(0.0, total_type_sec - already_elapsed)
        return remaining_type_sec + SPEECH_READ_HOLD_SEC

    async def _pace_public_ai_speech(self, speech: str, was_streamed: bool) -> None:
        delay = self._compute_speech_pace_delay(
            speech,
            stream_started_at=self._last_stream_started_at if was_streamed else None,
        )
        if delay > 0:
            await asyncio.sleep(delay)
        else:
            await self._pace_if_spectating()

    async def _pace_if_spectating(self) -> None:
        human = self.state.human_player()
        if human and not human.is_alive and self.spectator_delay > 0:
            await asyncio.sleep(self.spectator_delay)

    async def _run_game_loop(self) -> None:
        try:
            self.state.day = 0
            while self.state.phase != Phase.GAME_OVER:
                self.state.day += 1
                day_rec = DayRecord(day=self.state.day)
                self.state.history.append(day_rec)

                # Night Phase (Night 1 has NO kill — acquaintance night only!)
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

    async def _run_night_phase(self, day_rec: DayRecord) -> None:
        self.state.phase = Phase.NIGHT
        self.state.round_num = 1
        self.state.mafia_suggestion = None
        self.state.current_votes = {}

        # Night 1: Non-lethal acquaintance night!
        if self.state.day == 1:
            alive_mafia = self.state.alive_mafia()
            if len(alive_mafia) == 2:
                m1, m2 = alive_mafia[0], alive_mafia[1]
                sender = m2 if m1.is_human else m1
                other = m1 if m1.is_human else m2
                self._add_Message_direct(
                    sender_seat=sender.seat,
                    sender_name=f"{sender.name} (Мафия)",
                    text=f"Работаем в паре с {other.name}. Первая ночь без выстрела — днём ведём себя естественно и не выдаём связку.",
                    visibility="mafia",
                )
            self._add_system_message(
                "Первая ночь прошла мирно: мафия познакомилась друг с другом без выстрела. "
                "Наступает День 1 — все 7 игроков за столом!"
            )
            await self.push_state()
            return

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
            # AI-only Mafia night: 1 fast decision call (rotating lead mafia between nights)
            shooter = ai_mafia[(self.state.day - 1) % len(ai_mafia)]
            self.state.active_speaker_seat = shooter.seat
            await self.push_state()
            res = await self._ai_generate(
                player=shooter,
                system_prompt=build_system_prompt(shooter, self.state),
                user_prompt=build_night_kill_prompt(shooter, self.state, valid_targets),
                valid_targets=valid_targets,
                is_public_day=False,
                fallback_kind="night",
            )
            chosen_target_name = res["target"]
            self._add_Message_direct(
                sender_seat=shooter.seat,
                sender_name=f"{shooter.name} (Мафия)",
                text=f"{res['speech']} [Выстрел в: {chosen_target_name}]",
                visibility="mafia",
                thought=res["thought"],
            )

        self.state.active_speaker_seat = None
        self.state.mafia_suggestion = None

        # 2. Doctor night action (cannot heal the same player two nights in a row)
        healed_player: Optional[Player] = None
        doctor = self.state.alive_role(Role.DOCTOR)
        if doctor:
            heal_candidates = [
                p.name
                for p in self.state.alive_players()
                if p.seat != self.state.last_doctor_heal_seat
            ]
            if not heal_candidates:
                heal_candidates = [p.name for p in self.state.alive_players()]
            if doctor.is_human:
                self.state.active_speaker_seat = doctor.seat
                payload = await self._wait_for_human("night_heal", valid_targets=heal_candidates)
                healed_name = payload["target"]
            else:
                res_doc = await self._ai_generate(
                    player=doctor,
                    system_prompt=build_system_prompt(doctor, self.state),
                    user_prompt=build_doctor_heal_prompt(doctor, self.state, heal_candidates),
                    valid_targets=heal_candidates,
                    is_public_day=False,
                    fallback_kind="night_target",
                )
                healed_name = res_doc["target"]
            healed_player = self.state.find_by_name(healed_name or "")
            if healed_player:
                self.state.last_doctor_heal_seat = healed_player.seat
                self._add_Message_direct(
                    sender_seat=doctor.seat,
                    sender_name=f"{doctor.name} (Доктор)",
                    text=f"Ночное лечение: под защитой на эту ночь {healed_player.name}.",
                    visibility="doctor",
                )
            self.state.active_speaker_seat = None

        # 3. Commissar night action (checks one alive unchecked player)
        commissar = self.state.alive_role(Role.COMMISSAR)
        if commissar:
            check_candidates = [
                p.name
                for p in self.state.alive_players()
                if p.seat != commissar.seat and p.name not in self.state.commissar_checks
            ]
            if not check_candidates:
                check_candidates = [p.name for p in self.state.alive_players() if p.seat != commissar.seat]
            if check_candidates:
                if commissar.is_human:
                    self.state.active_speaker_seat = commissar.seat
                    payload = await self._wait_for_human("night_check", valid_targets=check_candidates)
                    checked_name = payload["target"]
                else:
                    res_com = await self._ai_generate(
                        player=commissar,
                        system_prompt=build_system_prompt(commissar, self.state),
                        user_prompt=build_commissar_check_prompt(commissar, self.state, check_candidates),
                        valid_targets=check_candidates,
                        is_public_day=False,
                        fallback_kind="night_target",
                    )
                    checked_name = res_com["target"]
                checked_player = self.state.find_by_name(checked_name or "")
                if checked_player:
                    verdict = "Мафия" if checked_player.role == Role.MAFIA else "Мирный"
                    self.state.commissar_checks[checked_player.name] = verdict
                    self._add_Message_direct(
                        sender_seat=commissar.seat,
                        sender_name=f"{commissar.name} (Комиссар)",
                        text=f"Результат проверки: {checked_player.name} — {verdict}!",
                        visibility="commissar",
                    )
                self.state.active_speaker_seat = None

        # 4. Resolve night kill vs Doctor's save
        victim = self.state.find_by_name(chosen_target_name or "")
        if victim and victim.is_alive:
            if healed_player and healed_player.seat == victim.seat:
                day_rec.night_saved = True
                self._add_system_message(
                    f"Наступает утро Дня {self.state.day}. Этой ночью мафия стреляла, "
                    "но Доктор вовремя спас жертву — все игроки остались живы!"
                )
                await self.push_state()
            else:
                victim.is_alive = False
                victim.death_reason = "night"
                victim.death_day = self.state.day
                day_rec.night_victim = victim.name
                killed_verb = "убит" if victim.gender == Gender.MALE else "убита"
                self._add_system_message(
                    f"Наступает утро Дня {self.state.day}. Этой ночью {killed_verb} {victim.name}. "
                    "По правилам закрытой игры роль не раскрывается."
                )
                if not self._is_victory_reached():
                    self.state.awaiting_last_word_seat = victim.seat
                    await self._run_last_word(victim, reason="night")
                else:
                    await self.push_state()
        else:
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
                    self.state.active_speaker_seat = None
                    await self.push_state()
                    if self.pace_speech:
                        await asyncio.sleep(0.8)
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
                    self.state.active_speaker_seat = None
                    await self.push_state()
                    await self._pace_public_ai_speech(res["speech"], was_streamed=True)

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
                reason_text = payload.get("text") or f"Мой голос — {target_name}."
                self._add_Message_direct(
                    sender_seat=player.seat,
                    sender_name=player.name,
                    text=reason_text,
                    visibility="public",
                    vote_target=target_name,
                )
            else:
                ai_vote_targets = list(valid_targets)
                # AI Commissar NEVER ignores known Mafia or votes out verified Citizens!
                if player.role == Role.COMMISSAR and self.state.commissar_checks:
                    known_mafia = [
                        t for t in ai_vote_targets if self.state.commissar_checks.get(t) == "Мафия"
                    ]
                    if known_mafia:
                        ai_vote_targets = known_mafia
                    else:
                        unchecked_or_susp = [
                            t for t in ai_vote_targets if self.state.commissar_checks.get(t) != "Мирный"
                        ]
                        if unchecked_or_susp:
                            ai_vote_targets = unchecked_or_susp
                # AI Mafia avoids voting against its own alive Mafia partner when Town targets exist
                elif player.role == Role.MAFIA:
                    non_mafia_targets = [
                        p.name
                        for p in self.state.alive_players()
                        if p.seat != player.seat and p.role != Role.MAFIA
                    ]
                    if non_mafia_targets:
                        ai_vote_targets = non_mafia_targets
                # Town AI does not vote to execute the sole uncontested revealed Commissar!
                elif player.role.is_town:
                    alive_claims = [
                        n for n, _txt, alive in _find_public_commissar_claims(self.state) if alive
                    ]
                    if len(alive_claims) == 1:
                        without_sole_com = [t for t in ai_vote_targets if t != alive_claims[0]]
                        if without_sole_com:
                            ai_vote_targets = without_sole_com

                sys_prompt = build_system_prompt(player, self.state)
                user_prompt = build_voting_task_prompt(player, self.state, ai_vote_targets)
                res = await self._ai_generate(
                    player=player,
                    system_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    valid_targets=ai_vote_targets,
                    is_public_day=True,
                    fallback_kind="vote",
                )
                target_name = res["target"]
                self.state.current_votes[player.name] = target_name
                day_rec.votes[player.name] = target_name
                self._add_Message_direct(
                    sender_seat=player.seat,
                    sender_name=player.name,
                    text=res["speech"],
                    visibility="public",
                    thought=res["thought"],
                    vote_target=target_name,
                )
                self.state.active_speaker_seat = None
                await self.push_state()
                await self._pace_public_ai_speech(res["speech"], was_streamed=False)
                continue

            self.state.current_votes[player.name] = target_name
            day_rec.votes[player.name] = target_name
            self.state.active_speaker_seat = None
            await self.push_state()
            if self.pace_speech:
                await asyncio.sleep(0.8)

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
                if not self._is_victory_reached():
                    self.state.awaiting_last_word_seat = executed_player.seat
                    await self._run_last_word(executed_player, reason="vote")
                else:
                    await self.push_state()
                return

        await self.push_state()
