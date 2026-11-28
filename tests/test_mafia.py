from __future__ import annotations

import asyncio
import unittest
from typing import Optional

from fastapi.testclient import TestClient

from backend.app import app
from backend.engine import GameEngine
from backend.llm_client import (
    parse_target_and_speech,
    resolve_target_name,
    soft_sanitize_speech,
    validate_russian_speech,
)
from backend.models import ChatMessage, GameState, Gender, Phase, Player, Role
from backend.personas import generate_ai_profiles
from backend.prompts import (
    build_discussion_task_prompt,
    build_doctor_heal_prompt,
    build_system_prompt,
    build_table_state_block,
    build_voting_task_prompt,
    find_direct_mentions_for_player,
)
from backend.session_manager import SessionManager


class FakeOllamaClient:
    """Deterministic Russian mock LLM that verifies prompt isolation on every AI call."""

    def __init__(self) -> None:
        self.model: str = "gemma3:4b"
        self.calls: list[dict] = []

    def set_model(self, model_name: str) -> None:
        if model_name.strip():
            self.model = model_name.strip()

    async def generate_turn(
        self,
        player: Player,
        system_prompt: str,
        user_prompt: str,
        valid_targets: Optional[list[str]] = None,
        is_public_day: bool = True,
        fallback_kind: str = "discussion",
        allowed_names: Optional[list[str]] = None,
        dead_names: Optional[list[str]] = None,
        unspoken_names: Optional[list[str]] = None,
        on_stream_text=None,
        model: Optional[str] = None,
    ) -> dict[str, str]:
        self.calls.append(
            {
                "player": player,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "valid_targets": valid_targets,
                "is_public_day": is_public_day,
                "fallback_kind": fallback_kind,
            }
        )
        full_prompt = f"{system_prompt}\n{user_prompt}"
        if player.role == Role.CITIZEN:
            assert "ТВОЯ СЕКРЕТНАЯ РОЛЬ: МИРНЫЙ ЖИТЕЛЬ" in full_prompt
            assert "СЕКРЕТНЫЙ НОЧНОЙ ЧАТ МАФИИ" not in full_prompt
            assert "Твой напарник по мафии:" not in full_prompt
        elif player.role == Role.COMMISSAR:
            assert "ТВОЯ СЕКРЕТНАЯ РОЛЬ: КОМИССАР" in full_prompt
            assert "СЕКРЕТНЫЙ НОЧНОЙ ЧАТ МАФИИ" not in full_prompt
        elif player.role == Role.DOCTOR:
            assert "ТВОЯ СЕКРЕТНАЯ РОЛЬ: ДОКТОР" in full_prompt
            assert "СЕКРЕТНЫЙ НОЧНОЙ ЧАТ МАФИИ" not in full_prompt
        else:
            assert "ТВОЯ СЕКРЕТНАЯ РОЛЬ: МАФИЯ" in full_prompt

        target = valid_targets[0] if valid_targets else ""
        if fallback_kind == "night_target":
            return {"thought": "Ночной выбор", "speech": "", "target": target}
        speech = (
            "Давайте спокойно послушаем всех за столом."
            if not target
            else f"Голосую против {target}, его позиция мне непонятна."
        )
        if on_stream_text is not None and valid_targets is None:
            await on_stream_text(speech[:12])
            await on_stream_text(speech)

        err = validate_russian_speech(
            speech,
            player=player,
            is_public_day=is_public_day,
            allowed_names=allowed_names,
            dead_names=dead_names,
            unspoken_names=unspoken_names,
        )
        assert err is None, f"Mock speech failed validation: {err}"

        res = {"thought": "Тестовый ход", "speech": speech}
        if valid_targets is not None:
            res["target"] = target
        return res


class MafiaGameTests(unittest.TestCase):
    def test_profile_randomization_unique_names_and_mixed_genders(self) -> None:
        batch = generate_ai_profiles(6, forbidden_name="Руслан")
        self.assertEqual(len(batch), 6)
        self.assertEqual(len({p[0] for p in batch}), 6)
        self.assertNotIn("Руслан", {p[0] for p in batch})
        genders = {p[1] for p in batch}
        self.assertIn(Gender.MALE, genders)
        self.assertIn(Gender.FEMALE, genders)

    def test_direct_question_detection_russian_cases(self) -> None:
        players = [
            Player(1, "Руслан", Gender.MALE, True, Role.CITIZEN),
            Player(2, "Кирилл", Gender.MALE, False, Role.MAFIA),
            Player(3, "Алексей", Gender.MALE, False, Role.CITIZEN),
            Player(4, "Ольга", Gender.FEMALE, False, Role.CITIZEN),
        ]
        state = GameState(phase=Phase.DAY_DISCUSSION, day=1, round_num=1, players=players)
        state.messages.append(
            ChatMessage("m1", 1, Phase.DAY_DISCUSSION, 1, 1, "Руслан", "Кирилл какой-то немногословный, может ты мафия?", "public")
        )
        state.messages.append(
            ChatMessage("m2", 1, Phase.DAY_DISCUSSION, 1, 1, "Руслан", "Вопрос к Алексею и Ольге: кого подозреваете?", "public")
        )

        kirill_mentions = find_direct_mentions_for_player(players[1], state)
        self.assertEqual(len(kirill_mentions), 1)
        prompt_kirill = build_discussion_task_prompt(players[1], state)
        self.assertIn("ТЕБЯ УПОМЯНУЛ ИГРОК Руслан", prompt_kirill)
        self.assertIn("откажись отвечать", prompt_kirill)

        # Once Kirill speaks, old mention before his turn is cleared
        state.messages.append(
            ChatMessage("m3", 1, Phase.DAY_DISCUSSION, 1, 2, "Кирилл", "Руслан, не буду отвечать на провокации.", "public")
        )
        self.assertEqual(len(find_direct_mentions_for_player(players[1], state)), 0)

        # Declensions Алексею / Ольге detected
        self.assertEqual(len(find_direct_mentions_for_player(players[2], state)), 1)
        self.assertEqual(len(find_direct_mentions_for_player(players[3], state)), 1)

    def test_russian_validator_and_sanitizer(self) -> None:
        mafia_p = Player(seat=2, name="Виктор", gender=Gender.MALE, is_human=False, role=Role.MAFIA)

        self.assertIsNone(
            validate_russian_speech("Друзья, мне кажется странным поведение Елены.", mafia_p, is_public_day=True)
        )
        self.assertIsNotNone(
            validate_russian_speech("Я думаю это Елена 觉得很奇怪.", mafia_p, is_public_day=True)
        )
        self.assertIsNotNone(
            validate_russian_speech("Я думаю Елена выглядит очень suspicious сегодня.", mafia_p, is_public_day=True)
        )
        self.assertIsNone(
            validate_russian_speech(
                "Мне кажется, что Alex сегодня уходит от прямых вопросов.",
                mafia_p,
                is_public_day=True,
                allowed_names=["Alex", "Виктор"],
            )
        )
        self.assertIsNotNone(
            validate_russian_speech("Ладно, признаюсь, я мафия!", mafia_p, is_public_day=True)
        )
        self.assertEqual(resolve_target_name("[3] Ольгу", ["Виктор", "Ольга", "Сергей"]), "Ольга")

        t, s = parse_target_and_speech(
            "ПРИЧИНА: Она слишком резко сменила позицию.\nГОЛОС: Ольга",
            ["Виктор", "Ольга", "Сергей"],
        )
        self.assertEqual(t, "Ольга")
        self.assertEqual(s, "Она слишком резко сменила позицию.")

        # Screenshot bug #4: Denis says Svetlana is right & Nikolay raises questions, but target tag said Ruslan -> reconciles to Nikolay!
        t_denis, _ = parse_target_and_speech(
            "ЦЕЛЬ: Руслан\nРЕЧЬ: Я думаю, что Светлана права. Николай действительно вызывает вопросы.",
            ["Татьяна", "Руслан", "Николай", "Наталья", "Дарья", "Светлана"],
            speaker_name="Денис",
        )
        self.assertEqual(t_denis, "Николай")

        # Screenshot bug #5: Nikolay says "голосуем за Дарью. [голос: Светлана]" -> reconciles to Дарья and strips [голос:...]!
        t_nik, s_nik = parse_target_and_speech(
            "ГОЛОС: Светлана\nПРИЧИНА: Я думаю, что голосуем за Дарью. [голос: Светлана] — Она слишком много говорит.",
            ["Денис", "Татьяна", "Руслан", "Наталья", "Дарья", "Светлана"],
            speaker_name="Николай",
        )
        self.assertEqual(t_nik, "Дарья")
        self.assertNotIn("[голос", s_nik)

        # Screenshot bug #1 & #3: 3rd-person self-reference rejected; self-address stripped; typo prefix stripped
        denis_p = Player(seat=1, name="Денис", gender=Gender.MALE, is_human=False, role=Role.CITIZEN)
        self.assertIsNotNone(
            validate_russian_speech("Денис сейчас вроде бы спокойный, я не знаю как он настроен.", denis_p, is_public_day=True)
        )
        sveta_clean = soft_sanitize_speech("Светлана, ну я как раз пытаюсь понять кто тут мафия.", speaker_name="Светлана")
        self.assertEqual(sveta_clean, "Ну я как раз пытаюсь понять кто тут мафия.")
        tanya_clean = soft_sanitize_speech("Тетяна: Ну что тут говорить... [голос: Денис] — он слишком спокоен.", speaker_name="Татьяна")
        self.assertNotIn("Тетяна", tanya_clean)
        self.assertNotIn("[голос", tanya_clean)

        cleaned = soft_sanitize_speech(
            "Виктор: Задаю вопрос Кириллу: почему темнишь ok. А ещё я хотел сказ",
            speaker_name="Виктор",
        )
        self.assertEqual(cleaned, "Кириллу, почему темнишь.")

        # Dead player suspicion/addressing rejected, while past-tense mention allowed
        sergey_p = Player(seat=7, name="Сергей", gender=Gender.MALE, is_human=False, role=Role.CITIZEN)
        self.assertIsNotNone(
            validate_russian_speech(
                "Мне кажется, Павел переживает и пытается скрыть что-то.",
                sergey_p,
                is_public_day=True,
                dead_names=["Павел", "Руслан"],
            )
        )
        self.assertIsNotNone(
            validate_russian_speech(
                "Я думаю, что Сергей прав насчет Павла, он пытается скрыть роль.",
                Player(seat=1, name="Наталья", gender=Gender.FEMALE, is_human=False, role=Role.CITIZEN),
                is_public_day=True,
                dead_names=["Павел"],
            )
        )
        self.assertIsNone(
            validate_russian_speech(
                "Вчера Павел активно спорил с Дарьей — кому из живых мешал Павел?",
                sergey_p,
                is_public_day=True,
                dead_names=["Павел"],
            )
        )

        # Unspoken player hallucination rejected (e.g. accusing Roman of talking a lot before he spoke!)
        darya_p = Player(seat=6, name="Дарья", gender=Gender.FEMALE, is_human=False, role=Role.MAFIA)
        self.assertIsNotNone(
            validate_russian_speech(
                "Я думаю, что Роман слишком активно пытается привлечь внимание и много говорит.",
                darya_p,
                is_public_day=True,
                unspoken_names=["Роман", "Елена"],
            )
        )
        self.assertIsNotNone(
            validate_russian_speech(
                "Дарья, ты права насчёт Романа — он действительно кажется взволнованным.",
                Player(seat=7, name="Алина", gender=Gender.FEMALE, is_human=False, role=Role.CITIZEN),
                is_public_day=True,
                unspoken_names=["Роман"],
            )
        )
        # Asking an unspoken player a simple question IS allowed
        self.assertIsNone(
            validate_russian_speech(
                "Анна бодро начала разговор. Роман, а какие у тебя первые мысли за столом?",
                darya_p,
                is_public_day=True,
                unspoken_names=["Роман"],
            )
        )

        # Silence accusations globally banned because speaking is strictly sequential!
        pavel_p = Player(seat=6, name="Павел", gender=Gender.MALE, is_human=False, role=Role.CITIZEN)
        self.assertIsNotNone(
            validate_russian_speech(
                "Я думаю, стоит обратить внимание на Николая. Он сегодня молчит, а это странно.",
                pavel_p,
                is_public_day=True,
            )
        )
        self.assertIsNotNone(
            validate_russian_speech(
                "Его молчаливость и то, что он мало говорит, вызывают подозрения.",
                pavel_p,
                is_public_day=True,
            )
        )

    def test_full_closed_game_no_night1_kill_and_role_isolation(self) -> None:
        async def run_sim() -> None:
            events: list[dict] = []

            async def collect(msg: dict) -> None:
                events.append(msg)

            fake_llm = FakeOllamaClient()
            engine = GameEngine(llm=fake_llm, broadcast=collect, spectator_delay=0.0)  # type: ignore[arg-type]

            await engine.start_new_game(human_name="Руслан", human_gender="M", ai_model="gemma3:4b")
            self.assertEqual(engine.state.ai_model, "gemma3:4b")
            self.assertEqual(len(engine.state.players), 7)
            self.assertEqual(sum(1 for p in engine.state.players if p.role == Role.MAFIA), 2)
            self.assertEqual(sum(1 for p in engine.state.players if p.role == Role.COMMISSAR), 1)
            self.assertEqual(sum(1 for p in engine.state.players if p.role == Role.DOCTOR), 1)
            self.assertEqual(sum(1 for p in engine.state.players if p.role == Role.CITIZEN), 3)

            checked_day1_all_alive = False
            steps = 0
            while engine.state.phase != Phase.GAME_OVER and steps < 250:
                steps += 1
                await asyncio.sleep(0.005)

                if engine.state.day == 1 and engine.state.phase == Phase.DAY_DISCUSSION:
                    # Night 1 must NEVER kill anyone! All 7 players must be alive on Day 1!
                    self.assertEqual(len(engine.state.alive_players()), 7)
                    self.assertIsNone(engine.state.history[0].night_victim)
                    checked_day1_all_alive = True

                if engine.state.phase != Phase.GAME_OVER:
                    snap = engine.serialize_for_human()
                    human = engine.state.human_player()
                    assert human is not None
                    # During NIGHT, AI active_speaker_seat must NEVER be exposed!
                    if engine.state.phase == Phase.NIGHT and engine.state.active_speaker_seat != human.seat:
                        self.assertIsNone(snap["active_speaker_seat"])
                    for pd in snap["players"]:
                        if human.is_alive and human.role.is_town and not pd["is_human"]:
                            self.assertIsNone(pd["role"])
                            self.assertEqual(pd["role_ru"], "Неизвестно")
                        # If human is dead and finished Last Word, ALL roles MUST be visible to human spectator!
                        if not human.is_alive and engine.state.waiting_for_human != "last_word":
                            self.assertIsNotNone(pd["role"])

                mode = engine.state.waiting_for_human
                if mode == "speak":
                    engine.submit_human_input("speak", text="Я мирный, давайте смотреть по голосам.")
                elif mode == "last_word":
                    # Before submitting last_word, roles must still be hidden for Town human
                    human = engine.state.human_player()
                    if human and human.role.is_town:
                        pre_snap = engine.serialize_for_human()
                        others = [x for x in pre_snap["players"] if not x["is_human"]]
                        self.assertTrue(all(x["role"] is None for x in others))
                    engine.submit_human_input("last_word", text="Город, присмотритесь к аргументам!")
                elif mode in ("vote", "night_kill", "night_heal", "night_check"):
                    target = engine.state.valid_targets[0]
                    engine.submit_human_input(mode, target=target, text=f"Выбираю {target}.")

            self.assertTrue(checked_day1_all_alive)
            self.assertEqual(engine.state.phase, Phase.GAME_OVER)
            self.assertIn(engine.state.winner, (Role.CITIZEN, Role.MAFIA))
            # Night 2 MUST either have a victim OR a Doctor save if game reached Day 2
            if len(engine.state.history) >= 2:
                self.assertTrue(
                    engine.state.history[1].night_victim is not None
                    or engine.state.history[1].night_saved
                )

            final_snap = engine.serialize_for_human()
            for pd in final_snap["players"]:
                self.assertIsNotNone(pd["role"])

            citizen_ai = next(p for p in engine.state.players if not p.is_human and p.role == Role.CITIZEN)
            table_block = build_table_state_block(citizen_ai, engine.state)
            self.assertIn("МЁРТВЫЕ", table_block)
            sys_block = build_system_prompt(citizen_ai, engine.state)
            self.assertIn("ИГРА В ЗАКРЫТУЮ", sys_block)

        asyncio.run(run_sim())

    def test_commissar_uses_mafia_check_and_town_reacts_to_reveal(self) -> None:
        players = [
            Player(1, "Ольга", Gender.FEMALE, False, Role.MAFIA),
            Player(2, "Анна", Gender.FEMALE, False, Role.CITIZEN),
            Player(3, "Полина", Gender.FEMALE, False, Role.DOCTOR),
            Player(4, "Сергей", Gender.MALE, False, Role.COMMISSAR),
            Player(5, "Татьяна", Gender.FEMALE, False, Role.MAFIA),
            Player(6, "Руслан", Gender.MALE, True, Role.CITIZEN, is_alive=False, death_reason="night", death_day=2),
            Player(7, "Дмитрий", Gender.MALE, False, Role.CITIZEN, is_alive=False, death_reason="vote", death_day=1),
        ]
        state = GameState(
            phase=Phase.DAY_DISCUSSION,
            day=2,
            round_num=1,
            players=players,
            commissar_checks={"Ольга": "Мафия"},
        )
        # Even if Polina argued with Sergey, Sergey (Commissar) MUST prioritize revealing Olga = Mafia on 5-player table!
        state.messages.append(
            ChatMessage("1", 2, Phase.DAY_DISCUSSION, 1, 3, "Полина", "Сергей, ты пытаешься меня запутать!", "public")
        )
        sergey = players[3]
        disc_prompt = build_discussion_task_prompt(sergey, state)
        self.assertIn("ВАЖНЕЙШИЙ ХОД КОМИССАРА", disc_prompt)
        self.assertIn("Ольга — МАФИЯ", disc_prompt)

        vote_prompt = build_voting_task_prompt(sergey, state, ["Ольга"])
        self.assertIn("ГОЛОС: Ольга", vote_prompt)

        # Once Sergey publicly reveals as Commissar, Town & Doctor prompts recognize his uncontested claim!
        state.messages.append(
            ChatMessage("2", 2, Phase.DAY_DISCUSSION, 1, 4, "Сергей", "Я Комиссар! Ночью я проверил Ольгу — она мафия!", "public")
        )
        polina = players[2]
        polina_vote_prompt = build_voting_task_prompt(polina, state, ["Ольга", "Анна", "Сергей", "Татьяна"])
        self.assertIn("единственный вскрывшийся Комиссар", polina_vote_prompt)

        doc_night_prompt = build_doctor_heal_prompt(polina, state, ["Ольга", "Анна", "Полина", "Сергей", "Татьяна"])
        self.assertIn("Сергей вскрылся Комиссаром", doc_night_prompt)

    def test_multi_session_isolation_and_max_4_limit(self) -> None:
        async def run_multi() -> None:
            fake_llm = FakeOllamaClient()
            mgr = SessionManager(llm=fake_llm, max_sessions=4)  # type: ignore[arg-type]

            sessions = [mgr.get_or_create(f"session_num_{i}") for i in range(1, 6)]
            # Start 4 independent games simultaneously
            for i in range(4):
                ok, err = await mgr.can_start_game(sessions[i])
                self.assertTrue(ok, err)
                await sessions[i].engine.start_new_game(
                    human_name=f"Игрок{i+1}", human_gender="M", ai_model="gemma3:12b"
                )

            self.assertEqual(mgr.slots_summary()["active_games"], 4)
            self.assertEqual(mgr.slots_summary()["free_slots"], 0)

            # Verify sessions do not leak players into one another
            names_s1 = {p.name for p in sessions[0].engine.state.players if p.is_human}
            names_s2 = {p.name for p in sessions[1].engine.state.players if p.is_human}
            self.assertEqual(names_s1, {"Игрок1"})
            self.assertEqual(names_s2, {"Игрок2"})

            # 5th concurrent game must be rejected while 4 tables are playing
            ok5, err5 = await mgr.can_start_game(sessions[4])
            self.assertFalse(ok5)
            self.assertIn("4", err5 or "")

            # Once session 1 resets to lobby, a slot frees up and session 5 can start
            await sessions[0].engine.reset_to_lobby()
            self.assertEqual(mgr.slots_summary()["active_games"], 3)
            ok5_after, _ = await mgr.can_start_game(sessions[4])
            self.assertTrue(ok5_after)

            await mgr.shutdown_all()

        asyncio.run(run_multi())

    def test_speech_pacing_delay_calculation(self) -> None:
        async def noop(_msg: dict) -> None:
            return None

        fake_llm = FakeOllamaClient()
        paced_engine = GameEngine(llm=fake_llm, broadcast=noop, spectator_delay=0.8)  # type: ignore[arg-type]
        unpaced_engine = GameEngine(llm=fake_llm, broadcast=noop, spectator_delay=0.0)  # type: ignore[arg-type]

        sample_speech = "А" * 110  # ~5.0s typing at 22 chars/sec + 2.2s reading hold = ~7.2s
        self.assertEqual(unpaced_engine._compute_speech_pace_delay(sample_speech), 0.0)

        # Fast GPU generation (0.5s elapsed since first streamed token): waits remaining ~4.5s + 2.2s hold
        gpu_delay = paced_engine._compute_speech_pace_delay(
            sample_speech, stream_started_at=100.0, now=100.5
        )
        self.assertGreater(gpu_delay, 6.0)

        # Slow CPU generation (9.0s elapsed since first streamed token): typing already done, waits only reading hold
        cpu_delay = paced_engine._compute_speech_pace_delay(
            sample_speech, stream_started_at=100.0, now=109.0
        )
        self.assertAlmostEqual(cpu_delay, 2.2, places=1)

    def test_http_and_websocket_endpoints(self) -> None:
        client = TestClient(app)
        res = client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("no-store", res.headers.get("cache-control", ""))
        self.assertIn("МАФИЯ", res.text)
        js_res = client.get("/static/app.js")
        self.assertEqual(js_res.status_code, 200)
        self.assertIn("last_word", js_res.text)
        self.assertIn("no-store", js_res.headers.get("cache-control", ""))

        with client.websocket_connect("/ws/game?session_id=test_client_01") as ws:
            sess_msg = ws.receive_json()
            self.assertEqual(sess_msg["type"], "session")
            self.assertEqual(sess_msg["session_id"], "test_client_01")
            self.assertEqual(sess_msg["slots"]["max_sessions"], 4)
            first = ws.receive_json()
            self.assertEqual(first["type"], "state")
            self.assertEqual(first["state"]["phase"], "lobby")
            second = ws.receive_json()
            self.assertEqual(second["type"], "health")
            self.assertIn("models_detail", second["health"])


if __name__ == "__main__":
    unittest.main()
