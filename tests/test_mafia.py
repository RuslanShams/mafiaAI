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
    build_system_prompt,
    build_table_state_block,
    find_direct_mentions_for_player,
)


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
        else:
            assert "ТВОЯ СЕКРЕТНАЯ РОЛЬ: МАФИЯ" in full_prompt

        target = valid_targets[0] if valid_targets else ""
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
                        if human.is_alive and human.role == Role.CITIZEN and not pd["is_human"]:
                            self.assertIsNone(pd["role"])
                            self.assertEqual(pd["role_ru"], "Неизвестно")
                        # If human is dead and finished Last Word, ALL roles MUST be visible to human spectator!
                        if not human.is_alive and engine.state.waiting_for_human != "last_word":
                            self.assertIsNotNone(pd["role"])

                mode = engine.state.waiting_for_human
                if mode == "speak":
                    engine.submit_human_input("speak", text="Я мирный, давайте смотреть по голосам.")
                elif mode == "last_word":
                    # Before submitting last_word, roles must still be hidden for Citizen human
                    human = engine.state.human_player()
                    if human and human.role == Role.CITIZEN:
                        pre_snap = engine.serialize_for_human()
                        others = [x for x in pre_snap["players"] if not x["is_human"]]
                        self.assertTrue(all(x["role"] is None for x in others))
                    engine.submit_human_input("last_word", text="Город, присмотритесь к молчунам!")
                elif mode in ("vote", "night_kill"):
                    target = engine.state.valid_targets[0]
                    engine.submit_human_input(mode, target=target, text=f"Голосую против {target}.")

            self.assertTrue(checked_day1_all_alive)
            self.assertEqual(engine.state.phase, Phase.GAME_OVER)
            self.assertIn(engine.state.winner, (Role.CITIZEN, Role.MAFIA))
            # Night 2 MUST have a victim if game reached Day 2
            if len(engine.state.history) >= 2:
                self.assertIsNotNone(engine.state.history[1].night_victim)

            final_snap = engine.serialize_for_human()
            for pd in final_snap["players"]:
                self.assertIsNotNone(pd["role"])

            citizen_ai = next(p for p in engine.state.players if not p.is_human and p.role == Role.CITIZEN)
            table_block = build_table_state_block(citizen_ai, engine.state)
            self.assertIn("МЁРТВЫЕ", table_block)
            sys_block = build_system_prompt(citizen_ai, engine.state)
            self.assertIn("ИГРА В ЗАКРЫТУЮ", sys_block)

        asyncio.run(run_sim())

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

        with client.websocket_connect("/ws/game") as ws:
            first = ws.receive_json()
            self.assertEqual(first["type"], "state")
            self.assertEqual(first["state"]["phase"], "lobby")
            second = ws.receive_json()
            self.assertEqual(second["type"], "health")
            self.assertIn("models_detail", second["health"])


if __name__ == "__main__":
    unittest.main()
