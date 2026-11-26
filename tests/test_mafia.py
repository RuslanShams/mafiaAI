from __future__ import annotations

import asyncio
import unittest
from typing import Optional

from fastapi.testclient import TestClient

from backend.app import app
from backend.engine import GameEngine
from backend.llm_client import resolve_target_name, validate_russian_speech
from backend.models import Gender, Phase, Player, Role
from backend.personas import generate_ai_profiles
from backend.prompts import build_system_prompt, build_table_state_block


class FakeOllamaClient:
    """Deterministic Russian mock LLM that verifies prompt isolation on every AI call."""

    def __init__(self) -> None:
        self.model: str = "qwen2.5:7b"
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
        # Validate strict information isolation inside prompts!
        if player.role == Role.CITIZEN:
            assert "ТВОЯ СЕКРЕТНАЯ РОЛЬ: МИРНЫЙ ЖИТЕЛЬ" in system_prompt
            assert "СЕКРЕТНЫЙ НОЧНОЙ ЧАТ МАФИИ" not in user_prompt
            assert "напарник по мафии:" not in system_prompt
        else:
            assert "ТВОЯ СЕКРЕТНАЯ РОЛЬ: МАФИЯ" in system_prompt

        target = valid_targets[0] if valid_targets else ""
        speech = (
            f"Я внимательно слушаю стол и слежу за аргументами."
            if not target
            else f"Считаю, что сейчас главный кандидат — {target}."
        )
        err = validate_russian_speech(
            speech,
            player=player,
            is_public_day=is_public_day,
            allowed_names=allowed_names,
        )
        assert err is None, f"Mock speech failed validation: {err}"

        res = {"thought": "Скрытая тактическая мысль.", "speech": speech}
        if valid_targets is not None:
            res["target"] = target
        return res


class MafiaGameTests(unittest.TestCase):
    def test_persona_randomization_unique_and_varied(self) -> None:
        batch1 = generate_ai_profiles(6, forbidden_name="Максим")
        batch2 = generate_ai_profiles(6, forbidden_name="Максим")

        self.assertEqual(len(batch1), 6)
        # Unique names and unique personas within a game
        self.assertEqual(len({p[0] for p in batch1}), 6)
        self.assertEqual(len({p[2].id for p in batch1}), 6)
        self.assertNotIn("Максим", {p[0] for p in batch1})

        # Both genders represented
        genders = {p[1] for p in batch1}
        self.assertIn(Gender.MALE, genders)
        self.assertIn(Gender.FEMALE, genders)

        # Across multiple games, persona assignments vary
        any_diff = [p[2].id for p in batch1] != [p[2].id for p in batch2]
        if not any_diff:
            batch3 = generate_ai_profiles(6, forbidden_name="Максим")
            any_diff = [p[2].id for p in batch1] != [p[2].id for p in batch3]
        self.assertTrue(any_diff)

    def test_russian_validator_and_target_resolver(self) -> None:
        mafia_p = Player(seat=2, name="Виктор", gender=Gender.MALE, is_human=False, role=Role.MAFIA)

        # Clean Russian passes
        self.assertIsNone(
            validate_russian_speech("Друзья, мне кажется странным поведение Елены.", mafia_p, is_public_day=True)
        )
        # CJK rejected
        self.assertIsNotNone(
            validate_russian_speech("Я думаю это Елена 觉得很奇怪.", mafia_p, is_public_day=True)
        )
        # English words rejected
        self.assertIsNotNone(
            validate_russian_speech("Я думаю Елена выглядит очень suspicious сегодня.", mafia_p, is_public_day=True)
        )
        # Latin player name allowed when in allowed_names
        self.assertIsNone(
            validate_russian_speech(
                "Мне кажется, что Alex сегодня слишком отмалчивается.",
                mafia_p,
                is_public_day=True,
                allowed_names=["Alex", "Виктор"],
            )
        )
        # Meta AI leak rejected
        self.assertIsNotNone(
            validate_russian_speech("Как языковая модель, я голосую против Анны.", mafia_p, is_public_day=True)
        )
        # Mafia confession in public day rejected
        self.assertIsNotNone(
            validate_russian_speech("Ладно, признаюсь, я мафия!", mafia_p, is_public_day=True)
        )
        # Target resolution handles inflected/decorated strings
        self.assertEqual(resolve_target_name("[3] Ольгу", ["Виктор", "Ольга", "Сергей"]), "Ольга")

    def test_full_closed_game_simulation_and_role_isolation(self) -> None:
        async def run_sim() -> None:
            events: list[dict] = []

            async def collect(msg: dict) -> None:
                events.append(msg)

            fake_llm = FakeOllamaClient()
            engine = GameEngine(llm=fake_llm, broadcast=collect)  # type: ignore[arg-type]

            await engine.start_new_game(human_name="Артём", human_gender="M", ai_model="gemma3:4b")
            self.assertEqual(engine.state.ai_model, "gemma3:4b")
            self.assertEqual(fake_llm.model, "gemma3:4b")
            self.assertEqual(len(engine.state.players), 7)
            self.assertEqual(sum(1 for p in engine.state.players if p.role == Role.MAFIA), 2)
            self.assertEqual(sum(1 for p in engine.state.players if p.role == Role.CITIZEN), 5)
            self.assertEqual(sum(1 for p in engine.state.players if p.is_human), 1)

            # Drive human turns automatically until GAME_OVER
            steps = 0
            while engine.state.phase != Phase.GAME_OVER and steps < 200:
                steps += 1
                await asyncio.sleep(0.01)

                # Verify closed-game privacy while game is running
                if engine.state.phase != Phase.GAME_OVER:
                    snap = engine.serialize_for_human()
                    human = engine.state.human_player()
                    assert human is not None
                    for pd in snap["players"]:
                        # Dead players' roles must remain hidden ("Неизвестно") for Citizen human!
                        if human.role == Role.CITIZEN and not pd["is_human"]:
                            self.assertIsNone(pd["role"])
                            self.assertEqual(pd["role_ru"], "Неизвестно")
                        # AI persona is hidden until GAME_OVER
                        if not pd["is_human"]:
                            self.assertIsNone(pd["persona_title"])

                mode = engine.state.waiting_for_human
                if mode == "speak":
                    engine.submit_human_input("speak", text="Я мирный житель, давайте рассуждать логически.")
                elif mode in ("vote", "night_kill"):
                    target = engine.state.valid_targets[0]
                    engine.submit_human_input(mode, target=target, text=f"Выбираю {target}.")

            self.assertEqual(engine.state.phase, Phase.GAME_OVER)
            self.assertIn(engine.state.winner, (Role.CITIZEN, Role.MAFIA))

            # Verify full reveal at GAME_OVER
            final_snap = engine.serialize_for_human()
            for pd in final_snap["players"]:
                self.assertIsNotNone(pd["role"])
                if not pd["is_human"]:
                    self.assertIsNotNone(pd["persona_title"])

            # Verify thoughts are revealed at GAME_OVER
            thoughts_shown = [m for m in final_snap["messages"] if m.get("thought")]
            self.assertGreater(len(thoughts_shown), 0)

            # Verify dead players in prompts never leak roles
            citizen_ai = next(p for p in engine.state.players if not p.is_human and p.role == Role.CITIZEN)
            table_block = build_table_state_block(citizen_ai, engine.state)
            self.assertIn("роль не раскрыта", table_block)
            sys_block = build_system_prompt(citizen_ai, engine.state)
            self.assertIn("ИГРА В ЗАКРЫТУЮ", sys_block)

        asyncio.run(run_sim())

    def test_http_and_websocket_endpoints(self) -> None:
        client = TestClient(app)
        res = client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("МАФИЯ", res.text)

        with client.websocket_connect("/ws/game") as ws:
            first = ws.receive_json()
            self.assertEqual(first["type"], "state")
            self.assertEqual(first["state"]["phase"], "lobby")
            second = ws.receive_json()
            self.assertEqual(second["type"], "health")
            self.assertIn("configured_model", second["health"])
            self.assertIn("models_detail", second["health"])

            ws.send_json({"action": "select_model", "model": "mistral:7b"})
            msg_a = ws.receive_json()
            msg_b = ws.receive_json()
            received_types = {msg_a["type"], msg_b["type"]}
            self.assertEqual(received_types, {"health", "state"})
            state_msg = msg_a if msg_a["type"] == "state" else msg_b
            self.assertEqual(state_msg["state"]["ai_model"], "mistral:7b")


if __name__ == "__main__":
    unittest.main()

