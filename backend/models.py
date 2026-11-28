from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Role(str, Enum):
    CITIZEN = "citizen"
    MAFIA = "mafia"
    COMMISSAR = "commissar"
    DOCTOR = "doctor"

    @property
    def label_ru(self) -> str:
        labels = {
            Role.CITIZEN: "Мирный житель",
            Role.MAFIA: "Мафия",
            Role.COMMISSAR: "Комиссар",
            Role.DOCTOR: "Доктор",
        }
        return labels[self]

    @property
    def is_town(self) -> bool:
        return self != Role.MAFIA


class Gender(str, Enum):
    MALE = "M"
    FEMALE = "F"

    @property
    def label_ru(self) -> str:
        return "М" if self == Gender.MALE else "Ж"


class Phase(str, Enum):
    LOBBY = "lobby"
    DAY_INTRO = "day_intro"
    NIGHT = "night"
    DAY_DISCUSSION = "day_discussion"
    DAY_VOTING = "day_voting"
    GAME_OVER = "game_over"

    @property
    def label_ru(self) -> str:
        labels = {
            Phase.LOBBY: "Лобби",
            Phase.DAY_INTRO: "Знакомство за столом",
            Phase.NIGHT: "Ночь",
            Phase.DAY_DISCUSSION: "Дневное обсуждение",
            Phase.DAY_VOTING: "Голосование",
            Phase.GAME_OVER: "Игра окончена",
        }
        return labels[self]


@dataclass(frozen=True)
class Persona:
    id: str
    title: str
    short_desc: str
    prompt_style: str


@dataclass
class Player:
    seat: int
    name: str
    gender: Gender
    is_human: bool
    role: Role
    persona: Optional[Persona] = None
    is_alive: bool = True
    death_reason: Optional[str] = None  # "night" | "vote"
    death_day: Optional[int] = None

    def to_public_dict(self, viewer: Optional[Player], reveal_all: bool = False) -> dict:
        """Serialize player strictly isolating secret role until game over."""
        can_see_role = (
            reveal_all
            or (viewer is not None and viewer.seat == self.seat)
            or (
                viewer is not None
                and viewer.role == Role.MAFIA
                and self.role == Role.MAFIA
            )
        )
        return {
            "seat": self.seat,
            "name": self.name,
            "gender": self.gender.value,
            "gender_ru": self.gender.label_ru,
            "is_human": self.is_human,
            "is_alive": self.is_alive,
            "death_reason": self.death_reason,
            "death_day": self.death_day,
            "role": self.role.value if can_see_role else None,
            "role_ru": self.role.label_ru if can_see_role else "Неизвестно",
        }


@dataclass
class ChatMessage:
    id: str
    day: int
    phase: Phase
    round_num: int
    sender_seat: Optional[int]
    sender_name: str
    text: str
    visibility: str = "public"  # "public" | "mafia" | "system"
    thought: Optional[str] = None
    vote_target: Optional[str] = None

    def to_client_dict(self, include_thought: bool = False) -> dict:
        data = {
            "id": self.id,
            "day": self.day,
            "phase": self.phase.value,
            "round_num": self.round_num,
            "sender_seat": self.sender_seat,
            "sender_name": self.sender_name,
            "text": self.text,
            "visibility": self.visibility,
            "vote_target": self.vote_target,
        }
        if include_thought and self.thought:
            data["thought"] = self.thought
        return data


@dataclass
class DayRecord:
    day: int
    night_victim: Optional[str] = None
    night_saved: bool = False
    votes: dict[str, str] = field(default_factory=dict)  # voter_name -> target_name
    executed: Optional[str] = None
    tie_no_execution: bool = False


@dataclass
class GameState:
    phase: Phase = Phase.LOBBY
    day: int = 0
    round_num: int = 0
    ai_model: str = "qwen2.5:7b"
    players: list[Player] = field(default_factory=list)
    messages: list[ChatMessage] = field(default_factory=list)
    history: list[DayRecord] = field(default_factory=list)
    winner: Optional[Role] = None
    win_reason: Optional[str] = None
    active_speaker_seat: Optional[int] = None
    awaiting_last_word_seat: Optional[int] = None
    waiting_for_human: Optional[str] = None  # None | "speak" | "last_word" | "vote" | "night_kill" | "night_heal" | "night_check"
    valid_targets: list[str] = field(default_factory=list)
    mafia_suggestion: Optional[str] = None
    current_votes: dict[str, str] = field(default_factory=dict)
    commissar_checks: dict[str, str] = field(default_factory=dict)  # target_name -> "Мафия" | "Мирный"
    last_doctor_heal_seat: Optional[int] = None

    def human_player(self) -> Optional[Player]:
        for p in self.players:
            if p.is_human:
                return p
        return None

    def alive_players(self) -> list[Player]:
        return [p for p in self.players if p.is_alive]

    def alive_mafia(self) -> list[Player]:
        return [p for p in self.players if p.is_alive and p.role == Role.MAFIA]

    def alive_citizens(self) -> list[Player]:
        """All alive Town-aligned players (Citizens + Commissar + Doctor)."""
        return [p for p in self.players if p.is_alive and p.role.is_town]

    def alive_role(self, role: Role) -> Optional[Player]:
        for p in self.players:
            if p.is_alive and p.role == role:
                return p
        return None

    def find_by_name(self, name: str) -> Optional[Player]:
        clean = name.strip().lower()
        for p in self.players:
            if p.name.lower() == clean:
                return p
        return None

