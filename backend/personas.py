from __future__ import annotations

import random

from backend.models import Gender

MALE_NAMES: list[str] = [
    "Алексей",
    "Виктор",
    "Дмитрий",
    "Игорь",
    "Максим",
    "Роман",
    "Сергей",
    "Андрей",
    "Павел",
    "Кирилл",
    "Артём",
    "Денис",
    "Николай",
    "Глеб",
]

FEMALE_NAMES: list[str] = [
    "Анна",
    "Елена",
    "Марина",
    "Ольга",
    "Светлана",
    "Татьяна",
    "Юлия",
    "Наталья",
    "Ирина",
    "Дарья",
    "Ксения",
    "Полина",
    "Алина",
    "Вероника",
]


def generate_ai_profiles(count: int, forbidden_name: str) -> list[tuple[str, Gender]]:
    """Generate `count` unique (name, gender) tuples randomized per game."""
    num_males = random.choice([2, 3, 4])
    num_females = count - num_males
    genders = [Gender.MALE] * num_males + [Gender.FEMALE] * num_females
    random.shuffle(genders)

    forbidden_lower = forbidden_name.strip().lower()
    avail_males = [n for n in MALE_NAMES if n.lower() != forbidden_lower]
    avail_females = [n for n in FEMALE_NAMES if n.lower() != forbidden_lower]

    chosen_males = iter(random.sample(avail_males, k=num_males))
    chosen_females = iter(random.sample(avail_females, k=num_females))

    profiles: list[tuple[str, Gender]] = []
    for gender in genders:
        name = next(chosen_males) if gender == Gender.MALE else next(chosen_females)
        profiles.append((name, gender))

    return profiles
