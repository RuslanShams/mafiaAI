(() => {
  let ws = null;
  let gameState = null;
  let selectedTarget = "";
  let showThoughts = false;
  let renderedMessageCount = 0;
  let lastPhase = null;

  // Elements
  const appShell = document.getElementById("appShell");
  const phaseBadge = document.getElementById("phaseBadge");
  const roleBadge = document.getElementById("roleBadge");
  const ollamaStatusBtn = document.getElementById("ollamaStatusBtn");
  const ollamaDot = document.getElementById("ollamaDot");
  const ollamaText = document.getElementById("ollamaText");
  const ollamaWarning = document.getElementById("ollamaWarning");
  const resetBtn = document.getElementById("resetBtn");

  const aliveCountLabel = document.getElementById("aliveCountLabel");
  const playerList = document.getElementById("playerList");

  const lobbyView = document.getElementById("lobbyView");
  const gameView = document.getElementById("gameView");
  const startForm = document.getElementById("startForm");
  const startGameBtn = document.getElementById("startGameBtn");
  const playerNameInput = document.getElementById("playerNameInput");
  const modelSelect = document.getElementById("modelSelect");
  const refreshModelsBtn = document.getElementById("refreshModelsBtn");
  const lobbyModelCode = document.getElementById("lobbyModelCode");

  const gameOverCard = document.getElementById("gameOverCard");
  const winnerTitle = document.getElementById("winnerTitle");
  const winnerReason = document.getElementById("winnerReason");
  const revealTableBody = document.getElementById("revealTableBody");
  const toggleThoughtsBtn = document.getElementById("toggleThoughtsBtn");
  const playAgainBtn = document.getElementById("playAgainBtn");

  const chatFeed = document.getElementById("chatFeed");
  const waitingStatusLine = document.getElementById("waitingStatusLine");
  const waitingStatusText = document.getElementById("waitingStatusText");
  const speakForm = document.getElementById("speakForm");
  const speakInput = document.getElementById("speakInput");
  const targetForm = document.getElementById("targetForm");
  const targetPromptLabel = document.getElementById("targetPromptLabel");
  const targetChips = document.getElementById("targetChips");
  const targetCommentInput = document.getElementById("targetCommentInput");
  const confirmTargetBtn = document.getElementById("confirmTargetBtn");

  function connectWebSocket() {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${window.location.host}/ws/game`);

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "state") {
        handleStateUpdate(msg.state);
      } else if (msg.type === "health") {
        renderHealth(msg.health);
      } else if (msg.type === "error") {
        alert(msg.message);
      }
    };

    ws.onclose = () => {
      ollamaDot.className = "dot err";
      ollamaText.textContent = "Переподключение к серверу...";
      setTimeout(connectWebSocket, 2000);
    };
  }

  function sendAction(payload) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
    }
  }

  function renderHealth(h) {
    const modelsDetail = h.models_detail && h.models_detail.length > 0
      ? h.models_detail
      : (h.available_models || []).map((name) => ({ name, size_label: "" }));

    const savedModel = localStorage.getItem("mafia_selected_model") || "";
    const availableNames = modelsDetail.map((m) => m.name);

    if (modelsDetail.length > 0) {
      let chosen = h.configured_model;
      if (savedModel && availableNames.includes(savedModel) && (!gameState || gameState.phase === "lobby")) {
        chosen = savedModel;
        if (chosen !== h.configured_model) {
          sendAction({ action: "select_model", model: chosen });
        }
      } else if (!availableNames.includes(chosen)) {
        chosen = availableNames[0];
      }

      modelSelect.innerHTML = "";
      modelsDetail.forEach((m) => {
        const opt = document.createElement("option");
        opt.value = m.name;
        opt.textContent = m.size_label ? `${m.name} (${m.size_label})` : m.name;
        if (m.name === chosen) opt.selected = true;
        modelSelect.appendChild(opt);
      });
      modelSelect.disabled = false;
      startGameBtn.disabled = false;
      lobbyModelCode.textContent = chosen;
    } else {
      modelSelect.innerHTML = `<option value="">Нет доступных моделей в Ollama</option>`;
      modelSelect.disabled = true;
      startGameBtn.disabled = !h.ollama_online;
    }

    const activeModelLabel =
      gameState && gameState.phase !== "lobby" && gameState.ai_model
        ? gameState.ai_model
        : modelSelect.value || h.configured_model;

    if (h.ollama_online && h.model_ready) {
      ollamaDot.className = "dot ok";
      ollamaText.textContent = `ИИ: ${activeModelLabel}`;
      ollamaWarning.classList.add("hidden");
    } else if (h.ollama_online && !h.model_ready) {
      ollamaDot.className = "dot err";
      ollamaText.textContent = "Нет моделей в Ollama";
      ollamaWarning.classList.remove("hidden");
      ollamaWarning.textContent =
        "Ollama запущена, но локальные модели не найдены. Скачайте модель, например: ollama pull qwen2.5:7b";
    } else {
      ollamaDot.className = "dot err";
      ollamaText.textContent = "Ollama не запущена";
      ollamaWarning.classList.remove("hidden");
      ollamaWarning.textContent =
        "Локальный сервер Ollama недоступен на 127.0.0.1:11434. Запустите `ollama serve` перед началом игры.";
    }
  }

  function handleStateUpdate(state) {
    const isNewGameOrReset =
      !gameState ||
      state.messages.length < renderedMessageCount ||
      (lastPhase !== "game_over" && state.phase === "game_over");

    gameState = state;

    // Night mode visual cue
    appShell.classList.toggle("night-mode", state.phase === "night");

    // Header badges
    if (state.phase === "lobby") {
      phaseBadge.textContent = "Лобби";
      roleBadge.classList.add("hidden");
      resetBtn.classList.add("hidden");
      lobbyView.classList.remove("hidden");
      gameView.classList.add("hidden");
      renderPlayers([]);
      renderedMessageCount = 0;
      lastPhase = state.phase;
      return;
    }

    lobbyView.classList.add("hidden");
    gameView.classList.remove("hidden");
    resetBtn.classList.remove("hidden");

    const dayLabel = state.day > 0 ? `День ${state.day} · ` : "";
    const roundLabel =
      state.phase === "day_discussion" ? ` (Круг ${state.round_num}/2)` : "";
    phaseBadge.textContent = `${dayLabel}${state.phase_ru}${roundLabel}`;
    if (state.ai_model && ollamaDot.classList.contains("ok")) {
      ollamaText.textContent = `ИИ: ${state.ai_model}`;
    }

    const human = state.players.find((p) => p.is_human);
    if (human && human.role) {
      roleBadge.classList.remove("hidden", "role-mafia", "role-citizen");
      roleBadge.classList.add(human.role === "mafia" ? "role-mafia" : "role-citizen");
      const partner =
        human.role === "mafia"
          ? state.players.find((p) => !p.is_human && p.role === "mafia")
          : null;
      roleBadge.textContent = partner
        ? `Ваша роль: Мафия (напарник: ${partner.name})`
        : `Ваша роль: ${human.role_ru}`;
    }

    renderPlayers(state.players);
    renderGameOver(state);
    renderMessages(state, isNewGameOrReset);
    renderActionBar(state, human);

    lastPhase = state.phase;
  }

  function renderPlayers(players) {
    if (!players || players.length === 0) {
      aliveCountLabel.textContent = "7 / 7";
      playerList.innerHTML = `<li class="empty-seats">Места распределятся в начале партии</li>`;
      return;
    }

    const aliveCount = players.filter((p) => p.is_alive).length;
    aliveCountLabel.textContent = `${aliveCount} / ${players.length} живы`;

    // Tally current day votes against each player
    const voteCounts = {};
    if (gameState && gameState.current_votes) {
      Object.values(gameState.current_votes).forEach((target) => {
        voteCounts[target] = (voteCounts[target] || 0) + 1;
      });
    }

    playerList.innerHTML = "";
    players.forEach((p) => {
      const li = document.createElement("li");
      li.className = "player-card";
      if (!p.is_alive) li.classList.add("dead");
      if (gameState.active_speaker_seat === p.seat) li.classList.add("active-speaker");

      const tags = [];
      if (p.is_human) tags.push(`<span class="mini-tag tag-you">Вы</span>`);
      if (p.role === "mafia" && (!p.is_human || gameState.phase === "game_over")) {
        tags.push(`<span class="mini-tag tag-mafia">Мафия</span>`);
      } else if (p.role === "citizen" && gameState.phase === "game_over") {
        tags.push(`<span class="mini-tag tag-citizen">Мирный</span>`);
      }
      if (voteCounts[p.name]) {
        tags.push(`<span class="mini-tag tag-votes">Голосов: ${voteCounts[p.name]}</span>`);
      }

      let statusText = "Жив";
      if (!p.is_alive) {
        statusText = p.death_reason === "night" ? `Убит(а) ночью ${p.death_day}` : `Казнён(а) днём ${p.death_day}`;
      } else if (gameState.active_speaker_seat === p.seat) {
        statusText = p.is_human ? "Ваш ход..." : "Думает...";
      }

      li.innerHTML = `
        <div class="player-top">
          <div class="player-name-wrap">
            <span class="seat-num">#${p.seat}</span>
            <span class="player-name">${escapeHtml(p.name)}</span>
            <span class="muted">(${p.gender_ru})</span>
          </div>
          <div>${tags.join(" ")}</div>
        </div>
        <div class="player-sub">
          <span class="${gameState.active_speaker_seat === p.seat ? "speaking-label" : ""}">${statusText}</span>
          <span>${p.persona_title ? escapeHtml(p.persona_title) : ""}</span>
        </div>
      `;
      playerList.appendChild(li);
    });
  }

  function renderGameOver(state) {
    if (state.phase !== "game_over") {
      gameOverCard.classList.add("hidden");
      showThoughts = false;
      toggleThoughtsBtn.textContent = "Показать скрытые мысли ИИ";
      return;
    }

    gameOverCard.classList.remove("hidden");
    winnerTitle.textContent =
      state.winner === "citizen" ? "Победа Мирных жителей!" : "Победа Мафии!";
    winnerReason.textContent = state.win_reason || "";

    revealTableBody.innerHTML = "";
    state.players.forEach((p) => {
      const tr = document.createElement("tr");
      const roleClass = p.role === "mafia" ? "tag-mafia" : "tag-citizen";
      const fate = p.is_alive
        ? "Выжил(а)"
        : p.death_reason === "night"
        ? `Убит(а) ночью ${p.death_day}`
        : `Казнён(а) днём ${p.death_day}`;

      const personaText = p.is_human
        ? "Человек"
        : `${escapeHtml(p.persona_title || "—")} — ${escapeHtml(p.persona_desc || "")}`;

      tr.innerHTML = `
        <td>#${p.seat}</td>
        <td><strong>${escapeHtml(p.name)}</strong> (${p.gender_ru})${p.is_human ? " · Вы" : ""}</td>
        <td><span class="mini-tag ${roleClass}">${escapeHtml(p.role_ru)}</span></td>
        <td>${personaText}</td>
        <td>${fate}</td>
      `;
      revealTableBody.appendChild(tr);
    });
  }

  function renderMessages(state, forceFullRebuild) {
    if (forceFullRebuild) {
      chatFeed.innerHTML = "";
      renderedMessageCount = 0;
    }

    const human = state.players.find((p) => p.is_human);
    for (let i = renderedMessageCount; i < state.messages.length; i++) {
      const m = state.messages[i];
      if (m.visibility === "system") {
        const div = document.createElement("div");
        div.className = "msg-system";
        div.textContent = m.text;
        chatFeed.appendChild(div);
      } else {
        const div = document.createElement("div");
        div.className = "msg-bubble";
        if (human && m.sender_seat === human.seat && m.visibility === "public") {
          div.classList.add("msg-human");
        }
        if (m.visibility === "mafia") {
          div.classList.add("msg-mafia");
        }

        const phaseTag = m.visibility === "mafia" ? `Ночь ${m.day} · Чат мафии` : m.day === 0 ? "Знакомство" : `День ${m.day}`;
        const voteHtml = m.vote_target
          ? `<div class="vote-pill">Голос против: ${escapeHtml(m.vote_target)}</div>`
          : "";
        const thoughtHtml =
          showThoughts && m.thought
            ? `<div class="msg-thought">Скрытая мысль ИИ: ${escapeHtml(m.thought)}</div>`
            : "";

        div.innerHTML = `
          <div class="msg-header">
            <span class="msg-author">${escapeHtml(m.sender_name)}</span>
            <span class="msg-meta">${phaseTag}</span>
          </div>
          <div class="msg-text">${escapeHtml(m.text)}</div>
          ${voteHtml}
          ${thoughtHtml}
        `;
        chatFeed.appendChild(div);
      }
    }

    if (state.messages.length > renderedMessageCount) {
      renderedMessageCount = state.messages.length;
      chatFeed.scrollTop = chatFeed.scrollHeight;
    }
  }

  function renderActionBar(state, human) {
    waitingStatusLine.classList.add("hidden");
    speakForm.classList.add("hidden");
    targetForm.classList.add("hidden");

    if (state.phase === "game_over") {
      waitingStatusLine.classList.remove("hidden");
      waitingStatusText.textContent = "Партия завершена. Все роли и характеры открыты выше.";
      return;
    }

    if (state.waiting_for_human === "speak") {
      speakForm.classList.remove("hidden");
      speakInput.focus();
      return;
    }

    if (state.waiting_for_human === "vote" || state.waiting_for_human === "night_kill") {
      targetForm.classList.remove("hidden");
      const isNight = state.waiting_for_human === "night_kill";
      const suggestionHint =
        isNight && state.mafia_suggestion
          ? ` (напарник предлагает: ${state.mafia_suggestion})`
          : "";
      targetPromptLabel.textContent = isNight
        ? `Ночной выстрел мафии — выберите жертву${suggestionHint}:`
        : "Открытое голосование — против кого вы голосуете?";
      confirmTargetBtn.textContent = isNight ? "Убрать выбранную цель" : "Проголосовать";

      if (!state.valid_targets.includes(selectedTarget)) {
        selectedTarget = state.mafia_suggestion && state.valid_targets.includes(state.mafia_suggestion)
          ? state.mafia_suggestion
          : "";
      }
      renderTargetChips(state.valid_targets, state.mafia_suggestion);
      return;
    }

    // Waiting for AI
    waitingStatusLine.classList.remove("hidden");
    const activePlayer = state.players.find((p) => p.seat === state.active_speaker_seat);
    if (human && !human.is_alive) {
      waitingStatusText.textContent = activePlayer
        ? `Вы выбыли (режим наблюдателя) · Сейчас ходит ${activePlayer.name}...`
        : "Вы выбыли (режим наблюдателя) · ИИ продолжают партию...";
    } else if (state.phase === "night") {
      waitingStatusText.textContent = "Город спит... Мафия выбирает жертву.";
    } else if (activePlayer) {
      waitingStatusText.textContent = `${activePlayer.name} обдумывает реплику...`;
    } else {
      waitingStatusText.textContent = "Обработка хода...";
    }
  }

  function renderTargetChips(targets, suggested) {
    targetChips.innerHTML = "";
    targets.forEach((name) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "target-chip";
      if (selectedTarget === name) btn.classList.add("selected");
      if (suggested === name) btn.classList.add("suggested");
      btn.textContent = suggested === name ? `${name} ★` : name;
      btn.addEventListener("click", () => {
        selectedTarget = name;
        renderTargetChips(targets, suggested);
      });
      targetChips.appendChild(btn);
    });
    confirmTargetBtn.disabled = !selectedTarget;
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // Event Listeners
  ollamaStatusBtn.addEventListener("click", () => {
    sendAction({ action: "check_health" });
  });

  refreshModelsBtn.addEventListener("click", () => {
    sendAction({ action: "check_health" });
  });

  modelSelect.addEventListener("change", () => {
    const chosen = modelSelect.value;
    if (chosen) {
      localStorage.setItem("mafia_selected_model", chosen);
      lobbyModelCode.textContent = chosen;
      sendAction({ action: "select_model", model: chosen });
    }
  });

  startForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const genderEl = startForm.querySelector('input[name="gender"]:checked');
    const chosenModel = modelSelect.value || "qwen2.5:7b";
    localStorage.setItem("mafia_selected_model", chosenModel);
    sendAction({
      action: "start_game",
      name: playerNameInput.value.trim() || "Игрок",
      gender: genderEl ? genderEl.value : "M",
      model: chosenModel,
    });
  });

  resetBtn.addEventListener("click", () => {
    if (confirm("Завершить текущую партию и вернуться в лобби?")) {
      sendAction({ action: "reset_lobby" });
    }
  });

  playAgainBtn.addEventListener("click", () => {
    sendAction({ action: "reset_lobby" });
  });

  toggleThoughtsBtn.addEventListener("click", () => {
    showThoughts = !showThoughts;
    toggleThoughtsBtn.textContent = showThoughts
      ? "Скрыть мысли ИИ"
      : "Показать скрытые мысли ИИ";
    if (gameState) {
      renderMessages(gameState, true);
    }
  });

  speakForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = speakInput.value.trim();
    if (!text) return;
    sendAction({ action: "speak", text });
    speakInput.value = "";
  });

  targetForm.addEventListener("submit", (e) => {
    e.preventDefault();
    if (!selectedTarget || !gameState) return;
    sendAction({
      action: gameState.waiting_for_human,
      target: selectedTarget,
      text: targetCommentInput.value.trim(),
    });
    selectedTarget = "";
    targetCommentInput.value = "";
  });

  connectWebSocket();
})();

