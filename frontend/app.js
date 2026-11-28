(() => {
  let ws = null;
  let gameState = null;
  let selectedTarget = "";
  let renderedMessageCount = 0;
  let firstRenderedMessageId = null;
  let lastPhase = null;
  let lastShownBubbleSeat = null;
  let serverSlots = { max_sessions: 4, active_games: 0, free_slots: 4, locked_model: null };

  // Smooth typewriter state (~22 chars/sec so humans comfortably read every comic speech bubble)
  const TYPEWRITER_CHAR_MS = 45;
  let lastTypedMsgId = null;
  const typewriter = {
    key: null,
    msgId: null,
    seat: null,
    senderName: "",
    day: 0,
    fullText: "",
    shownLen: 0,
    isFinal: false,
    voteTarget: null,
    isLastWord: false,
    isMafia: false,
    timerId: null,
  };

  function isTypewriterActive() {
    return Boolean(
      typewriter.key && (!typewriter.isFinal || typewriter.shownLen < typewriter.fullText.length)
    );
  }

  function stopTypewriter(flushDom = true) {
    if (typewriter.timerId) {
      clearInterval(typewriter.timerId);
      typewriter.timerId = null;
    }
    if (flushDom && typewriter.msgId && chatFeed) {
      const committedTextEl = chatFeed.querySelector(
        `.msg-bubble[data-msg-id="${typewriter.msgId}"] .msg-text`
      );
      if (committedTextEl) {
        committedTextEl.textContent = typewriter.fullText;
      }
      lastTypedMsgId = typewriter.msgId;
    }
    typewriter.key = null;
    typewriter.msgId = null;
    typewriter.seat = null;
    typewriter.senderName = "";
    typewriter.day = 0;
    typewriter.fullText = "";
    typewriter.shownLen = 0;
    typewriter.isFinal = false;
    typewriter.voteTarget = null;
    typewriter.isLastWord = false;
    typewriter.isMafia = false;
  }

  function tickTypewriter() {
    if (!typewriter.key) {
      stopTypewriter(false);
      return;
    }
    if (typewriter.shownLen < typewriter.fullText.length) {
      typewriter.shownLen += 1;
      const visible = typewriter.fullText.slice(0, typewriter.shownLen);
      const stillGoing = !typewriter.isFinal || typewriter.shownLen < typewriter.fullText.length;
      const displayStr = stillGoing ? visible + "▌" : visible;

      const comicEl = seatsLayer.querySelector(
        `.comic-speech[data-seat="${typewriter.seat}"] .comic-speech-text`
      );
      if (comicEl) {
        comicEl.textContent = displayStr;
      }

      const streamBubbleText = document.querySelector("#liveStreamBubble .msg-text");
      if (streamBubbleText) {
        streamBubbleText.textContent = displayStr;
        chatFeed.scrollTop = chatFeed.scrollHeight;
      } else if (typewriter.msgId) {
        const committedTextEl = chatFeed.querySelector(
          `.msg-bubble[data-msg-id="${typewriter.msgId}"] .msg-text`
        );
        if (committedTextEl) {
          committedTextEl.textContent = displayStr;
          chatFeed.scrollTop = chatFeed.scrollHeight;
        }
      }

      if (!stillGoing) {
        lastTypedMsgId = typewriter.msgId;
        clearInterval(typewriter.timerId);
        typewriter.timerId = null;
        typewriter.key = null;
        if (gameState) {
          const human = gameState.players.find((p) => p.is_human);
          renderTableScene(gameState);
          renderActionBar(gameState, human);
        }
      }
    } else if (typewriter.isFinal) {
      lastTypedMsgId = typewriter.msgId;
      clearInterval(typewriter.timerId);
      typewriter.timerId = null;
      typewriter.key = null;
      if (gameState) {
        const human = gameState.players.find((p) => p.is_human);
        renderTableScene(gameState);
        renderActionBar(gameState, human);
      }
    } else {
      // Caught up with latest stream_chunk; pause timer until next chunk or final state
      clearInterval(typewriter.timerId);
      typewriter.timerId = null;
    }
  }

  function ensureTypewriterTimer() {
    if (!typewriter.timerId && typewriter.key) {
      typewriter.timerId = setInterval(tickTypewriter, TYPEWRITER_CHAR_MS);
    }
  }

  function getOrCreateTabSessionId() {
    let sid = sessionStorage.getItem("mafia_tab_session_id");
    if (!sid || !/^[A-Za-z0-9_-]{6,64}$/.test(sid)) {
      sid =
        (window.crypto && window.crypto.randomUUID
          ? window.crypto.randomUUID().replace(/-/g, "").slice(0, 16)
          : Math.random().toString(36).slice(2, 10) + Date.now().toString(36));
      sessionStorage.setItem("mafia_tab_session_id", sid);
    }
    return sid;
  }

  const tabSessionId = getOrCreateTabSessionId();

  // Elements
  const appShell = document.getElementById("appShell");
  const phaseBadge = document.getElementById("phaseBadge");
  const roleBadge = document.getElementById("roleBadge");
  const aliveCountLabel = document.getElementById("aliveCountLabel");
  const slotsBadge = document.getElementById("slotsBadge");
  const ollamaStatusBtn = document.getElementById("ollamaStatusBtn");
  const ollamaDot = document.getElementById("ollamaDot");
  const ollamaText = document.getElementById("ollamaText");
  const ollamaWarning = document.getElementById("ollamaWarning");
  const resetBtn = document.getElementById("resetBtn");

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
  const playAgainBtn = document.getElementById("playAgainBtn");

  const tableIcon = document.getElementById("tableIcon");
  const tablePhaseTitle = document.getElementById("tablePhaseTitle");
  const tableNarratorText = document.getElementById("tableNarratorText");
  const seatsLayer = document.getElementById("seatsLayer");

  const chatFeed = document.getElementById("chatFeed");
  const waitingStatusLine = document.getElementById("waitingStatusLine");
  const waitingStatusText = document.getElementById("waitingStatusText");
  const speakForm = document.getElementById("speakForm");
  const speakInput = document.getElementById("speakInput");
  const speakSubmitBtn = document.getElementById("speakSubmitBtn");
  const skipLastWordBtn = document.getElementById("skipLastWordBtn");
  const targetForm = document.getElementById("targetForm");
  const targetPromptLabel = document.getElementById("targetPromptLabel");
  const targetChips = document.getElementById("targetChips");
  const targetCommentInput = document.getElementById("targetCommentInput");
  const confirmTargetBtn = document.getElementById("confirmTargetBtn");

  function connectWebSocket() {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(
      `${proto}//${window.location.host}/ws/game?session_id=${encodeURIComponent(tabSessionId)}`
    );

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "session") {
        if (msg.slots) renderSlots(msg.slots);
      } else if (msg.type === "slots") {
        renderSlots(msg.slots);
      } else if (msg.type === "state") {
        handleStateUpdate(msg.state);
      } else if (msg.type === "stream_chunk") {
        handleStreamChunk(msg);
      } else if (msg.type === "health") {
        if (msg.health && msg.health.slots) renderSlots(msg.health.slots);
        renderHealth(msg.health);
      } else if (msg.type === "error") {
        alert(msg.message);
      }
    };

    ws.onclose = () => {
      ollamaDot.className = "dot err";
      ollamaText.textContent = "Переподключение...";
      setTimeout(connectWebSocket, 2000);
    };
  }

  function sendAction(payload) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
    }
  }

  function renderSlots(slots) {
    if (!slots) return;
    serverSlots = slots;
    if (slotsBadge) {
      slotsBadge.textContent = `Столы: ${slots.active_games} / ${slots.max_sessions}`;
    }
    if (slots.locked_model && modelSelect && (!gameState || gameState.phase === "lobby")) {
      modelSelect.value = slots.locked_model;
      modelSelect.disabled = true;
      lobbyModelCode.textContent = slots.locked_model;
    } else if (modelSelect && modelSelect.options.length > 0 && modelSelect.options[0].value) {
      modelSelect.disabled = false;
    }
  }

  function renderHealth(h) {
    const modelsDetail =
      h.models_detail && h.models_detail.length > 0
        ? h.models_detail
        : (h.available_models || []).map((name) => ({ name, size_label: "" }));

    const savedModel = localStorage.getItem("mafia_selected_model") || "";
    const availableNames = modelsDetail.map((m) => m.name);

    if (modelsDetail.length > 0) {
      let chosen = serverSlots.locked_model || h.configured_model;
      if (
        !serverSlots.locked_model &&
        savedModel &&
        availableNames.includes(savedModel) &&
        (!gameState || gameState.phase === "lobby")
      ) {
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
      modelSelect.disabled = Boolean(serverSlots.locked_model);
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
        "Ollama запущена, но локальные модели не найдены. Скачайте модель, например: ollama pull gemma3:4b";
    } else {
      ollamaDot.className = "dot err";
      ollamaText.textContent = "Ollama не запущена";
      ollamaWarning.classList.remove("hidden");
      ollamaWarning.textContent =
        "Локальный сервер Ollama недоступен на 127.0.0.1:11434. Запустите `ollama serve` перед началом игры.";
    }
  }

  function handleStreamChunk(chunk) {
    const phase = gameState ? gameState.phase : "day_discussion";
    const round = gameState ? gameState.round_num : 1;
    const streamKey = `stream:${chunk.day}:${phase}:${round}:${chunk.seat}`;
    const isSameStream = typewriter.key === streamKey;

    if (!isSameStream) {
      stopTypewriter(true);
      typewriter.key = streamKey;
      typewriter.msgId = null;
      typewriter.seat = chunk.seat;
      typewriter.senderName = chunk.sender_name;
      typewriter.day = chunk.day;
      typewriter.fullText = chunk.text;
      typewriter.shownLen = Math.min(1, chunk.text.length);
      typewriter.isFinal = false;
      typewriter.voteTarget = null;
      typewriter.isLastWord = chunk.sender_name.includes("Последнее слово");
      typewriter.isMafia = false;
    } else {
      typewriter.fullText = chunk.text;
      if (typewriter.shownLen > chunk.text.length) {
        typewriter.shownLen = chunk.text.length;
      }
    }

    // Update chronicle streaming bubble with typewriter-paced prefix
    let bubble = document.getElementById("liveStreamBubble");
    if (!bubble) {
      bubble = document.createElement("div");
      bubble.id = "liveStreamBubble";
      bubble.className = "msg-bubble msg-streaming";
      bubble.innerHTML = `
        <div class="msg-header">
          <span class="msg-author"></span>
          <span class="msg-meta"></span>
        </div>
        <div class="msg-text"></div>
      `;
      chatFeed.appendChild(bubble);
    }
    bubble.querySelector(".msg-author").textContent = chunk.sender_name;
    bubble.querySelector(".msg-meta").textContent = `День ${chunk.day} · говорит...`;
    bubble.querySelector(".msg-text").textContent =
      typewriter.fullText.slice(0, typewriter.shownLen) + "▌";
    chatFeed.scrollTop = chatFeed.scrollHeight;

    if (!isSameStream && gameState) {
      renderTableScene(gameState);
      const human = gameState.players.find((p) => p.is_human);
      renderActionBar(gameState, human);
    }
    ensureTypewriterTimer();
  }

  function handleStateUpdate(state) {
    const wasInitialLoad = !gameState;
    const currentFirstId = state.messages.length > 0 ? state.messages[0].id : null;
    const isHardReset =
      !gameState ||
      state.messages.length < renderedMessageCount ||
      currentFirstId !== firstRenderedMessageId;
    const isNewGameOrReset =
      isHardReset ||
      (!gameState.spectator_reveal && state.spectator_reveal) ||
      (lastPhase !== "game_over" && state.phase === "game_over");

    if (state.phase === "lobby" || (isHardReset && !wasInitialLoad)) {
      stopTypewriter(false);
      lastTypedMsgId = null;
    }

    // Sync or initiate smooth typewriter for the latest spoken message
    let latestBubbleMsg = null;
    if (state.phase !== "night" && state.phase !== "lobby") {
      latestBubbleMsg = [...state.messages]
        .reverse()
        .find((m) => m.visibility === "public" && m.sender_seat !== null);
    } else if (state.phase === "night") {
      latestBubbleMsg = [...state.messages]
        .reverse()
        .find((m) => m.visibility === "mafia" && m.day === state.day && m.sender_seat !== null);
    }

    if (latestBubbleMsg) {
      const senderPlayer = (state.players || []).find((p) => p.seat === latestBubbleMsg.sender_seat);
      const isHumanMsg = Boolean(senderPlayer && senderPlayer.is_human);
      if (isHumanMsg || (wasInitialLoad && state.messages.length > 2) || state.phase === "game_over") {
        stopTypewriter(true);
        lastTypedMsgId = latestBubbleMsg.id;
      } else if (latestBubbleMsg.id !== lastTypedMsgId) {
        if (typewriter.key && !typewriter.isFinal && typewriter.seat === latestBubbleMsg.sender_seat) {
          // Seamlessly continue typing from live stream into committed message without jumping!
          typewriter.key = latestBubbleMsg.id;
          typewriter.msgId = latestBubbleMsg.id;
          typewriter.fullText = latestBubbleMsg.text;
          typewriter.shownLen = Math.min(typewriter.shownLen, latestBubbleMsg.text.length);
          typewriter.isFinal = true;
          typewriter.voteTarget = latestBubbleMsg.vote_target;
          typewriter.isLastWord = latestBubbleMsg.sender_name.includes("Последнее слово");
          typewriter.isMafia = latestBubbleMsg.visibility === "mafia";
          ensureTypewriterTimer();
        } else if (typewriter.msgId !== latestBubbleMsg.id) {
          stopTypewriter(true);
          if (!(state.phase === "night" && state.day === 1)) {
            typewriter.key = latestBubbleMsg.id;
            typewriter.msgId = latestBubbleMsg.id;
            typewriter.seat = latestBubbleMsg.sender_seat;
            typewriter.senderName = latestBubbleMsg.sender_name;
            typewriter.day = latestBubbleMsg.day;
            typewriter.fullText = latestBubbleMsg.text;
            typewriter.shownLen = Math.min(1, latestBubbleMsg.text.length);
            typewriter.isFinal = true;
            typewriter.voteTarget = latestBubbleMsg.vote_target;
            typewriter.isLastWord = latestBubbleMsg.sender_name.includes("Последнее слово");
            typewriter.isMafia = latestBubbleMsg.visibility === "mafia";
            ensureTypewriterTimer();
          } else {
            lastTypedMsgId = latestBubbleMsg.id;
          }
        }
      }
    }

    gameState = state;
    appShell.classList.toggle("night-mode", state.phase === "night");

    if (state.phase === "lobby") {
      phaseBadge.textContent = "Лобби";
      roleBadge.classList.add("hidden");
      aliveCountLabel.classList.add("hidden");
      resetBtn.classList.add("hidden");
      lobbyView.classList.remove("hidden");
      gameView.classList.add("hidden");
      renderedMessageCount = 0;
      firstRenderedMessageId = null;
      lastPhase = state.phase;
      return;
    }

    lobbyView.classList.add("hidden");
    gameView.classList.remove("hidden");
    resetBtn.classList.remove("hidden");
    aliveCountLabel.classList.remove("hidden");

    const aliveCount = state.players.filter((p) => p.is_alive).length;
    aliveCountLabel.textContent = `${aliveCount} / ${state.players.length} живы`;

    const dayLabel = state.day > 0 ? `День ${state.day} · ` : "";
    const roundLabel =
      state.phase === "day_discussion" ? ` (Круг ${state.round_num}/2)` : "";
    phaseBadge.textContent = `${dayLabel}${state.phase_ru}${roundLabel}`;
    if (state.ai_model && ollamaDot.classList.contains("ok")) {
      ollamaText.textContent = `ИИ: ${state.ai_model}`;
    }

    const human = state.players.find((p) => p.is_human);
    if (human && human.role) {
      roleBadge.classList.remove("hidden", "role-mafia", "role-citizen", "role-commissar", "role-doctor");
      roleBadge.classList.add(`role-${human.role}`);
      const partner =
        human.role === "mafia"
          ? state.players.find((p) => !p.is_human && p.role === "mafia")
          : null;
      roleBadge.textContent = partner
        ? `Мафия · напарник: ${partner.name}`
        : `Ваша роль: ${human.role_ru}`;
    }

    renderTableScene(state);
    renderGameOver(state);
    renderMessages(state, isNewGameOrReset);
    renderActionBar(state, human);

    lastPhase = state.phase;
  }

  /* =========================================================================
     ILLUSTRATED SVG CHARACTER PORTRAITS (Noir Comic Style)
     ========================================================================= */
  function hashName(name) {
    let h = 2166136261;
    for (let i = 0; i < name.length; i++) {
      h ^= name.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return Math.abs(h);
  }

  function buildAvatarSvg(player, isSpeaking, isThinking) {
    const h = hashName(player.name || String(player.seat));
    const isFemale = player.gender === "F";

    const bgPalettes = ["#243047", "#30263f", "#223836", "#3b2927", "#2b3240", "#342e22", "#263345"];
    const coatColors = ["#3a4b6e", "#6b313d", "#2f574b", "#5c4632", "#4b3b63", "#3e4c59", "#69422b"];
    const skinTones = ["#f2cbb0", "#e7b999", "#f6d7bf", "#dca785", "#ecc2a4"];
    const hairColors = ["#231d1b", "#4a3222", "#6e4b2a", "#191c24", "#8a5a32", "#3a2824", "#b08b57"];

    const bg = bgPalettes[h % bgPalettes.length];
    const coat = coatColors[(h >> 3) % coatColors.length];
    const skin = skinTones[(h >> 6) % skinTones.length];
    const hair = hairColors[(h >> 9) % hairColors.length];
    const styleIdx = (h >> 12) % 3;
    const hasGlasses = ((h >> 15) % 4) === 0;

    // Hair layers
    let backHair = "";
    let frontHair = "";
    if (isFemale) {
      if (styleIdx === 0) {
        // Long wavy hair
        backHair = `<path d="M18,34 C16,52 19,68 23,74 L77,74 C81,68 84,52 82,34 C80,14 68,10 50,10 C32,10 20,14 18,34 Z" fill="${hair}"/>`;
        frontHair = `<path d="M25,35 C27,22 39,17 50,17 C61,17 73,22 75,35 C70,27 61,25 50,26 C39,25 30,27 25,35 Z" fill="${hair}"/>`;
      } else if (styleIdx === 1) {
        // Sleek bob cut
        backHair = `<path d="M21,36 C20,52 24,62 30,64 L70,64 C76,62 80,52 79,36 C78,16 66,11 50,11 C34,11 22,16 21,36 Z" fill="${hair}"/>`;
        frontHair = `<path d="M24,36 C28,20 44,16 62,19 C70,21 75,28 76,36 C68,26 48,24 24,36 Z" fill="${hair}"/>`;
      } else {
        // High bun
        backHair = `<circle cx="50" cy="14" r="11" fill="${hair}"/><path d="M24,36 C24,18 35,14 50,14 C65,14 76,18 76,36 Z" fill="${hair}"/>`;
        frontHair = `<path d="M26,37 C29,24 40,20 50,20 C60,20 71,24 74,37 C67,29 58,27 50,27 C42,27 33,29 26,37 Z" fill="${hair}"/>`;
      }
    } else {
      if (styleIdx === 0) {
        // Classic short crop
        frontHair = `<path d="M26,37 C26,18 37,13 50,13 C63,13 74,18 74,37 C70,28 62,24 50,24 C38,24 30,28 26,37 Z" fill="${hair}"/>`;
      } else if (styleIdx === 1) {
        // Side part / noir wave
        frontHair = `<path d="M25,38 C24,19 37,12 52,13 C67,14 75,21 74,38 C69,28 58,23 41,25 C34,26 29,31 25,38 Z" fill="${hair}"/>`;
      } else {
        // Textured short + subtle stubble
        frontHair = `<path d="M26,36 C27,17 38,12 50,12 C62,12 73,17 74,36 L70,29 L63,25 L50,26 L37,25 L30,29 Z" fill="${hair}"/>`;
      }
    }

    // Eyes (glance up when thinking, X when dead, alert otherwise)
    let eyesSvg = "";
    if (!player.is_alive) {
      eyesSvg = `
        <path d="M36,41 L42,47 M42,41 L36,47" stroke="#221e20" stroke-width="2.4" stroke-linecap="round"/>
        <path d="M58,41 L64,47 M64,41 L58,47" stroke="#221e20" stroke-width="2.4" stroke-linecap="round"/>
      `;
    } else {
      const py = isThinking ? 42.5 : 44.5;
      eyesSvg = `
        <ellipse cx="39" cy="44" rx="4.2" ry="3.2" fill="#fff"/>
        <ellipse cx="61" cy="44" rx="4.2" ry="3.2" fill="#fff"/>
        <circle cx="39.5" cy="${py}" r="2.1" fill="#181c24"/>
        <circle cx="61.5" cy="${py}" r="2.1" fill="#181c24"/>
        <path d="M34,38.5 Q39,36.5 44,38.5" stroke="${hair}" stroke-width="2" fill="none" stroke-linecap="round"/>
        <path d="M56,38.5 Q61,36.5 66,38.5" stroke="${hair}" stroke-width="2" fill="none" stroke-linecap="round"/>
      `;
    }

    const glassesSvg = hasGlasses
      ? `<rect x="32.5" y="39.5" width="13" height="9" rx="3" fill="none" stroke="#1c202a" stroke-width="1.8"/>
         <rect x="54.5" y="39.5" width="13" height="9" rx="3" fill="none" stroke="#1c202a" stroke-width="1.8"/>
         <line x1="45.5" y1="44" x2="54.5" y2="44" stroke="#1c202a" stroke-width="1.8"/>`
      : "";

    // Mouth (open speaking vs thinking vs calm)
    let mouthSvg = "";
    if (!player.is_alive) {
      mouthSvg = `<line x1="44" y1="58" x2="56" y2="58" stroke="#4a2e2c" stroke-width="2.2" stroke-linecap="round"/>`;
    } else if (isSpeaking) {
      mouthSvg = `<ellipse cx="50" cy="57.5" rx="5.5" ry="3.8" fill="#592629" stroke="#2a1618" stroke-width="1.5"/>`;
    } else if (isThinking) {
      mouthSvg = `<path d="M45,58 Q50,56.5 55,58.5" stroke="#5b3630" stroke-width="2.2" fill="none" stroke-linecap="round"/>`;
    } else {
      mouthSvg = `<path d="M44,57 Q50,60.5 56,57" stroke="#5b3630" stroke-width="2.2" fill="none" stroke-linecap="round"/>`;
    }

    // Collar / outfit details
    const outfitDetail = isFemale
      ? `<path d="M39,72 L50,84 L61,72" fill="#e8ebf2" opacity="0.88"/>`
      : `<path d="M41,72 L50,86 L59,72" fill="#e8ebf2" opacity="0.9"/><path d="M48,75 L52,75 L51.2,92 L48.8,92 Z" fill="#a8323e"/>`;

    return `
      <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <rect width="100" height="100" fill="${bg}"/>
        ${backHair}
        <!-- Shoulders / Coat -->
        <path d="M12,102 C14,76 28,69 50,69 C72,69 86,76 88,102 Z" fill="${coat}" stroke="#141821" stroke-width="2.5"/>
        ${outfitDetail}
        <!-- Neck -->
        <rect x="43" y="59" width="14" height="15" rx="5" fill="${skin}"/>
        <!-- Ears -->
        <circle cx="27" cy="46" r="4.5" fill="${skin}"/>
        <circle cx="73" cy="46" r="4.5" fill="${skin}"/>
        <!-- Head -->
        <ellipse cx="50" cy="44" rx="22" ry="25" fill="${skin}" stroke="#1c1f29" stroke-width="2"/>
        ${frontHair}
        ${eyesSvg}
        ${glassesSvg}
        <!-- Nose -->
        <path d="M50,45 L48.5,51.5 L51.5,52" stroke="#b98667" stroke-width="1.6" fill="none" stroke-linecap="round"/>
        ${mouthSvg}
      </svg>
    `;
  }

  /* =========================================================================
     RENDER TABLETOP SCENE WITH 7 CHARACTERS + COMIC BUBBLES
     ========================================================================= */
  function renderTableScene(state) {
    // Center emblem
    if (state.phase === "night") {
      tableIcon.textContent = "☾";
      tablePhaseTitle.textContent = `Ночь ${state.day}`;
    } else if (state.phase === "day_voting") {
      tableIcon.textContent = "⚖";
      tablePhaseTitle.textContent = `День ${state.day} · Голосование`;
    } else if (state.phase === "game_over") {
      tableIcon.textContent = state.winner === "citizen" ? "★" : "☠";
      tablePhaseTitle.textContent =
        state.winner === "citizen" ? "Победа Мирных!" : "Победа Мафии!";
    } else {
      tableIcon.textContent = "☀";
      tablePhaseTitle.textContent = `День ${state.day} · Круг ${state.round_num}/2`;
    }

    // Latest narrator announcement in the center of the felt
    const sysMsgs = state.messages.filter((m) => m.visibility === "system");
    const lastSys = sysMsgs.length > 0 ? sysMsgs[sysMsgs.length - 1].text : "Партия началась";
    tableNarratorText.textContent = lastSys;

    // Order players so Human is always at Slot 0 (bottom center)
    const players = state.players || [];
    if (players.length === 0) {
      seatsLayer.innerHTML = "";
      return;
    }
    const humanIdx = players.findIndex((p) => p.is_human);
    const ordered =
      humanIdx >= 0
        ? [...players.slice(humanIdx), ...players.slice(0, humanIdx)]
        : [...players];

    // Vote tally per player
    const voteCounts = {};
    if (state.current_votes) {
      Object.values(state.current_votes).forEach((target) => {
        voteCounts[target] = (voteCounts[target] || 0) + 1;
      });
    }

    // Determine which single comic speech bubble should be displayed right now
    // Priority 1: Active typewriter animation (types smoothly at ~22 chars/sec)
    // Priority 2: Most recent completed message in the current day/phase
    let activeBubble = null;
    if (isTypewriterActive() && (state.phase !== "night" || typewriter.isMafia)) {
      activeBubble = {
        seat: typewriter.seat,
        text: typewriter.fullText.slice(0, typewriter.shownLen) + "▌",
        vote_target: typewriter.shownLen >= typewriter.fullText.length ? typewriter.voteTarget : null,
        isLastWord: typewriter.isLastWord,
        isMafia: typewriter.isMafia,
      };
    } else if (state.phase !== "night") {
      const recentSpoken = [...state.messages]
        .reverse()
        .find((m) => m.visibility === "public" && m.sender_seat !== null);
      if (recentSpoken) {
        activeBubble = {
          seat: recentSpoken.sender_seat,
          text: recentSpoken.text,
          vote_target: recentSpoken.vote_target,
          isLastWord: recentSpoken.sender_name.includes("Последнее слово"),
          isMafia: false,
        };
      }
    } else {
      // At night, if human is Mafia (or spectator) and AI partner whispered a suggestion, show it!
      const recentMafia = [...state.messages]
        .reverse()
        .find((m) => m.visibility === "mafia" && m.day === state.day && m.sender_seat !== null);
      if (recentMafia) {
        activeBubble = {
          seat: recentMafia.sender_seat,
          text: recentMafia.text,
          vote_target: null,
          isLastWord: false,
          isMafia: true,
        };
      }
    }

    const showAllRoles = Boolean(state.spectator_reveal || state.phase === "game_over");
    seatsLayer.innerHTML = "";

    ordered.forEach((p, slotIdx) => {
      const isVisibleActive =
        state.active_speaker_seat === p.seat &&
        (state.phase !== "night" || p.is_human);

      const isTypingThis = Boolean(isTypewriterActive() && typewriter.seat === p.seat);
      // Show comic thought cloud ("Думает...") ONLY when an AI is thinking during non-Night and hasn't started typing yet
      const isThinking = Boolean(
        isVisibleActive && !p.is_human && !isTypingThis && state.phase !== "night"
      );
      const hasSpeechBubble = Boolean(
        activeBubble && activeBubble.seat === p.seat && !isThinking
      );
      const isSpeaking = Boolean(isTypingThis || (hasSpeechBubble && isVisibleActive && p.is_human));

      const node = document.createElement("div");
      node.className = `seat-node seat-slot-${slotIdx}`;
      if (p.is_human) node.classList.add("is-human");
      if (!p.is_alive) node.classList.add("is-dead");
      if (isSpeaking) node.classList.add("is-speaking");
      if (isThinking) node.classList.add("is-thinking");

      const tags = [];
      if (p.is_human) tags.push(`<span class="mini-tag tag-you">Вы</span>`);
      if (p.role === "mafia" && (!p.is_human || showAllRoles)) {
        tags.push(`<span class="mini-tag tag-mafia">Мафия</span>`);
      } else if (p.role === "commissar" && (p.is_human || showAllRoles)) {
        tags.push(`<span class="mini-tag tag-commissar">★ Комиссар</span>`);
      } else if (p.role === "doctor" && (p.is_human || showAllRoles)) {
        tags.push(`<span class="mini-tag tag-doctor">✚ Доктор</span>`);
      } else if (p.role === "citizen" && showAllRoles) {
        tags.push(`<span class="mini-tag tag-citizen">Мирный</span>`);
      }
      if (voteCounts[p.name]) {
        tags.push(`<span class="mini-tag tag-votes">Голосов: ${voteCounts[p.name]}</span>`);
      }

      const deadOverlay = !p.is_alive
        ? `<div class="dead-stamp"><span>${p.death_reason === "night" ? "УБИТ" : "КАЗНЁН"}</span></div>`
        : "";

      // Comic thought cloud OR comic speech bubble
      let comicHtml = "";
      if (isThinking) {
        const thoughtLabel = !p.is_alive ? "Последнее слово" : "Думает";
        comicHtml = `
          <div class="comic-thought">
            <span>${thoughtLabel}</span>
            <span class="thought-dots"><i></i><i></i><i></i></span>
          </div>
        `;
      } else if (hasSpeechBubble) {
        const extraClass = activeBubble.isLastWord
          ? "is-last-word"
          : activeBubble.isMafia
          ? "is-mafia-whisper"
          : "";
        const noPopClass = lastShownBubbleSeat === p.seat ? "no-pop" : "";
        const labelHtml = activeBubble.isLastWord
          ? `<div class="comic-speech-label">Последнее слово</div>`
          : activeBubble.isMafia
          ? `<div class="comic-speech-label">Шёпот мафии</div>`
          : "";
        const voteHtml = activeBubble.vote_target
          ? `<div><span class="comic-vote-chip">Голос против: ${escapeHtml(activeBubble.vote_target)}</span></div>`
          : "";
        comicHtml = `
          <div class="comic-speech ${extraClass} ${noPopClass}" data-seat="${p.seat}">
            ${labelHtml}
            <div class="comic-speech-text">${escapeHtml(activeBubble.text)}</div>
            ${voteHtml}
          </div>
        `;
      }

      node.innerHTML = `
        ${comicHtml}
        <div class="avatar-frame">
          ${buildAvatarSvg(p, isSpeaking, isThinking)}
          ${deadOverlay}
        </div>
        <div class="seat-plate">
          <div class="seat-name-row">
            <span class="seat-name-text">${escapeHtml(p.name)}</span>
          </div>
          ${tags.length ? `<div class="seat-badges">${tags.join("")}</div>` : ""}
        </div>
      `;
      seatsLayer.appendChild(node);
    });

    lastShownBubbleSeat = activeBubble ? activeBubble.seat : null;
  }

  function renderGameOver(state) {
    if (state.phase !== "game_over") {
      gameOverCard.classList.add("hidden");
      return;
    }

    gameOverCard.classList.remove("hidden");
    winnerTitle.textContent =
      state.winner === "citizen" ? "Победа Мирных жителей!" : "Победа Мафии!";
    winnerReason.textContent = state.win_reason || "";

    revealTableBody.innerHTML = "";
    state.players.forEach((p) => {
      const tr = document.createElement("tr");
      const roleClass = `tag-${p.role || "citizen"}`;
      const fate = p.is_alive
        ? "Выжил(а)"
        : p.death_reason === "night"
        ? `Убит(а) ночью ${p.death_day}`
        : `Казнён(а) днём ${p.death_day}`;

      tr.innerHTML = `
        <td>#${p.seat}</td>
        <td><strong>${escapeHtml(p.name)}</strong> (${p.gender_ru})${p.is_human ? " · Вы" : ""}</td>
        <td><span class="mini-tag ${roleClass}">${escapeHtml(p.role_ru)}</span></td>
        <td>${fate}</td>
      `;
      revealTableBody.appendChild(tr);
    });
  }

  function renderMessages(state, forceFullRebuild) {
    const existingStream = document.getElementById("liveStreamBubble");
    if (existingStream) {
      existingStream.remove();
    }
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
        div.dataset.msgId = m.id;
        if (human && m.sender_seat === human.seat && m.visibility === "public") {
          div.classList.add("msg-human");
        }
        if (m.visibility === "mafia") {
          div.classList.add("msg-mafia");
        } else if (m.visibility === "commissar") {
          div.classList.add("msg-commissar");
        } else if (m.visibility === "doctor") {
          div.classList.add("msg-doctor");
        }

        const phaseTag =
          m.visibility === "mafia"
            ? `Ночь ${m.day} · Мафия`
            : m.visibility === "commissar"
            ? `Ночь ${m.day} · Проверка`
            : m.visibility === "doctor"
            ? `Ночь ${m.day} · Лечение`
            : `День ${m.day}`;
        const voteHtml = m.vote_target
          ? `<div class="vote-pill">Голос против: ${escapeHtml(m.vote_target)}</div>`
          : "";
        const initialMsgText =
          isTypewriterActive() && typewriter.msgId === m.id
            ? typewriter.fullText.slice(0, typewriter.shownLen) + "▌"
            : m.text;

        div.innerHTML = `
          <div class="msg-header">
            <span class="msg-author">${escapeHtml(m.sender_name)}</span>
            <span class="msg-meta">${phaseTag}</span>
          </div>
          <div class="msg-text">${escapeHtml(initialMsgText)}</div>
          ${voteHtml}
        `;
        chatFeed.appendChild(div);
      }
    }

    if (state.messages.length > renderedMessageCount) {
      renderedMessageCount = state.messages.length;
      firstRenderedMessageId = state.messages[0].id;
      chatFeed.scrollTop = chatFeed.scrollHeight;
    }
  }

  function renderActionBar(state, human) {
    waitingStatusLine.classList.add("hidden");
    speakForm.classList.add("hidden");
    targetForm.classList.add("hidden");

    if (state.phase === "game_over") {
      waitingStatusLine.classList.remove("hidden");
      waitingStatusText.textContent = "Партия завершена. Все роли открыты за столом и в таблице выше.";
      return;
    }

    if (state.waiting_for_human === "speak" || state.waiting_for_human === "last_word") {
      const isLastWord = state.waiting_for_human === "last_word";
      speakForm.classList.remove("hidden");
      speakInput.placeholder = isLastWord
        ? "Ваше последнее слово городу перед уходом (после этого откроются все роли)..."
        : "Ваша очередь говорить за столом...";
      speakSubmitBtn.textContent = isLastWord ? "Оставить завещание" : "Сказать";
      skipLastWordBtn.classList.toggle("hidden", !isLastWord);
      speakInput.focus();
      return;
    }

    if (
      state.waiting_for_human === "vote" ||
      state.waiting_for_human === "night_kill" ||
      state.waiting_for_human === "night_heal" ||
      state.waiting_for_human === "night_check"
    ) {
      targetForm.classList.remove("hidden");
      const mode = state.waiting_for_human;
      const isTargetOnly = mode === "night_heal" || mode === "night_check";
      targetCommentInput.classList.toggle("hidden", isTargetOnly);

      if (mode === "night_kill") {
        const suggestionHint = state.mafia_suggestion
          ? ` (напарник предлагает: ${state.mafia_suggestion})`
          : "";
        targetPromptLabel.textContent = `Ночной выстрел мафии — выберите жертву${suggestionHint}:`;
        confirmTargetBtn.textContent = "Убрать выбранную цель";
        confirmTargetBtn.className = "btn btn-danger";
      } else if (mode === "night_heal") {
        targetPromptLabel.textContent = "Ночной ход Доктора — кого защитить этой ночью?";
        confirmTargetBtn.textContent = "Вылечить игрока";
        confirmTargetBtn.className = "btn btn-primary";
      } else if (mode === "night_check") {
        targetPromptLabel.textContent = "Ночной ход Комиссара — кого проверить на мафию?";
        confirmTargetBtn.textContent = "Проверить игрока";
        confirmTargetBtn.className = "btn btn-primary";
      } else {
        targetPromptLabel.textContent = "Открытое голосование — против кого вы голосуете?";
        confirmTargetBtn.textContent = "Проголосовать";
        confirmTargetBtn.className = "btn btn-danger";
      }

      if (!state.valid_targets.includes(selectedTarget)) {
        selectedTarget =
          mode === "night_kill" &&
          state.mafia_suggestion &&
          state.valid_targets.includes(state.mafia_suggestion)
            ? state.mafia_suggestion
            : "";
      }
      renderTargetChips(state.valid_targets, mode === "night_kill" ? state.mafia_suggestion : null);
      return;
    }

    waitingStatusLine.classList.remove("hidden");
    const activePlayer = state.players.find((p) => p.seat === state.active_speaker_seat);
    const typingPlayer =
      isTypewriterActive() && typewriter.seat
        ? state.players.find((p) => p.seat === typewriter.seat)
        : null;
    if (human && !human.is_alive) {
      waitingStatusText.textContent =
        state.phase === "night"
          ? "Вы выбыли (режим наблюдателя — все роли раскрыты) · Ночь: мафия выбирает жертву..."
          : typingPlayer
          ? `Вы выбыли (режим наблюдателя — все роли раскрыты) · Говорит ${typingPlayer.name}...`
          : activePlayer
          ? `Вы выбыли (режим наблюдателя — все роли раскрыты) · Сейчас ходит ${activePlayer.name}...`
          : "Вы выбыли (режим наблюдателя — все роли раскрыты) · ИИ продолжают партию...";
    } else if (state.phase === "night") {
      waitingStatusText.textContent = "Город спит... Мафия выбирает жертву.";
    } else if (typingPlayer) {
      waitingStatusText.textContent = `${typingPlayer.name} говорит за столом...`;
    } else if (activePlayer) {
      waitingStatusText.textContent = `${activePlayer.name} обдумывает реплику...`;
    } else {
      waitingStatusText.textContent = "Пауза на чтение реплики...";
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
    const chosenModel = modelSelect.value || "gemma3:4b";
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

  speakForm.addEventListener("submit", (e) => {
    e.preventDefault();
    if (!gameState) return;
    const mode = gameState.waiting_for_human === "last_word" ? "last_word" : "speak";
    const text = speakInput.value.trim();
    if (!text && mode === "speak") return;
    sendAction({ action: mode, text });
    speakInput.value = "";
  });

  skipLastWordBtn.addEventListener("click", () => {
    if (!gameState || gameState.waiting_for_human !== "last_word") return;
    sendAction({ action: "last_word", text: "" });
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
