const API_URL = "https://crag-api-route-punit-09-dev.apps.rm2.thpm.p1.openshiftapps.com/query";

const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

userInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

userInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleQuery();
    }
});

sendBtn.addEventListener('click', handleQuery);

async function handleQuery() {
    const queryText = userInput.value.trim();
    if (!queryText) return;

    addMessage(queryText, 'user');
    userInput.value = '';
    userInput.style.height = 'auto';

    const { loaderId, stopStages } = addLoader();
    sendBtn.disabled = true;

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: queryText })
        });

        const data = await response.json();
        stopStages();
        removeElement(loaderId);

        if (!response.ok) {
            let errorMsg = data.detail || data.error || "An unknown error occurred.";
            if (response.status === 429) {
                errorMsg = "Daily rate limit reached. Please try again tomorrow.";
            }
            addMessage(`**Error:** ${errorMsg}`, 'system');
            return;
        }

        buildBotResponse(data);

    } catch (error) {
        stopStages();
        removeElement(loaderId);
        addMessage("**Network Error:** Failed to connect to the cluster.", 'system');
        console.error(error);
    } finally {
        sendBtn.disabled = false;
        userInput.focus();
    }
}

function buildBotResponse(data) {
    const bubble = document.createElement('div');
    bubble.className = 'message system';

    let contentHtml = `<div class="avatar">⚙️</div><div class="message-bubble">`;
    contentHtml += `<div>${marked.parse(data.answer)}</div>`;

    // Group citations by document to avoid repetitive tags
    if (data.sources && data.sources.length > 0) {
        const grouped = {};
        data.sources.forEach(src => {
            if (src.is_web) {
                grouped["Web Knowledge Fallback"] = grouped["Web Knowledge Fallback"] || [];
            } else {
                const name = src.doc_name;
                grouped[name] = grouped[name] || [];
                if (src.page && !grouped[name].includes(src.page)) {
                    grouped[name].push(src.page);
                }
            }
        });

        const sourceStrings = Object.entries(grouped).map(([doc, pages]) => {
            if (pages.length === 0) return doc;
            const sortedPages = pages.sort((a, b) => a - b).join(", ");
            return `${doc} (p. ${sortedPages})`;
        });

        contentHtml += `
            <div class="sources-footer">
                <span class="source-label">Referenced:</span>
                ${sourceStrings.map(str => `<span class="source-item">${str}</span>`).join('')}
            </div>
        `;
    }

    // Diagnostics Drawer
    const isCached = data.cached ? `<span class="metric-pill" style="border-color:#166534; color:#4ade80">⚡ Cached</span>` : '';
    const webUsed = data.web_search_used ? `<span class="metric-pill" style="border-color:#991b1b; color:#f87171">🌐 Web Search</span>` : '';

    contentHtml += `
        <details class="workflow-details">
            <summary>Diagnostics (${data.latency_seconds}s)</summary>
            <div class="workflow-metrics">
                <div>
                    <span class="metric-pill">⏱️ ${data.latency_seconds}s</span>
                    ${isCached}
                    ${webUsed}
                </div>
                <ul>
                    ${data.execution_trace.map(step => `<li>↳ ${step}</li>`).join('')}
                </ul>
            </div>
        </details>
    `;

    contentHtml += `</div>`;
    bubble.innerHTML = contentHtml;

    chatBox.appendChild(bubble);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function addMessage(text, sender) {
    const bubble = document.createElement('div');
    bubble.className = `message ${sender}`;

    const avatar = sender === 'user' ? '👤' : '⚙️';
    const parsedText = sender === 'user' ? text : marked.parse(text);

    bubble.innerHTML = `
        <div class="avatar">${avatar}</div>
        <div class="message-bubble">${parsedText}</div>
    `;

    chatBox.appendChild(bubble);
    chatBox.scrollTop = chatBox.scrollHeight;
}

// Progressive loader ticker to eliminate perceived latency
function addLoader() {
    const id = 'loader-' + Date.now();
    const bubble = document.createElement('div');
    bubble.id = id;
    bubble.className = 'message system';

    bubble.innerHTML = `
        <div class="avatar">⚙️</div>
        <div class="message-bubble">
            <div class="loading-wrapper">
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
                <div class="stage-status" id="${id}-status">Querying vector index & hybrid BM25...</div>
            </div>
        </div>
    `;
    chatBox.appendChild(bubble);
    chatBox.scrollTop = chatBox.scrollHeight;

    const stages = [
        { time: 3500, text: "Retrieving technical documentation candidates..." },
        { time: 8000, text: "Reranking chunks with cross-encoder..." },
        { time: 14000, text: "Evaluating relevance & context grounding..." },
        { time: 19000, text: "Synthesizing precise technical answer..." }
    ];

    const timeouts = [];
    stages.forEach(stage => {
        const t = setTimeout(() => {
            const el = document.getElementById(`${id}-status`);
            if (el) el.textContent = stage.text;
        }, stage.time);
        timeouts.push(t);
    });

    const stopStages = () => timeouts.forEach(clearTimeout);
    return { loaderId: id, stopStages };
}

function removeElement(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}