let audioContext;
let scriptProcessor;
let audioInput;
let socket;
let isRecording = false;

const recordBtn = document.getElementById('recordBtn');
const clearBtn = document.getElementById('clearBtn');
const asrChunkSelect = document.getElementById('asrChunkSelect');
const astChunkSelect = document.getElementById('astChunkSelect');
const langSelect = document.getElementById('langSelect');
const asrLangSelect = document.getElementById('asrLangSelect');
const enableASRCheckbox = document.getElementById('enableASR');
const enableASTCheckbox = document.getElementById('enableAST');

const asrFinalText = document.getElementById('asrFinalText');
const asrPartialText = document.getElementById('asrPartialText');
const asrDisplay = document.getElementById('asrDisplay');
const asrDot = document.getElementById('asrDot');
const asrStatus = document.getElementById('asrStatus');
const asrPanel = document.getElementById('asrPanel');

const astFinalText = document.getElementById('astFinalText');
const astPartialText = document.getElementById('astPartialText');
const astDisplay = document.getElementById('astDisplay');
const astDot = document.getElementById('astDot');
const astStatus = document.getElementById('astStatus');
const astPanel = document.getElementById('astPanel');

function updatePanelVisibility() {
    asrPanel.classList.toggle('panel-disabled', !enableASRCheckbox.checked);
    astPanel.classList.toggle('panel-disabled', !enableASTCheckbox.checked);
}
updatePanelVisibility();
enableASRCheckbox.addEventListener('change', () => { updatePanelVisibility(); sendConfig(); });
enableASTCheckbox.addEventListener('change', () => { updatePanelVisibility(); sendConfig(); });

function needsSpace(existing, incoming) {
    if (!existing || !incoming) return false;
    const lastChar = existing[existing.length - 1];
    const firstChar = incoming[0];
    const endPunctuation = /[\s.,!?;:，。！？；：、\n\r]$/;
    const startPunctuation = /^[\s.,!?;:，。！？；：、\n\r]/;
    if (endPunctuation.test(lastChar) || startPunctuation.test(firstChar)) return false;
    return true;
}

function appendText(finalEl, display, text) {
    const existing = finalEl.textContent;
    const space = needsSpace(existing, text) ? ' ' : '';
    finalEl.textContent += space + text;
    display.scrollTop = display.scrollHeight;
}

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    socket = new WebSocket(`${protocol}//${window.location.host}/ws/stream`);

    socket.onopen = () => {
        console.log("WebSocket connected.");
        sendConfig();
    };

    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.task === 'asr') {
            if (data.type === 'partial' || data.type === 'final') {
                appendText(asrFinalText, asrDisplay, data.text);
                asrPartialText.textContent = '';
            }
        } else if (data.task === 'ast') {
            if (data.type === 'partial' || data.type === 'final') {
                appendText(astFinalText, astDisplay, data.text);
                astPartialText.textContent = '';
            }
        }
    };

    socket.onclose = () => {
        console.log("WebSocket disconnected.");
        setTimeout(connectWebSocket, 1000);
    };
}

function sendConfig() {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
            type: "config",
            asr_num_chunks: parseInt(asrChunkSelect.value),
            ast_num_chunks: parseInt(astChunkSelect.value),
            lang: langSelect.value,
            asr_lang: asrLangSelect.value,
            enable_asr: enableASRCheckbox.checked,
            enable_ast: enableASTCheckbox.checked,
        }));
    }
}

asrChunkSelect.addEventListener('change', sendConfig);
astChunkSelect.addEventListener('change', sendConfig);
langSelect.addEventListener('change', sendConfig);
asrLangSelect.addEventListener('change', sendConfig);

async function toggleRecording() {
    if (!isRecording) {
        await startRecording();
    } else {
        stopRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 16000 }
        });

        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        audioInput = audioContext.createMediaStreamSource(stream);

        scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
        scriptProcessor.onaudioprocess = (event) => {
            if (!isRecording) return;
            const channelData = event.inputBuffer.getChannelData(0);
            if (socket && socket.readyState === WebSocket.OPEN) {
                socket.send(channelData.buffer);
            }
        };

        audioInput.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination);

        isRecording = true;
        recordBtn.classList.add('recording');

        if (enableASRCheckbox.checked) { asrDot.classList.add('active'); asrStatus.textContent = "Listening..."; }
        if (enableASTCheckbox.checked) { astDot.classList.add('active'); astStatus.textContent = "Listening..."; }

        sendConfig();
    } catch (err) {
        console.error("Microphone access denied or error:", err);
    }
}

function stopRecording() {
    isRecording = false;

    if (scriptProcessor) { scriptProcessor.disconnect(); scriptProcessor = null; }
    if (audioInput) { audioInput.disconnect(); audioInput = null; }
    if (audioContext) { audioContext.close(); audioContext = null; }

    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: "stop" }));
    }

    recordBtn.classList.remove('recording');
    asrDot.classList.remove('active'); asrStatus.textContent = "Ready";
    astDot.classList.remove('active'); astStatus.textContent = "Ready";
}

recordBtn.addEventListener('click', toggleRecording);

clearBtn.addEventListener('click', () => {
    asrFinalText.textContent = '';
    asrPartialText.textContent = '';
    astFinalText.textContent = '';
    astPartialText.textContent = '';
});

connectWebSocket();
