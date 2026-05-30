const { BrowserWindow, ipcMain } = require('electron');
const { spawn } = require('child_process');
const WebSocket = require('ws');
const { saveDebugAudio } = require('../audioUtils');
const { getSystemPrompt } = require('./prompts');
const { getApiKey } = require('../storage');
const { connectCloud, sendCloudAudio, sendCloudText, sendCloudImage, closeCloud, setOnTurnComplete } = require('./cloud');

// Lazy-loaded to avoid circular dependency (localai.js imports from gemini.js)
let _localai = null;
function getLocalAi() {
    if (!_localai) _localai = require('./localai');
    return _localai;
}

const OPENAI_REALTIME_MODEL = 'gpt-realtime';
const OPENAI_VISION_MODEL = 'gpt-5.4-mini';

// Provider mode: 'byok', 'cloud', or 'local'
let currentProviderMode = 'byok';

// Conversation tracking variables
let currentSessionId = null;
let currentProfile = null;
let currentCustomPrompt = null;
let currentSystemPrompt = null;
let conversationHistory = [];
let screenAnalysisHistory = [];
let isInitializingSession = false;

// OpenAI realtime state
let openAiWs = null;
let currentTranscription = '';
let currentTypedInput = '';
let activeResponseId = null;
let activeResponseText = '';
let responseInputById = new Map();
let pendingTypedInputs = [];

// Audio capture variables
let systemAudioProc = null;

// Reconnection variables
let isUserClosing = false;
let sessionParams = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 3;
const RECONNECT_DELAY = 2000;

function sendToRenderer(channel, data) {
    const windows = BrowserWindow.getAllWindows();
    if (windows.length > 0) {
        windows[0].webContents.send(channel, data);
    }
}

function formatSpeakerResults(results) {
    let text = '';
    for (const result of results) {
        if (result.transcript && result.speakerId) {
            const speakerLabel = result.speakerId === 1 ? 'Interviewer' : 'Candidate';
            text += `[${speakerLabel}]: ${result.transcript}\n`;
        }
    }
    return text;
}

module.exports.formatSpeakerResults = formatSpeakerResults;

function buildContextMessage() {
    const lastTurns = conversationHistory.slice(-20);
    const validTurns = lastTurns.filter(turn => turn.transcription?.trim() && turn.ai_response?.trim());

    if (validTurns.length === 0) return null;

    const contextLines = validTurns.map(turn => `[User]: ${turn.transcription.trim()}\n[Assistant]: ${turn.ai_response.trim()}`);

    return `Previous conversation context:\n\n${contextLines.join('\n\n')}`;
}

function initializeNewSession(profile = null, customPrompt = null) {
    currentSessionId = Date.now().toString();
    currentProfile = profile;
    currentCustomPrompt = customPrompt;
    conversationHistory = [];
    screenAnalysisHistory = [];
    currentTranscription = '';
    currentTypedInput = '';
    activeResponseId = null;
    activeResponseText = '';
    responseInputById = new Map();
    pendingTypedInputs = [];

    if (profile) {
        sendToRenderer('save-session-context', {
            sessionId: currentSessionId,
            profile,
            customPrompt: customPrompt || '',
        });
    }
}

function saveConversationTurn(transcription, aiResponse) {
    if (!currentSessionId) {
        initializeNewSession();
    }

    const normalizedTranscription = (transcription || '').trim();
    const normalizedResponse = (aiResponse || '').trim();
    if (!normalizedResponse) return;

    const conversationTurn = {
        timestamp: Date.now(),
        transcription: normalizedTranscription,
        ai_response: normalizedResponse,
    };

    conversationHistory.push(conversationTurn);

    sendToRenderer('save-conversation-turn', {
        sessionId: currentSessionId,
        turn: conversationTurn,
        fullHistory: conversationHistory,
    });
}

function saveScreenAnalysis(prompt, response, model) {
    if (!currentSessionId) {
        initializeNewSession();
    }

    const analysisEntry = {
        timestamp: Date.now(),
        prompt,
        response: (response || '').trim(),
        model,
    };

    screenAnalysisHistory.push(analysisEntry);

    sendToRenderer('save-screen-analysis', {
        sessionId: currentSessionId,
        analysis: analysisEntry,
        fullHistory: screenAnalysisHistory,
        profile: currentProfile,
        customPrompt: currentCustomPrompt,
    });
}

function getCurrentSessionData() {
    return {
        sessionId: currentSessionId,
        history: conversationHistory,
    };
}

async function getEnabledTools() {
    return [];
}

async function getStoredSetting(key, defaultValue) {
    return defaultValue;
}

function closeOpenAiSocket() {
    if (!openAiWs) return;
    try {
        openAiWs.removeAllListeners();
        openAiWs.close();
    } catch (error) {
        console.error('Error closing OpenAI realtime socket:', error);
    }
    openAiWs = null;
}

function queueResponseInput(text) {
    const value = (text || '').trim();
    if (value) {
        pendingTypedInputs.push(value);
    }
}

function extractResponseText(payload) {
    if (!payload) return '';
    if (typeof payload.output_text === 'string') {
        return payload.output_text.trim();
    }

    const texts = [];
    const visit = value => {
        if (!value) return;
        if (Array.isArray(value)) {
            value.forEach(visit);
            return;
        }
        if (typeof value === 'object') {
            if (typeof value.text === 'string' && (value.type === 'output_text' || value.type === 'text')) {
                texts.push(value.text);
            }
            Object.values(value).forEach(visit);
        }
    };

    visit(payload.output);
    return texts.join('').trim();
}

function handleRealtimeMessage(rawMessage) {
    let event;
    try {
        event = JSON.parse(rawMessage.toString());
    } catch (error) {
        console.error('Failed to parse OpenAI realtime event:', error);
        return;
    }

    switch (event.type) {
        case 'session.created':
        case 'session.updated':
            sendToRenderer('update-status', 'Listening...');
            return;

        case 'input_audio_buffer.speech_started':
            sendToRenderer('update-status', 'Listening...');
            return;

        case 'input_audio_buffer.speech_stopped':
            sendToRenderer('update-status', 'Thinking...');
            return;

        case 'conversation.item.input_audio_transcription.delta':
            currentTranscription += event.delta || '';
            return;

        case 'conversation.item.input_audio_transcription.completed':
            currentTranscription = (event.transcript || event.text || currentTranscription).trim();
            return;

        case 'response.created':
            activeResponseId = event.response?.id || null;
            activeResponseText = '';
            if (activeResponseId && !responseInputById.has(activeResponseId)) {
                responseInputById.set(activeResponseId, pendingTypedInputs.length > 0 ? pendingTypedInputs.shift() : '');
            }
            sendToRenderer('update-status', 'Generating response...');
            return;

        case 'response.output_text.delta':
            if (event.response_id) {
                activeResponseId = event.response_id;
            }
            activeResponseText += event.delta || '';
            sendToRenderer(activeResponseText.length === (event.delta || '').length ? 'new-response' : 'update-response', activeResponseText);
            return;

        case 'response.output_text.done':
            if (event.text && !activeResponseText) {
                activeResponseText = event.text;
                sendToRenderer('new-response', activeResponseText);
            }
            return;

        case 'response.done': {
            const responseId = event.response?.id || activeResponseId;
            const finalText = extractResponseText(event.response) || activeResponseText;
            const inputText = (responseId ? responseInputById.get(responseId) : '') || currentTypedInput || currentTranscription;

            if (finalText.trim()) {
                saveConversationTurn(inputText, finalText);
            }

            if (responseId) {
                responseInputById.delete(responseId);
            }

            activeResponseId = null;
            activeResponseText = '';
            currentTypedInput = '';
            currentTranscription = '';
            sendToRenderer('update-status', 'Listening...');
            return;
        }

        case 'error':
            console.error('OpenAI realtime error:', event.error || event);
            sendToRenderer('update-status', 'Error: ' + (event.error?.message || 'Realtime API error'));
            return;

        default:
            return;
    }
}

async function initializeGeminiSession(apiKey, customPrompt = '', profile = 'interview', language = 'en-US', isReconnect = false) {
    if (isInitializingSession) {
        return false;
    }

    isInitializingSession = true;
    if (!isReconnect) {
        sendToRenderer('session-initializing', true);
    }

    if (!isReconnect) {
        sessionParams = { apiKey, customPrompt, profile, language };
        reconnectAttempts = 0;
    }

    currentSystemPrompt = getSystemPrompt(profile, customPrompt, false);

    if (!isReconnect) {
        initializeNewSession(profile, customPrompt);
    }

    return await new Promise(resolve => {
        const ws = new WebSocket(`wss://api.openai.com/v1/realtime?model=${OPENAI_REALTIME_MODEL}`, {
            headers: {
                Authorization: `Bearer ${apiKey}`,
            },
        });

        let settled = false;

        const settle = value => {
            if (settled) return;
            settled = true;
            isInitializingSession = false;
            if (!isReconnect) {
                sendToRenderer('session-initializing', false);
            }
            resolve(value);
        };

        ws.on('open', () => {
            openAiWs = ws;

            ws.send(JSON.stringify({
                type: 'session.update',
                session: {
                    type: 'realtime',
                    instructions: currentSystemPrompt,
                    output_modalities: ['text'],
                    audio: {
                        input: {
                            format: {
                                type: 'audio/pcm',
                                rate: 24000,
                            },
                            turn_detection: {
                                type: 'server_vad',
                            },
                        },
                    },
                },
            }));

            const contextMessage = isReconnect ? buildContextMessage() : null;
            if (contextMessage) {
                ws.send(JSON.stringify({
                    type: 'conversation.item.create',
                    item: {
                        type: 'message',
                        role: 'user',
                        content: [
                            {
                                type: 'input_text',
                                text: contextMessage,
                            },
                        ],
                    },
                }));
            }

            settle(ws);
        });

        ws.on('message', handleRealtimeMessage);

        ws.on('error', error => {
            console.error('OpenAI realtime socket error:', error);
            if (!settled) {
                settle(null);
            } else {
                sendToRenderer('update-status', 'Error: ' + error.message);
            }
        });

        ws.on('close', () => {
            if (openAiWs === ws) {
                openAiWs = null;
            }

            if (!settled) {
                settle(null);
                return;
            }

            if (isUserClosing) {
                isUserClosing = false;
                sendToRenderer('update-status', 'Session closed');
                return;
            }

            if (sessionParams && reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
                attemptReconnect();
            } else {
                sendToRenderer('update-status', 'Session closed');
            }
        });
    });
}

async function attemptReconnect() {
    reconnectAttempts++;
    sendToRenderer('update-status', `Reconnecting... (${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})`);

    closeOpenAiSocket();
    activeResponseId = null;
    activeResponseText = '';

    await new Promise(resolve => setTimeout(resolve, RECONNECT_DELAY));

    try {
        const session = await initializeGeminiSession(
            sessionParams.apiKey,
            sessionParams.customPrompt,
            sessionParams.profile,
            sessionParams.language,
            true
        );

        if (session && global.geminiSessionRef) {
            global.geminiSessionRef.current = session;
            sendToRenderer('update-status', 'Reconnected! Listening...');
            return true;
        }
    } catch (error) {
        console.error(`Reconnection attempt ${reconnectAttempts} failed:`, error);
    }

    if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
        return attemptReconnect();
    }

    sendToRenderer('reconnect-failed', {
        message: 'Tried 3 times to reconnect. Must be upstream/network issues. Try restarting or download updated app from site.',
    });
    sessionParams = null;
    return false;
}

function killExistingSystemAudioDump() {
    return new Promise(resolve => {
        const killProc = spawn('pkill', ['-f', 'SystemAudioDump'], {
            stdio: 'ignore',
        });

        killProc.on('close', () => resolve());
        killProc.on('error', () => resolve());

        setTimeout(() => {
            killProc.kill();
            resolve();
        }, 2000);
    });
}

async function startMacOSAudioCapture(geminiSessionRef) {
    if (process.platform !== 'darwin') return false;

    await killExistingSystemAudioDump();

    const { app } = require('electron');
    const path = require('path');

    let systemAudioPath;
    if (app.isPackaged) {
        systemAudioPath = path.join(process.resourcesPath, 'SystemAudioDump');
    } else {
        systemAudioPath = path.join(__dirname, '../assets', 'SystemAudioDump');
    }

    systemAudioProc = spawn(systemAudioPath, [], {
        stdio: ['ignore', 'pipe', 'pipe'],
        env: { ...process.env },
    });

    if (!systemAudioProc.pid) {
        return false;
    }

    const CHUNK_DURATION = 0.1;
    const SAMPLE_RATE = 24000;
    const BYTES_PER_SAMPLE = 2;
    const CHANNELS = 2;
    const CHUNK_SIZE = SAMPLE_RATE * BYTES_PER_SAMPLE * CHANNELS * CHUNK_DURATION;

    let audioBuffer = Buffer.alloc(0);

    systemAudioProc.stdout.on('data', data => {
        audioBuffer = Buffer.concat([audioBuffer, data]);

        while (audioBuffer.length >= CHUNK_SIZE) {
            const chunk = audioBuffer.slice(0, CHUNK_SIZE);
            audioBuffer = audioBuffer.slice(CHUNK_SIZE);

            const monoChunk = CHANNELS === 2 ? convertStereoToMono(chunk) : chunk;

            if (currentProviderMode === 'cloud') {
                sendCloudAudio(monoChunk);
            } else if (currentProviderMode === 'local') {
                getLocalAi().processLocalAudio(monoChunk);
            } else {
                const base64Data = monoChunk.toString('base64');
                sendAudioToGemini(base64Data, geminiSessionRef);
            }

            if (process.env.DEBUG_AUDIO) {
                saveDebugAudio(monoChunk, 'system_audio');
            }
        }

        const maxBufferSize = SAMPLE_RATE * BYTES_PER_SAMPLE;
        if (audioBuffer.length > maxBufferSize) {
            audioBuffer = audioBuffer.slice(-maxBufferSize);
        }
    });

    systemAudioProc.stderr.on('data', data => {
        console.error('SystemAudioDump stderr:', data.toString());
    });

    systemAudioProc.on('close', () => {
        systemAudioProc = null;
    });

    systemAudioProc.on('error', err => {
        console.error('SystemAudioDump process error:', err);
        systemAudioProc = null;
    });

    return true;
}

function convertStereoToMono(stereoBuffer) {
    const samples = stereoBuffer.length / 4;
    const monoBuffer = Buffer.alloc(samples * 2);

    for (let i = 0; i < samples; i++) {
        const leftSample = stereoBuffer.readInt16LE(i * 4);
        monoBuffer.writeInt16LE(leftSample, i * 2);
    }

    return monoBuffer;
}

function stopMacOSAudioCapture() {
    if (systemAudioProc) {
        systemAudioProc.kill('SIGTERM');
        systemAudioProc = null;
    }
}

async function sendAudioToGemini(base64Data, geminiSessionRef) {
    if (!openAiWs || openAiWs.readyState !== WebSocket.OPEN) return;

    try {
        process.stdout.write('.');
        openAiWs.send(JSON.stringify({
            type: 'input_audio_buffer.append',
            audio: base64Data,
        }));
    } catch (error) {
        console.error('Error sending audio to OpenAI realtime:', error);
    }
}

async function sendImageToGeminiHttp(base64Data, prompt) {
    const apiKey = getApiKey();
    if (!apiKey) {
        return { success: false, error: 'No API key configured' };
    }

    try {
        const response = await fetch('https://api.openai.com/v1/responses', {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${apiKey}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model: OPENAI_VISION_MODEL,
                instructions: currentSystemPrompt,
                input: [
                    {
                        role: 'user',
                        content: [
                            { type: 'input_text', text: prompt || 'Analyze this screen.' },
                            { type: 'input_image', image_url: `data:image/jpeg;base64,${base64Data}` },
                        ],
                    },
                ],
            }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            return { success: false, error: `OpenAI error ${response.status}: ${errorText}` };
        }

        const payload = await response.json();
        const text = extractResponseText(payload);

        if (text) {
            sendToRenderer('new-response', text);
            saveScreenAnalysis(prompt, text, OPENAI_VISION_MODEL);
        }

        return { success: true, text, model: OPENAI_VISION_MODEL };
    } catch (error) {
        console.error('Error sending image to OpenAI Responses API:', error);
        return { success: false, error: error.message };
    }
}

function setupGeminiIpcHandlers(geminiSessionRef) {
    global.geminiSessionRef = geminiSessionRef;

    ipcMain.handle('initialize-cloud', async (event, token, profile, userContext) => {
        try {
            currentProviderMode = 'cloud';
            initializeNewSession(profile);
            setOnTurnComplete((transcription, response) => {
                saveConversationTurn(transcription, response);
            });
            sendToRenderer('session-initializing', true);
            await connectCloud(token, profile, userContext);
            sendToRenderer('session-initializing', false);
            return true;
        } catch (err) {
            console.error('[Cloud] Init error:', err);
            currentProviderMode = 'byok';
            sendToRenderer('session-initializing', false);
            return false;
        }
    });

    ipcMain.handle('initialize-openai', async (event, apiKey, customPrompt, profile = 'interview', language = 'en-US') => {
        currentProviderMode = 'byok';
        const session = await initializeGeminiSession(apiKey, customPrompt, profile, language);
        if (session) {
            geminiSessionRef.current = session;
            return true;
        }
        return false;
    });

    ipcMain.handle('initialize-local', async (event, ollamaHost, ollamaModel, whisperModel, profile, customPrompt) => {
        currentProviderMode = 'local';
        const success = await getLocalAi().initializeLocalSession(ollamaHost, ollamaModel, whisperModel, profile, customPrompt);
        if (!success) {
            currentProviderMode = 'byok';
        }
        return success;
    });

    ipcMain.handle('send-audio-content', async (event, { data }) => {
        if (currentProviderMode === 'cloud') {
            try {
                sendCloudAudio(Buffer.from(data, 'base64'));
                return { success: true };
            } catch (error) {
                return { success: false, error: error.message };
            }
        }

        if (currentProviderMode === 'local') {
            try {
                getLocalAi().processLocalAudio(Buffer.from(data, 'base64'));
                return { success: true };
            } catch (error) {
                return { success: false, error: error.message };
            }
        }

        if (!openAiWs || openAiWs.readyState !== WebSocket.OPEN) {
            return { success: false, error: 'No active OpenAI session' };
        }

        try {
            openAiWs.send(JSON.stringify({
                type: 'input_audio_buffer.append',
                audio: data,
            }));
            return { success: true };
        } catch (error) {
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('send-mic-audio-content', async (event, { data }) => {
        if (currentProviderMode === 'cloud') {
            try {
                sendCloudAudio(Buffer.from(data, 'base64'));
                return { success: true };
            } catch (error) {
                return { success: false, error: error.message };
            }
        }

        if (currentProviderMode === 'local') {
            try {
                getLocalAi().processLocalAudio(Buffer.from(data, 'base64'));
                return { success: true };
            } catch (error) {
                return { success: false, error: error.message };
            }
        }

        if (!openAiWs || openAiWs.readyState !== WebSocket.OPEN) {
            return { success: false, error: 'No active OpenAI session' };
        }

        try {
            openAiWs.send(JSON.stringify({
                type: 'input_audio_buffer.append',
                audio: data,
            }));
            return { success: true };
        } catch (error) {
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('send-image-content', async (event, { data, prompt }) => {
        try {
            if (!data || typeof data !== 'string') {
                return { success: false, error: 'Invalid image data' };
            }

            const buffer = Buffer.from(data, 'base64');
            if (buffer.length < 1000) {
                return { success: false, error: 'Image buffer too small' };
            }

            process.stdout.write('!');

            if (currentProviderMode === 'cloud') {
                const sent = sendCloudImage(data);
                if (!sent) {
                    return { success: false, error: 'Cloud connection not active' };
                }
                return { success: true, model: 'cloud' };
            }

            if (currentProviderMode === 'local') {
                return await getLocalAi().sendLocalImage(data, prompt);
            }

            return await sendImageToGeminiHttp(data, prompt);
        } catch (error) {
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('send-text-message', async (event, text) => {
        if (!text || typeof text !== 'string' || text.trim().length === 0) {
            return { success: false, error: 'Invalid text message' };
        }

        if (currentProviderMode === 'cloud') {
            try {
                sendCloudText(text.trim());
                return { success: true };
            } catch (error) {
                return { success: false, error: error.message };
            }
        }

        if (currentProviderMode === 'local') {
            try {
                return await getLocalAi().sendLocalText(text.trim());
            } catch (error) {
                return { success: false, error: error.message };
            }
        }

        if (!openAiWs || openAiWs.readyState !== WebSocket.OPEN) {
            return { success: false, error: 'No active OpenAI session' };
        }

        try {
            currentTypedInput = text.trim();
            queueResponseInput(currentTypedInput);

            openAiWs.send(JSON.stringify({
                type: 'conversation.item.create',
                item: {
                    type: 'message',
                    role: 'user',
                    content: [
                        {
                            type: 'input_text',
                            text: currentTypedInput,
                        },
                    ],
                },
            }));

            openAiWs.send(JSON.stringify({
                type: 'response.create',
                response: {
                    output_modalities: ['text'],
                },
            }));

            return { success: true };
        } catch (error) {
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('start-macos-audio', async event => {
        if (process.platform !== 'darwin') {
            return {
                success: false,
                error: 'macOS audio capture only available on macOS',
            };
        }

        try {
            const success = await startMacOSAudioCapture(geminiSessionRef);
            return { success };
        } catch (error) {
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('stop-macos-audio', async event => {
        try {
            stopMacOSAudioCapture();
            return { success: true };
        } catch (error) {
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('close-session', async event => {
        try {
            stopMacOSAudioCapture();

            if (currentProviderMode === 'cloud') {
                closeCloud();
                currentProviderMode = 'byok';
                return { success: true };
            }

            if (currentProviderMode === 'local') {
                getLocalAi().closeLocalSession();
                currentProviderMode = 'byok';
                return { success: true };
            }

            isUserClosing = true;
            sessionParams = null;
            closeOpenAiSocket();
            geminiSessionRef.current = null;
            return { success: true };
        } catch (error) {
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('get-current-session', async event => {
        try {
            return { success: true, data: getCurrentSessionData() };
        } catch (error) {
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('start-new-session', async event => {
        try {
            initializeNewSession();
            return { success: true, sessionId: currentSessionId };
        } catch (error) {
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('update-google-search-setting', async (event, enabled) => {
        return { success: true };
    });
}

module.exports = {
    initializeGeminiSession,
    getEnabledTools,
    getStoredSetting,
    sendToRenderer,
    initializeNewSession,
    saveConversationTurn,
    getCurrentSessionData,
    killExistingSystemAudioDump,
    startMacOSAudioCapture,
    convertStereoToMono,
    stopMacOSAudioCapture,
    sendAudioToGemini,
    sendImageToGeminiHttp,
    setupGeminiIpcHandlers,
    formatSpeakerResults,
};
