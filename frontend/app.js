/**
 * MedReport Copilot — Frontend Application Engine.
 * Handles tab switching, preset loading, PDF drag-and-drop, MediaRecorder audio,
 * REST API communication with FastAPI backend, timeline rendering, and Text-to-Speech (TTS).
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const reportTextInput = document.getElementById('reportTextInput');
    const pdfDropZone = document.getElementById('pdfDropZone');
    const pdfFileInput = document.getElementById('pdfFileInput');
    const pdfFileSelected = document.getElementById('pdfFileSelected');
    const recordMicBtn = document.getElementById('recordMicBtn');
    const recordStatus = document.getElementById('recordStatus');
    const recordingTimer = document.getElementById('recordingTimer');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const analyzeBtnText = document.getElementById('analyzeBtnText');
    const analyzeSpinner = document.getElementById('analyzeSpinner');
    
    const emptyState = document.getElementById('emptyState');
    const resultsContainer = document.getElementById('resultsContainer');
    const specialtyValue = document.getElementById('specialtyValue');
    const confidenceValue = document.getElementById('confidenceValue');
    const routeValue = document.getElementById('routeValue');
    const modelUsedText = document.getElementById('modelUsedText');
    const timelineSteps = document.getElementById('timelineSteps');
    const entitiesChips = document.getElementById('entitiesChips');
    const explanationText = document.getElementById('explanationText');
    const evidenceList = document.getElementById('evidenceList');
    const ttsPlayBtn = document.getElementById('ttsPlayBtn');
    const ttsStopBtn = document.getElementById('ttsStopBtn');

    // Audio & State
    let activeTab = 'textTab';
    let selectedPdfFile = null;
    let mediaRecorder = null;
    let audioChunks = [];
    let recordedAudioBlob = null;
    let timerInterval = null;
    let secondsRecorded = 0;
    let speechSynthesisUtterance = null;

    // Preset Data
    window.presets = {
        cardio: "Patient presents with chest discomfort radiating to left arm. ECG demonstrates sinus tachycardia at 104 bpm with non-specific T-wave inversions in leads V3-V5. Serum Troponin I elevated at 0.45 ng/mL. Echocardiography shows preserved LVEF of 58%. Assessment: Acute coronary syndrome evaluation.",
        neuro: "Patient evaluated for persistent left-sided hemicranial headache accompanied by photophobia. Neurological examination reveals intact cranial nerves II-XII. MRI Brain reveals non-specific T2/FLAIR white matter hyperintensities in cerebral hemispheres. EEG requested.",
        gastro: "Patient reports epigastric burning pain. Esophagogastroduodenoscopy (EGD) reveals patchy mucosal erythema in gastric antrum consistent with gastritis. Abdominal ultrasound shows Grade 1 hepatic steatosis. Liver enzymes: ALT 48 U/L, AST 42 U/L."
    };

    window.loadPreset = (key) => {
        if (window.presets[key]) {
            reportTextInput.value = window.presets[key];
            // Switch to text tab
            switchTab('textTab');
        }
    };

    // Tab Switching
    function switchTab(targetTabId) {
        activeTab = targetTabId;
        tabBtns.forEach(btn => {
            if (btn.dataset.tab === targetTabId) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        tabContents.forEach(content => {
            if (content.id === targetTabId) {
                content.classList.add('active');
            } else {
                content.classList.remove('active');
            }
        });
    }

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });

    // PDF File Selection
    pdfDropZone.addEventListener('click', () => pdfFileInput.click());
    pdfDropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        pdfDropZone.style.borderColor = 'var(--accent-teal)';
    });
    pdfDropZone.addEventListener('dragleave', () => {
        pdfDropZone.style.borderColor = 'var(--border-color)';
    });
    pdfDropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        pdfDropZone.style.borderColor = 'var(--border-color)';
        if (e.dataTransfer.files.length > 0) {
            handlePdfSelect(e.dataTransfer.files[0]);
        }
    });

    pdfFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handlePdfSelect(e.target.files[0]);
        }
    });

    function handlePdfSelect(file) {
        if (file.type !== 'application/pdf' && !file.name.endsWith('.pdf')) {
            alert('Please select a valid PDF file.');
            return;
        }
        selectedPdfFile = file;
        pdfFileSelected.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    }

    // Voice Dictation Recording
    recordMicBtn.addEventListener('click', async () => {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            stopRecording();
        } else {
            startRecording();
        }
    });

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) audioChunks.push(e.data);
            };

            mediaRecorder.onstop = () => {
                recordedAudioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                recordStatus.textContent = 'Recording completed. Ready for analysis.';
            };

            mediaRecorder.start();
            recordMicBtn.classList.add('recording');
            recordStatus.textContent = 'Recording dictation... Click again to stop.';
            
            secondsRecorded = 0;
            recordingTimer.textContent = '00:00';
            timerInterval = setInterval(() => {
                secondsRecorded++;
                const mins = String(Math.floor(secondsRecorded / 60)).padStart(2, '0');
                const secs = String(secondsRecorded % 60).padStart(2, '0');
                recordingTimer.textContent = `${mins}:${secs}`;
            }, 1000);

        } catch (err) {
            alert('Microphone access permission denied or unavailable: ' + err.message);
        }
    }

    function stopRecording() {
        if (mediaRecorder) {
            mediaRecorder.stop();
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
            recordMicBtn.classList.remove('recording');
            clearInterval(timerInterval);
        }
    }

    // Analyze Action Execution
    analyzeBtn.addEventListener('click', async () => {
        const formData = new FormData();

        if (activeTab === 'textTab') {
            const textContent = reportTextInput.value.trim();
            if (!textContent) {
                alert('Please enter clinical report text or select a preset sample.');
                return;
            }
            formData.append('text', textContent);
        } else if (activeTab === 'pdfTab') {
            if (!selectedPdfFile) {
                alert('Please select a PDF file first.');
                return;
            }
            formData.append('file', selectedPdfFile);
        } else if (activeTab === 'voiceTab') {
            if (!recordedAudioBlob) {
                alert('Please record voice dictation audio first.');
                return;
            }
            formData.append('file', recordedAudioBlob, 'dictation.wav');
        }

        // Show Loading UI
        setLoading(true);

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Analysis request failed.');
            }

            const data = await response.json();
            renderResults(data);

        } catch (err) {
            alert('Error processing report: ' + err.message);
        } finally {
            setLoading(false);
        }
    });

    function setLoading(isLoading) {
        if (isLoading) {
            analyzeBtnText.textContent = 'Orchestrating Agents...';
            analyzeSpinner.classList.remove('hidden');
            analyzeBtn.disabled = true;
        } else {
            analyzeBtnText.textContent = '⚡ Analyze Report with Copilot';
            analyzeSpinner.classList.add('hidden');
            analyzeBtn.disabled = false;
        }
    }

    // Render Results Panel
    function renderResults(data) {
        emptyState.classList.add('hidden');
        resultsContainer.classList.remove('hidden');

        // Specialty & Route Badges
        specialtyValue.textContent = data.specialty || 'General Medicine';
        confidenceValue.textContent = `${data.confidence_percentage}% Confidence`;
        routeValue.textContent = data.route_mode === 'FAST_PATH' ? '⚡ FAST PATH' : '🧠 DEEP AGENT PATH';
        modelUsedText.textContent = `Inference Model: ${data.model_used}`;

        // Timeline Steps
        timelineSteps.innerHTML = '';
        (data.execution_pipeline || []).forEach(step => {
            const item = document.createElement('div');
            item.className = 'timeline-item';
            item.innerHTML = `
                <div>
                    <span class="agent-name">${step.agent}:</span>
                    <span class="agent-detail">${step.detail}</span>
                </div>
                <span class="status-tag ${step.status}">${step.status}</span>
            `;
            timelineSteps.appendChild(item);
        });

        // Entity Chips
        entitiesChips.innerHTML = '';
        const ent = data.entities || {};
        
        (ent.symptoms || []).forEach(s => {
            const chip = document.createElement('span');
            chip.className = 'chip symptom';
            chip.textContent = `Symptom: ${s}`;
            entitiesChips.appendChild(chip);
        });

        (ent.measurements || []).forEach(m => {
            const chip = document.createElement('span');
            chip.className = 'chip measurement';
            chip.textContent = `Measurement: ${m}`;
            entitiesChips.appendChild(chip);
        });

        (ent.tests || []).forEach(t => {
            const chip = document.createElement('span');
            chip.className = 'chip test';
            chip.textContent = `Test: ${t}`;
            entitiesChips.appendChild(chip);
        });

        // Patient Explanation
        explanationText.textContent = data.explanation || 'Explanation pending.';

        // RAG Evidence
        evidenceList.innerHTML = '';
        (data.evidence || []).forEach(ev => {
            const evItem = document.createElement('div');
            evItem.className = 'evidence-item';
            evItem.innerHTML = `
                <div class="evidence-header">
                    <span>${ev.title}</span>
                    <span class="evidence-score">Relevance: ${(ev.relevance_score * 100).toFixed(1)}%</span>
                </div>
                <div>${ev.text}</div>
            `;
            evidenceList.appendChild(evItem);
        });
    }

    // Text-To-Speech (TTS) Integration
    ttsPlayBtn.addEventListener('click', () => {
        const text = explanationText.textContent;
        if (!text || !('speechSynthesis' in window)) {
            alert('Text-to-speech is not supported in this browser.');
            return;
        }

        window.speechSynthesis.cancel(); // Stop any active speech
        speechSynthesisUtterance = new SpeechSynthesisUtterance(text);
        speechSynthesisUtterance.rate = 0.95;
        speechSynthesisUtterance.pitch = 1.0;

        speechSynthesisUtterance.onend = () => {
            ttsPlayBtn.classList.remove('hidden');
            ttsStopBtn.classList.add('hidden');
        };

        window.speechSynthesis.speak(speechSynthesisUtterance);
        ttsPlayBtn.classList.add('hidden');
        ttsStopBtn.classList.remove('hidden');
    });

    ttsStopBtn.addEventListener('click', () => {
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
        }
        ttsPlayBtn.classList.remove('hidden');
        ttsStopBtn.classList.add('hidden');
    });

    // Check System Health on Load
    fetch('/api/health')
        .then(res => res.json())
        .then(data => {
            const badge = document.getElementById('systemHealthBadge');
            if (data.status === 'online') {
                badge.innerHTML = `<span class="dot"></span> Backend Active (${data.models.rag_chunks_indexed} KB chunks loaded)`;
            }
        })
        .catch(() => {
            const badge = document.getElementById('systemHealthBadge');
            badge.style.color = 'var(--accent-amber)';
            badge.innerHTML = `<span class="dot" style="background:var(--accent-amber)"></span> Backend Connecting...`;
        });
});
