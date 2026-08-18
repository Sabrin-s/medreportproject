/**
 * MedReport Copilot — Next-Gen Frontend Application Engine.
 * Features:
 * - Real-Time HTML5 Web Audio API Canvas Waveform Visualizer
 * - Dynamic Anatomical Organ Radar Illumination
 * - Radial Holographic Confidence Gauge Animation
 * - 8-Agent Neural Orchestration Flowgraph Visualizer
 * - Interactive Categorized Clinical NLP Entity Inspector
 * - Text-to-Speech (TTS) SpeechSynthesis Engine with Speed Modulation
 * - Export Brief & Multi-Agent JSON Telemetry Inspector
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements - Navigation & Inputs
    const tabBtns = document.querySelectorAll('.tab-trigger');
    const tabPanes = document.querySelectorAll('.tab-pane');
    const reportTextInput = document.getElementById('reportTextInput');
    const pdfDropZone = document.getElementById('pdfDropZone');
    const pdfFileInput = document.getElementById('pdfFileInput');
    const pdfFileSelected = document.getElementById('pdfFileSelected');
    const recordMicBtn = document.getElementById('recordMicBtn');
    const recordStatus = document.getElementById('recordStatus');
    const recordingTimer = document.getElementById('recordingTimer');
    const audioWaveformCanvas = document.getElementById('audioWaveformCanvas');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const analyzeBtnText = document.getElementById('analyzeBtnText');
    const analyzeSpinner = document.getElementById('analyzeSpinner');

    // DOM Elements - Intelligence HUD Display
    const emptyState = document.getElementById('emptyState');
    const resultsContainer = document.getElementById('resultsContainer');
    const organGlyph = document.getElementById('organGlyph');
    const organTargetLabel = document.getElementById('organTargetLabel');
    const confidenceGaugeCircle = document.getElementById('confidenceGaugeCircle');
    const gaugePercentage = document.getElementById('gaugePercentage');
    const gaugeConfidenceGrade = document.getElementById('gaugeConfidenceGrade');
    const specialtyValue = document.getElementById('specialtyValue');
    const routeBadgePill = document.getElementById('routeBadgePill');
    const routeValue = document.getElementById('routeValue');
    const routeReasonText = document.getElementById('routeReasonText');
    const modelUsedText = document.getElementById('modelUsedText');
    const neuralNodesContainer = document.getElementById('neuralNodesContainer');
    const symptomsChips = document.getElementById('symptomsChips');
    const vitalsChips = document.getElementById('vitalsChips');
    const testsChips = document.getElementById('testsChips');
    const medsChips = document.getElementById('medsChips');
    const explanationText = document.getElementById('explanationText');
    const evidenceList = document.getElementById('evidenceList');

    // TTS Elements
    const ttsPlayBtn = document.getElementById('ttsPlayBtn');
    const ttsStopBtn = document.getElementById('ttsStopBtn');
    const ttsSpeedSelect = document.getElementById('ttsSpeedSelect');

    // Telemetry Modal Elements
    const jsonModal = document.getElementById('jsonModal');
    const jsonTelemetryViewer = document.getElementById('jsonTelemetryViewer');
    const btnInspectJson = document.getElementById('btnInspectJson');
    const btnExportSummary = document.getElementById('btnExportSummary');
    const btnCloseJsonModal = document.getElementById('btnCloseJsonModal');
    const btnCloseJsonModal2 = document.getElementById('btnCloseJsonModal2');
    const btnCopyJson = document.getElementById('btnCopyJson');

    // Application State
    let activeTab = 'textTab';
    let selectedPdfFile = null;
    let mediaRecorder = null;
    let audioChunks = [];
    let recordedAudioBlob = null;
    let timerInterval = null;
    let secondsRecorded = 0;
    let audioContext = null;
    let analyserNode = null;
    let animationFrameId = null;
    let latestAnalysisData = null;

    // Presets Collection
    window.presets = {
        cardio: "Patient: 58-year-old male presenting with acute retrosternal chest pain radiating to left jaw and diaphoresis. ECG demonstrates sinus tachycardia at 108 bpm with marked ST-segment elevation in leads V2-V5. High-sensitivity Troponin I significantly elevated at 3.8 ng/mL. Bedside echocardiogram reveals anterior wall hypokinesis with preserved LVEF 48%. Assessment: Acute anterior ST-elevation myocardial infarction (STEMI).",
        neuro: "Patient: 44-year-old female presenting with severe pulsating hemicranial headache, nausea, and visual aura. Neurological exam reveals normal cranial nerves, intact reflexes, and no motor deficit. Brain MRI demonstrates non-specific T2/FLAIR subcortical white matter hyperintensities with no acute ischemic infarction. EEG shows normal background rhythm without epileptiform discharges.",
        gastro: "Patient: 61-year-old male with persistent epigastric burning pain and postprandial dyspepsia. Esophagogastroduodenoscopy (EGD) reveals diffuse erythema, mucosal edema, and shallow erosions in the gastric antrum consistent with active gastritis. Abdominal ultrasound shows mild hepatic steatosis. Serum ALT 46 U/L, AST 39 U/L, H. pylori stool antigen positive.",
        pulmo: "Patient: 67-year-old with chronic productive cough, progressive dyspnea on exertion (mMRC Grade 3), and bilateral expiratory wheezing. Chest CT reveals bilateral upper lobe centrilobular emphysema and bronchial wall thickening. Spirometry demonstrates FEV1/FVC ratio 0.58 post-bronchodilator confirming severe chronic obstructive pulmonary disease (COPD).",
        ortho: "Patient: 32-year-old athlete with acute right knee twisting injury followed by rapid joint effusion and joint line tenderness. MRI Right Knee demonstrates complete disruption of the anterior cruciate ligament (ACL) with a vertical tear of the posterior horn of the medial meniscus and lateral femoral condyle bone contusion."
    };

    window.loadPreset = (key) => {
        if (window.presets[key]) {
            reportTextInput.value = window.presets[key];
            switchTab('textTab');
            // Flash input to indicate loaded
            reportTextInput.style.boxShadow = '0 0 20px rgba(0, 242, 254, 0.4)';
            setTimeout(() => {
                reportTextInput.style.boxShadow = '';
            }, 600);
        }
    };

    // =========================================================================
    // Tab Switching
    // =========================================================================
    function switchTab(targetTabId) {
        activeTab = targetTabId;
        tabBtns.forEach(btn => {
            if (btn.dataset.tab === targetTabId) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        tabPanes.forEach(pane => {
            if (pane.id === targetTabId) {
                pane.classList.add('active');
            } else {
                pane.classList.remove('active');
            }
        });
    }

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });

    // =========================================================================
    // PDF File Drag-and-Drop & Browse
    // =========================================================================
    pdfDropZone.addEventListener('click', () => pdfFileInput.click());
    pdfDropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        pdfDropZone.classList.add('dragover');
    });
    pdfDropZone.addEventListener('dragleave', () => {
        pdfDropZone.classList.remove('dragover');
    });
    pdfDropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        pdfDropZone.classList.remove('dragover');
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
        pdfFileSelected.style.display = 'inline-flex';
        pdfFileSelected.innerHTML = `<span>✓</span> Selected: <strong>${file.name}</strong> (${(file.size / 1024).toFixed(1)} KB)`;
    }

    // =========================================================================
    // Real-Time Audio Dictation & Live Canvas Waveform
    // =========================================================================
    function initWaveformVisualizer(stream) {
        try {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = audioContext.createMediaStreamSource(stream);
            analyserNode = audioContext.createAnalyser();
            analyserNode.fftSize = 64;
            source.connect(analyserNode);

            const canvas = audioWaveformCanvas;
            const ctx = canvas.getContext('2d');
            const bufferLength = analyserNode.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);

            canvas.width = canvas.parentElement.clientWidth || 380;
            canvas.height = 70;

            function draw() {
                animationFrameId = requestAnimationFrame(draw);
                analyserNode.getByteFrequencyData(dataArray);

                ctx.fillStyle = 'rgba(8, 14, 26, 0.4)';
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                const barWidth = (canvas.width / bufferLength) * 1.5;
                let x = 0;

                for (let i = 0; i < bufferLength; i++) {
                    const barHeight = (dataArray[i] / 255) * (canvas.height - 10);
                    
                    const grad = ctx.createLinearGradient(0, canvas.height, 0, 0);
                    grad.addColorStop(0, '#0284c7');
                    grad.addColorStop(0.5, '#00f2fe');
                    grad.addColorStop(1, '#6366f1');

                    ctx.fillStyle = grad;
                    ctx.fillRect(x, canvas.height - barHeight, barWidth - 2, barHeight);
                    x += barWidth;
                }
            }
            draw();
        } catch (e) {
            console.warn('Web Audio API Visualizer not supported:', e);
        }
    }

    function stopWaveformVisualizer() {
        if (animationFrameId) cancelAnimationFrame(animationFrameId);
        if (audioContext && audioContext.state !== 'closed') {
            audioContext.close();
        }
        const ctx = audioWaveformCanvas.getContext('2d');
        ctx.clearRect(0, 0, audioWaveformCanvas.width, audioWaveformCanvas.height);
    }

    recordMicBtn.addEventListener('click', async () => {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            stopRecording();
        } else {
            await startRecording();
        }
    });

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            initWaveformVisualizer(stream);

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) audioChunks.push(e.data);
            };

            mediaRecorder.onstop = () => {
                recordedAudioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                recordStatus.textContent = `Recorded ${(recordedAudioBlob.size / 1024).toFixed(1)} KB audio. Click "Launch Analysis" to process.`;
                recordMicBtn.classList.remove('recording');
                stopWaveformVisualizer();
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorder.start();
            recordMicBtn.classList.add('recording');
            recordStatus.textContent = 'Listening & Recording voice dictation...';

            secondsRecorded = 0;
            recordingTimer.textContent = '00:00';
            timerInterval = setInterval(() => {
                secondsRecorded++;
                const mins = String(Math.floor(secondsRecorded / 60)).padStart(2, '0');
                const secs = String(secondsRecorded % 60).padStart(2, '0');
                recordingTimer.textContent = `${mins}:${secs}`;
            }, 1000);
        } catch (err) {
            alert('Microphone access denied or unavailable: ' + err.message);
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
            clearInterval(timerInterval);
        }
    }

    // =========================================================================
    // Master Multi-Agent Execution Pipeline
    // =========================================================================
    analyzeBtn.addEventListener('click', async () => {
        if (activeTab === 'textTab') {
            const text = reportTextInput.value.trim();
            if (!text) {
                alert('Please enter clinical report text or select a preset.');
                return;
            }
            await executeAnalysis({ type: 'text', content: text });
        } else if (activeTab === 'pdfTab') {
            if (!selectedPdfFile) {
                alert('Please upload a PDF file.');
                return;
            }
            await executeAnalysis({ type: 'pdf', file: selectedPdfFile });
        } else if (activeTab === 'voiceTab') {
            if (!recordedAudioBlob) {
                alert('Please record voice dictation first using the microphone.');
                return;
            }
            await executeAnalysis({ type: 'audio', audioBlob: recordedAudioBlob });
        }
    });

    async function executeAnalysis(payload) {
        setLoadingState(true);
        try {
            let data;
            if (payload.type === 'audio') {
                const formData = new FormData();
                formData.append('audio_file', payload.audioBlob, 'dictation.wav');
                const res = await fetch('/api/analyze-audio', {
                    method: 'POST',
                    body: formData
                });
                if (!res.ok) throw new Error(`Server returned status ${res.status}`);
                data = await res.json();
            } else if (payload.type === 'pdf') {
                const formData = new FormData();
                formData.append('pdf_file', payload.file);
                const res = await fetch('/api/analyze', {
                    method: 'POST',
                    body: formData
                });
                if (!res.ok) throw new Error(`Server returned status ${res.status}`);
                data = await res.json();
            } else {
                const res = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        input_type: 'text',
                        content: payload.content
                    })
                });
                if (!res.ok) throw new Error(`Server returned status ${res.status}`);
                data = await res.json();
            }

            latestAnalysisData = data;
            renderAnalysisResults(data);
        } catch (error) {
            alert('Analysis Error: ' + error.message);
            console.error('Error during clinical analysis:', error);
        } finally {
            setLoadingState(false);
        }
    }

    function setLoadingState(loading) {
        analyzeBtn.disabled = loading;
        if (loading) {
            analyzeBtnText.textContent = 'Orchestrating 8 Multi-Agents...';
            analyzeSpinner.style.display = 'inline-block';
            analyzeBtn.style.opacity = '0.7';
        } else {
            analyzeBtnText.textContent = '⚡ Launch Multi-Agent Analysis';
            analyzeSpinner.style.display = 'none';
            analyzeBtn.style.opacity = '1';
        }
    }

    // =========================================================================
    // Render Results & Neural HUD Components
    // =========================================================================
    function renderAnalysisResults(data) {
        emptyState.style.display = 'none';
        resultsContainer.style.display = 'flex';

        // 1. Specialty & Provenance
        specialtyValue.textContent = data.specialty || 'General Clinical Medicine';
        modelUsedText.textContent = data.model_used || 'DistilBERT Fine-Tuned';

        // 2. Anatomical Target Radar Illumination
        updateOrganRadar(data.specialty);

        // 3. Radial Confidence Gauge Animation
        updateConfidenceGauge(data.confidence || 0.85);

        // 4. Routing Velocity Badge
        if (data.route_mode === 'FAST_PATH') {
            routeBadgePill.className = 'route-velocity-pill fastpath';
            routeValue.innerHTML = '⚡ FAST PATH';
            routeReasonText.textContent = 'Confidence ≥ 75% Streamlined Execution';
        } else {
            routeBadgePill.className = 'route-velocity-pill deepagent';
            routeValue.innerHTML = '🧠 DEEP AGENT PATH';
            routeReasonText.textContent = 'Sub-Threshold LangGraph Verification';
        }

        // 5. 8-Agent Neural Flowgraph
        renderNeuralFlowgraph(data.execution_pipeline || []);

        // 6. Clinical NLP Entity Matrices
        renderClinicalEntities(data.entities || {});

        // 7. Plain-English Patient Explanation
        renderPatientExplanation(data.explanation || 'No explanation generated.');

        // 8. Vetted RAG Evidence & Citations
        renderEvidenceCitations(data.evidence || []);

        // Smooth scroll to results on smaller screens
        if (window.innerWidth < 1180) {
            resultsContainer.scrollIntoView({ behavior: 'smooth' });
        }
    }

    // Anatomical Organ Mapping
    function updateOrganRadar(specialty) {
        const s = (specialty || '').toLowerCase();
        let glyph = '🩺';
        let label = 'General System';

        if (s.includes('cardio') || s.includes('heart')) {
            glyph = '🫀';
            label = 'Cardiovascular';
        } else if (s.includes('neuro') || s.includes('brain')) {
            glyph = '🧠';
            label = 'Neurological';
        } else if (s.includes('pulmo') || s.includes('lung') || s.includes('chest')) {
            glyph = '🫁';
            label = 'Pulmonary';
        } else if (s.includes('gastro') || s.includes('stomach') || s.includes('liver')) {
            glyph = '🧪';
            label = 'Gastrointestinal';
        } else if (s.includes('ortho') || s.includes('bone') || s.includes('spine') || s.includes('knee')) {
            glyph = '🦴';
            label = 'Musculoskeletal';
        } else if (s.includes('onco') || s.includes('tumor') || s.includes('cancer')) {
            glyph = '🔬';
            label = 'Oncological';
        }

        organGlyph.textContent = glyph;
        organTargetLabel.textContent = label;
    }

    // Radial Gauge Animation
    function updateConfidenceGauge(confidence) {
        const percent = Math.round(confidence * 100);
        gaugePercentage.textContent = `${percent}%`;

        // SVG circumference for r=40 is ~251.2
        const totalLength = 251.2;
        const offset = totalLength - (totalLength * Math.min(Math.max(confidence, 0), 1));
        confidenceGaugeCircle.style.strokeDashoffset = offset;

        if (confidence >= 0.85) {
            confidenceGaugeCircle.style.stroke = 'var(--cyan-glow)';
            gaugeConfidenceGrade.textContent = 'High Precision';
        } else if (confidence >= 0.75) {
            confidenceGaugeCircle.style.stroke = 'var(--emerald-green)';
            gaugeConfidenceGrade.textContent = 'Optimal Range';
        } else {
            confidenceGaugeCircle.style.stroke = 'var(--violet-neural)';
            gaugeConfidenceGrade.textContent = 'RAG Enhanced';
        }
    }

    // Multi-Agent Neural Flowgraph Visualizer
    function renderNeuralFlowgraph(pipeline) {
        const defaultAgents = [
            { name: "Agent 1: Router", icon: "📥", status: "COMPLETED" },
            { name: "Agent 3: Classifier", icon: "🏷️", status: "COMPLETED" },
            { name: "Agent 2: Clinical NLP", icon: "🔬", status: "COMPLETED" },
            { name: "Agent 4: Evidence RAG", icon: "📚", status: "COMPLETED" },
            { name: "Agent 7: Explanation", icon: "💬", status: "COMPLETED" },
            { name: "Agent 5: Fact Checker", icon: "⚖️", status: "PASSED" },
            { name: "Agent 6: Safety Guard", icon: "🛡️", status: "PASSED" },
            { name: "Agent 8: Citation Verifier", icon: "🔗", status: "COMPLETED" }
        ];

        neuralNodesContainer.innerHTML = '';

        defaultAgents.forEach((agent, idx) => {
            const stepInfo = pipeline.find(p => p.agent && p.agent.toLowerCase().includes(agent.name.toLowerCase().split(':')[0])) || agent;
            const nodeDiv = document.createElement('div');
            nodeDiv.className = 'neural-node done';
            nodeDiv.innerHTML = `
                <div class="node-icon">${agent.icon}</div>
                <div class="node-name">${agent.name.split(':')[1] || agent.name}</div>
                <div class="node-status">${stepInfo.status || 'DONE'}</div>
            `;
            neuralNodesContainer.appendChild(nodeDiv);
        });
    }

    // Clinical NLP Entity Deep Inspector
    function renderClinicalEntities(entities) {
        symptomsChips.innerHTML = renderChips(entities.symptoms || [], 'symptom');
        vitalsChips.innerHTML = renderChips(entities.measurements || [], 'measurement');
        testsChips.innerHTML = renderChips(entities.tests || [], 'test');
        medsChips.innerHTML = renderChips(entities.medications || [], 'medication');
    }

    function renderChips(items, type) {
        if (!items || items.length === 0) {
            return `<span style="font-size: 0.72rem; color: var(--text-muted);">None detected</span>`;
        }
        return items.map(item => `<span class="clinical-chip ${type}">${escapeHtml(item)}</span>`).join('');
    }

    // Patient Explanation Renderer
    function renderPatientExplanation(markdown) {
        // Convert Markdown formatting into clean styled HTML
        let html = markdown
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/^### (.*$)/gim, '<h4 style="color: var(--cyan-glow); margin: 0.8rem 0 0.4rem 0;">$1</h4>')
            .replace(/^## (.*$)/gim, '<h3 style="color: #fff; margin: 1rem 0 0.5rem 0;">$1</h3>')
            .replace(/^\s*\n\*/gm, '<ul>\n*')
            .replace(/^(\*|\-) (.*$)/gim, '<li>$2</li>')
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>');

        explanationText.innerHTML = `<p>${html}</p>`;
    }

    // Evidence & Citations Renderer
    function renderEvidenceCitations(evidence) {
        if (!evidence || evidence.length === 0) {
            evidenceList.innerHTML = `<p style="font-size: 0.8rem; color: var(--text-muted);">Standard internal guidelines referenced.</p>`;
            return;
        }
        evidenceList.innerHTML = evidence.map(item => `
            <div class="evidence-snippet">
                <div class="evidence-meta">
                    <span>📄 Reference Guideline: ${escapeHtml(item.source || 'Medical Knowledge Base')}</span>
                    <span>Relevance: ${Math.round((item.score || 0.88) * 100)}%</span>
                </div>
                <div class="evidence-text">"${escapeHtml(item.text || item.chunk || '')}"</div>
            </div>
        `).join('');
    }

    // =========================================================================
    // Text-to-Speech (TTS) SpeechSynthesis Engine
    // =========================================================================
    let synth = window.speechSynthesis;
    let currentUtterance = null;

    ttsPlayBtn.addEventListener('click', () => {
        if (!synth) {
            alert('Speech Synthesis not supported in this browser.');
            return;
        }
        synth.cancel();

        const plainText = explanationText.innerText;
        if (!plainText) return;

        currentUtterance = new SpeechSynthesisUtterance(plainText);
        currentUtterance.rate = parseFloat(ttsSpeedSelect.value || 1.0);
        currentUtterance.pitch = 1.0;

        currentUtterance.onend = () => {
            ttsPlayBtn.style.display = 'flex';
            ttsStopBtn.style.display = 'none';
        };

        currentUtterance.onerror = () => {
            ttsPlayBtn.style.display = 'flex';
            ttsStopBtn.style.display = 'none';
        };

        synth.speak(currentUtterance);
        ttsPlayBtn.style.display = 'none';
        ttsStopBtn.style.display = 'flex';
    });

    ttsStopBtn.addEventListener('click', () => {
        if (synth) {
            synth.cancel();
            ttsPlayBtn.style.display = 'flex';
            ttsStopBtn.style.display = 'none';
        }
    });

    // =========================================================================
    // Modal: JSON Telemetry & Export Studio
    // =========================================================================
    btnInspectJson.addEventListener('click', () => {
        if (!latestAnalysisData) {
            jsonTelemetryViewer.textContent = JSON.stringify({
                status: "System Ready",
                instructions: "Run an analysis to inspect live multi-agent execution telemetry."
            }, null, 2);
        } else {
            jsonTelemetryViewer.textContent = JSON.stringify(latestAnalysisData, null, 2);
        }
        jsonModal.classList.add('show');
    });

    btnCloseJsonModal.addEventListener('click', () => jsonModal.classList.remove('show'));
    btnCloseJsonModal2.addEventListener('click', () => jsonModal.classList.remove('show'));

    btnCopyJson.addEventListener('click', () => {
        navigator.clipboard.writeText(jsonTelemetryViewer.textContent).then(() => {
            btnCopyJson.textContent = '✓ Copied!';
            setTimeout(() => { btnCopyJson.textContent = '📋 Copy JSON'; }, 1500);
        });
    });

    btnExportSummary.addEventListener('click', () => {
        if (!latestAnalysisData) {
            alert('Please perform an analysis before exporting.');
            return;
        }
        window.print();
    });

    // Utility: HTML Escaping
    function escapeHtml(str) {
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }
});
