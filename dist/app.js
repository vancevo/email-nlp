const state = { data: null, metric: "accuracy", filter: "all", query: "", page: 1, selectedId: null, classifier: null, inputMode: "text", imageFile: null, imageUrl: null, ocrText: "" };
const pageSize = 7;
const modelMeta = {
  "Word2Vec + LSTM": { short: "LSTM", note: "Skip-gram embeddings · sequence length 300" },
  "TF-IDF + Logistic Regression": { short: "TF-IDF + LR", note: "Sparse lexical baseline" },
  "Average Word2Vec + Logistic Regression": { short: "Avg W2V + LR", note: "Mean embedding baseline" },
};
const lexicalSignals = {
  spam: new Set(["free","prize","winner","win","cash","claim","urgent","offer","discount","bonus","lottery","bitcoin","crypto","viagra","casino","click","limited","congratulations","guaranteed","unsubscribe","password","verify","account","suspended","gift","reward","money","loan","cheap","promotion","sale","miễn","phí","trúng","thưởng","khuyến","mãi","quà","tặng","khẩn","cấp","xác","minh","tài","khoản","mật","khẩu","nhấp","vay"]),
  ham: new Set(["meeting","agenda","project","report","schedule","minutes","team","invoice","contract","attached","review","update","deadline","office","client","thanks","regards","họp","dự","án","báo","cáo","lịch","đính","kèm","hợp","đồng","công","việc","cảm","ơn"]),
};
const spamPhrases = ["click here","claim your","act now","limited time","free gift","you have won","verify your account","xác minh tài khoản","nhấn vào đây","trúng thưởng","miễn phí"];

const $ = (selector, scope = document) => scope.querySelector(selector);
const $$ = (selector, scope = document) => [...scope.querySelectorAll(selector)];
const percent = (value, digits = 2) => `${(value * 100).toLocaleString("vi-VN", { minimumFractionDigits: digits, maximumFractionDigits: digits })}%`;

async function loadData() {
  const response = await fetch("./data/results.json");
  if (!response.ok) throw new Error("Không thể tải dữ liệu Kaggle.");
  state.data = await response.json();
  $("#datasetSize").textContent = state.data.state.preprocessing.rows.toLocaleString("vi-VN");
  $("#testSize").textContent = state.data.state.split.test.toLocaleString("vi-VN");
  renderModels(); renderSamples(); renderTraining(); renderMatrix(); buildClassifier(); registerWebMcp();
}

function tokenize(value) {
  return (value.toLowerCase().normalize("NFKC").match(/[\p{L}\p{N}$%]+/gu) || [])
    .filter((token) => token.length > 1 && token.length < 32);
}

function buildClassifier() {
  const classes = {
    ham: { documents: 0, tokens: 0, counts: new Map() },
    spam: { documents: 0, tokens: 0, counts: new Map() },
  };
  const vocabulary = new Set();
  state.data.predictions.forEach((item) => {
    const bucket = classes[item.label];
    bucket.documents += 1;
    tokenize(item.subject).forEach((token) => {
      vocabulary.add(token);
      bucket.tokens += 1;
      bucket.counts.set(token, (bucket.counts.get(token) || 0) + 1);
    });
  });
  state.classifier = { classes, vocabularySize: vocabulary.size, documents: state.data.predictions.length };
}

function classifyText(value) {
  if (!state.classifier) throw new Error("Mô hình chưa sẵn sàng.");
  const frequency = new Map();
  tokenize(value).forEach((token) => frequency.set(token, Math.min(3, (frequency.get(token) || 0) + 1)));
  if (!frequency.size) throw new Error("Không tìm thấy đủ từ để phân loại.");

  const contributions = [];
  let datasetScore = 0;
  let lexicalScore = 0;
  frequency.forEach((count, token) => {
    const spam = state.classifier.classes.spam;
    const ham = state.classifier.classes.ham;
    const spamLikelihood = ((spam.counts.get(token) || 0) + 1) / (spam.tokens + state.classifier.vocabularySize);
    const hamLikelihood = ((ham.counts.get(token) || 0) + 1) / (ham.tokens + state.classifier.vocabularySize);
    const corpusWeight = count * Math.log(spamLikelihood / hamLikelihood);
    datasetScore += corpusWeight;
    const lexicalWeight = (lexicalSignals.spam.has(token) ? 1.8 : 0) - (lexicalSignals.ham.has(token) ? 1.35 : 0);
    lexicalScore += lexicalWeight;
    contributions.push({ token, weight: corpusWeight * .16 + lexicalWeight });
  });

  const normalized = value.toLowerCase().normalize("NFKC");
  spamPhrases.forEach((phrase) => {
    if (normalized.includes(phrase)) {
      lexicalScore += 2.2;
      contributions.push({ token: phrase, weight: 2.2 });
    }
  });
  const priorScore = Math.log(state.classifier.classes.spam.documents / state.classifier.classes.ham.documents);
  const corpusAverage = datasetScore / Math.max(1, [...frequency.values()].reduce((sum, count) => sum + count, 0));
  const delta = Math.max(-6, Math.min(6, priorScore + corpusAverage * .9 + lexicalScore));
  const spamProbability = 1 / (1 + Math.exp(-delta));
  const label = spamProbability >= .5 ? "spam" : "ham";
  const signals = contributions
    .filter((item) => Math.abs(item.weight) > .15)
    .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
    .slice(0, 6);
  return { label, spamProbability, confidence: label === "spam" ? spamProbability : 1 - spamProbability, signals, tokenCount: [...frequency.values()].reduce((sum, count) => sum + count, 0) };
}

function renderLiveResult(result, source) {
  const confidence = Math.round(result.confidence * 1000) / 10;
  const spamChance = Math.round(result.spamProbability * 1000) / 10;
  const signalMarkup = result.signals.length
    ? result.signals.map((item) => `<span class="signal ${item.weight > 0 ? "spam" : ""}">${escapeHtml(item.token)} · ${item.weight > 0 ? "+spam" : "+ham"}</span>`).join("")
    : `<span class="signal">Không có từ khóa nổi trội</span>`;
  $("#classifierResult").innerHTML = `<div class="live-result-head"><div><p class="eyebrow">KẾT QUẢ PHÂN LOẠI</p><span class="confidence-caption">Nguồn: ${source}</span></div><strong class="live-label ${result.label}">${result.label}</strong></div>
    <div class="confidence-number">${confidence.toLocaleString("vi-VN")}<sup>%</sup></div>
    <div class="confidence-caption">độ tin cậy cho nhãn ${result.label.toUpperCase()}</div>
    <div class="confidence-track ${result.label}"><i style="width:${spamChance}%"></i></div>
    <div class="confidence-scale"><span>0% spam</span><span>${spamChance.toLocaleString("vi-VN")}% spam</span><span>100% spam</span></div>
    <p class="signal-title">TỪ CÓ ẢNH HƯỞNG MẠNH</p><div class="signal-list">${signalMarkup}</div>
    <p class="result-summary">Đã phân tích ${result.tokenCount.toLocaleString("vi-VN")} từ hợp lệ. Điểm tin cậy phản ánh mô hình demo trên dữ liệu Enron, không phải xác suất đã hiệu chỉnh cho email ngoài thực tế.</p>`;
}

function renderModels() {
  const values = state.data.metrics.map((item) => item[state.metric]);
  const best = Math.max(...values);
  $("#modelList").innerHTML = state.data.metrics.map((item, index) => {
    const value = item[state.metric];
    const scaled = 72 + ((value - Math.min(...values)) / Math.max(best - Math.min(...values), .000001)) * 28;
    const meta = modelMeta[item.model];
    return `<article class="model-row ${value === best ? "best" : ""}">
      <div class="model-name"><strong>${index + 1}. ${meta.short}</strong><small>${meta.note}</small></div>
      <div class="bar-track" aria-hidden="true"><i style="width:${scaled}%"></i></div>
      <div class="model-value">${percent(value, 3)}</div>
    </article>`;
  }).join("");
}

function filteredPredictions() {
  const normalized = state.query.trim().toLowerCase();
  return state.data.predictions.filter((item) => {
    const matchesText = !normalized || item.subject.toLowerCase().includes(normalized);
    const wrong = item.models.lstm.prediction !== item.truth;
    const matchesFilter = state.filter === "all" || item.label === state.filter || (state.filter === "wrong" && wrong);
    return matchesText && matchesFilter;
  });
}

function renderSamples() {
  const rows = filteredPredictions();
  const pages = Math.max(1, Math.ceil(rows.length / pageSize));
  state.page = Math.min(state.page, pages);
  const visible = rows.slice((state.page - 1) * pageSize, state.page * pageSize);
  $("#resultCount").textContent = `${rows.length.toLocaleString("vi-VN")} mẫu`;
  $("#pageStatus").textContent = `${state.page} / ${pages}`;
  $("#prevPage").disabled = state.page === 1;
  $("#nextPage").disabled = state.page === pages;
  $("#sampleList").innerHTML = visible.length ? visible.map((item) => {
    const wrong = item.models.lstm.prediction !== item.truth;
    return `<button class="sample ${state.selectedId === item.id ? "selected" : ""}" data-id="${item.id}">
      <span class="sample-id">#${String(item.id).padStart(4, "0")}</span>
      <span class="sample-copy"><strong>${escapeHtml(item.subject || "(Không có subject)")}</strong><small>LSTM: ${percent(item.models.lstm.probability, 1)} xác suất spam</small></span>
      <span class="tag ${item.label} ${wrong ? "wrong" : ""}">${wrong ? "Sai · " : ""}${item.label}</span>
    </button>`;
  }).join("") : `<div class="empty-state"><h3>Không tìm thấy</h3><p>Thử một từ khóa hoặc bộ lọc khác.</p></div>`;
  $$(".sample").forEach((button) => button.addEventListener("click", () => selectPrediction(Number(button.dataset.id))));
}

function selectPrediction(id) {
  state.selectedId = id;
  const item = state.data.predictions.find((row) => row.id === id);
  const entries = [
    ["TF-IDF + LR", item.models.tfidf],
    ["Average W2V + LR", item.models.avgW2v],
    ["Word2Vec + LSTM", item.models.lstm],
  ];
  $("#detailPanel").innerHTML = `<p class="eyebrow">PREDICTION DETAIL · #${String(id).padStart(4, "0")}</p>
    <h3 class="detail-subject">${escapeHtml(item.subject || "(Không có subject)")}</h3>
    <div class="truth-row"><span>Nhãn thật</span><strong class="tag ${item.label}">${item.label}</strong></div>
    <div class="prediction-bars">${entries.map(([label, model]) => `<div class="pred-row ${model.prediction ? "spam" : "ham"}">
      <div class="pred-head"><span>${label} · ${model.prediction ? "spam" : "ham"}</span><span>${percent(model.probability, 2)}</span></div>
      <div class="prob-track"><i style="width:${Math.max(model.probability * 100, .4)}%"></i></div>
    </div>`).join("")}</div>
    <p class="detail-note">Thanh thể hiện xác suất thuộc lớp spam. Ngưỡng phân loại là 50%. Các giá trị được đọc trực tiếp từ <code>test_predictions.csv</code>.</p>`;
  renderSamples();
}

function renderTraining() {
  const svg = $("#trainingChart");
  const rows = state.data.history;
  const width = 760, height = 300, pad = { x: 54, y: 25, bottom: 44, right: 18 };
  const x = (i) => pad.x + i * ((width - pad.x - pad.right) / (rows.length - 1));
  const y = (v) => pad.y + (1 - ((v - .965) / .035)) * (height - pad.y - pad.bottom);
  const pathFor = (key) => rows.map((row, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(row[key]).toFixed(1)}`).join(" ");
  const grid = [.97,.98,.99,1].map((value) => `<line class="chart-grid" x1="${pad.x}" y1="${y(value)}" x2="${width-pad.right}" y2="${y(value)}"/><text class="chart-label" x="8" y="${y(value)+4}">${value.toFixed(2)}</text>`).join("");
  const ticks = rows.map((row,i) => `<text class="chart-label" text-anchor="middle" x="${x(i)}" y="${height-12}">${row.epoch+1}</text>`).join("");
  svg.innerHTML = `${grid}${ticks}<path class="chart-line" stroke="#2563eb" d="${pathFor("accuracy")}"/><path class="chart-line" stroke="#ff725e" d="${pathFor("val_accuracy")}"/>${rows.map((row,i) => `<circle class="chart-dot" fill="#2563eb" cx="${x(i)}" cy="${y(row.accuracy)}" r="4"/><circle class="chart-dot" fill="#ff725e" cx="${x(i)}" cy="${y(row.val_accuracy)}" r="4"/>`).join("")}<text class="chart-label" text-anchor="middle" x="${width/2}" y="${height}">Epoch</text>`;
}

function renderMatrix() {
  const data = state.data.confusion[$("#matrixModel").value];
  $("#matrix").innerHTML = [
    [data.tn,"Ham → Ham","good"], [data.fp,"Ham → Spam",data.fp ? "error" : ""],
    [data.fn,"Spam → Ham",data.fn ? "error" : ""], [data.tp,"Spam → Spam","good"],
  ].map(([value,label,kind]) => `<div class="matrix-cell ${kind}"><strong>${value.toLocaleString("vi-VN")}</strong><span>${label}</span></div>`).join("");
}

function escapeHtml(value) { const element = document.createElement("span"); element.textContent = value; return element.innerHTML; }

function registerWebMcp() {
  const context = document.modelContext;
  if (!context?.registerTool) return;
  context.registerTool({ name:"search_test_predictions", title:"Tìm dự đoán test", description:"Tìm email trong tập test Kaggle và cập nhật danh sách dự đoán đang hiển thị.", inputSchema:{ type:"object", properties:{ query:{ type:"string" }, label:{ type:"string", enum:["all","spam","ham","wrong"] } }, additionalProperties:false }, annotations:{ readOnlyHint:true, untrustedContentHint:true }, execute:(input)=>{ const query=typeof input?.query==="string"?input.query:""; const label=["all","spam","ham","wrong"].includes(input?.label)?input.label:"all"; state.query=query; state.filter=label; state.page=1; $("#searchInput").value=query; $$(".chip").forEach((chip)=>chip.classList.toggle("active",chip.dataset.filter===label)); renderSamples(); return { count:filteredPredictions().length, query, label }; } });
  context.registerTool({ name:"show_prediction_detail", title:"Mở chi tiết dự đoán", description:"Mở chi tiết một dự đoán theo ID hợp lệ trong tập test Kaggle.", inputSchema:{ type:"object", properties:{ id:{ type:"integer", minimum:1 } }, required:["id"], additionalProperties:false }, annotations:{ readOnlyHint:true, untrustedContentHint:true }, execute:(input)=>{ const id=Number(input?.id); if(!Number.isInteger(id)) throw new Error("ID phải là số nguyên."); const exists=state.data.predictions.some((row)=>row.id===id); if(!exists) throw new Error("Không tìm thấy dự đoán với ID này."); selectPrediction(id); return { found:true, id }; } });
}

function setInputMode(mode) {
  state.inputMode = mode;
  const textMode = mode === "text";
  $("#textTab").classList.toggle("active", textMode);
  $("#imageTab").classList.toggle("active", !textMode);
  $("#textTab").setAttribute("aria-selected", String(textMode));
  $("#imageTab").setAttribute("aria-selected", String(!textMode));
  $("#textInputPanel").hidden = !textMode;
  $("#imageInputPanel").hidden = textMode;
  $("#textInputPanel").classList.toggle("active", textMode);
  $("#imageInputPanel").classList.toggle("active", !textMode);
  $("#classifierHint").textContent = textMode ? "Nhập ít nhất 3 ký tự để bắt đầu." : "Chọn ảnh rõ nét để nhận dạng chính xác hơn.";
}

function updateCharacterCount() {
  $("#characterCount").textContent = `${$("#emailContent").value.length.toLocaleString("vi-VN")} ký tự`;
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function clearImage() {
  if (state.imageUrl) URL.revokeObjectURL(state.imageUrl);
  state.imageFile = null; state.imageUrl = null; state.ocrText = "";
  $("#imageInput").value = "";
  $("#imagePreview").hidden = true;
  $("#dropZone").hidden = false;
  $("#ocrStatus").innerHTML = "";
}

function acceptImage(file) {
  if (!file) return;
  if (!/^image\/(png|jpeg|webp)$/.test(file.type)) {
    $("#ocrStatus").textContent = "Chỉ hỗ trợ ảnh PNG, JPG hoặc WEBP.";
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    $("#ocrStatus").textContent = "Ảnh vượt quá giới hạn 10 MB.";
    return;
  }
  clearImage();
  state.imageFile = file;
  state.imageUrl = URL.createObjectURL(file);
  $("#previewImage").src = state.imageUrl;
  $("#imageName").textContent = file.name;
  $("#imageSize").textContent = formatBytes(file.size);
  $("#dropZone").hidden = true;
  $("#imagePreview").hidden = false;
  $("#ocrStatus").textContent = "Ảnh đã sẵn sàng. Bấm “Phân loại email” để đọc chữ.";
}

async function extractImageText() {
  if (!state.imageFile) throw new Error("Vui lòng chọn một ảnh email.");
  $("#ocrStatus").innerHTML = `Đang tải bộ nhận dạng chữ…<div class="ocr-progress"><i style="width:4%"></i></div>`;
  const Tesseract = (await import("https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.esm.min.js")).default;
  const result = await Tesseract.recognize(state.imageFile, "eng+vie", {
    logger(message) {
      if (message.status !== "recognizing text") return;
      const progress = Math.round((message.progress || 0) * 100);
      $("#ocrStatus").innerHTML = `Đang nhận dạng chữ… ${progress}%<div class="ocr-progress"><i style="width:${progress}%"></i></div>`;
    },
  });
  state.ocrText = result.data.text.trim();
  if (state.ocrText.length < 3) throw new Error("Không đọc được đủ nội dung từ ảnh. Hãy thử ảnh rõ nét hơn.");
  $("#ocrStatus").textContent = `Đã nhận dạng ${state.ocrText.length.toLocaleString("vi-VN")} ký tự.`;
  return state.ocrText;
}

async function runLiveClassification() {
  const button = $("#classifyButton");
  button.disabled = true;
  try {
    let content;
    let source;
    if (state.inputMode === "image") {
      content = state.ocrText || await extractImageText();
      source = "OCR từ ảnh email";
    } else {
      content = $("#emailContent").value.trim();
      if (content.length < 3) throw new Error("Vui lòng nhập ít nhất 3 ký tự.");
      source = "nội dung đã nhập";
    }
    renderLiveResult(classifyText(content), source);
  } catch (error) {
    if (state.inputMode === "image") $("#ocrStatus").textContent = error.message;
    $("#classifierResult").innerHTML = `<div class="result-placeholder"><span>!</span><h3>Chưa thể phân loại</h3><p>${escapeHtml(error.message)}</p></div>`;
  } finally {
    button.disabled = false;
  }
}

$$('[data-metric]').forEach((button) => button.addEventListener("click", () => { state.metric = button.dataset.metric; $$('[data-metric]').forEach((item) => item.classList.toggle("active", item === button)); renderModels(); }));
$$('[data-filter]').forEach((button) => button.addEventListener("click", () => { state.filter = button.dataset.filter; state.page = 1; $$('[data-filter]').forEach((item) => item.classList.toggle("active", item === button)); renderSamples(); }));
$("#searchInput").addEventListener("input", (event) => { state.query = event.target.value; state.page = 1; renderSamples(); });
$("#prevPage").addEventListener("click", () => { state.page -= 1; renderSamples(); });
$("#nextPage").addEventListener("click", () => { state.page += 1; renderSamples(); });
$("#matrixModel").addEventListener("change", renderMatrix);
$("#textTab").addEventListener("click", () => setInputMode("text"));
$("#imageTab").addEventListener("click", () => setInputMode("image"));
$("#emailContent").addEventListener("input", updateCharacterCount);
$("#clearContent").addEventListener("click", () => { $("#emailContent").value = ""; updateCharacterCount(); $("#emailContent").focus(); });
$("#imageInput").addEventListener("change", (event) => acceptImage(event.target.files[0]));
$("#removeImage").addEventListener("click", clearImage);
$("#classifyButton").addEventListener("click", runLiveClassification);
$("#dropZone").addEventListener("dragover", (event) => { event.preventDefault(); $("#dropZone").classList.add("dragging"); });
$("#dropZone").addEventListener("dragleave", () => $("#dropZone").classList.remove("dragging"));
$("#dropZone").addEventListener("drop", (event) => { event.preventDefault(); $("#dropZone").classList.remove("dragging"); acceptImage(event.dataTransfer.files[0]); });
document.addEventListener("keydown", (event) => { if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") { event.preventDefault(); $("#searchInput").focus(); } });
const sections = ["overview","models","classifier","predictions","training"];
const observer = new IntersectionObserver((entries) => { const visible = entries.filter((entry)=>entry.isIntersecting).sort((a,b)=>b.intersectionRatio-a.intersectionRatio)[0]; if (!visible) return; $$('nav a').forEach((link)=>link.classList.toggle("active",link.getAttribute("href")===`#${visible.target.id}`)); }, { rootMargin:"-25% 0px -55%", threshold:[.05,.25,.5] });
sections.forEach((id)=>observer.observe(document.getElementById(id)));

loadData().catch((error) => { $("#sampleList").innerHTML = `<div class="empty-state"><h3>Không tải được dữ liệu</h3><p>${escapeHtml(error.message)} Hãy chạy website qua một local server.</p></div>`; });
