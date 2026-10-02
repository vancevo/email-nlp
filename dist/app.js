const state = { data: null, metric: "accuracy", filter: "all", query: "", page: 1, selectedId: null };
const pageSize = 7;
const modelMeta = {
  "Word2Vec + LSTM": { short: "LSTM", note: "Skip-gram embeddings · sequence length 300" },
  "TF-IDF + Logistic Regression": { short: "TF-IDF + LR", note: "Sparse lexical baseline" },
  "Average Word2Vec + Logistic Regression": { short: "Avg W2V + LR", note: "Mean embedding baseline" },
};

const $ = (selector, scope = document) => scope.querySelector(selector);
const $$ = (selector, scope = document) => [...scope.querySelectorAll(selector)];
const percent = (value, digits = 2) => `${(value * 100).toLocaleString("vi-VN", { minimumFractionDigits: digits, maximumFractionDigits: digits })}%`;

async function loadData() {
  const response = await fetch("./data/results.json");
  if (!response.ok) throw new Error("Không thể tải dữ liệu Kaggle.");
  state.data = await response.json();
  $("#datasetSize").textContent = state.data.state.preprocessing.rows.toLocaleString("vi-VN");
  $("#testSize").textContent = state.data.state.split.test.toLocaleString("vi-VN");
  renderModels(); renderSamples(); renderTraining(); renderMatrix(); registerWebMcp();
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

$$('[data-metric]').forEach((button) => button.addEventListener("click", () => { state.metric = button.dataset.metric; $$('[data-metric]').forEach((item) => item.classList.toggle("active", item === button)); renderModels(); }));
$$('[data-filter]').forEach((button) => button.addEventListener("click", () => { state.filter = button.dataset.filter; state.page = 1; $$('[data-filter]').forEach((item) => item.classList.toggle("active", item === button)); renderSamples(); }));
$("#searchInput").addEventListener("input", (event) => { state.query = event.target.value; state.page = 1; renderSamples(); });
$("#prevPage").addEventListener("click", () => { state.page -= 1; renderSamples(); });
$("#nextPage").addEventListener("click", () => { state.page += 1; renderSamples(); });
$("#matrixModel").addEventListener("change", renderMatrix);
document.addEventListener("keydown", (event) => { if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") { event.preventDefault(); $("#searchInput").focus(); } });
const sections = ["overview","models","predictions","training"];
const observer = new IntersectionObserver((entries) => { const visible = entries.filter((entry)=>entry.isIntersecting).sort((a,b)=>b.intersectionRatio-a.intersectionRatio)[0]; if (!visible) return; $$('nav a').forEach((link)=>link.classList.toggle("active",link.getAttribute("href")===`#${visible.target.id}`)); }, { rootMargin:"-25% 0px -55%", threshold:[.05,.25,.5] });
sections.forEach((id)=>observer.observe(document.getElementById(id)));

loadData().catch((error) => { $("#sampleList").innerHTML = `<div class="empty-state"><h3>Không tải được dữ liệu</h3><p>${escapeHtml(error.message)} Hãy chạy website qua một local server.</p></div>`; });
