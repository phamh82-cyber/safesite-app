"use strict";
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const api = async (url, opt = {}) => {
  const r = await fetch(url, { headers: opt.body && !(opt.body instanceof FormData) ? { "Content-Type": "application/json" } : {}, ...opt });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
};
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const LABELS = {
  FALL: "Té ngã", FALL_IMMOBILE: "Bất động sau ngã", PERSON_DOWN: "Người nằm bất động",
  NEAR_MISS: "Nguy cơ va chạm người – xe", PERSON_VEHICLE_COLLISION: "Va chạm người – xe",
  VEHICLE_COLLISION: "Va chạm xe – xe", FALL_FROM_HEIGHT: "Ngã từ trên cao", OBJECT_FALL: "Vật rơi",
  STRUCK_BY: "Bị vật rơi đập trúng", FIRE_SMOKE: "Cháy / khói",
};
const SEV = { critical: "Nghiêm trọng", warning: "Cảnh báo" };
const STATUS = { new: "Mới", ack: "Đã xác nhận", false_alarm: "Báo nhầm", resolved: "Đã xử lý" };
const CAM_STATUS = { running: "Đang chạy", starting: "Đang khởi động", reconnecting: "Đang kết nối lại", stopped: "Đã dừng", finished: "Hết video", error: "Lỗi" };
const GROUND = { homography: "Đã hiệu chỉnh", "auto-perspective": "Tự ước lượng (phối cảnh)", "auto-scale": "Tự ước lượng (tỉ lệ)" };
const DETAIL_LABELS = {
  torso_angle: "Góc thân (°)", fall_speed: "Tốc độ rơi (H/s)", height_ratio: "Tỉ lệ chiều cao", reason: "Lý do",
  distance_m: "Khoảng cách (m)", ttc_s: "Thời gian tới va chạm (s)", vehicle_speed_ms: "Tốc độ xe (m/s)",
  closing_speed_ms: "Tốc độ tiến lại gần (m/s)", recovered_after_s: "Đứng dậy sau (s)", related: "Sự kiện liên quan",
  drop_h: "Độ rơi (× chiều cao người)", est_drop_m: "Độ cao rơi ước tính (m)", kind: "Loại", source: "Nguồn phát hiện",
  hits: "Số lần phát hiện", area_pct: "Diện tích (% khung)", speed_fh_s: "Tốc độ vật (khung/s)", drop_fh: "Quãng rơi (× khung)",
  nearest_person_h: "Cách người gần nhất (× chiều cao người)", object_speed_fh_s: "Tốc độ vật khi trúng (khung/s)",
};

const state = { cams: [], feed: [], unseen: 0, sound: true, page: 0, pageSize: 25, current: null };

// ------------------------------------------------------------------ điều hướng
function show(view) {
  $$(".nav button").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === "view-" + view));
  if (view === "events") loadEvents();
  if (view === "stats") loadStats();
  if (view === "settings") loadSettings();
  if (view === "cameras") renderCamRows();
  if (view === "monitor") { state.unseen = 0; updateBadge(); }
}
$$(".nav button").forEach((b) => b.addEventListener("click", () => show(b.dataset.view)));
document.addEventListener("click", (e) => {
  const g = e.target.closest("[data-goto]"); if (g) show(g.dataset.goto);
  const c = e.target.closest("[data-close]"); if (c) c.closest(".modal").classList.remove("open");
});
$$(".modal").forEach((m) => m.addEventListener("click", (e) => { if (e.target === m) m.classList.remove("open"); }));

// ------------------------------------------------------------------ websocket
function connect() {
  const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
  ws.onopen = () => { $("#ws-dot").classList.add("on"); $("#ws-text").textContent = "Đã kết nối máy chủ"; };
  ws.onclose = () => { $("#ws-dot").classList.remove("on"); $("#ws-text").textContent = "Mất kết nối – thử lại…"; setTimeout(connect, 2000); };
  ws.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (msg.kind === "status") { state.cams = msg.cameras; renderTiles(); if ($("#view-cameras").classList.contains("active")) renderCamRows(); }
    if (msg.kind === "event") onEvent(msg.event);
    if (msg.kind === "event_update") onEventUpdate(msg.event);
  };
}

// ------------------------------------------------------------------ giám sát
function renderTiles() {
  const box = $("#tiles");
  const running = state.cams.filter((c) => c.runtime?.status === "running").length;
  $("#monitor-sub").textContent = `${state.cams.length} camera · ${running} đang chạy`;
  if (!state.cams.length) { box.innerHTML = `<div class="card empty">Chưa có camera. Vào <a href="#" data-goto="cameras">Camera</a> để thêm file video, webcam hoặc RTSP.</div>`; return; }
  const ids = state.cams.map((c) => c.id);
  $$(".tile", box).forEach((t) => { if (!ids.includes(t.dataset.id)) t.remove(); });
  $(".empty", box)?.remove();
  for (const c of state.cams) {
    let t = $(`.tile[data-id="${c.id}"]`, box);
    if (!t) {
      t = document.createElement("div");
      t.className = "tile"; t.dataset.id = c.id;
      t.innerHTML = `<div class="head"><b></b><span class="chip st"></span><div class="meta"></div></div>
        <div class="screen"><img alt="Luồng camera" src="/api/cameras/${c.id}/stream"><div class="msg"></div></div><div class="bar"><i></i></div>`;
      box.appendChild(t);
      t.addEventListener("click", async (e) => {
        if (!e.target.closest("[data-start]")) return;
        await api(`/api/cameras/${c.id}/start`, { method: "POST" });
      });
    }
    const rt = c.runtime || {};
    $("b", t).textContent = c.name;
    const st = $(".st", t); st.textContent = CAM_STATUS[rt.status] || rt.status; st.className = "chip st " + (rt.status || "");
    $(".meta", t).innerHTML = rt.status === "running"
      ? `<span>${rt.fps} FPS</span><span>👷 ${rt.persons ?? 0}</span><span>🚚 ${rt.vehicles ?? 0}</span><span title="${GROUND[rt.ground?.mode] || ""}">${rt.ground?.mode === "homography" ? "📐 đã hiệu chỉnh" : "📐 tự động"}</span>${rt.shaking ? `<span title="Camera đang rung mạnh: phân tích vật rơi/cháy tạm dừng">📳 rung</span>` : ""}${rt.restarts ? `<span title="Số lần luồng tự khởi động lại sau lỗi">↻ ${rt.restarts}</span>` : ""}`
      : `<span>${esc(rt.error || "")}</span>`;
    $(".bar i", t).style.width = c.source_type === "file" ? (100 * (rt.progress || 0)).toFixed(1) + "%" : "0";
    const msg = $(".msg", t), live = ["running", "starting", "reconnecting"].includes(rt.status);
    msg.style.display = live ? "none" : "grid";
    if (!live) msg.innerHTML = `<div><div>${CAM_STATUS[rt.status] || "Đã dừng"}${rt.error ? " – " + esc(rt.error) : ""}</div><button class="btn sm" data-start style="margin-top:8px">▶ Chạy</button></div>`;
    else if (rt.status === "starting") { msg.style.display = "grid"; msg.innerHTML = "<div>Đang tải mô hình…</div>"; }
    else if (rt.status === "reconnecting") { msg.style.display = "grid"; msg.innerHTML = "<div>Mất hình – đang kết nối lại camera…</div>"; }
    const img = $("img", t);
    if (rt.status === "running" && img.dataset.live !== "1") { img.src = `/api/cameras/${c.id}/stream?ts=${Date.now()}`; img.dataset.live = "1"; }
    if (rt.status !== "running") img.dataset.live = "0";
  }
}

function alertHTML(e) {
  return `<div class="alert ${e.severity} ${e.status === "new" ? "new" : ""}" data-id="${e.id}">
    <img src="/media/snapshots/${encodeURIComponent(e.snapshot)}" alt="">
    <div><div class="t">${esc(e.label)} <span class="chip ${e.severity}">${SEV[e.severity]}</span></div>
    <div class="m">${esc(e.camera_name)} · ${e.created_at.slice(11)} · tin cậy ${(e.confidence * 100).toFixed(0)}% · ${e.track_ids.join(", ")}</div>
    <div class="m">${STATUS[e.status]}</div></div></div>`;
}
function renderFeed() {
  $("#feed").innerHTML = state.feed.length ? state.feed.map(alertHTML).join("") : `<div class="empty">Chưa có cảnh báo</div>`;
  $("#feed-count").textContent = state.feed.length;
}
$("#feed").addEventListener("click", (e) => { const a = e.target.closest(".alert"); if (a) openEvent(+a.dataset.id); });

function onEvent(e) {
  state.feed.unshift(e); state.feed = state.feed.slice(0, 50); renderFeed();
  if (!$("#view-monitor").classList.contains("active")) { state.unseen++; updateBadge(); }
  toast(e); beep(e.severity);
  const t = $(`.tile[data-id="${e.camera_id}"]`);
  if (t) { t.classList.remove("alarm"); void t.offsetWidth; t.classList.add("alarm"); }
  if (window.Notification && Notification.permission === "granted" && document.hidden)
    new Notification("⚠ " + e.label, { body: `${e.camera_name} – ${e.created_at.slice(11)}` });
}
function onEventUpdate(e) {
  const i = state.feed.findIndex((x) => x.id === e.id);
  if (i >= 0) { state.feed[i] = e; renderFeed(); }
  if (state.current?.id === e.id) fillEvent(e);
}
function updateBadge() { const b = $("#badge-new"); b.textContent = state.unseen || ""; b.dataset.n = state.unseen; }
function toast(e) {
  const d = document.createElement("div");
  d.className = "toast " + e.severity;
  d.innerHTML = `<b>⚠ ${esc(e.label)}</b><span>${esc(e.camera_name)} · ${e.created_at.slice(11)} · ${(e.confidence * 100).toFixed(0)}%</span>`;
  d.onclick = () => { openEvent(e.id); d.remove(); };
  $("#toasts").prepend(d); setTimeout(() => d.remove(), 8000);
}
let actx;
// Trình duyệt chỉ cho phát âm thanh sau một thao tác của người dùng; màn hình giám sát để treo cũng cần
// được "mở khóa" sẵn: lần bấm/chạm đầu tiên tạo và đánh thức AudioContext, đồng thời xin quyền thông báo.
document.addEventListener("pointerdown", () => {
  try { actx = actx || new AudioContext(); actx.resume(); } catch (_) { /* không có Web Audio */ }
  if (window.Notification && Notification.permission === "default") Notification.requestPermission();
}, { once: true });
function beep(sev) {
  if (!state.sound) return;
  try {
    actx = actx || new AudioContext();
    if (actx.state === "suspended") actx.resume();
    const n = sev === "critical" ? 3 : 1;
    for (let i = 0; i < n; i++) {
      const o = actx.createOscillator(), g = actx.createGain();
      o.frequency.value = sev === "critical" ? 880 : 620; o.connect(g); g.connect(actx.destination);
      const t0 = actx.currentTime + i * 0.25; g.gain.setValueAtTime(0.15, t0); g.gain.exponentialRampToValueAtTime(0.001, t0 + 0.2);
      o.start(t0); o.stop(t0 + 0.2);
    }
  } catch (_) { /* trình duyệt chặn âm thanh khi chưa tương tác */ }
}
$("#btn-sound").onclick = () => {
  state.sound = !state.sound; $("#btn-sound").textContent = "🔔 Âm thanh: " + (state.sound ? "bật" : "tắt");
};

// ------------------------------------------------------------------ sự kiện
Object.entries(LABELS).forEach(([k, v]) => $("#f-type").insertAdjacentHTML("beforeend", `<option value="${k}">${v}</option>`));
["#f-type", "#f-sev", "#f-status", "#f-cam"].forEach((s) => $(s).addEventListener("change", () => { state.page = 0; loadEvents(); }));
$("#pg-prev").onclick = () => { if (state.page > 0) { state.page--; loadEvents(); } };
$("#pg-next").onclick = () => { state.page++; loadEvents(); };
function evQuery(limit, offset) {
  return new URLSearchParams({ type: $("#f-type").value, severity: $("#f-sev").value, status: $("#f-status").value, camera: $("#f-cam").value, limit, offset });
}
async function loadEvents() {
  const camSel = $("#f-cam"), cur = camSel.value;
  camSel.innerHTML = `<option value="">Mọi camera</option>` + state.cams.map((c) => `<option value="${c.id}">${esc(c.name)}</option>`).join("");
  camSel.value = cur;
  const r = await api("/api/events?" + evQuery(state.pageSize, state.page * state.pageSize));
  if (!r.items.length && state.page > 0) { state.page--; return loadEvents(); }
  $("#events-sub").textContent = `${r.total} sự kiện`;
  $("#pg-info").textContent = `Trang ${state.page + 1} / ${Math.max(1, Math.ceil(r.total / state.pageSize))}`;
  $("#ev-rows").innerHTML = r.items.map((e) => `<tr data-id="${e.id}">
    <td><img src="/media/snapshots/${encodeURIComponent(e.snapshot)}" alt=""></td><td>${e.created_at.replace("T", " ")}</td>
    <td>${esc(e.label)}</td><td><span class="chip ${e.severity}">${SEV[e.severity]}</span></td><td>${esc(e.camera_name)}</td>
    <td>${e.track_ids.join(", ")}</td><td>${(e.confidence * 100).toFixed(0)}%</td><td>${STATUS[e.status]}</td></tr>`).join("")
    || `<tr><td colspan="8" class="empty">Không có sự kiện</td></tr>`;
}
$("#ev-rows").addEventListener("click", (e) => { const tr = e.target.closest("tr[data-id]"); if (tr) openEvent(+tr.dataset.id); });
$("#btn-export").onclick = async () => {
  const r = { items: [] };
  for (let off = 0; off < 20000; off += 500) {            // server giới hạn 500/lần ⇒ lấy lần lượt tới hết
    const part = await api("/api/events?" + evQuery(500, off));
    r.items.push(...part.items);
    if (part.items.length < 500) break;
  }
  const head = ["id", "created_at", "camera", "type", "severity", "confidence", "t_video", "track_ids", "status", "note", "details"];
  const rows = r.items.map((e) => [e.id, e.created_at, e.camera_name, e.type, e.severity, e.confidence, e.t_video, e.track_ids.join(" "), e.status, e.note, JSON.stringify(e.details)]);
  const csv = [head, ...rows].map((r) => r.map((x) => `"${String(x ?? "").replace(/"/g, '""')}"`).join(",")).join("\n");
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob(["﻿" + csv], { type: "text/csv" })); a.download = "su_kien.csv"; a.click();
};

async function openEvent(id) { fillEvent(await api("/api/events/" + id)); $("#m-event").classList.add("open"); }
function fillEvent(e) {
  state.current = e;
  $("#me-title").textContent = `#${e.id} · ${e.label}`;
  $("#me-sev").innerHTML = `<span class="chip ${e.severity}">${SEV[e.severity]}</span> <span class="chip">${STATUS[e.status]}</span>`;
  $("#me-img").src = "/media/snapshots/" + encodeURIComponent(e.snapshot);
  $("#me-clip").innerHTML = e.clip ? `<video controls autoplay muted loop src="/media/clips/${encodeURIComponent(e.clip)}"></video>`
    : `<div class="empty card">Đang ghi video ngắn quanh thời điểm sự kiện…</div>`;
  const kv = [["Thời gian", e.created_at.replace("T", " ")], ["Camera", e.camera_name], ["Thời điểm trong video", e.t_video.toFixed(1) + " s"],
    ["Đối tượng", e.track_ids.join(", ")], ["Độ tin cậy", (e.confidence * 100).toFixed(0) + "%"],
    ...Object.entries(e.details).map(([k, v]) => [DETAIL_LABELS[k] || k, v ?? "—"])];
  $("#me-kv").innerHTML = kv.map(([k, v]) => `<div>${esc(k)}</div><div>${esc(v)}</div>`).join("");
  $("#me-note").value = e.note || "";
}
$$("#m-event [data-status]").forEach((b) => b.onclick = async () => {
  const e = await api("/api/events/" + state.current.id, { method: "PATCH", body: JSON.stringify({ status: b.dataset.status, note: $("#me-note").value }) });
  fillEvent(e); if ($("#view-events").classList.contains("active")) loadEvents();
});

// ------------------------------------------------------------------ camera
$("#c-type").onchange = () => {
  const f = $("#c-type").value === "file";
  $("#c-file-wrap").style.display = f ? "" : "none"; $("#c-src-wrap").style.display = f ? "none" : "";
  $("#c-src").placeholder = $("#c-type").value === "webcam" ? "0 (webcam mặc định)" : "rtsp://user:pass@192.168.1.10:554/stream";
};
$("#c-add").onclick = async () => {
  const msg = $("#c-msg"); msg.textContent = "";
  try {
    const type = $("#c-type").value; let source = $("#c-src").value.trim();
    if (type === "file") {
      const f = $("#c-file").files[0]; if (!f) throw new Error("Chọn một file video");
      msg.textContent = "Đang tải video lên…";
      const fd = new FormData(); fd.append("file", f);
      source = (await api("/api/upload", { method: "POST", body: fd })).path;
    }
    if (type === "webcam" && !source) source = "0";
    if (!source) throw new Error("Nhập địa chỉ nguồn video");
    const name = $("#c-name").value.trim() || (type === "file" ? $("#c-file").files[0].name : "Camera " + (state.cams.length + 1));
    await api("/api/cameras", { method: "POST", body: JSON.stringify({ name, source_type: type, source, loop: $("#c-loop").checked, autostart: $("#c-auto").checked }) });
    msg.textContent = "Đã thêm. Lần chạy đầu sẽ tải mô hình YOLO (vài chục MB)…";
    state.cams = await api("/api/cameras"); renderCamRows(); renderTiles();
  } catch (err) { msg.textContent = "Lỗi: " + err.message; }
};
function renderCamRows() {
  $("#cam-rows").innerHTML = state.cams.map((c) => {
    const rt = c.runtime || {}, run = ["running", "starting", "reconnecting"].includes(rt.status);
    return `<tr data-id="${c.id}"><td><b>${esc(c.name)}</b></td><td class="hint">${esc(c.source_type)} · ${esc(c.source.split(/[\\/]/).pop())}</td>
      <td><span class="chip ${rt.status}">${CAM_STATUS[rt.status] || rt.status}</span> <span class="hint">${esc(rt.error || "")}</span></td>
      <td>${c.calibration ? "Đã hiệu chỉnh" : (GROUND[rt.ground?.mode] || "Tự động")}</td>
      <td class="row"><button class="btn sm" data-act="${run ? "stop" : "start"}">${run ? "Dừng" : "Chạy"}</button>
      <button class="btn sm" data-act="edit">Sửa</button><button class="btn sm" data-act="calib">Hiệu chỉnh</button><button class="btn sm danger" data-act="del">Xóa</button></td></tr>`;
  }).join("") || `<tr><td colspan="5" class="empty">Chưa có camera</td></tr>`;
}
$("#cam-rows").addEventListener("click", async (e) => {
  const b = e.target.closest("[data-act]"); if (!b) return;
  const id = b.closest("tr").dataset.id;
  if (b.dataset.act === "start") await api(`/api/cameras/${id}/start`, { method: "POST" });
  if (b.dataset.act === "stop") await api(`/api/cameras/${id}/stop`, { method: "POST" });
  if (b.dataset.act === "del" && confirm("Xóa camera này?")) await api(`/api/cameras/${id}`, { method: "DELETE" });
  if (b.dataset.act === "calib") return openCalib(id);
  if (b.dataset.act === "edit") return openEdit(id);
  state.cams = await api("/api/cameras"); renderCamRows(); renderTiles();
});

// ------------------------------------------------------------------ sửa camera
let editId = null;
function openEdit(id) {
  const c = state.cams.find((x) => x.id === id); if (!c) return;
  const file = c.source_type === "file";
  editId = id;
  $("#e-name").value = c.name;
  $("#e-type").value = file ? "rtsp" : c.source_type;
  $("#e-src").value = c.source;
  $("#e-loop").checked = !!c.loop; $("#e-auto").checked = !!c.autostart;
  // file video: chỉ sửa tên/tùy chọn, muốn đổi video thì thêm camera mới
  $("#e-type-wrap").style.display = $("#e-src-wrap").style.display = file ? "none" : "";
  $("#e-loop-wrap").style.display = file ? "" : "none";
  $("#e-hint").style.display = file ? "none" : "";
  $("#e-msg").textContent = "";
  $("#m-cam").classList.add("open");
}
$("#e-save").onclick = async () => {
  const c = state.cams.find((x) => x.id === editId); if (!c) return;
  const body = { name: $("#e-name").value, loop: $("#e-loop").checked, autostart: $("#e-auto").checked };
  if (c.source_type !== "file") { body.source_type = $("#e-type").value; body.source = $("#e-src").value; }
  try {
    await api(`/api/cameras/${editId}`, { method: "PUT", body: JSON.stringify(body) });
    $("#m-cam").classList.remove("open");
    state.cams = await api("/api/cameras"); renderCamRows(); renderTiles();
  } catch (err) { $("#e-msg").textContent = "Lỗi: " + err.message; }
};

// ------------------------------------------------------------------ hiệu chỉnh
const cal = { id: null, pts: [] };
function openCalib(id) {
  cal.id = id; cal.pts = [];
  const img = $("#calib-img");
  img.onerror = () => { $("#cal-msg").textContent = "Chưa có khung hình – hãy chạy camera trước."; };
  img.onload = drawCalib;
  img.src = `/api/cameras/${id}/frame?ts=${Date.now()}`;
  $("#m-calib").classList.add("open");
}
$("#calib-img").addEventListener("click", (e) => {
  if (cal.pts.length >= 4) return;
  const img = e.currentTarget, r = img.getBoundingClientRect();
  cal.pts.push([(e.clientX - r.left) * img.naturalWidth / r.width, (e.clientY - r.top) * img.naturalHeight / r.height]);
  drawCalib();
});
function drawCalib() {
  const img = $("#calib-img"), svg = $("#calib-svg");
  svg.setAttribute("viewBox", `0 0 ${img.naturalWidth} ${img.naturalHeight}`);
  const s = img.naturalWidth / 100;
  svg.innerHTML = (cal.pts.length > 1 ? `<polygon points="${cal.pts.map((p) => p.join(",")).join(" ")}" fill="rgba(59,130,246,.2)" stroke="#3b82f6" stroke-width="${s * 0.3}"/>` : "")
    + cal.pts.map((p, i) => `<circle cx="${p[0]}" cy="${p[1]}" r="${s * 0.8}" fill="#f59e0b"/><text x="${p[0] + s}" y="${p[1] - s}" fill="#fff" font-size="${s * 2.5}" font-weight="700">${"ABCD"[i]}</text>`).join("");
  $("#cal-msg").textContent = `Đã chọn ${cal.pts.length}/4 điểm`;
}
$("#cal-reset").onclick = () => { cal.pts = []; drawCalib(); };
$("#cal-save").onclick = async () => {
  if (cal.pts.length !== 4) { $("#cal-msg").textContent = "Cần đủ 4 điểm"; return; }
  const w = +$("#cal-w").value, l = +$("#cal-l").value;
  await api(`/api/cameras/${cal.id}/calibration`, { method: "PUT", body: JSON.stringify({ img_pts: cal.pts, ground_pts: [[0, 0], [w, 0], [w, l], [0, l]] }) });
  $("#m-calib").classList.remove("open"); state.cams = await api("/api/cameras"); renderCamRows();
};
$("#cal-clear").onclick = async () => {
  await api(`/api/cameras/${cal.id}/calibration`, { method: "PUT", body: JSON.stringify({}) });
  $("#m-calib").classList.remove("open"); state.cams = await api("/api/cameras"); renderCamRows();
};

// ------------------------------------------------------------------ thống kê
async function loadStats() {
  const s = await api("/api/stats");
  const total = Object.values(s.by_type).reduce((a, b) => a + b, 0);
  const crit = (s.by_severity || {}).critical || 0;
  const fa = s.by_status.false_alarm || 0, reviewed = (s.by_status.ack || 0) + (s.by_status.resolved || 0) + fa;
  $("#kpis").innerHTML = [["Tổng sự kiện", total], ["Nghiêm trọng", crit], ["Chưa xử lý", s.by_status.new || 0],
    ["Precision theo phản hồi", reviewed ? ((1 - fa / reviewed) * 100).toFixed(0) + "%" : "—"]]
    .map(([l, v]) => `<div class="kpi"><div class="l">${l}</div><div class="v">${v}</div></div>`).join("");
  const max = Math.max(1, ...Object.values(s.by_type));
  $("#bars-type").innerHTML = Object.keys(LABELS).map((k) => `<div class="b"><span>${LABELS[k]}</span><div class="track"><i style="width:${100 * (s.by_type[k] || 0) / max}%"></i></div><b>${s.by_type[k] || 0}</b></div>`).join("");
  const days = {}; s.by_day.forEach((r) => { days[r.day] = (days[r.day] || 0) + r.count; });
  const dmax = Math.max(1, ...Object.values(days));
  $("#bars-day").innerHTML = Object.entries(days).slice(0, 14).map(([d, c]) => `<div class="b"><span>${d}</span><div class="track"><i style="width:${100 * c / dmax}%"></i></div><b>${c}</b></div>`).join("") || `<div class="hint">Chưa có dữ liệu</div>`;
  $("#feedback").innerHTML = `<div class="hint">Đã xem xét ${reviewed} sự kiện: ${(s.by_status.ack || 0) + (s.by_status.resolved || 0)} đúng, ${fa} báo nhầm. Đánh dấu "Báo nhầm" trong chi tiết sự kiện để theo dõi tỉ lệ cảnh báo giả khi chạy thực tế.</div>`;
}
$("#btn-clear").onclick = async () => { if (confirm("Xóa toàn bộ sự kiện?")) { await api("/api/events", { method: "DELETE" }); state.feed = []; renderFeed(); loadStats(); } };

// ------------------------------------------------------------------ cài đặt
const SETTING_META = {
  fall: ["Té ngã", {
    descent_speed: ["Tốc độ rơi tối thiểu", "chiều cao người / giây"], height_drop: ["Tỉ lệ giảm chiều cao", "0–1, trong cửa sổ rơi"],
    descent_window: ["Cửa sổ rơi (s)"], lying_angle: ["Góc thân coi là nằm (°)"], upright_angle: ["Góc thân coi là đứng (°)"],
    lying_aspect: ["Tỉ lệ rộng/cao bbox coi là nằm", "dự phòng khi thiếu keypoint"], confirm_window: ["Thời gian rơi → nằm tối đa (s)"],
    lying_hold: ["Nằm liên tục để xác nhận (s)"], immobile_spread: ["Ngưỡng bất động", "độ dao động bbox / chiều cao người"],
    torso_rate: ["Tốc độ đổ thân (° trong 0,5 s)"],
    immobile_after: ["Bất động sau ngã để nâng mức (s)"], person_down_after: ["Nằm bất động không có pha rơi (s)"], recover_hold: ["Đứng lại để kết thúc (s)"] }],
  collision: ["Va chạm", {
    near_distance: ["Khoảng cách nguy hiểm (m)"], near_ttc: ["TTC nguy hiểm (s)"], near_hold: ["Duy trì điều kiện (s)"],
    min_vehicle_speed: ["Tốc độ xe tối thiểu (m/s)"], contact_distance: ["Khoảng cách tiếp xúc người–xe (m)"],
    confirm_window: ["Chờ dấu hiệu sau tiếp xúc (s)"], push_accel: ["Gia tốc bị đẩy (m/s²)"], vv_contact_distance: ["Tiếp xúc xe–xe (m)"],
    vv_min_closing: ["Tốc độ tiến lại gần tối thiểu (m/s)"], vv_speed_drop: ["Tỉ lệ giảm tốc sau va chạm"], vv_heading_change: ["Đổi hướng đột ngột (°)"],
    cooldown: ["Không báo lặp trong (s)"] }],
  pipeline: ["Xử lý video", {
    process_fps: ["Khung hình phân tích / giây"], imgsz: ["Kích thước ảnh vào mô hình"], person_conf: ["Ngưỡng tin cậy người"],
    vehicle_conf: ["Ngưỡng tin cậy xe"], pose_model: ["Mô hình pose", "áp dụng khi chạy lại camera"], det_model: ["Mô hình phát hiện xe"],
    focal_ratio: ["Tiêu cự / chiều rộng ảnh", "dùng khi tự ước lượng"], clip_pre: ["Ghi trước sự kiện (s)"], clip_post: ["Ghi sau sự kiện (s)"],
    show_bev: ["Hiện bản đồ mặt nền"], realtime_files: ["Phát file với tốc độ thật"],
    stabilize: ["Bù rung camera", "nên bật với điện thoại cầm/kẹp"], shake_level: ["Mức rung coi là mạnh", "tỉ lệ chiều rộng ảnh / khung"] }],
  height_fall: ["Ngã từ trên cao", {
    enabled: ["Bật phát hiện ngã từ trên cao"], min_speed: ["Tốc độ rơi tối thiểu", "chiều cao người / giây (rơi tự do 2 m ≈ 3,7)"],
    min_drop: ["Độ rơi tối thiểu", "chiều cao người; ngã cùng mặt bằng chỉ ≈ 0,5"], min_accel: ["Gia tốc tối thiểu (H/s²)", "rơi tự do ≈ 5,8"],
    max_growth: ["Bbox to lên tối đa", "lớn hơn = đang chạy lại gần camera, không phải rơi"], window: ["Cửa sổ đo (s)"],
    outcome_window: ["Chờ kết quả sau khi rơi (s)"], lost_confirm: ["Mất dấu bao lâu thì báo (s)"], cooldown: ["Không báo lặp trong (s)"] }],
  objects: ["Vật rơi / bị vật đập trúng", {
    enabled: ["Bật phát hiện vật rơi"], work_width: ["Chiều rộng ảnh phân tích (px)"], warmup: ["Thời gian học nền (s)"],
    min_area: ["Diện tích vật nhỏ nhất", "tỉ lệ khung"], max_area: ["Diện tích vật lớn nhất", "tỉ lệ khung"],
    min_speed: ["Tốc độ rơi tối thiểu", "chiều cao khung / giây"], min_drop: ["Quãng rơi tối thiểu", "tỉ lệ chiều cao khung"],
    max_lateral: ["Lệch ngang tối đa", "so với quãng rơi; loại chim, vật bay ngang"], min_samples: ["Số khung tối thiểu để kết luận"],
    gate: ["Bán kính ghép vệt"], max_fg_ratio: ["Bỏ khung khi nền đổi quá", "tỉ lệ khung (đổi sáng, rung)"],
    max_blobs: ["Bỏ khung khi quá nhiều vệt", "mưa, nhiễu"], contact_margin: ["Mở rộng vùng người khi xét trúng"],
    confirm_window: ["Chờ người phản ứng sau khi trúng (s)"], cooldown: ["Không báo lặp trong (s)"] }],
  hazard: ["Cháy / khói", {
    enabled: ["Bật phát hiện cháy/khói"], fire_model: ["Mô hình cháy/khói (.pt)", "vd models/fire_smoke_best.pt; trống = dùng bộ lọc màu; áp dụng khi chạy lại camera"],
    conf: ["Ngưỡng tin cậy mô hình"], every: ["Chạy mô hình mỗi N khung phân tích"], window: ["Cửa sổ xác nhận (s)"],
    min_hits: ["Số lần phát hiện cùng vị trí để báo"], cooldown: ["Không báo lặp trong (s)"],
    color_fallback: ["Dùng bộ lọc màu khi chưa có mô hình", "chỉ nhận ra lửa, dễ báo nhầm hơn"],
    color_min_area: ["Diện tích màu lửa nhỏ nhất", "tỉ lệ khung"], color_max_area: ["Diện tích màu lửa lớn nhất", "tỉ lệ khung"],
    color_flicker: ["Độ nhấp nháy tối thiểu", "0–1; đèn, áo cam đứng yên ≈ 0"] }],
};
async function loadSettings() {
  const s = await api("/api/settings");
  $("#settings-form").innerHTML = Object.entries(SETTING_META).map(([sec, [title, fields]]) =>
    `<div class="section-title">${title}</div><div class="card"><div class="body form">` +
    Object.entries(fields).map(([k, [lab, hint]]) => {
      const v = s[sec][k], id = `s-${sec}-${k}`;
      return typeof v === "boolean" ? `<label class="check"><input type="checkbox" id="${id}" ${v ? "checked" : ""}> ${lab}</label>`
        : `<label class="f">${lab}<input type="${typeof v === "number" ? "number" : "text"}" step="any" id="${id}" value="${esc(v)}">${hint ? `<small>${hint}</small>` : ""}</label>`;
    }).join("") + `</div></div>`).join("");
}
$("#btn-save-settings").onclick = async () => {
  const body = {};
  for (const [sec, [, fields]] of Object.entries(SETTING_META)) {
    body[sec] = {};
    for (const k of Object.keys(fields)) { const el = $(`#s-${sec}-${k}`); body[sec][k] = el.type === "checkbox" ? el.checked : el.type === "number" ? +el.value : el.value; }
  }
  const msg = $("#settings-msg"), btn = $("#btn-save-settings");
  msg.textContent = "";
  const bad = Object.entries(body).flatMap(([sec, o]) => Object.entries(o).filter(([, v]) => typeof v === "number" && !Number.isFinite(v)).map(([k]) => `${sec}.${k}`));
  if (bad.length) { msg.textContent = "Không lưu được: ô số bị trống hoặc sai (" + bad.join(", ") + ")"; return; }
  try {
    await api("/api/settings", { method: "PUT", body: JSON.stringify(body) });
    btn.textContent = "Đã lưu ✓"; setTimeout(() => btn.textContent = "Lưu cài đặt", 1500);
  } catch (err) { msg.textContent = "Không lưu được (chưa áp dụng gì): " + err.message; }
};

// ------------------------------------------------------------------ khởi động
(async () => {
  state.cams = await api("/api/cameras"); renderTiles();
  state.feed = (await api("/api/events?limit=30")).items; renderFeed();
  connect();
})();
