"use strict";

const COLORS = {
  ink: "#0F172A",
  primary: "#0F766E",
  secondary: "#D97706",
  accentBlue: "#2563EB",
  danger: "#DC2626",
  rushFill: "rgba(250, 204, 21, 0.18)",
};

const CATEGORY_COLORS = {
  "Lag Autoregresif": "#DC2626",
  "Siklik Temporal": "#16A34A",
  "Fisika Iklim": "#2563EB",
  "Cuaca Mentah": "#64748B",
};

const RUSH_HOURS = new Set([6, 7, 8, 9, 16, 17, 18, 19]);

const state = {
  summary: null,
  predictions: null,
  samples: null,
  activeTab: "tab1",
  mainCode: "F'",
  cmpCode: "C",
  preset: "14d",
  dateStart: "2026-04-22",
  dateEnd: "2026-05-05",
  onlyPeak: false,
  onlyRush: false,
  cycleOpt: "diurnal",
  xaiMode: "perm",
};

function escHtml(v) {
  return String(v)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function getSpec(code) {
  return state.summary.model_specs.find((m) => m.code === code) || { code, display_name: code };
}

function getLeaderboardRow(code) {
  return state.summary.leaderboard.find((r) => r.Kode === code) || null;
}

function computeMetrics(yTrue, yPred) {
  const n = yTrue.length;
  if (n === 0) return { rmse: null, mae: null, mape: null, r2: null };
  let sumSq = 0;
  let sumAbs = 0;
  let sumApe = 0;
  let apeCnt = 0;
  let sumY = 0;
  for (let i = 0; i < n; i++) {
    const yt = yTrue[i];
    const yp = yPred[i];
    const diff = yt - yp;
    sumSq += diff * diff;
    sumAbs += Math.abs(diff);
    sumY += yt;
    if (yt > 0) {
      sumApe += Math.abs(diff / yt) * 100.0;
      apeCnt++;
    }
  }
  const rmse = Math.sqrt(sumSq / n);
  const mae = sumAbs / n;
  const mape = apeCnt > 0 ? sumApe / apeCnt : null;
  let r2 = null;
  if (n >= 2) {
    const meanY = sumY / n;
    let ssTot = 0;
    for (let i = 0; i < n; i++) {
      const d = yTrue[i] - meanY;
      ssTot += d * d;
    }
    if (ssTot > 0) r2 = 1.0 - sumSq / ssTot;
  }
  return { rmse, mae, mape, r2 };
}

function basePlotlyLayout(title, height) {
  return {
    title: title ? { text: title, font: { size: 15, color: COLORS.ink }, x: 0.01, y: 0.96 } : undefined,
    template: "plotly_white",
    height,
    paper_bgcolor: "#FFFFFF",
    plot_bgcolor: "#FFFFFF",
    font: { family: "sans-serif", color: COLORS.ink, size: 12 },
    margin: { l: 54, r: 24, t: title ? 72 : 34, b: 48 },
    legend: { orientation: "h", yanchor: "bottom", y: 1.01, xanchor: "right", x: 1 },
    hovermode: "x unified",
    xaxis: { showgrid: true, gridcolor: "#F1F5F9", zeroline: false },
    yaxis: { showgrid: true, gridcolor: "#F1F5F9", zeroline: false },
  };
}

function formatErrorDelta(val, cmpVal, cmpCode, unit) {
  if (val === null || cmpVal === null || cmpVal === 0) {
    return '<div class="kpi-delta-neutral">Pembanding tidak tersedia</div>';
  }
  const diff = val - cmpVal;
  const pct = (diff / cmpVal) * 100.0;
  let cls = "kpi-delta-neutral";
  let arrow = "•";
  if (diff < -1e-6) {
    cls = "kpi-delta-good";
    arrow = "▼";
  } else if (diff > 1e-6) {
    cls = "kpi-delta-warn";
    arrow = "▲";
  }
  const signDiff = diff >= 0 ? `+${diff.toFixed(3)}` : diff.toFixed(3);
  const signPct = pct >= 0 ? `+${pct.toFixed(2)}` : pct.toFixed(2);
  return `<div class="${cls}">${arrow} ${signDiff}${unit} (${signPct}%) vs ${escHtml(cmpCode)}</div>`;
}

function formatR2Delta(val, cmpVal, cmpCode) {
  if (val === null || cmpVal === null) {
    return '<div class="kpi-delta-neutral">Butuh ≥ 2 titik bervariasi</div>';
  }
  const diff = val - cmpVal;
  let cls = "kpi-delta-neutral";
  let arrow = "•";
  if (diff > 1e-6) {
    cls = "kpi-delta-good";
    arrow = "▲";
  } else if (diff < -1e-6) {
    cls = "kpi-delta-warn";
    arrow = "▼";
  }
  const signDiff = diff >= 0 ? `+${diff.toFixed(4)}` : diff.toFixed(4);
  return `<div class="${cls}">${arrow} ${signDiff} vs ${escHtml(cmpCode)}</div>`;
}

function getDateWindow() {
  if (state.preset === "14d") return ["2026-04-22", "2026-05-05"];
  if (state.preset === "all") return ["2026-04-22", "2026-10-03"];
  const s = state.dateStart <= state.dateEnd ? state.dateStart : state.dateEnd;
  const e = state.dateEnd >= state.dateStart ? state.dateEnd : state.dateStart;
  return [s, e];
}

function formatShortDate(dateStr) {
  const dt = new Date(`${dateStr}T00:00:00`);
  return dt.toLocaleDateString("id-ID", { day: "2-digit", month: "short", year: "numeric" });
}

function renderTab1() {
  const { predictions, summary, mainCode, cmpCode } = state;
  const p95 = summary.split_metadata.p95_threshold;
  const [dStart, dEnd] = getDateWindow();

  const times = [];
  const yTrue = [];
  const yMain = [];
  const yCmp = [];
  const pMainAll = predictions.preds[mainCode];
  const pCmpAll = predictions.preds[cmpCode];

  for (let i = 0; i < predictions.times.length; i++) {
    const dt = predictions.dates[i];
    if (dt < dStart || dt > dEnd) continue;
    const act = predictions.actual[i];
    if (state.onlyPeak && act < p95) continue;
    if (state.onlyRush && !RUSH_HOURS.has(predictions.hours[i])) continue;
    times.push(predictions.times[i]);
    yTrue.push(act);
    yMain.push(pMainAll[i]);
    yCmp.push(pCmpAll[i]);
  }

  const nSelected = times.length;
  const emptyAlert = document.getElementById("emptyStateAlert");
  const contentBox = document.getElementById("tab1Content");
  if (nSelected === 0) {
    emptyAlert.classList.remove("hidden");
    contentBox.classList.add("hidden");
    return;
  }
  emptyAlert.classList.add("hidden");
  contentBox.classList.remove("hidden");

  const mMain = computeMetrics(yTrue, yMain);
  const mCmp = computeMetrics(yTrue, yCmp);
  const lbRow = getLeaderboardRow(mainCode) || { RMSE: "-", MAE: "-", "MAPE (%)": "-", R2: "-" };

  document.getElementById("kpiGrid").innerHTML = `
    <div class="kpi-card">
      <div class="kpi-label">RMSE (Data Uji Terfilter)</div>
      <div class="kpi-value">${mMain.rmse.toFixed(3)} µg/m³</div>
      ${formatErrorDelta(mMain.rmse, mCmp.rmse, cmpCode, " µg/m³")}
      <div class="kpi-foot">10-Seed Overall: ${escHtml(lbRow.RMSE)}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">MAE (Data Uji Terfilter)</div>
      <div class="kpi-value">${mMain.mae.toFixed(3)} µg/m³</div>
      ${formatErrorDelta(mMain.mae, mCmp.mae, cmpCode, " µg/m³")}
      <div class="kpi-foot">10-Seed Overall: ${escHtml(lbRow.MAE)}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">MAPE (%)</div>
      <div class="kpi-value">${mMain.mape !== null ? `${mMain.mape.toFixed(2)}%` : "N/A"}</div>
      ${formatErrorDelta(mMain.mape, mCmp.mape, cmpCode, "%")}
      <div class="kpi-foot">10-Seed Overall: ${escHtml(lbRow["MAPE (%)"])}%</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Koefisien R²</div>
      <div class="kpi-value">${mMain.r2 !== null ? mMain.r2.toFixed(3) : "N/A"}</div>
      ${formatR2Delta(mMain.r2, mCmp.r2, cmpCode)}
      <div class="kpi-foot">10-Seed Overall: ${escHtml(lbRow.R2)} · n=${nSelected.toLocaleString("id-ID")}</div>
    </div>
  `;

  // Interactive time-series trace mode & rush-hour bands when window <= 21 days
  const t0 = new Date(times[0].replace(" ", "T")).getTime();
  const t1 = new Date(times[times.length - 1].replace(" ", "T")).getTime();
  const spanHours = (t1 - t0) / 3600000 + 1;
  const spanDays = (t1 - t0) / 86400000;
  const isContiguous = nSelected > 1 && spanHours <= nSelected * 1.25;
  const traceMode = isContiguous ? "lines" : "markers";

  const traces = [
    {
      x: times,
      y: yTrue,
      name: "Observasi Aktual PM2.5 (CAMS)",
      type: "scatter",
      mode: traceMode,
      line: { color: COLORS.ink, width: 2.2 },
      marker: { size: 5 },
    },
    {
      x: times,
      y: yMain,
      name: getSpec(mainCode).display_name,
      type: "scatter",
      mode: traceMode,
      line: { color: COLORS.primary, width: 2.0 },
      marker: { size: 5 },
    },
  ];
  if (cmpCode !== mainCode) {
    traces.push({
      x: times,
      y: yCmp,
      name: getSpec(cmpCode).display_name,
      type: "scatter",
      mode: traceMode,
      line: { color: COLORS.secondary, width: 1.6, dash: "dash" },
      marker: { size: 4 },
    });
  }

  const shapes = [
    {
      type: "line",
      xref: "paper",
      x0: 0,
      x1: 1,
      yref: "y",
      y0: p95,
      y1: p95,
      line: { color: COLORS.danger, width: 1.5, dash: "dot" },
    },
  ];
  if (spanDays <= 21 && nSelected > 1) {
    const minTsStr = times[0];
    const maxTsStr = times[times.length - 1];
    const cur = new Date(`${minTsStr.slice(0, 10)}T00:00:00`);
    const endDt = new Date(`${maxTsStr.slice(0, 10)}T23:59:59`);
    while (cur <= endDt) {
      const ymd = cur.toISOString().slice(0, 10);
      for (const [hStart, hEnd] of [["06:00", "10:00"], ["16:00", "20:00"]]) {
        const bx0 = `${ymd} ${hStart}`;
        const bx1 = `${ymd} ${hEnd}`;
        if (bx1 <= minTsStr || bx0 >= maxTsStr) continue;
        shapes.push({
          type: "rect",
          xref: "x",
          yref: "paper",
          x0: bx0 < minTsStr ? minTsStr : bx0,
          x1: bx1 > maxTsStr ? maxTsStr : bx1,
          y0: 0,
          y1: 1,
          fillcolor: COLORS.rushFill,
          line: { width: 0 },
          layer: "below",
        });
      }
      cur.setDate(cur.getDate() + 1);
    }
  }

  const layoutTs = basePlotlyLayout(
    `Konsentrasi PM2.5 Aktual vs Prediksi (${formatShortDate(times[0].slice(0, 10))} – ${formatShortDate(times[times.length - 1].slice(0, 10))}, n=${nSelected.toLocaleString("id-ID")} jam)`,
    420
  );
  layoutTs.xaxis.title = { text: "Waktu (WIB)" };
  layoutTs.yaxis.title = { text: "Konsentrasi PM2.5 (µg/m³)" };
  layoutTs.shapes = shapes;
  layoutTs.annotations = [
    {
      xref: "paper",
      x: 0.01,
      yref: "y",
      y: p95,
      text: `Ambang Puncak P95 (${p95.toFixed(1)} µg/m³)`,
      showarrow: false,
      yanchor: "bottom",
      font: { size: 11, color: COLORS.danger },
    },
  ];
  Plotly.react("chartTimeSeries", traces, layoutTs, { responsive: true });

  // Sub-segment table
  const focusCodes = [...new Set([mainCode, cmpCode, "A", "C", "D'", "F'"])];
  const subRows = summary.subsegment_eval.filter((r) => focusCodes.includes(r.Kode));
  const subCols = [
    "Kode",
    "Model",
    "RMSE total",
    "RMSE puncak (>=P95)",
    "Bias puncak (pred-aktual)",
    "RMSE jam sibuk",
    "RMSE di luar jam sibuk",
    "MAE jam sibuk",
  ];
  let tblHtml = "<table><thead><tr>" + subCols.map((c) => `<th>${escHtml(c)}</th>`).join("") + "</tr></thead><tbody>";
  for (const r of subRows) {
    const hi = r.Kode === mainCode ? ' class="row-highlight"' : "";
    tblHtml += `<tr${hi}>` + subCols.map((c) => {
      const val = r[c];
      return `<td>${typeof val === "number" ? val.toFixed(3) : escHtml(val)}</td>`;
    }).join("") + "</tr>";
  }
  tblHtml += "</tbody></table>";
  document.getElementById("subsegmentTableWrap").innerHTML = tblHtml;

  const mainSub = summary.subsegment_eval.find((r) => r.Kode === mainCode);
  const rawSub = summary.subsegment_eval.find((r) => r.Kode === "C");
  const peakDiff = mainSub["RMSE puncak (>=P95)"] - rawSub["RMSE puncak (>=P95)"];
  const peakSign = peakDiff >= 0 ? `+${peakDiff.toFixed(3)}` : peakDiff.toFixed(3);
  const biasSign = mainSub["Bias puncak (pred-aktual)"] >= 0
    ? `+${mainSub["Bias puncak (pred-aktual)"].toFixed(3)}`
    : mainSub["Bias puncak (pred-aktual)"].toFixed(3);
  document.getElementById("subsegmentCaption").textContent =
    `Ambang P95 Data Uji = ${p95.toFixed(1)} µg/m³ (n=198 puncak; n=1.316 jam sibuk). Selisih RMSE puncak ${mainCode} vs Raw (C): ${peakSign} µg/m³; Bias puncak ${mainCode}: ${biasSign} µg/m³.`;

  renderCycleChart();
}

function renderCycleChart() {
  const { summary, cycleOpt, mainCode, cmpCode } = state;
  const layout = basePlotlyLayout("", 330);
  if (cycleOpt === "diurnal") {
    const dp = summary.diurnal_profile;
    const traces = [
      { x: dp.hours, y: dp.pm2_5, name: "Rata-rata PM2.5 (µg/m³)", type: "scatter", mode: "lines+markers", line: { color: COLORS.ink, width: 2.3 } },
      { x: dp.hours, y: dp.wind_speed_10m, name: "Kecepatan Angin (km/jam)", type: "scatter", mode: "lines", yaxis: "y2", line: { color: COLORS.accentBlue, dash: "dash" } },
      { x: dp.hours, y: dp.dew_point, name: "Titik Embun (°C)", type: "scatter", mode: "lines", yaxis: "y2", line: { color: COLORS.primary, dash: "dot", width: 2.2 } },
    ];
    layout.xaxis.title = { text: "Jam (WIB)" };
    layout.xaxis.dtick = 2;
    layout.yaxis.title = { text: "PM2.5 (µg/m³)" };
    layout.yaxis2 = { title: { text: "Angin / Titik Embun" }, overlaying: "y", side: "right", showgrid: false };
    layout.shapes = [
      { type: "rect", xref: "x", yref: "paper", x0: 5.5, x1: 9.5, y0: 0, y1: 1, fillcolor: COLORS.rushFill, line: { width: 0 }, layer: "below" },
      { type: "rect", xref: "x", yref: "paper", x0: 15.5, x1: 19.5, y0: 0, y1: 1, fillcolor: COLORS.rushFill, line: { width: 0 }, layer: "below" },
    ];
    Plotly.react("chartCycle", traces, layout, { responsive: true });
  } else if (cycleOpt === "monthly") {
    const mp = summary.monthly_profile;
    const traces = [
      {
        x: mp.months,
        y: mp.pm2_5,
        type: "bar",
        marker: { color: COLORS.accentBlue },
        text: mp.pm2_5.map((v) => v.toFixed(1)),
        textposition: "outside",
      },
    ];
    layout.yaxis.title = { text: "Rata-rata PM2.5 (µg/m³)" };
    layout.yaxis.range = [0, Math.max(...mp.pm2_5) * 1.15];
    Plotly.react("chartCycle", traces, layout, { responsive: true });
  } else {
    const hours = Array.from({ length: 24 }, (_, i) => i);
    const traces = [
      { x: hours, y: summary.hourly_mae[mainCode], name: `Model ${mainCode}`, type: "scatter", mode: "lines+markers", line: { color: COLORS.primary, width: 2.2 } },
    ];
    if (cmpCode !== mainCode) {
      traces.push({ x: hours, y: summary.hourly_mae[cmpCode], name: `Model ${cmpCode}`, type: "scatter", mode: "lines+markers", line: { color: COLORS.secondary, width: 1.8, dash: "dash" } });
    }
    layout.xaxis.title = { text: "Jam (WIB)" };
    layout.xaxis.dtick = 2;
    layout.yaxis.title = { text: "MAE per Jam (µg/m³)" };
    layout.shapes = [
      { type: "rect", xref: "x", yref: "paper", x0: 5.5, x1: 9.5, y0: 0, y1: 1, fillcolor: COLORS.rushFill, line: { width: 0 }, layer: "below" },
      { type: "rect", xref: "x", yref: "paper", x0: 15.5, x1: 19.5, y0: 0, y1: 1, fillcolor: COLORS.rushFill, line: { width: 0 }, layer: "below" },
    ];
    Plotly.react("chartCycle", traces, layout, { responsive: true });
  }
}

function renderTab2() {
  const { summary, samples } = state;
  const meta = summary.split_metadata;
  const tr = meta.splits.Train;
  const va = meta.splits.Validation;
  const te = meta.splits.Test;

  document.getElementById("splitCards").innerHTML = `
    <div class="kpi-card">
      <div class="kpi-label">Total Dataset Bersih</div>
      <div class="kpi-value">${meta.clean_rows.toLocaleString("id-ID")} Baris × ${meta.feature_cols + 1} Kolom</div>
      <div class="kpi-foot">29 Fitur Prediktor + Target pm2_5 (6 baris lag awal dibuang)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Segmen Latih &amp; Validasi</div>
      <div class="kpi-value">Train: ${tr.rows.toLocaleString("id-ID")} | Val: ${va.rows.toLocaleString("id-ID")}</div>
      <div class="kpi-foot">Rasio 70% : 15% · Rerata PM2.5: ${tr.mean_pm25} &amp; ${va.mean_pm25} µg/m³</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Segmen Uji (Out-of-Sample)</div>
      <div class="kpi-value">Test: ${te.rows.toLocaleString("id-ID")} Baris (15%)</div>
      <div class="kpi-foot">Evaluasi Kronologis Buta · Rerata PM2.5: ${te.mean_pm25} µg/m³</div>
    </div>
  `;

  const segKey = document.getElementById("segSelect").value;
  const grpKey = document.getElementById("grpSelect").value;
  const limit = Number(document.getElementById("rowsLimitSelect").value);

  const selectedCols = grpKey === "ALL" ? summary.all_columns : ["pm2_5", ...summary.feature_groups[grpKey]];
  const colIndexes = selectedCols.map((c) => samples.columns.indexOf(c));
  const segObj = samples.segments[segKey];
  const rowCount = Math.min(limit, segObj.times.length);

  let html = "<table><thead><tr><th>time</th>" + selectedCols.map((c) => `<th>${escHtml(c)}</th>`).join("") + "</tr></thead><tbody>";
  for (let i = 0; i < rowCount; i++) {
    const rowVals = segObj.rows[i];
    html += `<tr><td>${escHtml(segObj.times[i])}</td>` + colIndexes.map((idx) => `<td>${rowVals[idx]}</td>`).join("") + "</tr>";
  }
  html += "</tbody></table>";
  document.getElementById("datasetTableWrap").innerHTML = html;

  // Descriptive stats for segment & columns
  const segStats = summary.descriptive_stats[segKey];
  let dHtml = "<table><thead><tr><th>Variabel</th><th>count</th><th>mean</th><th>std</th><th>min</th><th>50%</th><th>max</th></tr></thead><tbody>";
  for (const col of selectedCols) {
    const s = segStats[col];
    dHtml += `<tr><td>${escHtml(col)}</td><td>${s.count.toLocaleString("id-ID")}</td><td>${s.mean}</td><td>${s.std}</td><td>${s.min}</td><td>${s.median}</td><td>${s.max}</td></tr>`;
  }
  dHtml += "</tbody></table>";
  document.getElementById("descStatsTableWrap").innerHTML = dHtml;

  // CAMS diagnostics table
  let cHtml = "<table><thead><tr><th>jam WIB mod 3</th><th>mean</th><th>median</th><th>frac |d2| &lt;= 0.25</th></tr></thead><tbody>";
  for (const r of summary.cams_diagnostics) {
    cHtml += `<tr><td>${r["jam WIB mod 3"]}</td><td>${r.mean}</td><td>${r.median}</td><td>${r["frac |d2| <= 0.25"]}</td></tr>`;
  }
  cHtml += "</tbody></table>";
  document.getElementById("camsDiagTableWrap").innerHTML = cHtml;
}

function downloadDatasetCsv() {
  const { summary, samples } = state;
  const segKey = document.getElementById("segSelect").value;
  const grpKey = document.getElementById("grpSelect").value;
  const limit = Number(document.getElementById("rowsLimitSelect").value);
  const selectedCols = grpKey === "ALL" ? summary.all_columns : ["pm2_5", ...summary.feature_groups[grpKey]];
  const colIndexes = selectedCols.map((c) => samples.columns.indexOf(c));
  const segObj = samples.segments[segKey];
  const rowCount = Math.min(limit, segObj.times.length);

  const lines = [["time", ...selectedCols].join(",")];
  for (let i = 0; i < rowCount; i++) {
    lines.push([`"${segObj.times[i]}"`, ...colIndexes.map((idx) => segObj.rows[i][idx])].join(","));
  }
  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `jakarta_pm25_${segKey}_slice.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

function parseMeanStd(text) {
  const s = String(text).trim();
  if (s.includes("±")) {
    const [m, sd] = s.split("±");
    return [Number(m.trim()), Number(sd.trim())];
  }
  return [Number(s), 0];
}

function renderTab3() {
  const { summary, xaiMode } = state;
  const lb = summary.leaderboard;
  const codes = lb.map((r) => r.Kode);
  const rmseParsed = lb.map((r) => parseMeanStd(r.RMSE));
  const maeParsed = lb.map((r) => parseMeanStd(r.MAE));
  const barColors = codes.map((c) =>
    c === "F'" ? COLORS.primary : c === "D'" ? COLORS.accentBlue : ["A", "B", "B2"].includes(c) ? "#94A3B8" : "#64748B"
  );

  const tracesLb = [
    {
      x: codes,
      y: rmseParsed.map((v) => v[0]),
      error_y: { type: "data", array: rmseParsed.map((v) => v[1]), visible: true },
      type: "bar",
      marker: { color: barColors },
      text: rmseParsed.map((v) => v[0].toFixed(2)),
      textposition: "outside",
      xaxis: "x",
      yaxis: "y",
      hovertemplate: "%{x}: %{y:.3f} µg/m³<extra></extra>",
    },
    {
      x: codes,
      y: maeParsed.map((v) => v[0]),
      error_y: { type: "data", array: maeParsed.map((v) => v[1]), visible: true },
      type: "bar",
      marker: { color: barColors },
      text: maeParsed.map((v) => v[0].toFixed(2)),
      textposition: "outside",
      xaxis: "x2",
      yaxis: "y2",
      hovertemplate: "%{x}: %{y:.3f} µg/m³<extra></extra>",
    },
  ];
  const layoutLb = {
    template: "plotly_white",
    height: 340,
    paper_bgcolor: "#FFFFFF",
    plot_bgcolor: "#FFFFFF",
    showlegend: false,
    font: { family: "sans-serif", color: COLORS.ink, size: 12 },
    margin: { l: 54, r: 24, t: 56, b: 40 },
    grid: { rows: 1, columns: 2, pattern: "independent" },
    yaxis: { title: { text: "RMSE (µg/m³)" }, range: [0, Math.max(...rmseParsed.map((v) => v[0])) * 1.18], gridcolor: "#F1F5F9" },
    yaxis2: { title: { text: "MAE (µg/m³)" }, range: [0, Math.max(...maeParsed.map((v) => v[0])) * 1.18], gridcolor: "#F1F5F9" },
    annotations: [
      { text: "RMSE Data Uji (mean ± std)", x: 0.22, y: 1.07, xref: "paper", yref: "paper", showarrow: false, font: { size: 13, color: COLORS.ink } },
      { text: "MAE Data Uji (mean ± std)", x: 0.78, y: 1.07, xref: "paper", yref: "paper", showarrow: false, font: { size: 13, color: COLORS.ink } },
    ],
  };
  Plotly.react("chartLeaderboard", tracesLb, layoutLb, { responsive: true });

  const lbCols = ["Kode", "Model", "RMSE", "MAE", "MAPE (%)", "R2", "Delta vs Raw (sama regime)", "Delta efek tuning (set sama)"];
  let lbHtml = "<table><thead><tr>" + lbCols.map((c) => `<th>${escHtml(c)}</th>`).join("") + "</tr></thead><tbody>";
  for (const r of lb) {
    const hi = r.Kode === "F'" || r.Kode === "D'" ? ' class="row-highlight"' : "";
    lbHtml += `<tr${hi}>` + lbCols.map((c) => `<td>${escHtml(r[c])}</td>`).join("") + "</tr>";
  }
  lbHtml += "</tbody></table>";
  document.getElementById("leaderboardTableWrap").innerHTML = lbHtml;

  // XAI horizontal bars
  renderXaiChart(xaiMode);

  // Category totals table
  const catCols = ["Kategori", "PermImp (naik RMSE)", "Gain Importance"];
  let catHtml = "<table><thead><tr>" + catCols.map((c) => `<th>${escHtml(c)}</th>`).join("") + "</tr></thead><tbody>";
  for (const r of summary.category_totals) {
    catHtml += `<tr><td>${escHtml(r.Kategori)}</td><td>${Number(r["PermImp (naik RMSE)"]).toFixed(3)}</td><td>${Number(r["Gain Importance"]).toFixed(3)}</td></tr>`;
  }
  catHtml += "</tbody></table>";
  document.getElementById("categoryTotalsTableWrap").innerHTML = catHtml;

  // Tuning summary
  const t = summary.tuning_summary;
  const nameMap = { C: "Raw", D: "+Temporal", E: "+Iklim", F: "Full / Model G" };
  let cfgHtml = "<table><thead><tr><th>Skenario</th><th>n_estimators</th><th>max_depth</th><th>lr</th><th>subsample</th><th>colsample</th><th>Val RMSE</th></tr></thead><tbody>";
  for (const k of ["C", "D", "E", "F"]) {
    const cfg = t.best_cfg[k];
    const valRmse = t.top5[k][0]["Val RMSE"];
    cfgHtml += `<tr><td>${k}' (${nameMap[k]})</td><td>${cfg.n_estimators}</td><td>${cfg.max_depth}</td><td>${cfg.learning_rate}</td><td>${cfg.subsample}</td><td>${cfg.colsample_bytree}</td><td>${Number(valRmse).toFixed(3)}</td></tr>`;
  }
  cfgHtml += "</tbody></table>";
  document.getElementById("bestCfgTableWrap").innerHTML = cfgHtml;

  let top5Html = "<table><thead><tr><th>Peringkat</th><th>max_depth</th><th>lr</th><th>subsample</th><th>n_estimators</th><th>Val RMSE</th><th>Val MAE</th></tr></thead><tbody>";
  t.top5.F.forEach((r, idx) => {
    top5Html += `<tr><td>${idx + 1}</td><td>${r.max_depth}</td><td>${r.learning_rate}</td><td>${r.subsample}</td><td>${r.n_estimators}</td><td>${Number(r["Val RMSE"]).toFixed(3)}</td><td>${Number(r["Val MAE"]).toFixed(3)}</td></tr>`;
  });
  top5Html += "</tbody></table>";
  document.getElementById("top5TuningTableWrap").innerHTML = top5Html;

  document.getElementById("tuningProtocolCaption").textContent =
    `Grid Search ${t.grid_count} kombinasi/set · Early Stopping ${t.early_stopping_rounds} ronde · ${t.seeds.length} Random Seeds.`;
}

function renderXaiChart(mode) {
  const colName = mode === "perm" ? "PermImp (naik RMSE)" : "Gain Importance";
  const xTitle = mode === "perm" ? "Kenaikan RMSE saat fitur diacak (µg/m³)" : "Proporsi Gain Importance";
  const showErr = mode === "perm";

  const sorted = [...state.summary.feature_importance]
    .sort((a, b) => b[colName] - a[colName])
    .slice(0, 15)
    .reverse();

  const traces = [];
  for (const [catName, catColor] of Object.entries(CATEGORY_COLORS)) {
    const sub = sorted.filter((r) => r.Kategori === catName);
    if (sub.length === 0) continue;
    traces.push({
      y: sub.map((r) => r.Fitur),
      x: sub.map((r) => r[colName]),
      type: "bar",
      orientation: "h",
      name: catName,
      marker: { color: catColor },
      error_x: showErr ? { type: "data", array: sub.map((r) => r.Std), visible: true } : undefined,
      text: sub.map((r) => Number(r[colName]).toFixed(3)),
      textposition: "outside",
      cliponaxis: false,
    });
  }

  const maxVal = Math.max(...sorted.map((r) => r[colName]));
  const layout = basePlotlyLayout("", 440);
  layout.margin = { l: 195, r: 45, t: 45, b: 48 };
  layout.barmode = "overlay";
  layout.hovermode = "y unified";
  layout.yaxis = { categoryorder: "array", categoryarray: sorted.map((r) => r.Fitur), showgrid: true, gridcolor: "#F1F5F9" };
  layout.xaxis = { title: { text: xTitle }, range: [0, maxVal * 1.24], showgrid: true, gridcolor: "#F1F5F9" };
  Plotly.react("chartXai", traces, layout, { responsive: true });
}

function populateControls() {
  const { summary } = state;
  const mainSel = document.getElementById("mainModelSelect");
  const cmpSel = document.getElementById("cmpModelSelect");
  const optsHtml = summary.model_specs
    .map((m) => `<option value="${escHtml(m.code)}">${escHtml(m.display_name)}</option>`)
    .join("");
  mainSel.innerHTML = optsHtml;
  cmpSel.innerHTML = optsHtml;
  mainSel.value = state.mainCode;
  cmpSel.value = state.cmpCode;

  const p95 = summary.split_metadata.p95_threshold;
  document.getElementById("lblPeak").textContent = `Puncak Polusi (>= P95: ${p95.toFixed(1)} µg/m³)`;

  const grpSel = document.getElementById("grpSelect");
  grpSel.innerHTML =
    '<option value="ALL">Semua (Target + 29 Fitur)</option>' +
    Object.keys(summary.feature_groups)
      .map((k) => `<option value="${escHtml(k)}">${escHtml(k)}</option>`)
      .join("");
}

function bindEvents() {
  const sidebar = document.getElementById("sidebar");
  document.getElementById("openSidebarBtn").addEventListener("click", () => sidebar.classList.add("open"));
  document.getElementById("closeSidebarBtn").addEventListener("click", () => sidebar.classList.remove("open"));

  document.getElementById("mainModelSelect").addEventListener("change", (e) => {
    state.mainCode = e.target.value;
    renderTab1();
  });
  document.getElementById("cmpModelSelect").addEventListener("change", (e) => {
    state.cmpCode = e.target.value;
    renderTab1();
  });

  document.querySelectorAll('input[name="datePreset"]').forEach((el) => {
    el.addEventListener("change", (e) => {
      state.preset = e.target.value;
      document.getElementById("customDatesBox").classList.toggle("hidden", state.preset !== "custom");
      renderTab1();
    });
  });

  document.getElementById("dateStart").addEventListener("change", (e) => {
    state.dateStart = e.target.value;
    renderTab1();
  });
  document.getElementById("dateEnd").addEventListener("change", (e) => {
    state.dateEnd = e.target.value;
    renderTab1();
  });
  document.getElementById("chkPeak").addEventListener("change", (e) => {
    state.onlyPeak = e.target.checked;
    renderTab1();
  });
  document.getElementById("chkRush").addEventListener("change", (e) => {
    state.onlyRush = e.target.checked;
    renderTab1();
  });

  document.querySelectorAll('input[name="cycleOpt"]').forEach((el) => {
    el.addEventListener("change", (e) => {
      state.cycleOpt = e.target.value;
      renderCycleChart();
    });
  });

  ["segSelect", "grpSelect", "rowsLimitSelect"].forEach((id) => {
    document.getElementById(id).addEventListener("change", renderTab2);
  });
  document.getElementById("downloadCsvBtn").addEventListener("click", downloadDatasetCsv);

  document.querySelectorAll('input[name="xaiMode"]').forEach((el) => {
    el.addEventListener("change", (e) => {
      state.xaiMode = e.target.value;
      renderXaiChart(state.xaiMode);
    });
  });

  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetTab = btn.dataset.tab;
      state.activeTab = targetTab;
      document.querySelectorAll(".tab-btn").forEach((b) => {
        const active = b.dataset.tab === targetTab;
        b.classList.toggle("active", active);
        b.setAttribute("aria-selected", String(active));
      });
      document.querySelectorAll(".tab-panel").forEach((panel) => {
        panel.classList.toggle("hidden", panel.id !== `panel-${targetTab}`);
        panel.classList.toggle("active", panel.id === `panel-${targetTab}`);
      });
      requestAnimationFrame(() => {
        if (targetTab === "tab1") {
          Plotly.Plots.resize("chartTimeSeries");
          Plotly.Plots.resize("chartCycle");
        } else if (targetTab === "tab3") {
          Plotly.Plots.resize("chartLeaderboard");
          Plotly.Plots.resize("chartXai");
        }
      });
    });
  });
}

async function initApp() {
  try {
    const [sumRes, predRes, sampRes] = await Promise.all([
      fetch("data/summary_artifacts.json"),
      fetch("data/predictions_test.json"),
      fetch("data/dataset_samples.json"),
    ]);
    if (!sumRes.ok || !predRes.ok || !sampRes.ok) {
      throw new Error("Gagal mengambil file data JSON statis.");
    }
    state.summary = await sumRes.json();
    state.predictions = await predRes.json();
    state.samples = await sampRes.json();

    document.getElementById("loadingBanner").classList.add("hidden");
    populateControls();
    bindEvents();
    renderTab1();
    renderTab2();
    renderTab3();
  } catch (err) {
    document.getElementById("loadingBanner").classList.add("hidden");
    const errBox = document.getElementById("errorBanner");
    errBox.textContent = `Gagal memuat dashboard: ${err.message}`;
    errBox.classList.remove("hidden");
  }
}

document.addEventListener("DOMContentLoaded", initApp);
