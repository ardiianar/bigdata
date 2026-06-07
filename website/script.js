async function loadCSV(filePath) {
  const response = await fetch(filePath);

  if (!response.ok) {
    throw new Error("File tidak ditemukan: " + filePath);
  }

  const text = await response.text();

  if (!text.trim()) {
    return [];
  }

  return text
    .trim()
    .split("\n")
    .map(row => row.split(","));
}

function cleanText(value) {
  return String(value || "").replaceAll("\r", "").trim();
}

function formatLabel(text) {
  return cleanText(text)
    .replaceAll("_", " ")
    .toUpperCase();
}

function formatNumber(value) {
  const cleanValue = cleanText(value);

  if (cleanValue !== "" && !isNaN(cleanValue)) {
    return Number(cleanValue).toLocaleString("id-ID");
  }

  return cleanValue;
}

function createTable(elementId, rows, limit = null) {
  const table = document.getElementById(elementId);
  table.innerHTML = "";

  if (!rows || rows.length === 0) {
    table.innerHTML = "<tr><td>Data tidak tersedia</td></tr>";
    return;
  }

  const displayRows = limit ? rows.slice(0, limit + 1) : rows;

  displayRows.forEach((row, index) => {
    const tr = document.createElement("tr");

    row.forEach(cell => {
      const cellElement = document.createElement(index === 0 ? "th" : "td");

      if (index === 0) {
        cellElement.textContent = formatLabel(cell);
      } else {
        cellElement.textContent = formatNumber(cell);
      }

      tr.appendChild(cellElement);
    });

    table.appendChild(tr);
  });
}

function createKPICards(elementId, rows) {
  const container = document.getElementById(elementId);
  container.innerHTML = "";

  if (!rows || rows.length <= 1) {
    container.innerHTML = "<p>Data tidak tersedia</p>";
    return;
  }

  rows.slice(1).forEach(row => {
    const card = document.createElement("div");
    card.className = "kpi-card";

    const title = document.createElement("h3");
    title.textContent = formatLabel(row[0]);

    const value = document.createElement("p");
    value.textContent = formatNumber(row[1]);

    card.appendChild(title);
    card.appendChild(value);
    container.appendChild(card);
  });
}

async function initDashboard() {
  try {
    const kpi = await loadCSV("data/dashboard_kpi.csv");
    createKPICards("kpiCards", kpi);

    const borough = await loadCSV("data/dashboard_borough_analysis.csv");
    createTable("boroughTable", borough);

    const risk = await loadCSV("data/dashboard_risk_distribution.csv");
    createTable("riskTable", risk);

    const evaluation = await loadCSV("data/model_evaluation.csv");
    createKPICards("evaluationCards", evaluation);

    const confusion = await loadCSV("data/confusion_matrix.csv");
    createTable("confusionTable", confusion);

    const forecast = await loadCSV("data/forecast_risk_next_month.csv");
    createTable("forecastTable", forecast);

  } catch (error) {
    console.error(error);
    alert("Dashboard gagal dimuat. Pastikan semua file CSV sudah berada di folder website/data.");
  }
}

initDashboard();
