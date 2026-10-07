const form = document.getElementById('credit-form');
const clearBtn = document.getElementById('clear-btn');
const riskValue = document.getElementById('risk-value');
const statusBadge = document.querySelector('.status-badge');
const gaugeBar = document.getElementById('gauge-bar');

const circumference = 2 * Math.PI * 88;

gaugeBar.style.strokeDasharray = String(circumference);

const setGauge = (probability, isHigh) => {
  const percentage = Math.max(0, Math.min(probability * 100, 100));
  const offset = circumference * (1 - percentage / 100);

  gaugeBar.style.strokeDashoffset = String(offset);
  gaugeBar.classList.toggle('gauge-high', isHigh);
  gaugeBar.classList.toggle('gauge-low', !isHigh);
};

const setStatus = (isHigh) => {
  statusBadge.classList.remove('status-low', 'status-high');
  statusBadge.classList.add(isHigh ? 'status-high' : 'status-low');
  statusBadge.innerHTML = '<span class="status-dot"></span>' + (isHigh ? 'HIGH RISK' : 'LOW RISK');
};

const resetResult = () => {
  riskValue.textContent = '0%';
  setGauge(0, false);
  setStatus(false);
};

const resetForm = () => {
  const fields = form.querySelectorAll('input, select');

  fields.forEach((field) => {
    if (field.type === 'hidden') {
      return;
    }

    if (field.tagName === 'SELECT') {
      field.selectedIndex = -1;
      field.value = '';
      return;
    }

    field.value = '';
  });

  resetResult();
};

const setResultState = (risk, probability) => {
  const isHigh = risk === 'High';
  const value = Number(probability) || 0;

  riskValue.textContent = `${(value * 100).toFixed(0)}%`;
  setGauge(value, isHigh);
  setStatus(isHigh);
};

clearBtn.addEventListener('click', () => {
  resetForm();
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  const formData = new FormData(form);

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    setResultState(data.risco, data.probabilidade);
  } catch (error) {
    riskValue.textContent = '--';
    setGauge(0, true);
    setStatus(true);
    console.error(error);
  }
});

resetResult();
