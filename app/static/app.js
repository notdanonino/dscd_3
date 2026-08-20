const form = document.getElementById('form');
const resultDiv = document.getElementById('result');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const payload = {
    Car_Name: document.getElementById('Car_Name').value.trim(),
    Year: parseInt(document.getElementById('Year').value, 10),
    Kms_Driven: parseInt(document.getElementById('Kms_Driven').value, 10),
    Fuel_Type: document.getElementById('Fuel_Type').value,
    Seller_Type: document.getElementById('Seller_Type').value,
    Transmission: document.getElementById('Transmission').value,
    Owner: parseInt(document.getElementById('Owner').value, 10),
  };

  try {
    const res = await fetch('/predict-price', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Error desconocido');
    resultDiv.innerHTML = `
      <strong>Precio estimado:</strong> ${data.estimated_price} ${data.currency}
      <br/>
      <small>Modelo: ${data.model_version}</small>
    `;
  } catch (err) {
    resultDiv.innerHTML = `<strong>Error:</strong> ${err.message}`;
  }
});
