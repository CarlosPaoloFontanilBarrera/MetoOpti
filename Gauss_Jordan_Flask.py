from flask import Flask, render_template_string, request
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Función de Gauss-Jordan
def gj(m):
    n = len(m)
    for j in range(n):
        if m[j][j] == 0:
            try:
                swap = next(i for i in range(j + 1, n) if m[i][j] != 0)
                m[j], m[swap] = m[swap], m[j]
            except:
                raise ValueError("No solución única")
        m[j] = [x / m[j][j] for x in m[j]]
        for i in range(n):
            if i != j:
                m[i] = [a - m[i][j] * b for a, b in zip(m[i], m[j])]
    return [round(r[-1], 6) for r in m]

# HTML con visualización de gráfica
html = '''
<!doctype html>
<html>
<head>
  <title>Resolver Sistema Lineal</title>
  <style>
    body { font-family: Arial; padding: 20px; max-width: 700px; margin: auto; }
    input { margin-bottom: 10px; width: 100%; padding: 8px; }
    .warning { color: darkorange; }
    .error { color: red; }
  </style>
</head>
<body>
<h2>Resolver sistema de ecuaciones (Gauss-Jordan)</h2>
<form method="post">
  Número de incógnitas: <input type="number" name="n" value="{{ n or '' }}" required><br>
  {% if n %}
    {% for i in range(n) %}
      E{{ i+1 }} (coeficientes incluyendo el término independiente):<br>
      <input name="eq{{ i }}" required><br>
    {% endfor %}
    <label><input type="checkbox" name="restricciones" value="1" {{ 'checked' if restricciones }}> ¿Restringir valores?</label><br><br>
    {% if restricciones %}
      {% for i in range(n) %}
        Rango para x{{ i+1 }} (min,max):<br>
        <input name="rango{{ i }}" required><br>
      {% endfor %}
    {% endif %}
    <br>
    <input type="submit" value="Resolver">
  {% else %}
    <input type="submit" value="Continuar">
  {% endif %}
</form>

{% if solucion %}
  <h3>✅ Solución:</h3>
  <ul>
  {% for i in range(solucion|length) %}
    <li>x{{ i+1 }} = {{ solucion[i] }}</li>
  {% endfor %}
  </ul>
  {% if advertencias %}
    <h4 class="warning">⚠️ Advertencias:</h4>
    <ul>
    {% for msg in advertencias %}
      <li>{{ msg }}</li>
    {% endfor %}
    </ul>
  {% endif %}
{% elif error %}
  <p class="error">❌ Error: {{ error }}</p>
{% endif %}

{% if grafica_url %}
  <h3>📊 Gráfica de las ecuaciones:</h3>
  <img src="data:image/png;base64,{{ grafica_url }}" style="max-width:100%;">
{% endif %}

</body>
</html>
'''

@app.route("/", methods=["GET", "POST"])
def index():
    n = None
    solucion = []
    advertencias = []
    error = ""
    restricciones = False
    grafica_url = None
    m = []

    if request.method == "POST":
        try:
            if "n" not in request.form:
                raise ValueError("Debe especificar el número de incógnitas")

            n = int(request.form["n"])

            if all(f"eq{i}" in request.form for i in range(n)):
                m = []
                for i in range(n):
                    eq_str = request.form[f"eq{i}"].strip()
                    eq = list(map(float, eq_str.split()))
                    if len(eq) != n + 1:
                        raise ValueError(f"La ecuación {i+1} debe tener {n+1} números")
                    m.append(eq)

                restricciones = "restricciones" in request.form
                rangos = []

                if restricciones:
                    for i in range(n):
                        r_str = request.form[f"rango{i}"].strip()
                        r = tuple(map(float, r_str.split(",")))
                        if len(r) != 2:
                            raise ValueError(f"Rango inválido para x{i+1}")
                        rangos.append(r)
                else:
                    rangos = [None] * n

                solucion = gj([row[:] for row in m])  # Copia para preservar original

                if restricciones:
                    for i, (v, r) in enumerate(zip(solucion, rangos)):
                        if r and not (r[0] <= v <= r[1]):
                            advertencias.append(f"x{i+1} = {v} está fuera del rango [{r[0]}, {r[1]}]")

                # Generar gráfica solo si hay 2 incógnitas
                if n == 2:
                    fig, ax = plt.subplots()
                    x_vals = list(range(-10, 11))
                    for i in range(n):
                        a, b, c = m[i]
                        if b != 0:
                            y_vals = [(c - a * x) / b for x in x_vals]
                            ax.plot(x_vals, y_vals, label=f"Ecuación {i+1}")
                        else:
                            x = c / a
                            ax.axvline(x, label=f"Ecuación {i+1}")

                    ax.plot(solucion[0], solucion[1], 'ro', label='Solución')
                    ax.set_xlabel('x1')
                    ax.set_ylabel('x2')
                    ax.grid(True)
                    ax.legend()

                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    grafica_url = base64.b64encode(buf.read()).decode('utf-8')
                    plt.close()
        except Exception as e:
            error = str(e)

    return render_template_string(html, n=n, solucion=solucion,
                                  advertencias=advertencias,
                                  error=error, restricciones=restricciones,
                                  grafica_url=grafica_url)

if __name__ == "__main__":
    app.run(debug=True)
