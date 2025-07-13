import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import sympy as sp

# Configuración para gráficos en español
plt.rcParams['font.size'] = 12
plt.style.use('seaborn-v0_8')

class OptimizacionVentas:
    def __init__(self):
        """
        Inicializa el problema de optimización de ventas
        Función objetivo: V(x,y) = 60x + 30y - 2x² - y²
        Restricción: x + y = 30
        """
        self.presupuesto_total = 30
        
    def funcion_ventas(self, x, y):
        """
        Función de ventas a maximizar
        V(x,y) = 60x + 30y - 2x² - y²
        """
        return 60*x + 30*y - 2*x**2 - y**2
    
    def restriccion(self, x, y):
        """
        Restricción presupuestaria: x + y = 30
        """
        return x + y - self.presupuesto_total
    
    def resolver_lagrange_simbolico(self):
        """
        Resuelve el problema usando multiplicadores de Lagrange de forma simbólica
        """
        print("=== SOLUCIÓN CON MULTIPLICADORES DE LAGRANGE ===\n")
        
        # Definir variables simbólicas
        x, y, lam = sp.symbols('x y lambda')
        
        # Función objetivo
        V = 60*x + 30*y - 2*x**2 - y**2
        
        # Restricción
        g = x + y - 30
        
        # Función lagrangiana: L = V - λ*g
        L = V - lam * g
        
        print("Función de ventas: V(x,y) =", V)
        print("Restricción: g(x,y) =", g, "= 0")
        print("Lagrangiana: L(x,y,λ) =", L)
        print()
        
        # Calcular gradientes
        dL_dx = sp.diff(L, x)
        dL_dy = sp.diff(L, y)
        dL_dlam = sp.diff(L, lam)
        
        print("Condiciones de primer orden:")
        print("∂L/∂x =", dL_dx, "= 0")
        print("∂L/∂y =", dL_dy, "= 0")
        print("∂L/∂λ =", dL_dlam, "= 0")
        print()
        
        # Resolver sistema de ecuaciones
        solucion = sp.solve([dL_dx, dL_dy, dL_dlam], [x, y, lam])
        
        x_opt = float(solucion[x])
        y_opt = float(solucion[y])
        lambda_opt = float(solucion[lam])
        
        print("SOLUCIÓN ÓPTIMA:")
        print(f"x* = {x_opt} (miles de dólares en redes sociales)")
        print(f"y* = {y_opt} (miles de dólares en TV)")
        print(f"λ* = {lambda_opt} (multiplicador de Lagrange)")
        print()
        
        # Calcular ventas máximas
        ventas_max = self.funcion_ventas(x_opt, y_opt)
        print(f"Ventas máximas: V({x_opt}, {y_opt}) = {ventas_max}")
        print()
        
        # Interpretación del multiplicador
        print("INTERPRETACIÓN:")
        print(f"λ = {lambda_opt} indica que aumentar el presupuesto total en $1000")
        print("no mejora las ventas (derivada marginal = 0)")
        print()
        
        return x_opt, y_opt, lambda_opt, ventas_max
    
    def resolver_scipy(self):
        """
        Verificación usando scipy.optimize
        """
        print("=== VERIFICACIÓN CON SCIPY ===\n")
        
        # Función objetivo a minimizar (negativa para maximizar)
        def objetivo(vars):
            x, y = vars
            return -(60*x + 30*y - 2*x**2 - y**2)
        
        # Restricción de igualdad
        def restriccion_eq(vars):
            x, y = vars
            return x + y - 30
        
        # Restricciones
        constraints = {'type': 'eq', 'fun': restriccion_eq}
        
        # Punto inicial
        x0 = [15, 15]
        
        # Resolver
        resultado = minimize(objetivo, x0, method='SLSQP', constraints=constraints)
        
        if resultado.success:
            x_opt, y_opt = resultado.x
            ventas_max = -resultado.fun
            print(f"Solución verificada:")
            print(f"x* = {x_opt:.6f}")
            print(f"y* = {y_opt:.6f}")
            print(f"Ventas máximas = {ventas_max:.6f}")
        else:
            print("Error en la optimización")
        
        return resultado
    
    def graficar_solucion(self, x_opt, y_opt):
        """
        Crea visualizaciones del problema y su solución
        """
        # Crear figura con subplots
        fig = plt.figure(figsize=(16, 6))
        
        # Gráfico 3D
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Crear malla para la función
        x_range = np.linspace(0, 30, 50)
        y_range = np.linspace(0, 30, 50)
        X, Y = np.meshgrid(x_range, y_range)
        Z = self.funcion_ventas(X, Y)
        
        # Superficie de la función
        surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
        
        # Línea de restricción en 3D
        x_rest = np.linspace(0, 30, 100)
        y_rest = 30 - x_rest
        z_rest = self.funcion_ventas(x_rest, y_rest)
        ax1.plot(x_rest, y_rest, z_rest, 'r-', linewidth=3, label='Restricción: x + y = 30')
        
        # Punto óptimo
        z_opt = self.funcion_ventas(x_opt, y_opt)
        ax1.scatter([x_opt], [y_opt], [z_opt], color='red', s=100, label=f'Punto óptimo ({x_opt}, {y_opt})')
        
        ax1.set_xlabel('Redes sociales (x)')
        ax1.set_ylabel('TV (y)')
        ax1.set_zlabel('Ventas V(x,y)')
        ax1.set_title('Función de Ventas: V(x,y) = 60x + 30y - 2x² - y²')
        ax1.legend()
        
        # Gráfico de contorno
        ax2 = fig.add_subplot(122)
        
        # Crear malla más densa para contornos
        x_range_2d = np.linspace(0, 30, 100)
        y_range_2d = np.linspace(0, 30, 100)
        X2, Y2 = np.meshgrid(x_range_2d, y_range_2d)
        Z2 = self.funcion_ventas(X2, Y2)
        
        # Curvas de nivel
        contours = ax2.contour(X2, Y2, Z2, levels=20, colors='blue', alpha=0.6)
        ax2.clabel(contours, inline=True, fontsize=8)
        
        # Restricción
        x_rest_2d = np.linspace(0, 30, 100)
        y_rest_2d = 30 - x_rest_2d
        ax2.plot(x_rest_2d, y_rest_2d, 'r-', linewidth=3, label='Restricción: x + y = 30')
        
        # Punto óptimo
        ax2.scatter([x_opt], [y_opt], color='red', s=100, zorder=5, label=f'Punto óptimo ({x_opt}, {y_opt})')
        
        ax2.set_xlabel('Redes sociales (x)')
        ax2.set_ylabel('TV (y)')
        ax2.set_title('Curvas de Nivel y Restricción')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 30)
        ax2.set_ylim(0, 30)
        
        plt.tight_layout()
        plt.show()
    
    def analisis_sensibilidad(self, x_opt, y_opt):
        """
        Análisis de sensibilidad del presupuesto
        """
        print("=== ANÁLISIS DE SENSIBILIDAD ===\n")
        
        presupuestos = np.linspace(20, 40, 21)
        ventas_optimas = []
        
        for presupuesto in presupuestos:
            # Resolver para cada presupuesto
            x, y = sp.symbols('x y')
            V = 60*x + 30*y - 2*x**2 - y**2
            g = x + y - presupuesto
            
            # Usando simetría del problema, sabemos que x* = y* = presupuesto/2
            x_temp = presupuesto/2
            y_temp = presupuesto/2
            ventas_temp = self.funcion_ventas(x_temp, y_temp)
            ventas_optimas.append(ventas_temp)
        
        # Graficar
        plt.figure(figsize=(10, 6))
        plt.plot(presupuestos, ventas_optimas, 'b-', linewidth=2, marker='o')
        plt.axvline(x=30, color='r', linestyle='--', label='Presupuesto actual')
        plt.axhline(y=self.funcion_ventas(x_opt, y_opt), color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Presupuesto Total (miles de dólares)')
        plt.ylabel('Ventas Máximas')
        plt.title('Análisis de Sensibilidad: Ventas vs Presupuesto')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print(f"Para el presupuesto actual de $30,000:")
        print(f"- Ventas máximas: {self.funcion_ventas(x_opt, y_opt)}")
        print(f"- Distribución óptima: 50% redes sociales, 50% TV")

def main():
    """
    Función principal para ejecutar el análisis completo
    """
    print("OPTIMIZACIÓN DE VENTAS CON MULTIPLICADORES DE LAGRANGE")
    print("="*60)
    print()
    
    # Crear instancia del problema
    optimizer = OptimizacionVentas()
    
    # Resolver usando multiplicadores de Lagrange
    x_opt, y_opt, lambda_opt, ventas_max = optimizer.resolver_lagrange_simbolico()
    
    # Verificar con scipy
    optimizer.resolver_scipy()
    
    # Crear visualizaciones
    optimizer.graficar_solucion(x_opt, y_opt)
    
    # Análisis de sensibilidad
    optimizer.analisis_sensibilidad(x_opt, y_opt)
    
    print("\n" + "="*60)
    print("RESUMEN EJECUTIVO:")
    print("="*60)
    print(f"• Inversión óptima en redes sociales: ${x_opt*1000:,.0f}")
    print(f"• Inversión óptima en TV: ${y_opt*1000:,.0f}")
    print(f"• Ventas máximas esperadas: {ventas_max}")
    print(f"• Estrategia: Distribuir el presupuesto equitativamente")
    print("="*60)

if __name__ == "__main__":
    main()