import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. PARAMETRIZACIÓN DEL MUESTREO TEMPORAL
# =============================================================================
Fs = 1000          # Frecuencia de muestreo (1000 muestras por segundo)
T = 1 / Fs         # Periodo de muestreo
L = 1000           # Longitud de la señal (1 segundo de duración total)
t = np.linspace(0, (L - 1) * T, L)  # Vector de tiempo continuo discreto

# =============================================================================
# 2. GENERACIÓN DE SEÑALES ELEMENTALES
# =============================================================================
# Señal 1: Onda Senoidal Pura (Frecuencia fundamental de 10 Hz)
f0 = 10
senal_senoidal = np.sin(2 * np.pi * f0 * t)

# Señal 2: Pulso Rectangular (Activo de 0.4s a 0.6s)
senal_rectangular = np.where((t >= 0.4) & (t <= 0.6), 1.0, 0.0)

# Señal 3: Función Escalón Unitario (Activo a partir de t = 0.3s)
senal_escalon = np.where(t >= 0.3, 1.0, 0.0)

# Estructura de datos para automatizar el procesamiento de gráficos
diccionario_senales = {
    "Onda Senoidal (10 Hz)": senal_senoidal,
    "Pulso Rectangular": senal_rectangular,
    "Escalón Unitario": senal_escalon
}

# =============================================================================
# 3. CÁLCULO DE LA FFT Y GENERACIÓN DE GRÁFICOS
# =============================================================================
# Vector de frecuencias para el eje horizontal
frecuencias = np.fft.fftfreq(L, T)
frecuencias_positivas = frecuencias[:L // 2]  # Lado positivo del espectro

for nombre, signal in diccionario_senales.items():
    # Cálculo de la Transformada de Fourier
    X_f = np.fft.fft(signal)
    
    # Magnitud normalizada (solo frecuencias positivas)
    magnitud = np.abs(X_f) / L
    magnitud_positiva = magnitud[:L // 2]
    
    # Fase en radianes (solo frecuencias positivas)
    fase = np.angle(X_f)
    fase_positiva = fase[:L // 2]
    
    # Despliegue de ventanas gráficas individuales
    fig, axs = plt.subplots(3, 1, figsize=(8, 9))
    fig.suptitle(f'Análisis Espectral: {nombre}', fontsize=12, fontweight='bold')
    
    # Gráfico del Dominio del Tiempo
    axs[0].plot(t, signal, color='tab:blue', linewidth=2)
    axs[0].set_title('Dominio del Tiempo')
    axs[0].set_xlabel('Tiempo (s)')
    axs[0].set_ylabel('Amplitud')
    axs[0].grid(True)
    
    # Gráfico de la Magnitud en Frecuencia
    axs[1].stem(frecuencias_positivas, magnitud_positiva, linefmt='tab:red', markerfmt='ro')
    axs[1].set_title('Espectro de Frecuencia (Magnitud)')
    axs[1].set_xlabel('Frecuencia (Hz)')
    axs[1].set_ylabel('|X(f)|')
    axs[1].set_xlim(0, 50)  # Acercamiento a bajas frecuencias para legibilidad
    axs[1].grid(True)
    
    # Gráfico de la Fase en Frecuencia
    axs[2].plot(frecuencias_positivas, fase_positiva, color='tab:green')
    axs[2].set_title('Espectro de Frecuencia (Fase)')
    axs[2].set_xlabel('Frecuencia (Hz)')
    axs[2].set_ylabel('Fase (rad)')
    axs[2].set_xlim(0, 50)
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# 4. COMPROBACIÓN NUMÉRICA DE PROPIEDADES DE LA TRANSFORMADA
# =============================================================================
print("\n=== COMPROBACIÓN EXPERIMENTAL DE PROPIEDADES ===")

# --- PROPIEDAD A: LINEALIDAD ---
# Escalares de combinación
a, b = 2.5, -1.8  

# Lado Izquierdo: FFT de la combinación lineal en el tiempo
FFT_combinacion_tiempo = np.fft.fft(a * senal_senoidal + b * senal_rectangular)

# Lado Derecho: Combinación lineal de las FFT independientes
FFT_combinacion_espectros = a * np.fft.fft(senal_senoidal) + b * np.fft.fft(senal_rectangular)

# Validación matemática (con tolerancia a redondeos decimales)
linealidad_verificada = np.allclose(FFT_combinacion_tiempo, FFT_combinacion_espectros)
print(f"1. ¿Se cumple la Propiedad de Linealidad?: {linealidad_verificada}")


# --- PROPIEDAD B: DESPLAZAMIENTO TEMPORAL ---
# Desplazar el pulso rectangular 0.15 segundos a la derecha
t0 = 0.15 
muestras_desplazamiento = int(t0 * Fs)
print(f"2. Desplazando la señal temporal en {muestras_desplazamiento} muestras...")

# Generación del pulso desplazado mediante rotación de arreglo
senal_desplazada = np.roll(senal_rectangular, muestras_desplazamiento)

# Obtención de magnitudes espectrales
mag_original = np.abs(np.fft.fft(senal_rectangular))
mag_desplazada = np.abs(np.fft.fft(senal_desplazada))

# Validación: La magnitud debe permanecer invariable frente a traslaciones temporales
desplazamiento_verificado = np.allclose(mag_original, mag_desplazada)
print(f"   ¿La magnitud espectral se mantiene idéntica?: {desplazamiento_verificado}")
print("   Resultado: Verificado. El retraso solo afecta a la fase espectral de la señal.")
