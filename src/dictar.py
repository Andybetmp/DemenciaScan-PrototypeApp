import os
import tempfile
import wave
import pyaudio
import keyboard
import pyautogui
from groq import Groq
import pyperclip

client = Groq(api_key="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")

def grabar_audio(frecuencia_muestreo=16000, canales=1, fragmento=1024):
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=canales,
        rate=frecuencia_muestreo,
        input=True,
        frames_per_buffer=fragmento,)
    print("Presiona o manten presionado el boton INS para comenzar a grabar")
    frames = []
    keyboard.wait("insert")
    print("Grabando... (suelta INS para deterner)")
    while keyboard.is_pressed("insert"):
        data = stream.read(fragmento)
        frames.append(data)
    print("Garbación finalizada")
    stream.stop_stream()
    stream.close()
    p.terminate()
    return frames, frecuencia_muestreo
def guardar_audio(frames, frecuencia_muestreo):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_temp:
        wf = wave.open(audio_temp.name, mode="wb")
        wf.setnchannels(1)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wf.setframerate(frecuencia_muestreo)
        wf.writeframes(b"".join(frames))
        wf.close()
        return audio_temp.name

def transcribir_audio(ruta_archivo_audio):
    try:
        with open(ruta_archivo_audio, "rb") as archivo:
            transcripcion = client.audio.transcriptions.create(
                file=(os.path.basename(ruta_archivo_audio), archivo.read()),
                model="whisper-large-v3",
                prompt= "Transcribe el audio de una conversación en español con alta precisión, prestando especial atención a capturar pausas, repeticiones, autocorrecciones, errores de pronunciación y cualquier anomalía en el habla que pueda ser relevante para un análisis clínico de posibles signos de demencia o deterioro cognitivo. Marca claramente las pausas, incluyendo incluso las pausas muy breves o mínimas (por ejemplo, desde 0.2 segundos de silencio o menos), usando símbolos como '...' para pausas prolongadas y '[pausa breve]' para pausas cortas. También indica repeticiones (por ejemplo, palabras repetidas) y reformulaciones. El objetivo es obtener una transcripción detallada y fiel que permita un análisis lingüístico y discursivo posterior.",
                response_format="text",
                language="es"
            )
        return transcripcion
    except Exception as e:
        print(f"Ocurrio un error: {str(e)}")
        return None
    
def copiar_transcripcion_al_portapapeles(texto):
    pyperclip.copy(texto)
    pyautogui.hotkey("ctrl", "v")

def main():
    while True:
        frames, frecuencia_mestreo = grabar_audio()
        archivo_audio_temp = guardar_audio(frames, frecuencia_mestreo)
        print("Transcribiendo audio...")
        transcripcion = transcribir_audio(archivo_audio_temp)
        if transcripcion:
            print("\nTranscripción:")
            print("Copiando transcripcion al portapapeles. . . ")
            copiar_transcripcion_al_portapapeles(transcripcion)
            print("transcripcion copiada al portapapeles y pegado en la aplicacion")
        else:
            print("transcripcion fallida")

        os.unlink(archivo_audio_temp)
        print("\nListo para la proxima grabacion. Presiona INS para comenzar")


if __name__ == "__main__":
    main()