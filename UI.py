import os
import wave
import re
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
from vosk import Model, KaldiRecognizer
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ruaccent import RUAccent



# Настройки
model_path = "vosk-model-v1"
output_folder = 'outputs'
output_wav_path = os.path.join(output_folder, 'output_audio.wav')
transcription_file = os.path.join(output_folder, 'transcription.txt')
spectrogram_file = os.path.join(output_folder, 'spectrogram.png')
zcr_file = os.path.join(output_folder, 'zcr_graph.png')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

model = Model(model_path)
rec = KaldiRecognizer(model, 16000)

def replace_plus_with_stress(text):
    # Регулярное выражение для поиска плюса перед гласной
    pattern = r'\+([аеёиоуыэюяАЕЁИОУЫЭЮЯ])'
    # Замена плюса на знак ударения
    replacement = r"\1́"
    return re.sub(pattern, replacement, text)
# Функция выбора микрофона
def select_microphone():
    audio = pyaudio.PyAudio()
    mic_names = []
    mic_count = audio.get_device_count()
    for i in range(mic_count):
        info = audio.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            mic_names.append(info['name'].encode('cp1251').decode('utf-8'))
    audio.terminate()
    return mic_names

# Класс для работы с аудио
class AudioProcessor:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.wav_file = None
        self.transcription = []

    def start_stream(self, mic_index):
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=16000,
                                  input=True, frames_per_buffer=8000,
                                  input_device_index=mic_index)
        self.wav_file = wave.open(output_wav_path, 'wb')
        self.wav_file.setnchannels(1)
        self.wav_file.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        self.wav_file.setframerate(16000)

    def stop_stream(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.wav_file.close()
            self.stream = None
            self.wav_file = None

    def record_audio(self):
        data = self.stream.read(4000)
        self.wav_file.writeframes(data)
        if rec.AcceptWaveform(data):
            result = rec.Result()
            text = re.search(r'"text" : "(.*)"', result).group(1)
            self.transcription.append(text)
            return text
        return None

    def get_transcription(self):
        return " ".join(self.transcription)

    def __del__(self):
        self.p.terminate()

# Функция генерации спектрограммы
def generate_spectrogram(audio_path):
    with wave.open(audio_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        frame_rate = wf.getframerate()
        n_frames = wf.getnframes()
        samples = wf.readframes(n_frames)

    samples = np.frombuffer(samples, dtype=np.int16)
    if n_channels > 1:
        samples = samples[::n_channels]

    plt.figure(figsize=(12, 8))
    plt.specgram(samples, NFFT=2048, Fs=frame_rate, noverlap=128, cmap='inferno')
    plt.title('Спектрограмма')
    plt.ylabel('Частота (Гц)')
    plt.xlabel('Время (с)')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(spectrogram_file)
    plt.close()

# Функция генерации графика ZCR
def generate_zcr_graph(audio_path):
    with wave.open(audio_path, 'rb') as wf:
        n_frames = wf.getnframes()
        samples = wf.readframes(n_frames)

    samples = np.frombuffer(samples, dtype=np.int16)
    zero_crossings = np.nonzero(np.diff(np.sign(samples)))[0]
    zcr = len(zero_crossings) / len(samples)

    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(len(samples)) / 16000, samples)
    plt.axhline(0, color='r', linestyle='--')
    plt.title("Скорость пересечения нуля")
    plt.xlabel("Время (с)")
    plt.ylabel("Амплитуда")
    plt.grid()
    plt.savefig(zcr_file)
    plt.close()

# Функция для фонетической транскрипции
def transliterate_word(word):
    rules = {
        'е': "э",
        'ё': "о",
        'ю': "у",
        'я': "а",
        'и': "'и"
    }
    additional_rules = {
        'е': 'йэ',
        'ё': 'йо',
        'ю': 'йу',
        'я': 'йа',
        'и': 'и'
    }
    exceptions = {
        'й': 'й',
        'ч': 'ч\'',
        'щ': 'щ\''
    }
    vowels = 'аоуэыи'
    soft_consonants = 'бвгдзклмнпрстфх'
    result = ''
    i = 0
    while i < len(word):
        char = word[i].lower()
        if char in exceptions:
            result += exceptions[char]
        elif char == 'и' and i > 0 and word[i - 1].lower() in 'шж':
            result += 'ы'
        elif char in 'ъь':
            result += '\''
            if i + 1 < len(word) and word[i + 1].lower() in 'еёюя':
                result += 'й' + rules[word[i + 1].lower()]
                i += 1
        elif char in rules:
            if i == 0 or (i > 0 and word[i - 1].lower() in vowels):
                result += additional_rules[char]
            else:
                if char in 'еёюя' and i > 0 and word[i - 1].lower() in soft_consonants:
                    result += '\'' + rules[char]
                else:
                    result += rules[char]
        else:
            result += char
        i += 1
    return result

def transliterate(russian_string):
    words = russian_string.split()
    transliterated_words = [transliterate_word(word) for word in words]
    formatted_output = ''
    for word in transliterated_words:
        if word.endswith(','):
            formatted_output += f"[{word[:-1]}], "
        elif word.endswith('!'):
            formatted_output += f"[{word[:-1]}]!"
        else:
            formatted_output += f"[{word}] "
    return formatted_output.strip()

# Класс интерфейса с использованием Tkinter
class RecorderApp:
    def __init__(self, root):
        self.audio_processor = AudioProcessor()
        self.root = root
        self.root.title("Транскриптизатор")
        self.root.geometry("600x700")
        self.root.configure(bg="#f2f2f2")

        # Заголовок
        self.title_label = ttk.Label(self.root, text="Транскриптизатор", font=("Helvetica", 16, "bold"), background="#f2f2f2")
        self.title_label.pack(pady=10)

        # Основной фрейм
        self.frame = ttk.Frame(self.root, padding="10", style="TFrame")
        self.frame.pack(expand=True, fill=tk.BOTH)

        # Кнопка начала записи
        self.start_button = ttk.Button(self.frame, text="Начать", command=self.start_recording, style="TButton")
        self.start_button.pack(pady=10)

        # Кнопка остановки записи
        self.stop_button = ttk.Button(self.frame, text="Закончить", command=self.stop_recording, style="TButton")
        self.stop_button.pack(pady=10)

        # Выпадающий список микрофонов
        self.mic_label = ttk.Label(self.frame, text="Выберите микрофон:", background="#f2f2f2")
        self.mic_label.pack(pady=5)

        self.mic_spinner = ttk.Combobox(self.frame, values=select_microphone())
        self.mic_spinner.pack(pady=5)

        # Поле вывода транскрипции
        self.transcription_label = ttk.Label(self.frame, text="Транскрипция:", background="#f2f2f2")
        self.transcription_label.pack(pady=10)

        self.transcription_text = tk.Text(self.frame, height=5, width=50, wrap=tk.WORD, bg="#e6e6e6", font=("Helvetica", 10))
        self.transcription_text.pack(pady=5)

        # Поле вывода фонетической транскрипции
        self.phonetic_label = ttk.Label(self.frame, text="Фонетическая транскрипция:", background="#f2f2f2")
        self.phonetic_label.pack(pady=10)

        self.phonetic_text = tk.Text(self.frame, height=5, width=50, wrap=tk.WORD, bg="#e6e6e6", font=("Helvetica", 10))
        self.phonetic_text.pack(pady=5)

        # Поле для отображения спектрограммы
        self.spectrogram_label = ttk.Label(self.frame, text="Спектрограмма:", background="#f2f2f2")
        self.spectrogram_label.pack(pady=10)

        self.spectrogram_canvas = tk.Label(self.frame, background="#e6e6e6")
        self.spectrogram_canvas.pack(pady=5)

        # Поле для отображения графика ZCR
        self.zcr_label = ttk.Label(self.frame, text="Скорость пересечения нуля:", background="#f2f2f2")
        self.zcr_label.pack(pady=10)

        self.zcr_canvas = tk.Label(self.frame, background="#e6e6e6")
        self.zcr_canvas.pack(pady=5)

        # Поток записи
        self.recording = False

    def start_recording(self):
        mic_index = self.mic_spinner.current()
        if mic_index == -1:
            tk.messagebox.showerror("Ошибка", "Выберите микрофон.")
            return

        self.audio_processor.start_stream(mic_index)
        self.recording = True
        self.record_audio()

    def record_audio(self):
        if self.recording:
            transcribed_text = self.audio_processor.record_audio()
            if transcribed_text:
                self.transcription_text.insert(tk.END, transcribed_text + "\n")
                phonetic_transcription = transliterate(transcribed_text)
                self.phonetic_text.delete(1.0, tk.END)
                self.phonetic_text.insert(tk.END, phonetic_transcription + "\n")

            self.root.after(100, self.record_audio)

    def stop_recording(self):
        self.audio_processor.stop_stream()
        self.recording = False
        transcribed_text = self.audio_processor.get_transcription()
############################################################################################################
        accentizer = RUAccent()
        accentizer.load(omograph_model_size='tiny', use_dictionary=True)
        text = transcribed_text
        transcribed_text = accentizer.process_all(text)

        input_text = transcribed_text
        output_text = replace_plus_with_stress(input_text)
        transcribed_text = output_text
############################################################################################################
        phonetic_transcription = transliterate(transcribed_text)
        self.phonetic_text.delete(1.0, tk.END)
        self.phonetic_text.insert(tk.END, phonetic_transcription)

        # Генерация спектрограммы и графика ZCR
        generate_spectrogram(output_wav_path)
        generate_zcr_graph(output_wav_path)
        self.show_spectrogram()
        self.show_zcr_graph()

        # Сохранение транскрипции в файл
        with open(transcription_file, 'w', encoding='utf-8') as f:
            f.write(f"Текст: {transcribed_text}\n")
            f.write(f"Фонетическая транскрипция: {phonetic_transcription}\n")
        messagebox.showinfo("Сохранение", "Транскрипция сохранена.")

    def show_spectrogram(self):
        img = Image.open(spectrogram_file)
        img = img.resize((400, 300), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        self.spectrogram_canvas.config(image=img_tk)
        self.spectrogram_canvas.image = img_tk

    def show_zcr_graph(self):
        img = Image.open(zcr_file)
        img = img.resize((400, 300), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        self.zcr_canvas.config(image=img_tk)
        self.zcr_canvas.image = img_tk

if __name__ == '__main__':
    root = tk.Tk()
    app = RecorderApp(root)
    root.mainloop()
