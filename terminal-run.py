import os
import wave
import re
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
from vosk import Model, KaldiRecognizer
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
    pattern = r'\+([аеёиоуыэюяАЕЁИОУЫЭЮЯ])'
    replacement = r"\1́"
    return re.sub(pattern, replacement, text)

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

def transliterate_word(word):
    rules = {'е': "э", 'ё': "о", 'ю': "у", 'я': "а", 'и': "'и"}
    additional_rules = {'е': 'йэ', 'ё': 'йо', 'ю': 'йу', 'я': 'йа', 'и': 'и'}
    exceptions = {'й': 'й', 'ч': 'ч\'', 'щ': 'щ\''}
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

def main():
    print("Выберите микрофон:")
    mics = select_microphone()
    for i, mic in enumerate(mics):
        print(f"{i}: {mic}")
    mic_index = int(input("Введите номер микрофона: "))

    audio_processor = AudioProcessor()
    audio_processor.start_stream(mic_index)

    print("Запись началась. Нажмите Enter для завершения.")
    try:
        while True:
            text = audio_processor.record_audio()
            if text:
                print(f"Транскрипция: {text}")
    except KeyboardInterrupt:
        pass

    audio_processor.stop_stream()
    transcription = audio_processor.get_transcription()

    # Добавление акцентации
    accentizer = RUAccent()
    accentizer.load(omograph_model_size='tiny', use_dictionary=True)
    accented_text = accentizer.process_all(transcription)
    accented_text = replace_plus_with_stress(accented_text)

    # Фонетическая транскрипция
    phonetic_transcription = transliterate(accented_text)
    print(f"Транскрипция с ударениями: {accented_text}")
    print(f"Фонетическая транскрипция: {phonetic_transcription}")

    # Сохранение результатов
    generate_spectrogram(output_wav_path)
    generate_zcr_graph(output_wav_path)

    with open(transcription_file, 'w', encoding='utf-8') as f:
        f.write(f"Транскрипция: {accented_text}\n")
        f.write(f"Фонетическая транскрипция: {phonetic_transcription}\n")

    print("Результаты сохранены.")

if __name__ == "__main__":
    main()
