from sklearn.neural_network import MLPClassifier
import python_speech_features as sf
import pyaudio
import wave
import numpy

CHANNEL = 1
FORMAT = pyaudio.paInt16
RATE = 11025
CHUNK = 1024
voice_data = []
voice_id = []
with open("sample_num.txt", 'r', encoding="UTF-8") as f:
    sample_num = f.read()

for i in range(int(sample_num)):
    f = wave.open(f"beautify/{i}.wav", "rb")
    params = f.getparams()
    n_channels, sample_width, frame_rate, n_frames = params[:4]
    str_data = f.readframes(n_frames)
    f.close()
    wave_data = numpy.frombuffer(str_data, dtype=numpy.short)
    mfcc = sf.mfcc(wave_data, frame_rate)
    voice_data.append(mfcc.ravel())
    voice_id.append(0)

for i in range(int(sample_num)):
    f = wave.open(f"no_make/{i}.wav", "rb")
    params = f.getparams()
    n_channels, sample_width, frame_rate, n_frames = params[:4]
    str_data = f.readframes(n_frames)
    f.close()
    wave_data = numpy.frombuffer(str_data, dtype=numpy.short)
    mfcc = sf.mfcc(wave_data, frame_rate)
    voice_data.append(mfcc.ravel())
    voice_id.append(1)

# for i in range(int(sample_num)):
#     f = wave.open(f"shoot/{sample_num}.wav", "rb")
#     params = f.getparams()
#     n_channels, sample_width, frame_rate, n_frames = params[:4]
#     str_data = f.readframes(n_frames)
#     f.close()
#     wave_data = numpy.fromstring(str_data, dtype=numpy.int8, sep=" ")
#     mfcc = sf.mfcc(wave_data, frame_rate)
#     voice_data.append(mfcc.ravel())
#     voice_id.append(2)
while True:
    cmd = input("欢迎您来测试我们组的语音识别模型\n您说一段话，我们将尝试识别您说的是“人像美化”还是“使用原图”\n"
                "输入c继续，输入exit退出\n")
    if cmd == 'c':
        clf = MLPClassifier(solver="lbfgs", hidden_layer_sizes=(10, 10))
        clf.fit(voice_data, voice_id)

        RECORD_SECONDS = 3
        p = pyaudio.PyAudio()
        stream = p.open(channels=CHANNEL, format=FORMAT, rate=RATE,
                        frames_per_buffer=CHUNK, input=True)
        cmd = input("输入r开始录音，录音最大时长为3秒\n")
        if cmd == 'r':
            print("录音中...")
            frames = []
            for i in range(int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
            print("录音结束，开始识别")
            stream.stop_stream()
            stream.close()
            p.terminate()

            wf = wave.open("test_voice.wav", "wb")
            wf.setnchannels(CHANNEL)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            f = wave.open("test_voice.wav", "rb")
            params = f.getparams()
            n_channels, sample_width, frame_rate, n_frames = params[:4]
            str_data = f.readframes(n_frames)
            f.close()
            wave_data = numpy.frombuffer(str_data, dtype=numpy.short)
            mfcc = sf.mfcc(wave_data, frame_rate)
            X1 = [mfcc.ravel()]
            preY1 = clf.predict(X1)
            if preY1 == 0:
                print("您说的是“人像美化”")
            elif preY1 == 1:
                print("您说的是“使用原图”")
    elif cmd == "exit":
        break
    else:
        print("输入错误")
    print()
