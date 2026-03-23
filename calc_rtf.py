"""计算 Python 版 ViSQOL 的 RTF (Real-Time Factor)"""
import soundfile as sf
import time
from visqol.api import VisqolApi

td = '/Users/jiangshan/Desktop/test/0210/visqol/testdata/conformance_testdata_subset/'
speech_td = '/Users/jiangshan/Desktop/test/0210/visqol/testdata/clean_speech/'

# Audio mode tests
audio_tests = [
    ('strauss48_stereo.wav', 'strauss48_stereo_lp35.wav', 'strauss_lp35'),
    ('steely48_stereo.wav', 'steely48_stereo_lp7.wav', 'steely_lp7'),
    ('guitar48_stereo.wav', 'guitar48_stereo_64kbps_aac.wav', 'guitar_64aac'),
    ('glock48_stereo.wav', 'glock48_stereo_48kbps_aac.wav', 'glock_48aac'),
    ('contrabassoon48_stereo.wav', 'contrabassoon48_stereo_24kbps_aac.wav', 'contrabassoon_24aac'),
    ('castanets48_stereo.wav', 'castanets48_stereo.wav', 'castanets_identity'),
]

# Speech mode tests
speech_tests = [
    ('CA01_01.wav', 'transcoded_CA01_01.wav', 'speech_CA01'),
]

print("=" * 90)
print(f"{'Test':<30s} {'Duration':>8s} {'Process':>8s} {'RTF':>8s} {'Mode':<8s}")
print("=" * 90)

# Audio mode
api = VisqolApi()
api.create(mode='audio')

total_dur_audio = 0.0
total_time_audio = 0.0

for ref, deg, name in audio_tests:
    ref_path = td + ref
    deg_path = td + deg
    info = sf.info(ref_path)
    dur = info.duration

    t0 = time.time()
    result = api.measure(ref_path, deg_path)
    elapsed = time.time() - t0

    rtf = elapsed / dur
    total_dur_audio += dur
    total_time_audio += elapsed
    print(f"{name:<30s} {dur:>7.2f}s {elapsed:>7.2f}s {rtf:>7.3f}x  Audio")

# Speech mode
api2 = VisqolApi()
api2.create(mode='speech')

total_dur_speech = 0.0
total_time_speech = 0.0

for ref, deg, name in speech_tests:
    ref_path = speech_td + ref
    deg_path = speech_td + deg
    info = sf.info(ref_path)
    dur = info.duration

    t0 = time.time()
    result = api2.measure(ref_path, deg_path)
    elapsed = time.time() - t0

    rtf = elapsed / dur
    total_dur_speech += dur
    total_time_speech += elapsed
    print(f"{name:<30s} {dur:>7.2f}s {elapsed:>7.2f}s {rtf:>7.3f}x  Speech")

print("=" * 90)

avg_rtf_audio = total_time_audio / total_dur_audio
avg_rtf_speech = total_time_speech / total_dur_speech

print(f"\n=== Python 版 RTF 汇总 ===")
print(f"Audio  模式平均 RTF: {avg_rtf_audio:.3f}x  (总音频 {total_dur_audio:.1f}s, 总处理 {total_time_audio:.1f}s)")
print(f"Speech 模式平均 RTF: {avg_rtf_speech:.3f}x  (总音频 {total_dur_speech:.1f}s, 总处理 {total_time_speech:.1f}s)")

print(f"\n=== C++ 版 RTF 估算 (基于 C++ 通常 Audio ~1-2s, Speech ~0.1-0.3s) ===")
cpp_audio_time = 1.5  # typical per file
cpp_speech_time = 0.2
avg_dur_audio = total_dur_audio / len(audio_tests)
avg_dur_speech = total_dur_speech / len(speech_tests)
print(f"Audio  模式估算 RTF: ~{cpp_audio_time / avg_dur_audio:.3f}x")
print(f"Speech 模式估算 RTF: ~{cpp_speech_time / avg_dur_speech:.3f}x")
print(f"\n=== Python / C++ 速度比 ===")
print(f"Audio  模式: Python 约慢 {avg_rtf_audio / (cpp_audio_time / avg_dur_audio):.1f} 倍")
print(f"Speech 模式: Python 约慢 {avg_rtf_speech / (cpp_speech_time / avg_dur_speech):.1f} 倍")
