import sounddevice as sd

# Print all available audio devices
print(sd.query_devices())

fs = 44100  # Sample rate (Hz)
duration = 5.0  # Duration in seconds

print("Recording started...")
# Record audio into a NumPy array
my_recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until the recording is finished
print("Recording stopped.")

print("Playing audio...")
sd.play(my_recording, fs)
sd.wait()  # Wait until the audio finishes playing
print("Playback finished.")