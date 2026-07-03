import numpy as np
import librosa
import soundfile as sf
from scipy.signal import fftconvolve
import pysofaconventions

def generate_8d_audio(input_file="song.mp3", sofa_file="sofa.sofa", output_file="8d_output.wav", rotation_time=8.0):
    print(f"Loading audio from {input_file}...")
    audio, sr = librosa.load(input_file, sr=44100, mono=True)
    
    print(f"Loading HRTF data from {sofa_file}...")
    sofa = pysofaconventions.SOFAFile(sofa_file, 'r')
    
    irs = sofa.getDataIR()
    positions = sofa.getVariableValue('SourcePosition')
    
    # ---------------------------------------------------------
    # NEW FIX: Convert SOFA spherical coordinates to 3D Cartesian (X, Y, Z)
    # This prevents the "side-to-side stopping" issue by creating a true 3D map.
    # ---------------------------------------------------------
    azimuths_rad = np.radians(positions[:, 0])
    elevations_rad = np.radians(positions[:, 1])
    
    # Map all points in the SOFA file to a 3D sphere
    sofa_x = np.cos(azimuths_rad) * np.cos(elevations_rad)
    sofa_y = np.sin(azimuths_rad) * np.cos(elevations_rad)
    sofa_z = np.sin(elevations_rad)

    # Audio processing parameters
    chunk_duration = 0.05  # 50ms chunks
    chunk_size = int(sr * chunk_duration)
    hop_size = chunk_size // 2  # 50% overlap for smooth crossfading
    
    ir_length = irs.shape[2]
    output_length = len(audio) + ir_length
    output_audio = np.zeros((2, output_length))
    
    window = np.hanning(chunk_size)
    
    print("Applying true continuous 3D circular panning...")
    
    for i in range(0, len(audio) - chunk_size, hop_size):
        chunk = audio[i:i + chunk_size] * window
        
        # Calculate the current angle of the sound source
        current_time = i / sr
        current_angle = (current_time / rotation_time * 360) % 360
        target_rad = np.radians(current_angle)
        
        # Calculate where the sound SHOULD be on our perfect 3D circle
        target_x = np.cos(target_rad)
        target_y = np.sin(target_rad)
        target_z = 0.0 # Stay on the horizontal plane
        
        # Find the single HRTF point in the SOFA file closest to our target in 3D space
        # This completely ignores coordinate system formatting quirks
        distances = (sofa_x - target_x)**2 + (sofa_y - target_y)**2 + (sofa_z - target_z)**2
        angle_idx = distances.argmin()
        
        current_ir = irs[angle_idx]
        
        # Convolve the audio chunk with the exact 3D spatial filter
        left_chunk = fftconvolve(chunk, current_ir[0])
        right_chunk = fftconvolve(chunk, current_ir[1])
        
        # Smoothly overlap and add the chunks
        output_audio[0, i:i + len(left_chunk)] += left_chunk
        output_audio[1, i:i + len(right_chunk)] += right_chunk

    # Normalize audio to prevent clipping distortion
    max_val = np.max(np.abs(output_audio))
    if max_val > 0:
        output_audio = output_audio / max_val * 0.9

    print(f"Saving continuous 8D audio to {output_file}...")
    sf.write(output_file, output_audio.T, sr)
    print("Done! The audio will now orbit perfectly without stopping.")

if __name__ == "__main__":
    generate_8d_audio(
        input_file="song.mp3", 
        sofa_file="sofa.sofa", 
        output_file="8d_song.wav",
        rotation_time=10.0 # Change this to make the rotation faster or slower
    )
