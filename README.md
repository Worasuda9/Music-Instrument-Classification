# 🎶 Musical Instrument Classification Using CNN and Mel-Spectrograms
This project performs classification of musical instruments using deep learning. Audio files are transformed into Mel-spectrograms and used to train a Convolutional Neural Network (CNN) model for instrument prediction.

📁 Dataset
The dataset consists of .wav and .mp3 audio files organized in directories by instrument name (e.g., Drum, Piano, Guitar, Violin).

🧰 Requirements

Libraries:

librosa

numpy

pandas

matplotlib

tensorflow

scikit-learn

psutil

⚠️ Important Reminder

Do NOT run Threading and ThreadPoolExecutor simultaneously.
- Running both in the same session can lead to inaccurate timing results due to shared resource contention.
- You should choose one method at a time to run.
- Always restart the runtime/session before switching between threading methods to ensure a clean and fair comparison.

📝 Recommended Usage Guidelines
To better understand and use the code:

1. Copy the code from .py file and look up at .ipynb file to see the content

2. Edit Paths Before Running
  
Make sure to update the following paths in the code before execution:
- 🔧 Dataset Path in the configuration/setup section
- 🎵 new_audio_path in the instrument prediction part (used for testing new audio)
- 📈 Waveform Path used for visualization or feature extraction

These edits are necessary for the code to run correctly in your environment.

🧪 Workflow Overview

1. Data Preprocessing
- Extract Mel-spectrograms from waveform using librosa.
- Pad or truncate to fixed dimensions (128 × 216).
- Flatten Mel features into a DataFrame for optional CSV export.

2. Multithreading for Efficient Feature Extraction
- Utilizes ThreadPoolExecutor and native Python threading for faster processing.
- Memory usage and time are logged.

🧵 Threading vs ThreadPoolExecutor

We compare the performance of:

- Native Python multithreading (threading.Thread)
- ThreadPoolExecutor from the concurrent.futures module

🔍 Goals:
- Measure execution time of audio feature extraction
- Track memory usage using psutil

This comparative study helps assess how parallelism strategies affect real-world audio preprocessing workloads in ML pipelines.

3. Data Preparation
- Feature arrays reshaped to fit CNN input: (samples, 128, 216, 1)
- Labels are one-hot encoded.

4. Model Architecture

A CNN model with the following structure:

Conv2D(32) → MaxPooling → Dropout  
Conv2D(64) → MaxPooling → Dropout  
Conv2D(128) → MaxPooling → Dropout  
Flatten → Dense(128) → Dropout → Output Layer (Softmax)

5. Training
- Epochs: 20

6. Evaluation
- Accuracy and loss metrics on validation set.
- Visualize training history if needed.

🔍 Prediction

To predict the instrument of a new audio file:
- predict_instrument("path/to/audio.wav", model, label_encoder)

📊 Example Visualizations

Mel-Spectrogram:
- librosa.display.specshow(mel, x_axis='time', y_axis='mel', sr=SAMPLE_RATE)

Waveform (optional):
- y, sr = librosa.load(path)
- plt.plot(y)
  
📄 Output Example

Predicted Instrument: Violin

💾 Saving Extracted Features (Optional)

Export features to CSV:
- mel_db_df.to_csv('mel_spectrograms_table.csv', index=False)
  
🔚 Conclusion

This project demonstrates how to build a CNN-based classifier using spectrograms extracted from audio. With multithreading and efficient preprocessing, even moderate datasets can be processed quickly and used for accurate instrument recognition.

