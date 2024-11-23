<h3>Speech Spectrograms using Short Time Fourier Transforms</h3>
<p>
This is a web application written in Go that make use of the html/template package to dynamically create the web page.
Start the web server at bin\speechspectrogram.exe and connect to it from your web browser at http://127.0.0.1:8080/speechspectrogram.
The program reads in audio WAV files and creates spectrograms from them.  Spectrograms are 3-dimensional plots of the time-frequency
content of the audio.  The spectral power at a particular frequency and time is shown as a grayscale color, with black having the 
greatest power and white having the least.  Short-time Fourier transforms (STFT) are used in 32 ms time intervals with 50% overlap
between FFTs.  The sections are multiplied with various window types to minimize spectral leakage due to the Gibbs phenomenon.
Time domain plots of the audio waveform can be displayed as well as the spectrogram.  The user can enter audio of not more than
two seconds by selecting the <i>New Speech</i> radio button, as well as some previously created WAV file.  Upon selecting the <i>Play Speech</i> radio button, the user will hear the contents of the audio WAV file.  This program requires the <b>fmedia</b> program
to be in the PATH environmental variable.
<p>
The FFT size is 256.  The sampling rate is 8,000 Hz which produces a Nyquist critical frequency of 4,000 Hz.
The checkbox <i>Words Only</i> can be checked to eliminate the spectral noise for the intervals when no
audio is present.  This occurs at the beginning and end of the audio and in-between words.  The <i>Word Window</i>
select dropdown determines the sample window to separate the beginning and ending of the words in the audio.
This is used to eliminate the noise that is present in the spectrogram.  The window types used to reduce
the spectral leakage from the high sidelobes of the rectangular window are Hamming, Welch, Hanning and 
The grayscale consists of five colors.  As stated above, black RGB(0,0,0), has the greatest power in the PSD bin.
PSD bins with power more than 10dB down from the maximum bin power are white RGB(255,255,255).
</p>

<h4>Time domain plot of <i>I read the book yesterday</i></h4>
![image](https://github.com/user-attachments/assets/faa6206a-333e-4527-87ef-1485e1dbc338)
<h4>Spectrogram of <i>I read the book yesterday</i>, Hamming window, noise spectrum</h4>
![image](https://github.com/user-attachments/assets/388e2273-e422-4636-8126-504f33450161)
<h4>Spectrogram of <i>I read the book yesterday</i>, Hamming window, 50 ms word window, no noise spectrum</h4>
![image](https://github.com/user-attachments/assets/6fd0d0d6-eaf7-4c28-8c3d-612e0379274f)
