import os
from scipy.io.wavfile import write
from IPython.display import display, Audio 

def delete(file):
    if os.path.isfile(file):
        os.remove(file)

def play_audio(X):
    file = 'temp.wav'
    try:
        write(file, 8000, X)
        display(Audio(file))
    except Exception as e:
        delete(file)
        raise RuntimeError('Unable to play audio') from e
    finally:
        delete(file)