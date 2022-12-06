import streamlit as st
import requests
import streamlit as st
import cohere
import numpy as np
from scipy.io.wavfile import write

from PIL import Image
image = Image.open('1_I8jjjv3K4gY1mwdA4SNkdA.png')



co = cohere.Client('2N3UcA7d1YNSBfns9i1F6hyoDJKMx3unPamYmDn0')
import numpy as np

st.write(st.config.get_option("server.enableCORS"))
st.sidebar.header("prompt Tune")

samplerate = 44100 #Frequecy in Hz

def get_wave(freq, duration=0.5):
    '''
    Function takes the "frequecy" and "time_duration" for a wave 
    as the input and returns a "numpy array" of values at all points 
    in time
    '''
    
    amplitude = 4096
    t = np.linspace(0, duration, int(samplerate * duration))
    wave = amplitude * np.sin(2 * np.pi * freq * t)
    
    return wave

# To get a 1 second long wave of frequency 440Hz
a_wave = get_wave(440, 1)


def get_piano_notes():
    '''
    Returns a dict object for all the piano 
    note's frequencies
    '''
    # White keys are in Uppercase and black keys (sharps) are in lowercase
    octave = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B'] 
    
    base_freq = 261.63 #Frequency of Note C4
    
    note_freqs = {octave[i]: base_freq * pow(2,(i/12)) for i in range(len(octave))}        
    note_freqs[''] = 0.0 # silent note
    
    return note_freqs
  
# To get the piano note's frequencies
note_freqs = get_piano_notes()

def get_song_data(music_notes):
    '''
    Function to concatenate all the waves (notes)
    '''
    note_freqs = get_piano_notes() # Function that we made earlier
    song = [get_wave(note_freqs[note]) for note in music_notes.split('-')]
    song = np.concatenate(song)
    return song



def cohere_generate(genre):
  response = co.generate(
    model='large',
    prompt= f"""  
 This program generates a new tune given music notes.

  Genre : Children Music  
  keys: E-D-C-D-E-E-E--D-D-D--E-G-G--E-D-C-D-E-E-E--D-D-E-D-C  
  --  
  Genre: Children Music  
  keys: C-C-G-G-A-A-G--F-F-E-E-D-D-C--G-G-F-F-E-E-D--G-G-F-F-E-E-D--C-C-G-G-A-A-G--F-F-E-E-D-D-C  
  --  
  Genre: Ambient Music  
  keys: E-E-F-G--G-F-E-D--C-C-D-E-E-D-C--E-E-F-G--G-F-E-D--C-C-D-E-D-C-C  
  --  
  Genre: Ambient Music   
  keys: C-D-E-E--D-C-D-E-C-g--C-D-E-E--D-C-D-E-C--C-g-C-E-G-G-G--G-A-G-F-E-D-C--G-G-G--g-g-g--G-G-G--g-g-g--G-F-E-D  
  --  
  Genre: Ambient Music  
  keys: C-C-C-D-E-E--E-D-C-D-E-C--E-E-F-G-G--F-E-F-G-E--A-A-A-G-E--D-E-D-C-a-g-g-g--C-C-C-D-E-E--E-D-C-D-E-C  
  --  
  Genre: electronic
  keys: c-d-c-B-B-c-B-g--B-c-d-E d-c--c-d-c-B-B-c-B-g--B-c-d-E-d-c--c-d-c-B-B-c-B-g--E-f-E-d-c-c-d--f-E-d-c-c-d--d-c-B-d-c--B-c-c-d-B-c--c-d-d-c-B-a-g-c-d-d
  --
  Genre:Pop
  keys:D-B-A-G-G-G-G-G--D-B-A-G-A-A-A-A--D-B-A-G-G-G-G-G--G-G-G-A-A-B-A-D-B-A-G-G--D-B-A-G-G-G-G-G--D-B-A-G-A-A-A-A--D-B-A-G-G-G-G-G--G-G-G-A-A-B-B-D--B-A-G-G--D-B-A-G-D-B-A-G
  --
  Genre:Pop
  keys:a-C-D-a-F-d-D-C-D--a-C-D-a-F-d-D-C-a--a-C-D-a-F-G-D-C-D--D-D-d-D-C-a-a--G-a-d-F-D-C-a--D-G-F-F-G-D-C-a-D-F-F-G-F-d-D
  --
  Genre:Pop
  keys:D-a-A-G-F-D-C-a-D-F-G-D-C-D-a-C-a-a--D-C-a-D-F-F-G-D-D-D-a-a-A-G-a-A-G--G-G-a-G-G-D-C-C-G-F-D-C-a-A-G--a-G
  --
  Genre:Rock
  keys:F-G-A-G-F-G-F-G-A-G-F-D--F-G-A-G-G-F-G-A-G-F--F-C-A-G-G-G-F-G-A-G-F--G-F-G-A-G-F--F-G-A-G-G-G-G-F-G-A-G-F
  --
  Genre:Rock
  keys:F-G-A-G-G-G-G-G-G-F-G-A-G-F--F-C-A-G-G-G-G-F-G-A-G-F--F-F-G-F-G-A-G-F--C- C-C-C-A-A-G-G-A--C-C-C-A-A-G-G-F-F-E-D
  --
  Genre:Jazz
  keys:F-C-a-a-A-A-g-g-A--F-G-G-A-A-A-A-G,-F-F-G-G-G--F-G-A-A-A-A-G-G--G-G-G-A-D--C-a-A-a-A-G-G--A-C-D-C-D-C-D-E
  --
  Genre:Jazz
  keys:C-F-E-D-E-C-D-C-F-F-F-G--A-A-F-E-D-E-D-C--A-A-F-E-D-E-D-C--A-C-D-C-D-C-D-E
  --
  Genre:Rock
  keys:E-B-c-B-A-g-f-f-f-E-E--E-E-E-B-c-B-A-g-g-f-E-f-f-E-E--E-E-c-f-g-E-c-E--E-c-E-E-E-E-c-E--E-E-E-c-E-E-E-c-E-c-E
  --
  Genre:Rock
  keys:E-E-E-E-E-c-E--E-E-E-E-f-f-E-c-B--E-E-E-E-E-E-E--E-c-E-E-c-E-E-E-B--E-E-E-c-E
  --
  Genre:hip hop
  keys:C-G-g-a-F-F--C-G-g-a-F-F--C-a-C-d-C--C-a-g-F--g-a-d-c--C-F-d-C--a-C-d-C--C-a-g-F--g-a-d-C--C-F-d-C
  -- 
  Genre: Pop
  keys:C-C-C-C-D-CA--G-A-a-A-G-F-FA-G-A--C-C-C-C-D-C-C-G--G-G-A-a-A-G-F-F--F-C-C-C-C-C-C-C-A-A
  --
  Genre: Electronic
  keys:G-G-G-G-A-a-A-G-F-F-G-A-G-A--F-F-C-C-C-C-C-C-C-A-A
  --
  Genre: {genre} 
  keys:""",
  max_tokens=100
  )


  new_song = response.generations[0].text
  if new_song.find("\n"):
    new_song = new_song.split("\n")[0]
  print("---------")
  final_song = new_song.strip("-")
  print("Song notes --> ")
  print(final_song)
  return final_song




def get_song_data(music_notes,samplerate):
    '''
    Function to concatenate all the waves (notes)
    '''
    note_freqs = get_piano_notes() # Function that we made earlier
    song = [get_wave(note_freqs[note]) for note in music_notes.split('-')]
    song = np.concatenate(song)
    
    write('new_song.wav', samplerate, song.astype(np.int16))
    return 'new_song.wav'


st.title("Prompt Tune to generate custom music notes and music using Cohere 🎹")
st.subheader("Welcome!😀")
st.text("Define your music mood or Genre for the model to generate a customized music for you 🎹")
st.info("Example : I want to hear a relaxing music")
query = st.text_input('Please Enter the type of tune you want to generate',value="")


with st.sidebar:
    st.header("Made by:")
    st.subheader("Ribence Kadel")
    st.subheader('Goutam Vignesh')
    st.subheader('Ajay Surya')
    st.subheader('Saira Ali')
    st.subheader('Aqsa Ashfaq ')

    
if st.button('Search'):
    with st.spinner('Generating Custom music based on the input '):
        
            
        generated_notes=cohere_generate(query)
        st.info("Music Notes 🎶")
        st.text_area(value=generated_notes,label="Generated Music Notes by the Model is ")
        data = get_song_data(generated_notes.strip(),samplerate)
        audio_file = open(data, 'rb')
        audio_bytes = audio_file.read()


        st.info("Generated Music")
        st.audio(audio_bytes, format='audio/ogg')
        
        st.image(image, caption='Piano Octave Layout')


        
    
