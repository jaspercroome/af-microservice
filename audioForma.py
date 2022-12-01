from flask import Flask, request
from flask_cors import CORS 

import os
from urllib.request import urlopen

from librosa import load, cqt, get_duration
from librosa.effects import hpss

import importlib.util
import sys
import audioread

from pandas import read_csv, DataFrame

from numpy import abs

app=Flask(__name__)
CORS(app)

@app.route('/', methods=['GET','POST'])
def songdata():
    if not request.json or not 'songUrlId' in request.json:
        return('Welcome to the AudioForma Microservice!\n Send a POST request with \n{\n\t\"songUrlID\":\"xxxxxx\"\n}\n in the body for analysis')
    data = request.json
    song_url_id = data["songUrlId"]
    filepath = "https://gist.githubusercontent.com/Jasparr77/f365c49929bc275f15c82684f85921ca/raw/12084bd77c4bb6bc46acf36e6acf830b48443138/midinotes.csv"
    
    song_url="https://p.scdn.co/mp3-preview/"+song_url_id+".mp3"

    notes = read_csv(urlopen(filepath))

    sample_30s = urlopen(song_url)

    mp3_filepath = "tmp/"+song_url_id+".mp3"
    
    output = open(f'{mp3_filepath}', 'wb')

    output.write(sample_30s.read())

    file_stats = os.stat(mp3_filepath)

    print(file_stats)
    # return(mp3_filepath)

    y, sr = load(mp3_filepath)
    duration = get_duration(y=y, sr=sr)
    # split out the harmonic and percussive audio
    y_harmonic = hpss(y)[0]
    # map out the values into an array
    cqt_h = abs(cqt(y, sr=sr,
                                fmin=16.35, n_bins=108, bins_per_octave=12))
    c_df_h = DataFrame(notes).join(DataFrame(cqt_h), lsuffix='n').melt(
        id_vars={'MIDI Note', 'Octave', 'Note','Viz_Angle', 'circle_fifths_X', 'circle_fifths_Y'}).rename(columns={'variable': 'note_time', 'Octave': 'octave', 'Note': 'note_name', 'value': 'magnitude'})
    # Time transformation
    time_int = duration / cqt_h.shape[1]
    c_df_h['note_time'] = c_df_h['note_time'] * time_int * 1000

    c_df_h_final = c_df_h[c_df_h['magnitude'].astype(float) >= .01]

    song_data = c_df_h_final.to_csv(index=False)

    return(song_data, 200)
    
if __name__ == "__main__":
    app.run(debug=True)
