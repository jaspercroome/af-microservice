from quart import Quart, request, Response, stream_with_context
from quart_cors import cors
import os
import aiohttp
import aiofiles
import librosa
import numpy as np
import pandas as pd
import json
import io
import asyncio

app = Quart(__name__)
app = cors(app)

async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.read()

async def read_local_csv(file_path):
    async with aiofiles.open(file_path, mode='r') as file:
        return await file.read()

async def process_audio(y, sr, notes):
    duration = librosa.get_duration(y=y, sr=sr)
    y_harmonic, _ = librosa.effects.hpss(y)
    cqt_h = np.abs(librosa.cqt(y_harmonic, sr=sr, fmin=16.35, n_bins=108, bins_per_octave=12))
    time_int = duration / cqt_h.shape[1]
    
    for i, col in enumerate(cqt_h.T):
        c_df_h = pd.DataFrame(notes).assign(magnitude=col)
        c_df_h['note_time'] = i * time_int * 1000
        c_df_h_final = c_df_h[c_df_h['magnitude'] >= 0.01]
        
        if not c_df_h_final.empty:
            yield c_df_h_final.to_dict(orient='records')

@app.route('/', methods=['GET', 'POST'])
async def songdata():
    if not await request.is_json or 'songUrlId' not in (await request.json):
        return 'Invalid request. Send a POST request with {"songUrlId":"xxxxxx"} in the body for analysis', 400
    
    data = await request.json
    song_url_id = data["songUrlId"]
    notes_file_path = os.path.join(os.path.dirname(__file__), 'data', 'midinotes.csv')
    song_url = f"https://p.scdn.co/mp3-preview/{song_url_id}.mp3"

    # Read local CSV file and fetch song data concurrently
    notes_data, song_data = await asyncio.gather(
        read_local_csv(notes_file_path),
        fetch_url(aiohttp.ClientSession(), song_url)
    )

    # Parse notes data
    notes = pd.read_csv(io.StringIO(notes_data))

    # Load audio data
    y, sr = librosa.load(io.BytesIO(song_data))

    async def generate():
        for chunk in process_audio(y, sr, notes):
            yield json.dumps(chunk) + '\n'

    return Response(await stream_with_context(generate)(), content_type='application/json')

if __name__ == '__main__':
    app.run(debug=True)