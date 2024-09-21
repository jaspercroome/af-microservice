from quart import Quart, request, jsonify
from quart_cors import cors
from flask_caching import Cache 
import io
import aiohttp
import librosa
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

app=Quart(__name__)
cors(app)

# Configure caching
app.config['CACHE_TYPE'] = 'SimpleCache'
app.config['CACHE_DEFAULT_TIMEOUT'] = 1 * 60 * 60  # Cache timeout of 1 hr
cache = Cache(app)


@app.route('/song_data', methods=['POST'])
async def song_data():

    data = await request.data
    if not data:
        return('no data in request, Send a POST request with \n{\n\t\"songUrlID\":\"xxxxxx\"\n}\n in the body for analysis',400)
    
    json_data = await request.get_json() 
    if not json_data: 
        return('no json in request, Send a POST request with \n{\n\t\"songUrlID\":\"xxxxxx\"\n}\n in the body for analysis',400)
    if 'songUrlId' not in json_data:
        return('no \'songUrlId\' in request, Send a POST request with \n{\n\t\"songUrlID\":\"xxxxxx\"\n}\n in the body for analysis',400)
    
    data = await request.get_json()
    song_url_id = data["songUrlId"]

    cached_data = cache.get(song_url_id)
    if cached_data:
        return cached_data, 200

    filepath = "data/midinotes.csv"    
    notes = pd.read_csv(filepath)

    song_url="https://p.scdn.co/mp3-preview/"+song_url_id+".mp3"
    # fetch the song_data
    async with aiohttp.ClientSession() as session:
        async with session.get(song_url) as resp:
            mp3_data = await resp.read()
    
    mp3_file = io.BytesIO(mp3_data)    

    y, sr = librosa.load(mp3_file,sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    # split out the harmonic and percussive audio
    y_harmonic = librosa.effects.hpss(y)[0]
    # map out the values into an array
    cqt_h = np.abs(librosa.cqt(y_harmonic, sr=sr,
                                fmin=16.35, n_bins=108, bins_per_octave=12))
    c_df_h = pd.DataFrame(notes).join(pd.DataFrame(cqt_h), lsuffix='n').melt(
        id_vars={'MIDI Note', 'Octave', 'Note','Viz_Angle', 'circle_fifths_X', 'circle_fifths_Y'}).rename(columns={'variable': 'note_time', 'Octave': 'octave', 'Note': 'note_name', 'value': 'magnitude'})
    # Time transformation
    time_int = duration / cqt_h.shape[1]
    c_df_h['note_time'] *= time_int * 1000

    c_df_h_final = c_df_h[c_df_h['magnitude'] >= 0.01]

    song_data = c_df_h_final.groupby('note_time').apply(lambda x: x.to_json(orient='records')).to_json(orient='records')
    
    cache.set(song_url_id, song_data, timeout=3600) 
    return(song_data, 200)

@app.route('/3d_data',methods=['POST'])
async def get_3d_data():
    data = await request.get_json()
    if not data or 'tracks' not in data:
        return 'Invalid request. Send a POST request with {"tracks": [...]} in the body for analysis', 400

    tracks_data = data['tracks']
    
    # Prepare features
    features = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
    
    df = pd.DataFrame(tracks_data)
    
    X = df[features]

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reduce dimensions using PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    # Reduce dimensions using t-SNE
    tsne = TSNE(n_components=3, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    # Merge results with track IDs
    merged_data = [
        {
            'id': track_id,
            'name': name,
            'pca': pca_coords.tolist(),
            'tsne': tsne_coords.tolist()
        }
        for track_id, name, pca_coords, tsne_coords in zip(df['id'], df['name'], X_pca, X_tsne)
    ]
    
    # Prepare response
    response = {
        'tracks': merged_data
    }

    return jsonify(response), 200

@app.route('/',methods=['GET'])
async def home():
    return "Hiya! check out /3d_data or /song_data  with a POST for actual endpoints",200

if __name__ == '__main__':
    app.run(debug=True)