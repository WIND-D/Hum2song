from flask import Flask, request, jsonify, render_template
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
from database_manager import DatabaseManager
import tempfile

app = Flask(__name__)
db_manager = DatabaseManager()

# Taylor Swift songs database (you'll need to add actual song files)
TAYLOR_SWIFT_SONGS = {
    "shake_it_off": {
        "title": "Shake It Off",
        "album": "1989",
        "year": 2014
    },
    "blank_space": {
        "title": "Blank Space",
        "album": "1989",
        "year": 2014
    },
    "love_story": {
        "title": "Love Story",
        "album": "Fearless",
        "year": 2008
    },
    "you_belong_with_me": {
        "title": "You Belong With Me",
        "album": "Fearless",
        "year": 2008
    },
    "bad_blood": {
        "title": "Bad Blood",
        "album": "1989",
        "year": 2014
    },
    "wildest_dreams": {
        "title": "Wildest Dreams",
        "album": "1989",
        "year": 2014
    },
    "style": {
        "title": "Style",
        "album": "1989",
        "year": 2014
    },
    "look_what_you_made_me_do": {
        "title": "Look What You Made Me Do",
        "album": "reputation",
        "year": 2017
    },
    "cardigan": {
        "title": "cardigan",
        "album": "folklore",
        "year": 2020
    },
    "willow": {
        "title": "willow",
        "album": "evermore",
        "year": 2020
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/record', methods=['POST'])
def record_audio():
    # Accept uploaded audio file from the frontend
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400
    audio_file = request.files['audio']

    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
        audio_file.save(temp_file.name)
        temp_path = temp_file.name

    # Convert webm to wav using ffmpeg (requires ffmpeg installed)
    wav_path = temp_path + '.wav'
    os.system(f'ffmpeg -y -i "{temp_path}" -ar 16000 -ac 1 "{wav_path}"')
    os.unlink(temp_path)

    # Generate embedding for the recorded audio
    query_embedding = db_manager.get_audio_embedding(wav_path)
    
    # Print the embedding information
    print("\n=== Recording Embedding Information ===")
    print(f"Embedding shape: {query_embedding.shape}")
    print(f"Embedding type: {type(query_embedding)}")
    print(f"First few values: {query_embedding[:5]}")
    print("=====================================\n")

    # Search for similar songs
    results = db_manager.search_similar_songs(query_embedding)
    
    # Clean up temporary wav file
    os.unlink(wav_path)

    # Process results
    matches = []
    for result in results:
        song_id = result['song_id']
        if song_id in TAYLOR_SWIFT_SONGS:
            song_info = TAYLOR_SWIFT_SONGS[song_id]
            
            # Get both distance and similarity from the result
            distance = result['distance']
            similarity = result['similarity']
            score_percentage = round(similarity * 100, 2)
            
            # Determine match quality based on the similarity score
            if score_percentage >= 70:
                match_quality = "Excellent match"
            elif score_percentage >= 50:
                match_quality = "Good match"
            elif score_percentage >= 30:
                match_quality = "Moderate match"
            else:
                match_quality = "Weak match"
                
            print(f"\nMatch with {song_info['title']}:")
            print(f"  - Score: {score_percentage}% ({match_quality})")
            print(f"  - Original distance: {distance:.4f}")
            print(f"  - Album: {song_info['album']} ({song_info['year']})")
            
            matches.append({
                'title': song_info['title'],
                'album': song_info['album'],
                'year': song_info['year'],
                'score': score_percentage,
                'distance': distance,
                'quality': match_quality
            })

    return jsonify({
        'matches': matches,
        'message': 'Recording processed successfully'
    })

if __name__ == '__main__':
    app.run(debug=True) 