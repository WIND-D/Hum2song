Refer to Notion page for detailed Questions logged : https://www.notion.so/trychroma/Build-with-Chroma-1d558a6d8191802f83f4e7bc43c81255?pvs=4
# Taylor Swift Hum Challenge

A fun web application that lets users hum Taylor Swift songs and get matched with the closest song in the database, along with a similarity score.

## Features

- Record humming through your browser
- Match your humming with Taylor Swift songs
- Get similarity scores for matches
- Modern, responsive UI
- Real-time audio processing

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Web browser with microphone access
- Taylor Swift song files (MP3 or WAV format)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd hum2song
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create a `songs` directory and add your Taylor Swift song files:
```bash
mkdir songs
# Add your .mp3 or .wav files to the songs directory
```

5. Populate the database with songs:
```bash
python populate_database.py
```

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Click the "Start Recording" button and hum your favorite Taylor Swift song
4. Wait for the results to appear showing your matches and scores

## How It Works

1. The application uses Wav2Vec2 for audio embedding generation
2. ChromaDB is used for vector similarity search
3. The web interface captures audio through the browser
4. The backend processes the audio and compares it with the song database
5. Results are returned with similarity scores

## Notes

- Make sure your microphone is properly configured
- The recording duration is set to 10 seconds
- Song files should be in MP3 or WAV format
- The application works best with clear humming

## License

This project is licensed under the MIT License - see the LICENSE file for details. 