from database_manager import DatabaseManager
import os

def populate_database(force_rebuild=True):
    db_manager = DatabaseManager()
    
    # Directory containing Taylor Swift song files
    songs_dir = "songs"
    
    # Ensure the songs directory exists
    if not os.path.exists(songs_dir):
        os.makedirs(songs_dir)
        print(f"Created {songs_dir} directory. Please add Taylor Swift song files there.")
        return
    
    # If force_rebuild, try to delete the collection first
    if force_rebuild:
        try:
            print("Rebuilding database with improved embeddings...")
            db_manager.client.delete_collection(name="taylor_swift_songs")
            db_manager.collection = db_manager.client.create_collection(name="taylor_swift_songs")
            print("Old collection deleted, creating new collection with improved embeddings")
        except Exception as e:
            print(f"Could not delete collection: {str(e)}")
            print("Creating or using existing collection")
            db_manager.collection = db_manager.client.get_or_create_collection(name="taylor_swift_songs")
    
    # Process each song file in the directory
    for filename in os.listdir(songs_dir):
        if filename.endswith((".mp3", ".wav")):
            song_path = os.path.join(songs_dir, filename)
            song_id = os.path.splitext(filename)[0].lower().replace(" ", "_")
            
            # Get song metadata from the filename
            # Assuming filename format: "song_name.mp3"
            song_name = os.path.splitext(filename)[0]
            
            metadata = {
                "title": song_name,
                "file_path": song_path
            }
            
            try:
                print(f"\nProcessing: {song_name}")
                db_manager.add_song(song_id, song_path, metadata)
                print(f"Added {song_name} to the database with improved embedding")
            except Exception as e:
                print(f"Error processing {song_name}: {str(e)}")
    
    # Print summary after completion
    try:
        count = db_manager.collection.count()
        print(f"\nDatabase populated with {count} songs using improved embeddings")
        print("You should now get better search results!")
    except Exception as e:
        print(f"Error getting collection count: {str(e)}")

if __name__ == "__main__":
    populate_database() 