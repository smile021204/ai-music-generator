# AI Music Generator

This project is an AI Music Generator built with Python and Flask. It uses Markov Chains to generate a sequence of musical chords, which are then converted into a melody. The generated melody is saved as a MIDI file and can be viewed and played in the browser.

## Features

- Generate a sequence of musical chords using Markov Chains.
- Convert the chord sequence into a melody.
- Save the generated melody as a MIDI file.
- View and play the generated melody in the browser.

## Technologies Used

- Python
- Flask
- Music21
- NumPy
- Pickle

## Installation

1. Clone the repository from GitHub:

```bash
git clone https://github.com/smile021204/ai-music-generator.git
```

2. Navigate to the project directory:

```bash
cd ai-music-generator
```

3. Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Flask app:

```bash
python main.py
```

2. Open a web browser and navigate to `http://localhost:5000`.

3. Use the web interface to generate a melody.

## Project Structure

- `main.py`: The main script that runs the Flask app and contains the logic for generating melodies.
- `static/midi`: Directory where generated MIDI files are saved.
- `static/musicxml`: Directory where MusicXML files are saved for viewing in the browser.
- `templates`: Directory containing HTML templates for the web interface.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License.
