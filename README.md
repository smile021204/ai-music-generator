# Music Generator Application

## Overview

This Flask-based web application dynamically generates music based on selected genres, including classical composers like Bach and Beethoven, as well as modern K-pop. Utilizing a trained model for each genre, the application can create complex duets by layering two melodies from the same or different genres, according to user preferences.

## Features

- **Genre Selection:** Users can select one or multiple genres to influence the generated music.
- **Custom Note Length:** Specify the number of notes for the generated melodies.
- **Adjustable Tempo:** Users can set the tempo of the output music.
- **Dynamic Music Generation:** Utilises pre-trained models for each genre to generate music on the fly.
- **Music Display:** Generated music is displayed in a standard music notation format, thanks to integration with MuseScore.

## Prerequisites

- Python 3.6 or higher
- Flask
- music21
- NumPy
- Pickle

## Installation

1. **Clone the repository:**

   ```
   git clone https://gitlab.aber.ac.uk/mik53/ai-music-generator.git
   cd music-generator-application
   ```

2. **Install dependencies:**

   It's recommended to use a virtual environment:

   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Set up MuseScore for music21 (optional):**

   If you want to visualise the generated music scores, ensure MuseScore is installed and its path is correctly set in the script:

   ```python
   from music21 import environment
   env = environment.Environment()
   env['musicxmlPath'] = '/path/to/your/MuseScore4.exe'
   ```

## Running the Application

To run the application, execute:

```
flask run
```

Navigate to `http://127.0.0.1:5000/` in your web browser to access the music generator.

## Usage

1. Select the desired genre(s) from the provided options.
2. Specify the number of notes and tempo for your melody.
3. Click the "Generate" button to create the music.
4. The generated music will be displayed, which you can view and listen to through MuseScore or any compatible musicXML viewer.

## Contributing

Contributions to enhance the music generator or add new genres are welcome. Please fork the repository, make your changes, and submit a pull request.

## License
