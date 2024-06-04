import os
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from music21 import converter, note, chord, stream, tempo, meter, instrument
from chord_generator import ChordGenerator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variable to track the last generated file
last_generated_file = None

# Define the base directory for file handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MIDI_DIR = os.path.join(BASE_DIR, 'static', 'midi')
XML_DIR = os.path.join(BASE_DIR, 'static', 'musicxml')

# Ensure directories exist
os.makedirs(MIDI_DIR, exist_ok=True)
os.makedirs(XML_DIR, exist_ok=True)


@app.route('/midi/<path:filename>')
def midi_file(filename):
    """Serve a MIDI file from the static directory."""
    return send_from_directory(MIDI_DIR, filename)


@app.route('/musicxml/<path:filename>')
def musicxml_file(filename):
    """Serve a MusicXML file from the static directory."""
    return send_from_directory(XML_DIR, filename)


@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')


@app.route('/index.html')
def index():
    """Render the home page."""
    return render_template('index.html')


@app.route('/styleSelection.html')
def style_selection():
    """Render the style selection page."""
    return render_template('styleSelection.html')


@app.route('/osmd_viewer')
def osmd_viewer():
    """Render the OSMD viewer page."""
    global last_generated_file
    if last_generated_file:
        midi_filename = os.path.basename(last_generated_file).replace(".xml", ".mid")
        musicxml_filename = os.path.basename(last_generated_file)
        midi_url = url_for('midi_file', filename=midi_filename)
        musicxml_url = url_for('musicxml_file', filename=musicxml_filename)
        download_midi_url = url_for('midi_file', filename=midi_filename, _external=True)
        download_xml_url = url_for('musicxml_file', filename=musicxml_filename, _external=True)
    else:
        midi_url, musicxml_url, download_midi_url, download_xml_url = None, None, None, None
    return render_template('osmd_viewer.html', midi_file=midi_url, musicxml_file=musicxml_url,
                           download_midi_url=download_midi_url, download_xml_url=download_xml_url)


def convert_chords_to_melody(chord_sequence, user_tempo):
    """Convert a chord sequence to a melody using the given tempo."""
    score = stream.Score()
    melody_part = stream.Part()
    melody_part.insert(0, tempo.MetronomeMark(number=user_tempo))
    melody_part.insert(0, meter.TimeSignature('4/4'))
    melody_part.insert(0, instrument.Piano())
    rhythmic_patterns = [1, 0.5, 1, 0.5, 2, 1, 1.5, 0.5]

    for i, chord_str in enumerate(chord_sequence):
        pitches = [note.Note(midi=int(p)).pitch for p in chord_str.split('-')]
        current_chord = chord.Chord(pitches)

        if i % 4 == 0:
            melody_note = current_chord.root()
        else:
            melody_note = np.random.choice(current_chord.pitches)

        pattern_index = i % len(rhythmic_patterns)
        mel_note = note.Note(melody_note,
                             quarterLength=rhythmic_patterns[pattern_index])
        melody_part.append(mel_note)

    score.append(melody_part)
    return score


def create_baseline_melody(chord_sequence, melody_duration, rhythm_pattern=[1, 2, 1.5, 0.5], walking=False):
    """Create a baseline melody based on the chord sequence."""
    baseline = stream.Part()
    rhythm_index = 0
    total_duration = 0
    baseline.insert(0, instrument.Piano())

    for i, chord_str in enumerate(chord_sequence):
        if total_duration >= melody_duration:
            break

        pitches = [note.Note(midi=int(p)).pitch for p in chord_str.split('-')]
        current_chord = chord.Chord(pitches)
        root_note = current_chord.root()

        current_duration = rhythm_pattern[rhythm_index % len(rhythm_pattern)]
        if total_duration + current_duration > melody_duration:
            current_duration = melody_duration - total_duration

        base_note = note.Note(root_note, quarterLength=current_duration)

        if walking and i < len(chord_sequence) - 1:
            next_chord_str = chord_sequence[i + 1]
            next_root_pitch = chord.Chord([note.Note(midi=int(p)) for p in next_chord_str.split('-')]).root()
            while base_note.pitch.midi < next_root_pitch.midi - 12:
                base_note.transpose(12, inPlace=True)
            while base_note.pitch.midi > next_root_pitch.midi + 12:
                base_note.transpose(-12, inPlace=True)

        if i < len(chord_sequence) - 1:
            next_chord_str = chord_sequence[i + 1]
            next_root_pitch = chord.Chord([note.Note(midi=int(p)) for p in next_chord_str.split('-')]).root()
            while abs(base_note.pitch.midi - next_root_pitch.midi) > 12:
                if base_note.pitch.midi > next_root_pitch.midi:
                    base_note.transpose(-12, inPlace=True)
                else:
                    base_note.transpose(12, inPlace=True)

        baseline.append(base_note)
        total_duration += current_duration
        rhythm_index += 1

    return baseline


def generate_combined_sequence(genre_models, num_chords, chaotic_level):
    """Generate a combined chord sequence from multiple genre models."""
    sequences = [model.generate_chord_sequence(num_chords // len(genre_models), chaotic_level) for model in
                 genre_models]

    blended_sequence = []
    for chords in zip(*sequences):
        blended_sequence.extend(chords)

    return blended_sequence


@app.route('/generate', methods=['POST'])
def generate():
    """Generate a combined melody based on user input."""
    global last_generated_file
    num_chords = int(request.form.get('num_notes', 100))
    tempo_input = int(request.form.get('tempo', 120))
    genres = request.form.getlist('genre')
    chaotic_level = int(request.form.get('chaotic_level', 10))

    genre_descriptor = "_".join(sorted(genres))

    models = []
    for genre in genres:
        model_filename = os.path.join(BASE_DIR, 'static', f'{genre}_chord_model.pkl')
        logger.info(f"Checking for model file at {model_filename}")
        if os.path.exists(model_filename):
            logger.info(f"Model file found. Loading model from {model_filename}")
            model = ChordGenerator.load_model(model_filename)
        else:
            logger.info(f"Model file not found. Creating new model for genre: {genre}")
            midi_path = os.path.join(BASE_DIR, 'midi_files', genre)
            chords = ChordGenerator.load_midi(midi_path)
            unique_chords = list(set(chords))
            model = ChordGenerator(unique_chords)
            model.train_model(chords)
            model.save_model(model_filename)
            logger.info(f"Model saved at {model_filename}")
        models.append(model)

    blended_sequence = generate_combined_sequence(models, num_chords, chaotic_level)
    main_melody_score = convert_chords_to_melody(blended_sequence, tempo_input)
    melody_duration = sum(n.duration.quarterLength for n in main_melody_score.flatten().notes)
    baseline_melody = create_baseline_melody(blended_sequence, melody_duration, walking=True)

    combined_score = stream.Score()
    combined_score.append(main_melody_score.parts[0])
    combined_score.append(baseline_melody)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    midi_file_name = f'combined_melody_{genre_descriptor}_{timestamp}.mid'
    midi_file_path = os.path.join(MIDI_DIR, midi_file_name)
    combined_score.write('midi', fp=midi_file_path)

    xml_file_path = convert_midi_to_musicxml(midi_file_path)
    last_generated_file = xml_file_path

    return redirect(url_for('osmd_viewer'))


def convert_midi_to_musicxml(midi_file_path, output_xml_path=None):
    """Convert a MIDI file to MusicXML format."""
    if output_xml_path is None:
        xml_base_path = XML_DIR
        midi_file_name = os.path.basename(midi_file_path)
        xml_file_name = midi_file_name.replace('.mid', '.xml')
        output_xml_path = os.path.join(xml_base_path, xml_file_name)
    score = converter.parse(midi_file_path)
    score.write('musicxml', fp=output_xml_path)
    return output_xml_path


if __name__ == "__main__":
    app.run()
