import os
import pickle
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from music21 import environment, converter, note, chord, stream, tempo, meter, instrument

# Initialize Flask app
app = Flask(__name__)

# Setup music21 environment
env = environment.Environment()
env['musicxmlPath'] = '/Applications/MuseScore 4.app/'

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
    return send_from_directory(MIDI_DIR, filename)

@app.route('/musicxml/<path:filename>')
def musicxml_file(filename):
    return send_from_directory(XML_DIR, filename)



class ChordGenerator:
    def __init__(self, unique_chords):
        self.unique_chords = unique_chords
        self.chord_indices = {chord: i for i, chord in enumerate(unique_chords)}
        self.start_probabilities = np.zeros(len(unique_chords))
        self.transition_probabilities = np.zeros((len(unique_chords), len(unique_chords)))

    @staticmethod
    def load_midi(midi_folder):
        chords = []
        for file in os.listdir(midi_folder):
            if file.endswith(".mid"):
                midi_path = os.path.join(midi_folder, file)
                midi = converter.parse(midi_path)
                for element in midi.flat.getElementsByClass('Chord'):
                    chord_name = '-'.join(str(n.midi) for n in element.pitches)
                    chords.append(chord_name)
        return chords

    def train_model(self, chords, order_arg=1, smoothing=1):
        self._calculate_start_probabilities(chords, order_arg, smoothing)
        self._calculate_transition_probabilities(chords, order_arg, smoothing)

    def _calculate_start_probabilities(self, chords, order_arg=1, smoothing=1):
        count = np.zeros(len(self.unique_chords))
        for chord in chords[:len(chords) - order_arg]:
            index = self.chord_indices[chord]
            count[index] += 1
        total = count.sum() + smoothing * len(count)
        self.start_probabilities = (count + smoothing) / total

    def _calculate_transition_probabilities(self, chords, order_arg=1, smoothing=1):
        count_matrix = np.zeros((len(self.unique_chords), len(self.unique_chords)))
        for i in range(len(chords) - order_arg):
            start_index = self.chord_indices[chords[i]]
            end_index = self.chord_indices[chords[i + order_arg]]
            count_matrix[start_index][end_index] += 1
        row_sums = count_matrix.sum(axis=1) + smoothing * len(self.unique_chords)
        self.transition_probabilities = (count_matrix + smoothing) / row_sums[:, None]

    def roulette_wheel_selection(self, probabilities):
        """Perform Roulette Wheel Selection to choose an index based on given probabilities."""
        cumulative_probabilities = np.cumsum(probabilities)
        random_number = np.random.rand()
        chosen_index = np.where(cumulative_probabilities >= random_number)[0][0]
        return chosen_index

    def generate_chord_sequence(self, length=100):
        sequence = [np.random.choice(self.unique_chords, p=self.start_probabilities)]
        for _ in range(1, length):
            current_chord_index = self.chord_indices[sequence[-1]]
            probabilities = self.transition_probabilities[current_chord_index]
            next_chord_index = self.roulette_wheel_selection(probabilities)
            sequence.append(self.unique_chords[next_chord_index])
        return sequence

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')


def convert_chords_to_melody(chord_sequence, user_tempo):
    score = stream.Score()
    melody_part = stream.Part()
    melody_part.insert(0, tempo.MetronomeMark(number=user_tempo))
    melody_part.insert(0, meter.TimeSignature('4/4'))
    melody_part.insert(0, instrument.Piano())
    rhythmic_patterns = [1, 0.5, 1, 0.5, 2, 1, 1.5, 0.5]  # More varied patterns can be added

    for i, chord_str in enumerate(chord_sequence):
        pitches = [note.Note(midi=int(p)).pitch for p in chord_str.split('-')]
        current_chord = chord.Chord(pitches)

        if i % 4 == 0:  # Start of a new phrase
            melody_note = current_chord.root()  # Anchoring the phrase start to the root of the chord
        else:
            melody_note = np.random.choice(current_chord.pitches)

        pattern_index = i % len(rhythmic_patterns)
        mel_note = note.Note(melody_note,
                             quarterLength=rhythmic_patterns[pattern_index])
        melody_part.append(mel_note)

    score.append(melody_part)
    return score


def create_baseline_melody(chord_sequence, melody_duration, rhythm_pattern=[1, 2, 1.5, 0.5], walking=False):
    baseline = stream.Part()
    rhythm_index = 0
    total_duration = 0
    baseline.insert(0, instrument.Piano())

    for i, chord_str in enumerate(chord_sequence):
        if total_duration >= melody_duration:
            break  # Stop adding notes if the baseline reaches the melody duration

        pitches = [note.Note(midi=int(p)).pitch for p in chord_str.split('-')]
        current_chord = chord.Chord(pitches)
        root_note = current_chord.root()

        current_duration = rhythm_pattern[rhythm_index % len(rhythm_pattern)]
        if total_duration + current_duration > melody_duration:
            current_duration = melody_duration - total_duration  # Adjust the last note's duration

        base_note = note.Note(root_note, quarterLength=current_duration)

        if walking and i < len(chord_sequence) - 1:
            next_chord_str = chord_sequence[i + 1]
            next_root_pitch = chord.Chord([note.Note(midi=int(p)) for p in next_chord_str.split('-')]).root()
            while base_note.pitch.midi < next_root_pitch.midi - 12:
                base_note.transpose(12, inPlace=True)
            while base_note.pitch.midi > next_root_pitch.midi + 12:
                base_note.transpose(-12, inPlace=True)

        # Check if the next note is more than an octave away
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


def generate_combined_sequence(genre_models, num_chords):
    """ Generate a blended chord sequence from multiple genre models """
    sequences = [model.generate_chord_sequence(num_chords // len(genre_models)) for model in genre_models]

    blended_sequence = []
    for chords in zip(*sequences):
        blended_sequence.extend(chords)

    return blended_sequence


@app.route('/generate', methods=['POST'])
def generate():
    global last_generated_file
    num_chords = int(request.form.get('num_notes', 100))
    tempo_input = int(request.form.get('tempo', 120))
    genres = request.form.getlist('genre')

    # Create a filename-friendly genre descriptor
    genre_descriptor = "_".join(sorted(genres))  # Sort genres to keep the name consistent regardless of order

    models = []
    for genre in genres:
        model_filename = os.path.join(BASE_DIR, 'pickle_files', f'{genre}_chord_model.pkl')  # Change this line
        if os.path.exists(model_filename):
            model = ChordGenerator.load_model(model_filename)
        else:
            midi_path = os.path.join(BASE_DIR, 'midi_files', genre)
            chords = ChordGenerator.load_midi(midi_path)
            unique_chords = list(set(chords))
            model = ChordGenerator(unique_chords)
            model.train_model(chords)
            model.save_model(model_filename)
        models.append(model)

    blended_sequence = generate_combined_sequence(models, num_chords)
    main_melody_score = convert_chords_to_melody(blended_sequence, tempo_input)

    melody_duration = sum(n.duration.quarterLength for n in main_melody_score.flat.notes)
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


def display_chords(chord_sequence, user_tempo, file_name):
    score = stream.Score()
    score.append(tempo.MetronomeMark(number=user_tempo))
    part = stream.Part()
    for chord_str in chord_sequence:
        pitches = [note.Note(midi=int(p)).pitch for p in chord_str.split('-')]
        ch = chord.Chord(pitches)
        part.append(ch)
    score.append(part)
    score.write('midi', fp=file_name)


def convert_midi_to_musicxml(midi_file_path, output_xml_path=None):
    if output_xml_path is None:
        xml_base_path = XML_DIR
        midi_file_name = os.path.basename(midi_file_path)
        xml_file_name = midi_file_name.replace('.mid', '.xml')
        output_xml_path = os.path.join(xml_base_path, xml_file_name)
    score = converter.parse(midi_file_path)
    score.write('musicxml', fp=output_xml_path)
    return output_xml_path


@app.route('/osmd_viewer')
def osmd_viewer():
    global last_generated_file
    if last_generated_file:
        midi_filename = os.path.basename(last_generated_file).replace(".xml", ".mid")
        musicxml_filename = os.path.basename(last_generated_file)
        midi_file = url_for('midi_file', filename=midi_filename)
        musicxml_file = url_for('musicxml_file', filename=musicxml_filename)
    else:
        midi_file = None
        musicxml_file = None
    return render_template('osmd_viewer.html', midi_file=midi_file, musicxml_file=musicxml_file)




if __name__ == "__main__":
    app.run(debug=True)
