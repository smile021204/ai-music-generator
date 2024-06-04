import os
import numpy as np
import pickle

from music21 import converter

"""This file contains the ChordGenerator class that can be used to train a chord model and generate chord 
sequences."""
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
                for element in midi.flatten().getElementsByClass('Chord'):
                    chord_name = '-'.join(str(n.midi) for n in element.pitches)
                    chords.append(chord_name)
        return chords

    def train_model(self, chords, order_arg=10, smoothing=3):
        self._calculate_start_probabilities(chords, order_arg, smoothing)
        self._calculate_transition_probabilities(chords, order_arg, smoothing)

    def _calculate_start_probabilities(self, chords, order_arg=10, smoothing=3):
        """Calculate the start probabilities of the chord model."""
        count = np.zeros(len(self.unique_chords))
        for chord in chords[:len(chords) - order_arg]:
            index = self.chord_indices[chord]
            count[index] += 1
        total = count.sum() + smoothing * len(count)
        self.start_probabilities = (count + smoothing) / total

    def _calculate_transition_probabilities(self, chords, order_arg=10, smoothing=3):
        """Calculate the transition probabilities of the chord model."""
        count_matrix = np.zeros((len(self.unique_chords), len(self.unique_chords)))
        for i in range(len(chords) - order_arg):
            start_index = self.chord_indices[chords[i]]
            end_index = self.chord_indices[chords[i + order_arg]]
            count_matrix[start_index][end_index] += 1
        row_sums = count_matrix.sum(axis=1) + smoothing * len(self.unique_chords)
        epsilon = np.finfo(float).eps
        self.transition_probabilities = (count_matrix + smoothing) / (row_sums[:, None] + epsilon)

    def roulette_wheel_selection(self, probabilities, chaotic_level):
        if chaotic_level == 1:
            chosen_index = np.argmax(probabilities)
        else:
            adjusted_probabilities = probabilities ** (10 - chaotic_level)
            adjusted_probabilities /= adjusted_probabilities.sum()
            cumulative_probabilities = np.cumsum(adjusted_probabilities)
            random_number = np.random.rand()
            chosen_index = np.where(cumulative_probabilities >= random_number)[0][0]
        return chosen_index

    def generate_chord_sequence(self, length=100, chaotic_level=10):
        sequence = [np.random.choice(self.unique_chords, p=self.start_probabilities)]
        for _ in range(1, length):
            current_chord_index = self.chord_indices[sequence[-1]]
            probabilities = self.transition_probabilities[current_chord_index]
            next_chord_index = self.roulette_wheel_selection(probabilities, chaotic_level)
            sequence.append(self.unique_chords[next_chord_index])
        return sequence

    def save_model(self, filename):
        """Save the chord model to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filename):
        """Load a chord model from a file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)