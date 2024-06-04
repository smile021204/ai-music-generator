import os
import unittest

import numpy as np
from flask_testing import TestCase

from main import app, MIDI_DIR, XML_DIR
from chord_generator import ChordGenerator


class TestFlaskRoutes(TestCase):
    def create_app(self):
        """Create the Flask app for testing."""
        app.config['TESTING'] = True
        return app

    def setUp(self):
        """Create dummy MIDI and MusicXML files for testing."""
        self.dummy_midi_path = os.path.join(MIDI_DIR, 'test.mid')
        self.dummy_xml_path = os.path.join(XML_DIR, 'test.xml')
        with open(self.dummy_midi_path, 'w') as f:
            f.write("Dummy MIDI content")
        with open(self.dummy_xml_path, 'w') as f:
            f.write("Dummy XML content")

    def tearDown(self):
        """Remove the dummy MIDI and MusicXML files after testing."""
        os.remove(self.dummy_midi_path)
        os.remove(self.dummy_xml_path)
        super().tearDown()

    def test_midi_file_serving(self):
        """Test serving a MIDI file from the static directory."""
        response = self.client.get('/midi/test.mid')
        self.assertEqual(response.status_code, 200)

    def test_musicxml_file_serving(self):
        """Test serving a MusicXML file from the static directory."""
        response = self.client.get('/musicxml/test.xml')
        self.assertEqual(response.status_code, 200)


class TestChordGenerator(unittest.TestCase):
    def setUp(self):
        """Create a ChordGenerator instance for testing."""
        self.chords = ['C-E-G', 'D-F-A', 'E-G-B']
        self.generator = ChordGenerator(self.chords)
        self.generator.train_model(self.chords, order_arg=1, smoothing=1)

    def test_probabilities_sum_to_one(self):
        """Test that the probabilities sum to one."""
        self.assertAlmostEqual(sum(self.generator.start_probabilities), 1.0)
        self.assertTrue(all(np.isclose(sum(row), 1.0) for row in self.generator.transition_probabilities))

    def test_generate_chord_sequence(self):
        """Test generating a chord sequence."""
        sequence = self.generator.generate_chord_sequence(10)
        self.assertEqual(len(sequence), 10)
        self.assertTrue(all(chord in self.chords for chord in sequence))

    def test_load_midi(self):
        """Test loading chords from a MIDI file."""
        self.generator.load_midi = lambda x: self.chords
        self.assertEqual(self.generator.load_midi("dummy_path"), self.chords)

    def test_train_model(self):
        """Test training the chord model."""
        self.generator.train_model(self.chords, order_arg=1, smoothing=0)
        self.assertNotEqual(sum(self.generator.start_probabilities), 0)

    def test_generate_chord_sequence(self):
        """Test generating a chord sequence."""
        sequence = self.generator.generate_chord_sequence(10)
        self.assertEqual(len(sequence), 10)
        self.assertTrue(all(chord in self.chords for chord in sequence))

    def test_save_and_load_model(self):
        """Test saving and loading a chord model."""
        filename = 'test.pkl'
        self.generator.save_model(filename)
        loaded_model = ChordGenerator.load_model(filename)
        os.remove(filename)
        self.assertEqual(loaded_model.unique_chords, self.generator.unique_chords)

    def test_probability_normalization(self):
        """Test that the probabilities are normalized."""
        self.generator.train_model(self.chords, order_arg=1, smoothing=1)
        self.assertAlmostEqual(sum(self.generator.start_probabilities), 1)
        for row in self.generator.transition_probabilities:
            self.assertAlmostEqual(sum(row), 1)

        bad_chords = ['C', 'C', 'C']
        self.generator = ChordGenerator(bad_chords)
        self.generator.train_model(bad_chords, order_arg=1, smoothing=1)
        for row in self.generator.transition_probabilities:
            if np.sum(row) > 0:
                self.assertAlmostEqual(np.sum(row), 1)
            else:
                self.assertTrue(np.all(row == 0))

    def test_empty_chord_list(self):
        """Test that an empty chord list raises a ValueError."""
        empty_generator = ChordGenerator([])
        with self.assertRaises(ValueError):
            empty_generator.generate_chord_sequence(10)

    def test_roulette_wheel_selection_boundaries(self):
        """Test that the roulette wheel selection boundaries are correct."""
        np.random.seed(0)
        self.generator.train_model(self.chords, order_arg=1, smoothing=0)
        first_chord = self.generator.roulette_wheel_selection(self.generator.start_probabilities, chaotic_level=1)
        last_chord = self.generator.roulette_wheel_selection(self.generator.start_probabilities, chaotic_level=10)
        self.assertNotEqual(first_chord, last_chord)


if __name__ == '__main__':
    unittest.main()
