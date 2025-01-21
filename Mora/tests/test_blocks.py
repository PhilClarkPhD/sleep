import unittest
from datetime import timedelta
from pydantic import ValidationError
from mora.blocks import TimeBlockUnit, TimeBlocks


class TestTimeBlockUnit(unittest.TestCase):
    def test_valid_timeblockunit(self):
        """Test valid creation of a TimeBlockUnit."""
        block = TimeBlockUnit(duration=3661, phase="Light")
        self.assertEqual(block.duration, timedelta(hours=1, minutes=1, seconds=1))
        self.assertEqual(block.phase, "Light")

    def test_invalid_timeblockunit_duration(self):
        """Test TimeBlockUnit with end time before start time."""
        with self.assertRaises(ValidationError) as context:
            TimeBlockUnit(duration=-1000, phase="Dark")
        self.assertIn(
            "Value error, Duration must be greater than zero. [type=value_error, input_value={'duration': -1000, 'phase': 'Dark'}, input_type=dict]",
            str(context.exception),
        )

    def test_invalid_timeblockunit_invalid_phase(self):
        """Test TimeBlockUnit with an invalid phase."""
        with self.assertRaises(ValidationError) as context:
            TimeBlockUnit(duration=3661, phase="Bright")
        self.assertIn(
            "Input should be 'Light' or 'Dark' [type=literal_error, input_value='Bright', input_type=str]",
            str(context.exception),
        )


class TestTimeBlocks(unittest.TestCase):
    def test_valid_timeblocks(self):
        """Test valid creation of a TimeBlocks object."""
        blocks_data = {
            1: TimeBlockUnit(duration=40, phase="Light"),
            2: TimeBlockUnit(duration=5050, phase="Dark"),
        }
        time_blocks = TimeBlocks(blocks=blocks_data)
        self.assertEqual(len(time_blocks.blocks), 2)
        self.assertEqual(time_blocks.blocks[1].phase, "Light")
        self.assertEqual(time_blocks.blocks[2].phase, "Dark")

    def test_invalid_timeblocks_non_integer_key(self):
        """Test TimeBlocks with a non-integer key."""
        blocks_data = {
            "A": TimeBlockUnit(duration=2340, phase="Light"),
        }
        with self.assertRaises(ValidationError) as context:
            TimeBlocks(blocks=blocks_data)
        self.assertIn(
            "Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='A', input_type=str]",
            str(context.exception),
        )

    def test_invalid_timeblocks_non_dict_blocks(self):
        """Test TimeBlocks with a non-dictionary blocks input."""
        with self.assertRaises(ValidationError) as context:
            TimeBlocks(blocks=["not", "a", "dict"])
        self.assertIn(
            "Input should be a valid dictionary [type=dict_type, input_value=['not', 'a', 'dict'], input_type=list]",
            str(context.exception),
        )


if __name__ == "__main__":
    unittest.main()
