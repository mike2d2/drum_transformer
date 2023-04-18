from etl import MidiDatasetTokenizer, MidiTokenizer
data_dir = '../data_raw/'
dataset_name = 'default_hi_318'
num_accompaniment = 4
cap = 0

tokenizer = MidiTokenizer()
dataset_tokenizer = MidiDatasetTokenizer(data_dir, tokenizer, num_accompaniment)
dataset_tokenizer.write_tokenized(dataset_name, cap)