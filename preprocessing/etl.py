import json
import pathlib
from copy import deepcopy

import miditoolkit
import miditok
from miditok.constants import ADDITIONAL_TOKENS

import h5py
from tqdm import tqdm


# ------------------------------------------------------------------------------------------------------
# GLOBAL VARIABLES
# ------------------------------------------------------------------------------------------------------

ACCOMPANIMENT_BUCKETS: dict[range, tuple[str, int]] = {

    range(0,2): ('Acoustic Piano', 0),
    range(2,6): ('Electric Piano', 1),
    range(16,24): ('Organ', 2),
    range(24,26): ('Acoustic Guitar', 3),
    range(26,31): ('Electric Guitar', 4),
    range(32,40): ('Bass', 5),
    range(40,48): ('Strings', 6),
    range(48,56): ('Ensemble', 7),
    range(56,64): ('Brass', 8),
    range(64,72): ('Reed', 9),
    range(80,88): ('Synth Lead', 10),
    range(88,96): ('Synth Pad', 11)

}

BUCKET_DEFAULT_PROGRAMS: dict[str, int] = {

    'Acoustic Piano': 0,
    'Electric Piano': 2,
    'Organ': 18,
    'Acoustic Guitar': 24,
    'Electric Guitar': 27,
    'Bass': 33,
    'Strings': 40,
    'Ensemble': 48,
    'Brass': 56,
    'Reed': 66,
    'Synth Lead': 80,
    'Synth Pad': 88

}

BUCKET_NAMES: dict[int, str] = {
    bucket_id: bucket_name for bucket_name, bucket_id in ACCOMPANIMENT_BUCKETS.values()
}

ADDTL_TOKENS = ADDITIONAL_TOKENS.copy()
ADDTL_TOKENS["Tempo"] = True


# ------------------------------------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ------------------------------------------------------------------------------------------------------

def get_bucket_key(program: int) -> int | None:
    '''Given a program number for a track get the bucket id for accompaniment if applicable.'''

    bucket_id: int | None = None
    for key_range in ACCOMPANIMENT_BUCKETS:
        if program in key_range:
            bucket_id = ACCOMPANIMENT_BUCKETS[key_range][1]
            break
    return bucket_id

def get_bucket_name(program: int) -> str:
    '''Given a program number for a track get the bucket name for accompaniment if applicable.'''

    for key_range in ACCOMPANIMENT_BUCKETS:
        if program in key_range:
            bucket_name: str = ACCOMPANIMENT_BUCKETS[key_range][0]
            break
    return bucket_name

def create_midi_shell(midifile: miditoolkit.MidiFile) -> miditoolkit.MidiFile:
    '''Given a MIDI file of a song create a blank copy with no instruments'''

    new: miditoolkit.MidiFile = deepcopy(midifile)
    new.instruments = []
    return new


# ------------------------------------------------------------------------------------------------------
# MIDI EXTRACTOR CLASSES
# ------------------------------------------------------------------------------------------------------

class MidiTokenizer(object):

    def __init__(self, inst_buckets: dict[range, tuple[str, int]] = ACCOMPANIMENT_BUCKETS, addtl_tokens: dict[str, type] = ADDTL_TOKENS) -> None:

        # Base tokenizer for the instruments
        self.oore_tokenizer: miditok.MIDILike = miditok.MIDILike(
            additional_tokens=addtl_tokens,
            special_tokens=[]
        )
        self.base_vocab_size: int = len(self.oore_tokenizer.vocab)

        # Get token IDs for beginning/end of sequence/instrument events
        self.seq_ids: dict[str, int] = {
            'BOS': self.base_vocab_size + 0,
            'EOS': self.base_vocab_size + 1,
        }

        # Now add token IDs for instrument change events
        self.inst_ids: dict[str, int] = {}
        self.inst_buckets: dict[range, tuple[str, int]] = inst_buckets
        for b_name, b_id in self.inst_buckets.values():
            self.inst_ids[b_name] = self.base_vocab_size + 2 + b_id
        self.token_to_program: dict[int, int] = {
            token_num: BUCKET_DEFAULT_PROGRAMS[b_name] for b_name, token_num in self.inst_ids.items()
        }

        # Finally get ovr vocab size
        self.total_vocab_size: int = self.base_vocab_size + 2 + len(self.inst_buckets)

    def tokenize_drums(self, drums_midifile: miditoolkit.MidiFile) -> list[int]:
        '''Given a prepped drums midifile generate the event sequence with BOS and EOS tokens'''

        # Preprocess file and run initial tokenization
        self.oore_tokenizer.preprocess_midi(drums_midifile)
        tokenized_arr: list[miditok.TokSequence] = self.oore_tokenizer(drums_midifile)
        if not tokenized_arr:
            return []
        tokenized: list[int] = self.oore_tokenizer(drums_midifile)[0].ids

        # Add extra tokens and return
        tokenized = [self.seq_ids['BOS']] + tokenized + [self.seq_ids['EOS']]
        return tokenized
    
    def tokenize_accompaniment(self, accomp_midifile: miditoolkit.MidiFile) -> list[int]:
        '''Given a prepped accompaniment midifile generate the event sequence with extra tokens'''

        # Start overall seqeuence
        ovr_tokenized = [self.seq_ids['BOS']]

        # Preprocess file and run initial tokenization
        self.oore_tokenizer.preprocess_midi(accomp_midifile)
        tokenized_arr: list[miditok.TokSequence] = self.oore_tokenizer(accomp_midifile)
        if not tokenized_arr:
            return []
        programs: list[tuple[int, bool]] = miditok.utils.get_midi_programs(accomp_midifile)

        # Iterate through tracks and build seq
        for tokseq, program_tup in zip(tokenized_arr, programs):
            inst_id: int = self.inst_ids[get_bucket_name(program_tup[0])]
            tokenized_track: list[int] = [inst_id] + tokseq.ids
            ovr_tokenized += tokenized_track
        ovr_tokenized += [self.seq_ids['EOS']]

        return ovr_tokenized
    
    def compile_drums(self, drums_eventseq: list[int], ticks_per_beat: int, out_path: str) -> miditoolkit.MidiFile:
        '''Convert a drum event sequence into midi and download'''

        oore_seq: list[int] = drums_eventseq[1:-1]
        midi_obj: miditoolkit.MidiFile = self.oore_tokenizer.tokens_to_midi(
            [oore_seq],
            programs=[(0, True)],
            output_path=None,
            time_division=ticks_per_beat
        )
        midi_obj.dump(out_path)

        return midi_obj
        
    def compile_accompaniment(self, accomp_eventseq: list[int], ticks_per_beat: int, out_path: str) -> miditoolkit.MidiFile:
        '''Convert an accompaniment sequence into midi and download'''

        ids: list[list[int]] = []
        programs: list[tuple[int, bool]] = []

        oore_seqs: list[int] = accomp_eventseq[1:-1]
        switches: list[tuple[int, int]] = []
        for i,x in enumerate(oore_seqs):
            if x in self.token_to_program.keys():
                switches.append((i, self.token_to_program[x]))
        for k, (switch_ind, program) in enumerate(switches):
            inst_seq: list[int]
            if k != len(switches) - 1:
                inst_seq = oore_seqs[switch_ind+1:switches[k + 1][0]]
            else:
                inst_seq = oore_seqs[switch_ind+1:]
            ids.append(inst_seq)
            programs.append((program, False))
        
        midi_obj: miditoolkit.MidiFile = self.oore_tokenizer.tokens_to_midi(
            ids,
            programs,
            output_path=None,
            time_division=ticks_per_beat
        )
        midi_obj.dump(out_path)

        return midi_obj
    
    def compile_both(self, drums_eventseq: list[int], accomp_eventseq: list[int], ticks_per_beat: int, out_path: str) -> miditoolkit.MidiFile:
        '''Convert a drums + accompaniment sequence into midi and download'''

        ids: list[list[int]] = []
        programs: list[tuple[int, bool]] = []

        oore_seqs: list[int] = accomp_eventseq[1:-1]
        switches: list[tuple[int, int]] = []
        for i,x in enumerate(oore_seqs):
            if x in self.token_to_program.keys():
                switches.append((i, self.token_to_program[x]))
        for k, (switch_ind, program) in enumerate(switches):
            inst_seq: list[int]
            if k != len(switches) - 1:
                inst_seq = oore_seqs[switch_ind+1:switches[k + 1][0]]
            else:
                inst_seq = oore_seqs[switch_ind+1:]
            ids.append(inst_seq)
            programs.append((program, False))

        drums_seq: list[int] = drums_eventseq[1:-1]
        ids.append(drums_seq)
        programs.append((0, True))
        
        midi_obj: miditoolkit.MidiFile = self.oore_tokenizer.tokens_to_midi(
            ids,
            programs,
            output_path=None,
            time_division=ticks_per_beat
        )
        midi_obj.dump(out_path)

        return midi_obj


class MidiFileInfo(object):

    def __init__(self, file_path: str) -> None:
        
        # File path/id information
        self.file_path: pathlib.Path = pathlib.Path(file_path)
        self.file_id: str = self.file_path.parts[-1]
        parts_temp: list[str] = list(self.file_path.parent.parts)
        parts_temp[2] = 'lmd_matched_h5'
        parts_temp[-1] = parts_temp[-1] + '.h5'
        self.metadata_path: pathlib.Path = pathlib.Path('/'.join(parts_temp))

    def get_genre_tags(self) -> list[tuple[str, float]]:
        '''Extract genre tags for this file from the lmd_matched_h5 metadata.'''

        genre_tags: list = []
        metadata_file: h5py.File = h5py.File(self.metadata_path)
        tags: list[str] = list(metadata_file['metadata']['artist_terms'])
        weights: list[float] = list(metadata_file['metadata']['artist_terms_weight'])
        for tag, weight in zip(tags, weights):
            genre_tags.append((tag, weight))
        
        return genre_tags


class MidiExtractor(MidiFileInfo):

    def __init__(self, file_path: str, num_accompaniment: int) -> None:
        
        # File path/id information
        super().__init__(file_path)

        # Load in the MIDI file and identify instruments
        self.miditk_obj: miditoolkit.MidiFile
        try:
            self.miditk_obj = miditoolkit.MidiFile(self.file_path)
        except:
            self.miditk_obj = miditoolkit.MidiFile()
        self.instruments = self.miditk_obj.instruments
        self.has_drums: bool = any(x.is_drum for x in self.instruments)

        # Params
        self.num_accompaniment: int = num_accompaniment

    def extract_drums(self) -> list[tuple[int, miditoolkit.Instrument]]:
        '''Extract the drum tracks from the MIDI file with note counts'''

        filtered: list[miditoolkit.Instrument] = list(filter(lambda x: x.is_drum, self.instruments))
        with_notes: list[tuple[int, miditoolkit.Instrument]] = list(map(lambda x: (len(x.notes), x), filtered))
        return with_notes
    
    def extract_other(self) -> list[tuple[int, int | None, miditoolkit.Instrument]]:
        '''Extract the accompaniment tracks that match a bucket with note counts and bucket maps'''

        filtered: list[miditoolkit.Instrument] = list(filter(
            lambda x: (not x.is_drum) and any(x.program in k for k in ACCOMPANIMENT_BUCKETS) and get_bucket_key(x.program), 
            self.instruments
        ))
        with_notes_ids: list[tuple[int, int | None, miditoolkit.Instrument]] = list(map(
            lambda x: (len(x.notes), get_bucket_key(x.program), x), 
            filtered
        ))
        return with_notes_ids
    
    def prepare_splits(self, notes_thres_1: float = 0.5, notes_thres_2: float = 0.2) -> tuple[miditoolkit.MidiFile, miditoolkit.MidiFile]:
        '''Return the split MIDI files (drum tracks and "best" accompaniment tracks)'''

        # Initialize midi toolkit objects
        miditk_drums: miditoolkit.MidiFile = deepcopy(self.miditk_obj)
        miditk_accomp: miditoolkit.MidiFile = deepcopy(self.miditk_obj)

        # Get drum track with highest note count and assign to toolkit object
        drums_track: miditoolkit.Instrument = sorted(
            self.extract_drums(),
            key=lambda x: x[0],
            reverse=True
        )[0][1]
        miditk_drums.instruments = [drums_track]

        # Sort accompaniment tracks by note count and calculate note thresholds (proportions of max note track)
        sorted_accomp: list[tuple[int, int | None, miditoolkit.Instrument]] = sorted(
            self.extract_other(),
            key=lambda x: x[0],
            reverse=True
        )
        if not sorted_accomp:
            return (miditk_drums, miditoolkit.MidiFile())
        notes_threshold_1: float = sorted_accomp[0][0]*notes_thres_1
        notes_threshold_2: float = sorted_accomp[0][0]*notes_thres_2

        # Initialize accompaniment instrument container
        instruments: list[miditoolkit.Instrument] = []
        tracks_remaining: int = self.num_accompaniment

        # Prioritize bass and piano if suff notes (> threshold 1)
        piano_accomp: list[tuple[int, int | None, miditoolkit.Instrument]] = list(filter(
            lambda x: x[1] in [0,1] and x[0] > notes_threshold_1,
            sorted_accomp
        ))
        if len(piano_accomp) > 2:
            piano_accomp = piano_accomp[0:2]
        for track_tup in piano_accomp:
            tracks_remaining -= 1
            instruments.append(track_tup[2])
            sorted_accomp.remove(track_tup)
        bass_accomp: list[tuple[int, int | None, miditoolkit.Instrument]] = list(filter(
            lambda x: x[1] == 5 and x[0] > notes_threshold_1,
            sorted_accomp
        ))
        if bass_accomp:
            tracks_remaining -= 1
            instruments.append(bass_accomp[0][2])
            sorted_accomp.remove(bass_accomp[0])

        # Now take remaining instruments
        other_accomp: list[tuple[int, int | None, miditoolkit.Instrument]] = list(filter(
            lambda x: x[0] > notes_threshold_2,
            sorted_accomp
        ))
        if len(other_accomp) > tracks_remaining:
            other_accomp = other_accomp[0:tracks_remaining]
        for track_tup in other_accomp:
            tracks_remaining -= 1
            instruments.append(track_tup[2])

        # Assign instruments to accompaniment toolkit object
        miditk_accomp.instruments = instruments

        # Return two miditoolkit objects prepared for tokenization
        return (miditk_drums, miditk_accomp)
    
    def download_drums_orig(self, out_path: str) -> None:
        '''Extract just the drums from the file and download to given path. 
        
        If tokenizer provided performs cleanup work before dumping so timings match with generated outputs.
        '''

        miditk_drums_orig = deepcopy(self.miditk_obj)
        miditk_drums_orig.instruments = [inst for inst in miditk_drums_orig.instruments if inst.is_drum]
        miditk_drums_orig.dump(out_path)

    def download_accomp_orig(self, out_path: str) -> None:
        '''Extract just the non-drums from the file and download to given path.
        
        If tokenizer provided performs cleanup work before dumping so timings match with generated outputs.
        '''

        miditk_accomp_orig = deepcopy(self.miditk_obj)
        miditk_accomp_orig.instruments = [inst for inst in miditk_accomp_orig.instruments if not inst.is_drum]
        miditk_accomp_orig.dump(out_path)


class MidiDatasetTokenizer(object):

    def __init__(self, data_directory: str, tokenizer: MidiTokenizer, num_accompaniment: int) -> None:
        
        self.file_paths: map = map(
            str,
            pathlib.Path(data_directory).rglob('*.mid')
        )
        self.tokenizer: MidiTokenizer = tokenizer
        self.num_accompaniment: int = num_accompaniment

    def write_tokenized(self, dataset_name: str, cap: int = 0) -> None:
        '''Given a tokenizer and a directory of midi files, write cleaned data to given output'''

        # Open output file and start counter
        dataset_path: str = f'../data_cleaned/{dataset_name}.txt'
        count: int = 0

        f = open(dataset_path, 'w')

        # Iterate through dir
        for file in tqdm(self.file_paths):

            # Create extractor
            extractor: MidiExtractor = MidiExtractor(file, self.num_accompaniment)

            # Check that drums exist and valid file
            if (not extractor.has_drums) or (not extractor.miditk_obj.instruments):
                continue

            # Separate into distinct midi files
            drums, accomp = extractor.prepare_splits()

            # Check if we have accompanying instruments
            if (not accomp.instruments) or (not drums.instruments):
                continue

            # Tokenize
            drums_tok: list[int] = self.tokenizer.tokenize_drums(drums)
            accomp_tok: list[int] = self.tokenizer.tokenize_accompaniment(accomp)

            # Check if we have valid tokenizations
            if (not drums_tok) or (not accomp_tok):
                continue

            # Get other info
            ticks_per_beat: int = extractor.miditk_obj.ticks_per_beat
            metadata_path: str = str(extractor.metadata_path)

            # Write to file
            f.write(json.dumps({
                'drums': drums_tok,
                'accomp': accomp_tok,
                'ticks': ticks_per_beat,
                'metadata': metadata_path
            }))
            f.write('\n')

            # If capped check if we should finish
            if cap:
                count += 1
                if count >= cap:
                    break
        
        f.close()
