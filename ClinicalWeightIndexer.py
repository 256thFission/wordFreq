import pandas as pd
import numpy as np
from collections import defaultdict
import math
import re
import json
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm
import gc
from typing import Dict, List, Set
import multiprocessing as mp
from functools import partial
import warnings

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='clinical_indexer.log'
)


class ClinicalWeightIndexer:
    def __init__(self):
        self.term_freq = defaultdict(int)  # Raw frequency
        self.co_occurrences = defaultdict(lambda: defaultdict(int))
        self.total_words = 0
        self.total_segments = 0
        self.tf_idf = {}
        self.pmi_scores = defaultdict(dict)  # Pointwise Mutual Information
        self.clinical_segments = []

    def preprocess_note(self, text):
        """Extract logical segments with clinical meaning"""
        # Split on meaningful boundaries while preserving clinical context
        segments = []
        current_segment = []

        for line in text.split('\n'):
            line = line.strip()
            if not line:
                if current_segment:
                    segments.append(' '.join(current_segment))
                    current_segment = []
                continue

            # Check if line starts new segment
            if (re.match(r'^[A-Z\s]{4,}:', line) or  # Header pattern
                    re.match(r'^\d+\.|^\-|\*', line) or  # List items
                    re.match(r'^[A-Z][a-z]+:', line)):  # Assessment/Plan style
                if current_segment:
                    segments.append(' '.join(current_segment))
                    current_segment = []

            current_segment.append(line)

        if current_segment:
            segments.append(' '.join(current_segment))

        return segments

    def analyze_segments(self, segments):
        """Detailed analysis of clinical segments"""
        for segment in segments:
            # Clean while preserving clinical meaning
            words = self._clean_clinical_text(segment)
            self.total_segments += 1

            # Track frequencies and co-occurrences
            for i, word in enumerate(words):
                self.term_freq[word] += 1
                self.total_words += 1

                # Analyze context window (adaptive size based on punctuation)
                window_size = self._get_adaptive_window(words, i)
                start = max(0, i - window_size)
                end = min(len(words), i + window_size + 1)

                # Update co-occurrences with context
                for j in range(start, end):
                    if i != j:
                        self.co_occurrences[word][words[j]] += 1

            self.clinical_segments.append(words)

    def _clean_clinical_text(self, text):
        """Preserve clinical significance while cleaning"""
        # Preserve patterns with clinical significance
        text = re.sub(r'(\d+\.?\d*)\s*(mg|g|mL|L|mmol|mcg|kg|cm|mm|BP|HR|RR|O2)', r'\1_\2', text)

        # Preserve medical abbreviations
        text = re.sub(r'\b([A-Z]{2,})\b', r'_\1_', text)

        # Clean while keeping clinical meaning
        text = re.sub(r'[^\w\s_-]', ' ', text)

        return text.lower().split()

    def _get_adaptive_window(self, words, position):
        """Determine context window size based on text structure"""
        base_window = 5
        # Adjust window based on surrounding punctuation/structure
        return base_window  # Could be made more sophisticated

    def calculate_weights(self):
        """Calculate comprehensive weight index"""
        weights = {}

        # Calculate TF-IDF
        self._calculate_tf_idf()

        # Calculate PMI scores
        self._calculate_pmi()

        # Calculate final weights
        for term in self.term_freq:
            # Base frequency weight
            freq_weight = math.log(1 + self.term_freq[term])

            # TF-IDF component
            tf_idf_weight = self.tf_idf.get(term, 0)

            # Co-occurrence weight (based on PMI)
            co_occur_weight = self._get_co_occurrence_weight(term)

            # Clinical pattern weight
            pattern_weight = self._get_clinical_pattern_weight(term)

            # Combine weights with learned coefficients
            weights[term] = self._combine_weights(
                freq_weight,
                tf_idf_weight,
                co_occur_weight,
                pattern_weight
            )

        return weights

    def _calculate_tf_idf(self):
        """Calculate TF-IDF scores"""
        for term in self.term_freq:
            # Term frequency in entire corpus
            tf = self.term_freq[term] / self.total_words

            # Inverse document frequency
            doc_freq = sum(1 for seg in self.clinical_segments if term in set(seg))
            idf = math.log(self.total_segments / (1 + doc_freq))

            self.tf_idf[term] = tf * idf

    def _calculate_pmi(self):
        """Calculate Pointwise Mutual Information scores"""
        for term1 in self.co_occurrences:
            for term2, co_occur_count in self.co_occurrences[term1].items():
                pmi = math.log(
                    (co_occur_count * self.total_words) /
                    (self.term_freq[term1] * self.term_freq[term2])
                )
                self.pmi_scores[term1][term2] = pmi

    def _get_co_occurrence_weight(self, term):
        """Calculate weight based on co-occurrence patterns"""
        if term not in self.pmi_scores:
            return 1.0

        # Get average PMI with clinical terms
        clinical_pmis = [
            pmi for t, pmi in self.pmi_scores[term].items()
            if self._is_clinical_term(t)
        ]

        if not clinical_pmis:
            return 1.0

        return 1.0 + (sum(clinical_pmis) / len(clinical_pmis))

    def _get_clinical_pattern_weight(self, term):
        """Weight based on clinical patterns"""
        weight = 1.0

        # Check for measurement patterns
        if re.search(r'\d+_[a-zA-Z]+', term):
            weight *= 1.5

        # Check for medical abbreviations
        if re.search(r'_[A-Z]{2,}_', term):
            weight *= 1.3

        # Check for medical affixes
        medical_prefixes = {'hyper', 'hypo', 'anti', 'brady', 'tachy'}
        medical_suffixes = {'itis', 'emia', 'osis', 'pathy', 'ectomy'}

        if any(term.startswith(prefix) for prefix in medical_prefixes):
            weight *= 1.2
        if any(term.endswith(suffix) for suffix in medical_suffixes):
            weight *= 1.2

        return weight

    def _combine_weights(self, freq_w, tfidf_w, cooccur_w, pattern_w):
        """Combine different weight components"""
        # Could be tuned based on performance
        return (
                0.3 * freq_w +
                0.3 * tfidf_w +
                0.2 * cooccur_w +
                0.2 * pattern_w
        )

    def _is_clinical_term(self, term):
        """Identify likely clinical terms"""
        return any([
            re.search(r'\d+_[a-zA-Z]+', term),  # Measurements
            re.search(r'_[A-Z]{2,}_', term),  # Abbreviations
            term in self._get_clinical_term_set()
        ])

    def _get_clinical_term_set(self):
        """Get set of high-confidence clinical terms"""
        # Could be expanded based on patterns observed in data
        return {word for word, freq in self.term_freq.items()
                if any([
                re.search(r'_[A-Z]{2,}_', word),
                re.search(r'\d+_[a-zA-Z]+', word)
            ])}


def process_chunk(args):
    """Process a chunk of clinical notes"""
    chunk, chunk_id = args  # Unpack the arguments
    indexer = ClinicalWeightIndexer()

    for note in tqdm(chunk['text'], desc=f'Processing chunk {chunk_id}'):
        try:
            if pd.isna(note):
                continue
            segments = indexer.preprocess_note(note)
            indexer.analyze_segments(segments)
        except Exception as e:
            logging.error(f"Error processing note in chunk {chunk_id}: {str(e)}")
            continue

    return {
        'term_freq': dict(indexer.term_freq),
        'co_occurrences': {k: dict(v) for k, v in indexer.co_occurrences.items()},
        'total_words': indexer.total_words,
        'total_segments': indexer.total_segments
    }


def merge_results(results: List[Dict]) -> Dict:
    """Merge results from multiple chunks"""
    merged = {
        'term_freq': defaultdict(int),
        'co_occurrences': defaultdict(lambda: defaultdict(int)),
        'total_words': 0,
        'total_segments': 0
    }

    for result in results:
        # Merge term frequencies
        for term, freq in result['term_freq'].items():
            merged['term_freq'][term] += freq

        # Merge co-occurrences
        for term1, co_occurs in result['co_occurrences'].items():
            for term2, count in co_occurs.items():
                merged['co_occurrences'][term1][term2] += count

        # Merge totals
        merged['total_words'] += result['total_words']
        merged['total_segments'] += result['total_segments']

    return merged


def save_weights(weights: Dict, filename: str):
    """Save weights with metadata"""
    output = {
        'weights': weights,
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'num_terms': len(weights),
            'max_weight': max(weights.values()),
            'min_weight': min(weights.values()),
            'avg_weight': sum(weights.values()) / len(weights)
        }
    }

    with open(filename, 'w') as f:
        json.dump(output, f)


def main():
    # Configuration
    CHUNK_SIZE = 1000  # Adjust based on available memory
    NUM_PROCESSES = max(1, mp.cpu_count() - 1)  # Leave one CPU free
    INPUT_FILE = 'discharge.csv'
    OUTPUT_FILE = 'clinical_weights.json'

    logging.info(f"Starting processing with {NUM_PROCESSES} processes")

    try:
        # Calculate number of chunks
        total_rows = sum(1 for _ in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE))
        num_chunks = (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE

        logging.info(f"Processing {total_rows} notes in {num_chunks} chunks")

        # Process chunks in parallel
        results = []
        with mp.Pool(NUM_PROCESSES) as pool:
            # Create iterator of (chunk, chunk_id) tuples
            chunks = enumerate(pd.read_csv(
                INPUT_FILE,
                chunksize=CHUNK_SIZE,
                usecols=['text']
            ))

            # Process chunks
            for result in tqdm(
                    pool.imap(process_chunk, ((chunk, i) for i, chunk in chunks)),
                    total=num_chunks,
                    desc="Processing chunks"
            ):
                results.append(result)
                gc.collect()  # Free memory

                logging.info(f"Completed chunk {i + 1}/{num_chunks}")

        logging.info("Merging results...")
        merged = merge_results(results)

        # Create final indexer for weight calculation
        final_indexer = ClinicalWeightIndexer()
        final_indexer.term_freq = defaultdict(int, merged['term_freq'])
        final_indexer.co_occurrences = defaultdict(
            lambda: defaultdict(int),
            merged['co_occurrences']
        )
        final_indexer.total_words = merged['total_words']
        final_indexer.total_segments = merged['total_segments']

        logging.info("Calculating final weights...")
        weights = final_indexer.calculate_weights()

        # Save results
        logging.info(f"Saving weights to {OUTPUT_FILE}")
        save_weights(weights, OUTPUT_FILE)

        # Print summary statistics
        logging.info("Processing complete. Summary:")
        logging.info(f"Total terms processed: {len(weights)}")
        logging.info(f"Total words processed: {merged['total_words']}")
        logging.info(f"Total segments processed: {merged['total_segments']}")

        # Print top weighted terms
        print("\nTop 20 weighted terms:")
        top_terms = sorted(
            weights.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]

        for term, weight in top_terms:
            print(f"{term}: {weight:.3f}")

    except Exception as e:
        logging.error(f"Error in main processing: {str(e)}")
        raise


if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Starting processing at {start_time}")

    main()

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\nProcessing completed in {duration}")