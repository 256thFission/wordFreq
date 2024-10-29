import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import json
from pathlib import Path
import math
from dataclasses import dataclass, asdict


@dataclass
class MorphemeData:
    count: int = 0
    terms: List[str] = None
    diagnostic_value: float = 0.0
    specialty_bias: Dict[str, float] = None
    weight: float = 1.0

    def __post_init__(self):
        if self.terms is None:
            self.terms = []
        if self.specialty_bias is None:
            self.specialty_bias = {}


class MedicalMorphemeAnalyzer:
    def __init__(self, terms_file: str):
        # Configuration
        self.SPECIALTY_CATEGORIES = {
            'cardiology': ['cardi', 'angi', 'arter'],
            'neurology': ['neur', 'enceph', 'myel'],
            'orthopedics': ['oste', 'arthr', 'chondr'],
            'gastroenterology': ['gastr', 'enter', 'hepat'],
            'dermatology': ['derm', 'cut', 'papill'],
            'ophthalmology': ['opth', 'ocul', 'retin'],
            'pulmonology': ['pulmon', 'pneum', 'bronch'],
        }

        self.COMMON_PREFIXES = {
            'hyper', 'hypo', 'anti', 'peri', 'endo', 'epi',
            'hemi', 'poly', 'sub', 'trans', 'inter', 'intra'
        }

        self.COMMON_SUFFIXES = {
            'itis', 'emia', 'ectomy', 'osis', 'pathy',
            'plasty', 'trophy', 'tomy', 'opsy', 'oma'
        }

        # Load terms
        self.terms = self._load_terms(terms_file)
        self.morpheme_data = defaultdict(lambda: MorphemeData())

    def _load_terms(self, file_path: str) -> List[str]:
        """Load medical terms from file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip().lower() for line in f if line.strip()]

    def _extract_morphemes(self, term: str) -> Set[str]:
        """Extract potential morphemes from a term."""
        morphemes = set()

        # Check for known prefixes
        for prefix in self.COMMON_PREFIXES:
            if term.startswith(prefix):
                morphemes.add(prefix)
                term = term[len(prefix):]
                break

        # Check for known suffixes
        for suffix in self.COMMON_SUFFIXES:
            if term.endswith(suffix):
                morphemes.add(suffix)
                term = term[:-len(suffix)]
                break

        # Extract potential root morphemes (3+ characters)
        for specialty_roots in self.SPECIALTY_CATEGORIES.values():
            for root in specialty_roots:
                if root in term:
                    morphemes.add(root)

        return morphemes

    def analyze_frequencies(self):
        """Analyze morpheme frequencies in the medical terms."""
        # First pass: count frequencies
        for term in self.terms:
            morphemes = self._extract_morphemes(term)
            for morpheme in morphemes:
                self.morpheme_data[morpheme].count += 1
                self.morpheme_data[morpheme].terms.append(term)

    def calculate_specialty_distribution(self):
        """Calculate specialty distribution for each morpheme."""
        for morpheme, data in self.morpheme_data.items():
            terms = data.terms
            total_terms = len(terms)

            for specialty, indicators in self.SPECIALTY_CATEGORIES.items():
                specialty_count = sum(
                    1 for term in terms
                    if any(i in term for i in indicators)
                )
                if total_terms > 0:
                    data.specialty_bias[specialty] = specialty_count / total_terms

    def calculate_diagnostic_value(self):
        """Calculate diagnostic value for each morpheme."""
        total_terms = len(self.terms)

        for morpheme, data in self.morpheme_data.items():
            # Calculate frequency in medical corpus
            medical_freq = data.count / total_terms

            # Diagnostic value increases with:
            # 1. Higher frequency in medical terms
            # 2. Longer morpheme length (more specific)
            # 3. Specialty specificity

            length_factor = len(morpheme) / 4  # Normalize by typical morpheme length
            specialty_factor = max(data.specialty_bias.values()) if data.specialty_bias else 0

            data.diagnostic_value = medical_freq * length_factor * (1 + specialty_factor)

    def calculate_final_weights(self):
        """Calculate final weights for each morpheme."""
        max_count = max(data.count for data in self.morpheme_data.values())
        max_diagnostic = max(data.diagnostic_value for data in self.morpheme_data.values())

        for morpheme, data in self.morpheme_data.items():
            # Normalize counts and diagnostic values
            norm_count = data.count / max_count
            norm_diagnostic = data.diagnostic_value / max_diagnostic

            # Calculate base weight
            base_weight = (norm_count + norm_diagnostic) / 2

            # Adjust weight based on morpheme type
            if morpheme in self.COMMON_PREFIXES:
                base_weight *= 1.2  # Boost prefixes
            elif morpheme in self.COMMON_SUFFIXES:
                base_weight *= 1.1  # Boost suffixes

            # Adjust for specialty specificity
            max_specialty_bias = max(data.specialty_bias.values()) if data.specialty_bias else 0
            specialty_multiplier = 1 + (max_specialty_bias * 0.5)

            # Set final weight
            data.weight = base_weight * specialty_multiplier

    def analyze(self) -> Dict:
        """Run complete analysis pipeline."""
        print("Analyzing frequencies...")
        self.analyze_frequencies()

        print("Calculating specialty distributions...")
        self.calculate_specialty_distribution()

        print("Calculating diagnostic values...")
        self.calculate_diagnostic_value()

        print("Calculating final weights...")
        self.calculate_final_weights()

        # Prepare results
        results = {
            'morphemes': {
                morpheme: asdict(data)
                for morpheme, data in self.morpheme_data.items()
                if data.count > 1  # Filter out rare morphemes
            }
        }

        return results

    def save_results(self, results: Dict, output_file: str):
        """Save analysis results to JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)


def main():
    # File paths
    terms_file = "wordlist.txt"  # Input file with one term per line
    output_file = "morpheme_weights.json"

    # Create and run analyzer
    analyzer = MedicalMorphemeAnalyzer(terms_file)
    results = analyzer.analyze()

    # Save results
    analyzer.save_results(results, output_file)

    # Print summary
    print(f"\nAnalysis complete. Found {len(results['morphemes'])} significant morphemes.")

    # Print top 10 morphemes by weight
    print("\nTop 10 morphemes by weight:")
    top_morphemes = sorted(
        results['morphemes'].items(),
        key=lambda x: x[1]['weight'],
        reverse=True
    )[:10]

    for morpheme, data in top_morphemes:
        print(f"{morpheme}: {data['weight']:.3f} (found in {data['count']} terms)")


if __name__ == "__main__":
    main()