# query_cli.py
import argparse
from typing import List, Dict
import json
from .db import YogaVectorDB  # From previous implementation
from tabulate import tabulate
import warnings

class YogaCLI:
    def __init__(self, db_path: str = "./yoga_db"):
        self.db = YogaVectorDB(persist_dir=db_path)
        self.sequence_template = {
            'warmup': 2,
            'standing': 5,
            'seated': 5,
            'cool_down': 3
        }
        
    def _filter_contraindications(self, poses: List[Dict], conditions: List[str]) -> List[Dict]:
        """Filter poses based on health contraindications"""
        safe_poses = []
        for pose in poses:
            contraindications = pose['metadata']['Contraindications'].lower()
            if any(c.lower() in contraindications for c in conditions):
                continue
            safe_poses.append(pose)
        return safe_poses

    def generate_sequence(self, results: List[Dict]) -> List[Dict]:
        """Organize poses into logical sequence"""
        return sorted(results, key=lambda x: (
            -len(x['metadata']['Variations']),  # Poses with variations first
            x['metadata']['Level'] == 'Beginners',  # Beginner-friendly first
            len(x['document'])
        ))[:15]

    def query(self, conditions: List[str], goals: str, output_format: str = 'table'):
        """Main CLI query method"""
        # Construct semantic query
        query_text = f"Yoga poses for {', '.join(conditions)} that help with {goals}"
        
        # Get initial recommendations
        raw_results = self.db.query_poses(
            query_text=query_text,
            max_results=25
        )
        
        # Process results
        poses = [{
            'pose': r['document'].split('\n')[0].replace('Pose Name: ', ''),
            'duration': '3-5 mins' if 'Breathing' in r['document'] else '1-2 mins',
            'metadata': r['metadata'],
            'instructions': '\n'.join(r['document'].split('\n')[1:])
        } for r in raw_results['metadatas'][0]]

        # Safety filtering
        safe_poses = self._filter_contraindications(poses, conditions)
        
        if not safe_poses:
            warnings.warn("No safe poses found for your conditions. Consult a yoga therapist.")
            return []

        # Generate sequence
        sequence = self.generate_sequence(safe_poses)[:15]
        
        # Add sequence numbers and timing
        for i, pose in enumerate(sequence):
            pose['order'] = i + 1
            pose['type'] = self._categorize_pose(pose)
        
        # Output formatting
        if output_format == 'json':
            return json.dumps(sequence, indent=2)
        elif output_format == 'table':
            return tabulate(
                [[p['order'], p['pose'], p['type'], p['duration']] for p in sequence],
                headers=['#', 'Pose', 'Category', 'Duration'],
                tablefmt='grid'
            )
        return sequence

    def _categorize_pose(self, pose: Dict) -> str:
        """Categorize pose by type"""
        desc = pose['metadata']['Description'].lower()
        if 'standing' in desc:
            return 'standing'
        if 'seated' in desc or 'sit' in desc:
            return 'seated'
        if 'balance' in desc:
            return 'balance'
        return 'general'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Yoga Sequence Recommender')
    parser.add_argument('-c', '--conditions', nargs='+', required=True,
                        help='Health conditions (e.g., arthritis hypertension)')
    parser.add_argument('-g', '--goals', required=True,
                        help='Yoga goals (e.g., "improve flexibility reduce stress")')
    parser.add_argument('-f', '--format', choices=['table', 'json'], default='table',
                        help='Output format')
    
    args = parser.parse_args()
    
    recommender = YogaCLI()
    result = recommender.query(
        conditions=args.conditions,
        goals=args.goals,
        output_format=args.format
    )
    
    print(result)
