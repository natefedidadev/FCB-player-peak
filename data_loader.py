import os
import pickle
from pathlib import Path
from kloppy import sportscode
import pandas as pd
from typing import Union, List, Dict


"""
HOW TO LOAD THE DATA

from data_loader import MatchDataLoader

# Set your local path to the data folder
loader = MatchDataLoader(base_path="C:/Users/YourName/hackathon/data")

# Load all matches (will take time first time, then cached)
all_events = loader.load_all_matches_events()
"""

class MatchDataLoader:
    """
    Complete data loader for More than a Hack 2026
    Loads match data from XML/TXT files with automatic caching
    """
    
    def __init__(self, base_path: str, cache_dir: str = "cache", use_cache: bool = True):
        """
        Initialize the data loader
        
        Args:
            base_path: Path to the data folder (e.g., "C:/hackathon/data")
            cache_dir: Directory to store cached data
            use_cache: Whether to use caching (speeds up repeated loads)
        """
        self.base_path = Path(base_path)
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        
        # Create cache directory if it doesn't exist
        if self.use_cache:
            self.cache_dir.mkdir(exist_ok=True)
        
        # Auto-discover all matches
        self.matches = self._discover_matches()
        print(f"Found {len(self.matches)} matches")
    
    def _discover_matches(self) -> List[str]:
        """Automatically find all match folders"""
        matches = []
        masculino_path = self.base_path / "Masculi"
        
        if not masculino_path.exists():
            print(f"Warning: Path {masculino_path} does not exist!")
            return matches
        
        # Get all directories (each is a match)
        for item in masculino_path.iterdir():
            if item.is_dir():
                matches.append(item.name)
        
        return sorted(matches)
    
    def _get_cache_path(self, match_name: str, data_type: str) -> Path:
        """Get the cache file path for a match and data type"""
        safe_name = match_name.replace(" ", "_").replace("(", "").replace(")", "")
        return self.cache_dir / f"{safe_name}_{data_type}.pkl"
    
    def _load_from_cache(self, cache_path: Path):
        """Load data from cache file"""
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load cache {cache_path}: {e}")
                return None
        return None
    
    def _save_to_cache(self, data, cache_path: Path):
        """Save data to cache file"""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Warning: Could not save cache {cache_path}: {e}")
    
    def load_match_events(self, match_identifier: Union[int, str]) -> pd.DataFrame:
        """
        Load Smart Tagging events for a match
        
        Args:
            match_identifier: Match index (0-18) or match name
            
        Returns:
            DataFrame with event data
        """
        # Get match name
        if isinstance(match_identifier, int):
            if match_identifier >= len(self.matches):
                raise ValueError(f"Match index {match_identifier} out of range (0-{len(self.matches)-1})")
            match_name = self.matches[match_identifier]
        else:
            match_name = match_identifier
            if match_name not in self.matches:
                raise ValueError(f"Match '{match_name}' not found")
        
        # Check cache
        cache_path = self._get_cache_path(match_name, "events")
        if self.use_cache:
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                print(f"✓ Loaded events for '{match_name}' from cache")
                return cached_data
        
        # Load from file
        print(f"⟳ Loading events for '{match_name}' from files...")
        match_path = self.base_path / "Masculi" / match_name
        
        # Find pattern XML file
        pattern_files = list(match_path.glob("*_pattern.xml"))
        if not pattern_files:
            raise FileNotFoundError(f"No pattern.xml file found in {match_path}")
        
        pattern_file = pattern_files[0]
        
        # Load with kloppy
        dataset = sportscode.load(str(pattern_file))
        df = dataset.to_df()
        
        # Add match identifier
        df['match_name'] = match_name
        
        # Save to cache
        if self.use_cache:
            self._save_to_cache(df, cache_path)
            print(f"✓ Cached events for '{match_name}'")
        
        return df
    
    def load_match_tracking(self, match_identifier: Union[int, str]) -> str:
        """
        Load tracking data file path for a match
        
        Args:
            match_identifier: Match index (0-18) or match name
            
        Returns:
            Path to tracking XML file
        """
        # Get match name
        if isinstance(match_identifier, int):
            match_name = self.matches[match_identifier]
        else:
            match_name = match_identifier
        
        match_path = self.base_path / "Masculi" / match_name
        
        # Find tracking XML file (adjust pattern as needed)
        tracking_files = list(match_path.glob("*_tracking.xml")) or list(match_path.glob("*tracking*.xml"))
        
        if not tracking_files:
            print(f"Warning: No tracking file found for {match_name}")
            return None
        
        return str(tracking_files[0])
    
    def load_match_metadata(self, match_identifier: Union[int, str]) -> str:
        """
        Load metadata text file for a match
        
        Args:
            match_identifier: Match index (0-18) or match name
            
        Returns:
            Contents of metadata file
        """
        # Get match name
        if isinstance(match_identifier, int):
            match_name = self.matches[match_identifier]
        else:
            match_name = match_identifier
        
        match_path = self.base_path / "Masculi" / match_name
        
        # Find TXT file
        txt_files = list(match_path.glob("*.txt"))
        
        if not txt_files:
            print(f"Warning: No metadata file found for {match_name}")
            return None
        
        with open(txt_files[0], 'r', encoding='utf-8') as f:
            return f.read()
    
    def load_match_complete(self, match_identifier: Union[int, str]) -> Dict:
        """
        Load all data for a match (events, tracking path, metadata)
        
        Args:
            match_identifier: Match index (0-18) or match name
            
        Returns:
            Dictionary with all match data
        """
        if isinstance(match_identifier, int):
            match_name = self.matches[match_identifier]
        else:
            match_name = match_identifier
        
        return {
            'name': match_name,
            'events_df': self.load_match_events(match_name),
            'tracking_path': self.load_match_tracking(match_name),
            'metadata': self.load_match_metadata(match_name)
        }
    
    def load_all_matches_events(self) -> pd.DataFrame:
        """
        Load events from all matches into a single DataFrame
        
        Returns:
            Combined DataFrame with all events from all matches
        """
        cache_path = self.cache_dir / "all_matches_events.pkl"
        
        # Check cache
        if self.use_cache:
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                print(f"✓ Loaded all matches from cache ({len(cached_data)} total events)")
                return cached_data
        
        print(f"⟳ Loading all {len(self.matches)} matches...")
        all_dfs = []
        
        for i, match_name in enumerate(self.matches):
            print(f"  [{i+1}/{len(self.matches)}] {match_name}")
            df = self.load_match_events(match_name)
            all_dfs.append(df)
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Save to cache
        if self.use_cache:
            self._save_to_cache(combined_df, cache_path)
            print(f"✓ Cached all matches ({len(combined_df)} total events)")
        
        return combined_df
    
    def get_match_list(self) -> List[str]:
        """Get list of all available matches"""
        return self.matches.copy()
    
    def clear_cache(self):
        """Clear all cached data"""
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            print(f"✓ Cleared cache directory: {self.cache_dir}")
    
    def get_cache_info(self):
        """Print information about cached files"""
        if not self.cache_dir.exists():
            print("No cache directory found")
            return
        
        cache_files = list(self.cache_dir.glob("*.pkl"))
        if not cache_files:
            print("No cached files")
            return
        
        print(f"Cache directory: {self.cache_dir}")
        print(f"Cached files: {len(cache_files)}")
        
        total_size = sum(f.stat().st_size for f in cache_files)
        print(f"Total cache size: {total_size / 1024 / 1024:.2f} MB")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Initialize the loader
    # Each teammate should change this path to their local data folder
    loader = MatchDataLoader(
        base_path="C:/hackathon/data",  # Change this to your path!
        cache_dir="cache",
        use_cache=True
    )
    
    # Example 1: List all matches
    print("\n" + "="*60)
    print("Available matches:")
    for i, match in enumerate(loader.get_match_list()):
        print(f"  {i}: {match}")
    
    # Example 2: Load single match events
    print("\n" + "="*60)
    print("Loading single match...")
    df_match_0 = loader.load_match_events(0)  # By index
    print(f"Match 0 events shape: {df_match_0.shape}")
    print(df_match_0.head())
    
    # Example 3: Load complete match data
    print("\n" + "="*60)
    print("Loading complete match data...")
    match_data = loader.load_match_complete(0)
    print(f"Match name: {match_data['name']}")
    print(f"Events: {len(match_data['events_df'])} rows")
    print(f"Tracking file: {match_data['tracking_path']}")
    print(f"Metadata: {match_data['metadata'][:100] if match_data['metadata'] else 'N/A'}...")
    
    # Example 4: Load all matches (this will take time first time, then fast with cache)
    print("\n" + "="*60)
    print("Loading all matches...")
    all_events = loader.load_all_matches_events()
    print(f"Total events across all matches: {len(all_events)}")
    print(f"Matches included: {all_events['match_name'].nunique()}")
    
    # Example 5: Cache info
    print("\n" + "="*60)
    loader.get_cache_info()
    
    # Example 6: Basic analysis
    print("\n" + "="*60)
    print("Event distribution across matches:")
    print(all_events.groupby('match_name').size().sort_values(ascending=False))
