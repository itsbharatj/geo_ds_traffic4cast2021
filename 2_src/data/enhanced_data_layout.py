#  Copyright 2021 Institute of Advanced Research in Artificial Intelligence (IARAI) GmbH.
#  IARAI licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License. You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import itertools
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from collections import defaultdict
import h5py

# Original channel configuration
offset_map = {"N": (-1, 0), "NE": (-1, 1), "E": (0, 1), "SE": (1, 1), "S": (1, 0), "SW": (1, -1), "W": (0, -1), "NW": (-1, -1)}
layer_indices_from_offset = {v: i + 1 for i, v in enumerate(offset_map.values())}  # noqa

heading_list = ["NE", "SE", "SW", "NW"]
channel_labels = list(itertools.chain.from_iterable([[f"volume_{h}", f"speed_{h}"] for h in heading_list])) + ["incidents"]
static_channel_labels = ["base_map"] + [f"connectivity_{d}" for d in offset_map.keys()]

# Enhanced channel configuration with extra temporal channels
enhanced_channel_labels = channel_labels + [
    "avg_traffic_time_dow",      # Average traffic for time of day on day of week
    "avg_traffic_dow"            # Average traffic for day of week
]

volume_channel_indices = [ch for ch, l in enumerate(channel_labels) if "volume" in l]
speed_channel_indices = [ch for ch, l in enumerate(channel_labels) if "speed" in l]

# New indices for enhanced channels
avg_traffic_time_dow_idx = len(channel_labels)
avg_traffic_dow_idx = len(channel_labels) + 1

class TrafficStatisticsCalculator:
    """Calculator for traffic statistics used in enhanced channels"""
    
    def __init__(self, data_root: str):
        """
        Initialize with data root directory
        
        Args:
            data_root: Root directory containing traffic data files
        """
        self.data_root = Path(data_root)
        self.city_stats = {}  # Cache for computed statistics
        self.logger = logging.getLogger(__name__)
        
    def parse_filename(self, filename: str) -> Tuple[datetime, str]:
        """
        Parse filename to extract date and city
        
        Args:
            filename: Filename like "2019-01-02_BANGKOK_8ch.h5"
            
        Returns:
            Tuple of (datetime, city_name)
        """
        parts = filename.split('_')
        date_str = parts[0]
        city = parts[1] if len(parts) > 1 else "UNKNOWN"
        
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            self.logger.warning(f"Could not parse date from filename: {filename}")
            date = datetime(2019, 1, 1)  # Default date
            
        return date, city
        
    def get_time_metadata(self, date: datetime, time_slot: int) -> Dict:
        """
        Get time metadata for a given date and time slot
        
        Args:
            date: Date of the data
            time_slot: Time slot index (0-287, representing 5-minute intervals in a day)
            
        Returns:
            Dictionary with time metadata
        """
        # Calculate time of day from slot (each slot is 5 minutes)
        minutes_from_midnight = time_slot * 5
        hours = minutes_from_midnight // 60
        minutes = minutes_from_midnight % 60
        
        # Create datetime for this specific time
        current_time = date.replace(hour=hours, minute=minutes, second=0, microsecond=0)
        
        return {
            'datetime': current_time,
            'hour': hours,
            'minute': minutes,
            'day_of_week': date.weekday(),  # 0=Monday, 6=Sunday
            'day_name': date.strftime('%A'),
            'time_slot': time_slot,
            'time_of_day_minutes': minutes_from_midnight
        }
        
    def load_traffic_data_for_period(self, city: str, start_date: datetime, 
                                   end_date: datetime) -> Dict[str, np.ndarray]:
        """
        Load traffic data for a specific city and date range
        
        Args:
            city: City name
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary mapping date strings to traffic data arrays
        """
        data = {}
        current_date = start_date
        
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            filename = f"{date_str}_{city}_8ch.h5"
            file_path = self.data_root / city / "training" / filename
            
            if file_path.exists():
                try:
                    with h5py.File(file_path, 'r') as f:
                        # Load the data - assuming 'array' is the dataset name
                        if 'array' in f:
                            traffic_data = f['array'][:]
                            data[date_str] = traffic_data
                        else:
                            # Try to find the data array with different possible names
                            keys = list(f.keys())
                            if keys:
                                traffic_data = f[keys[0]][:]
                                data[date_str] = traffic_data
                except Exception as e:
                    self.logger.warning(f"Could not load {file_path}: {e}")
                    
            current_date += timedelta(days=1)
            
        return data
        
    def compute_traffic_statistics(self, city: str, target_date: datetime, 
                                 window_days: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute traffic statistics for enhanced channels
        
        Args:
            city: City name
            target_date: Target date for prediction
            window_days: Number of days to look back for statistics
            
        Returns:
            Tuple of (avg_traffic_time_dow, avg_traffic_dow) arrays
        """
        cache_key = f"{city}_{target_date.strftime('%Y-%m-%d')}_{window_days}"
        
        if cache_key in self.city_stats:
            return self.city_stats[cache_key]
            
        # Define the period for statistics (1 month before target date)
        end_date = target_date - timedelta(days=1)  # Don't include target date
        start_date = end_date - timedelta(days=window_days - 1)
        
        self.logger.info(f"Computing statistics for {city} from {start_date} to {end_date}")
        
        # Load traffic data for the period
        traffic_data = self.load_traffic_data_for_period(city, start_date, end_date)
        
        if not traffic_data:
            self.logger.warning(f"No traffic data found for {city} in the specified period")
            # Return zero arrays as fallback
            return np.zeros((495, 436)), np.zeros((495, 436))
            
        # Get shape from first available data
        first_data = next(iter(traffic_data.values()))
        height, width = first_data.shape[1], first_data.shape[2]
        
        # Initialize aggregation arrays
        # For average traffic during time of day on day of week
        time_dow_counts = defaultdict(lambda: defaultdict(int))
        time_dow_sums = defaultdict(lambda: defaultdict(lambda: np.zeros((height, width))))
        
        # For average traffic across day of week
        dow_counts = defaultdict(int)
        dow_sums = defaultdict(lambda: np.zeros((height, width)))
        
        # Process each day's data
        for date_str, daily_data in traffic_data.items():
            try:
                date = datetime.strptime(date_str, '%Y-%m-%d')
                dow = date.weekday()
                
                # Process each time slot in the day
                for time_slot in range(daily_data.shape[0]):
                    time_meta = self.get_time_metadata(date, time_slot)
                    hour = time_meta['hour']
                    
                    # Get traffic intensity (sum of volume channels)
                    traffic_intensity = np.sum(daily_data[time_slot, :, :, volume_channel_indices], axis=2)
                    
                    # Aggregate for time of day + day of week
                    key = (hour, dow)
                    time_dow_sums[key] += traffic_intensity
                    time_dow_counts[key] += 1
                    
                    # Aggregate for day of week
                    dow_sums[dow] += traffic_intensity
                    dow_counts[dow] += 1
                    
            except Exception as e:
                self.logger.warning(f"Error processing {date_str}: {e}")
                continue
        
        # Calculate target time metadata
        target_time_meta = self.get_time_metadata(target_date, 0)  # We'll update this for specific slots
        target_dow = target_date.weekday()
        
        # Compute averages - we'll use current hour and day of week for the target
        # For now, let's compute a general average for the target day of week
        avg_traffic_dow = np.zeros((height, width))
        if target_dow in dow_sums and dow_counts[target_dow] > 0:
            avg_traffic_dow = dow_sums[target_dow] / dow_counts[target_dow]
        
        # For time-of-day average, we'll compute an average across all hours for the target dow
        avg_traffic_time_dow = np.zeros((height, width))
        total_counts = 0
        total_sum = np.zeros((height, width))
        
        for hour in range(24):
            key = (hour, target_dow)
            if key in time_dow_sums and time_dow_counts[key] > 0:
                total_sum += time_dow_sums[key]
                total_counts += time_dow_counts[key]
                
        if total_counts > 0:
            avg_traffic_time_dow = total_sum / total_counts
            
        # Cache the results
        self.city_stats[cache_key] = (avg_traffic_time_dow, avg_traffic_dow)
        
        self.logger.info(f"Computed statistics for {city}: shapes {avg_traffic_time_dow.shape}, {avg_traffic_dow.shape}")
        
        return avg_traffic_time_dow, avg_traffic_dow
        
    def get_enhanced_channels_for_slot(self, city: str, date: datetime, 
                                     time_slot: int, window_days: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get enhanced channel data for a specific time slot
        
        Args:
            city: City name
            date: Date of the data
            time_slot: Time slot index
            window_days: Window for computing statistics
            
        Returns:
            Tuple of enhanced channel arrays
        """
        # Get base statistics
        avg_traffic_time_dow, avg_traffic_dow = self.compute_traffic_statistics(
            city, date, window_days)
        
        # For more sophisticated implementation, you could adjust these based on 
        # the specific time slot, but for now we return the computed averages
        return avg_traffic_time_dow, avg_traffic_dow


def create_enhanced_data_array(original_data: np.ndarray, 
                             enhanced_channels: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Combine original traffic data with enhanced channels
    
    Args:
        original_data: Original traffic data with shape (T, H, W, 8)
        enhanced_channels: Tuple of (avg_traffic_time_dow, avg_traffic_dow)
        
    Returns:
        Enhanced data array with shape (T, H, W, 10)
    """
    T, H, W, C = original_data.shape
    enhanced_data = np.zeros((T, H, W, C + 2), dtype=original_data.dtype)
    
    # Copy original channels
    enhanced_data[:, :, :, :C] = original_data
    
    # Add enhanced channels (same for all time steps)
    avg_traffic_time_dow, avg_traffic_dow = enhanced_channels
    for t in range(T):
        enhanced_data[t, :, :, C] = avg_traffic_time_dow
        enhanced_data[t, :, :, C + 1] = avg_traffic_dow
        
    return enhanced_data