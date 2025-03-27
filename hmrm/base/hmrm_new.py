import numpy as np
import pandas as pd
from scipy import sparse 
from numpy.linalg import inv as inverse
from numpy.linalg import norm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
import tempfile
import h5py
from tqdm import tqdm
import gc
import numba
from functools import partial
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages memory for large arrays with disk-based storage when needed"""

    def __init__(self, work_dir=None):
        self.work_dir = work_dir or tempfile.mkdtemp(prefix="hmrm_")
        os.makedirs(self.work_dir, exist_ok=True)
        self.storage_file = os.path.join(self.work_dir, "hmrm_storage.h5")
        self.arrays = {}

    def store_array(self, name, array, force_disk=False):
        """Store array in memory or on disk based on size and available RAM"""
        # Calculate array size in MB
        if isinstance(array, np.ndarray):
            size_mb = array.nbytes / (1024 * 1024)
        elif sparse.issparse(array):
            size_mb = array.data.nbytes / (1024 * 1024)
        else:
            size_mb = 0

        # If array is large or force_disk is True, store on disk
        if size_mb > 500 or force_disk:
            with h5py.File(self.storage_file, 'a') as f:
                if name in f:
                    del f[name]

                if sparse.issparse(array):
                    # Store sparse matrix in COO format
                    coo = array.tocoo()
                    grp = f.create_group(name)
                    grp.create_dataset("data", data=coo.data)
                    grp.create_dataset("row", data=coo.row)
                    grp.create_dataset("col", data=coo.col)
                    grp.attrs["shape"] = coo.shape
                    grp.attrs["format"] = "coo"
                    self.arrays[name] = {"on_disk": True, "format": "sparse"}
                else:
                    # Store dense array
                    f.create_dataset(name, data=array, chunks=True, compression="gzip", compression_opts=4)
                    self.arrays[name] = {"on_disk": True, "format": "dense"}
                logger.debug(f"Stored {name} on disk ({size_mb:.2f} MB)")
        else:
            # Store in memory
            self.arrays[name] = {"data": array, "on_disk": False}
            logger.debug(f"Stored {name} in memory ({size_mb:.2f} MB)")

    def get_array(self, name):
        """Retrieve array from memory or disk"""
        if name not in self.arrays:
            raise KeyError(f"Array {name} not found")

        if self.arrays[name]["on_disk"]:
            with h5py.File(self.storage_file, 'r') as f:
                if self.arrays[name]["format"] == "sparse":
                    grp = f[name]
                    data = grp["data"][:]
                    row = grp["row"][:]
                    col = grp["col"][:]
                    shape = grp.attrs["shape"]
                    return sparse.coo_matrix((data, (row, col)), shape=shape).tolil()
                else:
                    return f[name][:]
        else:
            return self.arrays[name]["data"]

    def cleanup(self):
        """Clean up temporary files"""
        if os.path.exists(self.storage_file):
            os.remove(self.storage_file)


class Optimizer:
    def __init__(self, num_workers=None, memory_limit_mb=None):
        # Initialize parameters
        self._weight = 0.001

        # Determine number of workers for parallel processing
        self.num_workers = num_workers or max(1, mp.cpu_count() - 1)

        # Set memory limit (default to 80% of available system memory)
        if memory_limit_mb is None:
            available_memory = psutil.virtual_memory().available / (1024 * 1024)
            self.memory_limit_mb = int(available_memory * 0.8)
        else:
            self.memory_limit_mb = memory_limit_mb

        # Initialize memory manager
        self.memory_manager = MemoryManager()

        # Track computation time
        self.timings = {}

    def _time_execution(func):
        """Decorator to track execution time of methods"""

        def wrapper(self, *args, **kwargs):
            start_time = datetime.now()
            result = func(self, *args, **kwargs)
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            self.timings[func.__name__] = execution_time
            logger.info(f"{func.__name__} completed in {execution_time:.2f} seconds")
            return result

        return wrapper

    def process_user_location_chunk(self, chunk_data, total_users, total_places):
        user_ids, place_ids = chunk_data
        chunk_matrix = sparse.lil_matrix((total_users, total_places))
        for u, p in zip(user_ids, place_ids):
            chunk_matrix[u, p] += 1
        return chunk_matrix

    @_time_execution
    def _create_user_location_frequency_matrix(self, users_checkins):
        """Create user-location frequency matrix with parallelization for large datasets"""
        placeids = users_checkins["placeid"].tolist()
        userids = users_checkins["userid"].tolist()
        total_users = len(users_checkins["userid"].unique())
        total_places = len(users_checkins["placeid"].unique())
        logger.info(f'Total places: {total_places}, total users: {total_users}')

        # For very large datasets, process in chunks using parallel processing
        if len(placeids) > 1000000:
            # Create matrix in memory
            user_location_frequency = sparse.lil_matrix((total_users, total_places))

            # Process in chunks
            chunk_size = len(placeids) // self.num_workers
            chunks = [(userids[i:i + chunk_size], placeids[i:i + chunk_size])
                      for i in range(0, len(placeids), chunk_size)]

            # Process chunks in parallel
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                results = list(tqdm(
                    executor.map(
                        partial(self.process_user_location_chunk, total_users=total_users, total_places=total_places),
                        chunks),
                    total=len(chunks),
                    desc="Creating user-location matrix"
                ))

            # Combine results
            for chunk_result in results:
                user_location_frequency += chunk_result
        else:
            # For smaller datasets, process directly
            user_location_frequency = sparse.lil_matrix((total_users, total_places))
            for u, p in tqdm(zip(userids, placeids), total=len(placeids), desc="Creating user-location matrix"):
                user_location_frequency[u, p] += 1

        # Store matrix using memory manager
        self.memory_manager.store_array("user_location_frequency", user_location_frequency)

    @_time_execution
    def _create_user_time_frequency_matrix(self, users_checkins):
        """Create user-time frequency matrix optimized for performance"""
        users_ids = users_checkins["userid"].tolist()
        datetimes = pd.to_datetime(users_checkins["datetime"]).tolist()
        total_users = len(users_checkins["userid"].unique())

        # Use numpy for faster computation
        user_time_frequency = np.zeros((total_users, 48), dtype=np.float32)

        # Vectorize datetime processing for better performance
        # Create lists for indices and values
        weekend_indices = []
        weekday_indices = []
        weekend_values = []
        weekday_values = []

        for i, dt in tqdm(enumerate(datetimes), total=len(datetimes), desc="Processing timestamps"):
            user_id = users_ids[i]
            if dt.weekday() >= 5:  # Weekend
                weekend_indices.append((user_id, dt.hour + 24))
                weekend_values.append(1)
            else:  # Weekday
                weekday_indices.append((user_id, dt.hour))
                weekday_values.append(1)

        # Process in batches
        for indices, values in [(weekend_indices, weekend_values), (weekday_indices, weekday_values)]:
            for (user_id, time_slot), value in zip(indices, values):
                user_time_frequency[user_id, time_slot] += value

        # Store the matrix
        self.memory_manager.store_array("user_time_frequency", user_time_frequency)

    @staticmethod
    @numba.njit(parallel=True)
    def _numba_pmi_calculation(co_ocurrency, number_of_locations, sum_of_dl, l_occurrency, c_occurrency):
        """Use numba to accelerate PMI calculation"""
        result = np.zeros((number_of_locations, number_of_locations), dtype=np.float32)
        for i in numba.prange(number_of_locations):
            line = co_ocurrency[i]
            for j in range(number_of_locations):
                if line[j] > 0:
                    pmi = np.log2(max(line[j] * sum_of_dl, 1) / (l_occurrency[i] * c_occurrency[j]))
                    result[i, j] = max(pmi, 0)
        return result

    def process_cooc_chunk(self, number_of_locations, locations, chunk_data):
        locs, start_idx = chunk_data
        # Use LIL format which is efficient for incremental construction
        chunk_matrix = sparse.lil_matrix((number_of_locations, number_of_locations))
        for i in range(len(locs)):
            actual_idx = start_idx + i
            # Look backwards
            for j in range(1, 6):
                if (actual_idx - j) < 0:
                    break
                chunk_matrix[locs[i], locations[actual_idx - j]] += 1
            # Look forward
            for j in range(1, 6):
                if (actual_idx + j) >= len(locations):
                    break
                chunk_matrix[locs[i], locations[actual_idx + j]] += 1
        return chunk_matrix

    @_time_execution
    def _create_location_coocurrency_matrix(self, users_checkins):
        """Create location co-occurrence matrix with optimized PMI calculation"""
        try:
            # Sort by datetime for correct sequence
            users_checkins_sorted = users_checkins.sort_values(by=["datetime"])
            locations = users_checkins_sorted["placeid"].tolist()
            number_of_locations = len(users_checkins["placeid"].unique())

            location_co_ocurrency = sparse.lil_matrix((number_of_locations, number_of_locations), dtype=np.float32)

            # Process in parallel for large datasets
            if len(locations) > 500000:
                chunk_size = len(locations) // self.num_workers
                chunks = [(locations[i:i + chunk_size], i)
                          for i in range(0, len(locations), chunk_size)]

                # Process chunks in parallel
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    results = list(tqdm(
                        executor.map(
                            partial(self.process_cooc_chunk, number_of_locations, locations),
                            chunks),
                        total=len(chunks),
                        desc="Creating co-occurrence chunks"
                    ))

                # Combine results
                for chunk_result in results:
                    location_co_ocurrency += chunk_result
            else:
                # For smaller datasets, process directly with vectorized operations where possible
                for i in tqdm(range(len(locations)), desc="Creating co-occurrence matrix"):
                    # Look backwards
                    for j in range(1, 6):
                        if (i - j) < 0:
                            break
                        location_co_ocurrency[locations[i], locations[i - j]] += 1
                    # Look forward
                    for j in range(1, 6):
                        if (i + j) >= len(locations):
                            break
                        location_co_ocurrency[locations[i], locations[i + j]] += 1

            # Convert to dense for PMI calculation - for large matrices, process in chunks
            if number_of_locations > 10000:
                # Process PMI calculation in chunks to avoid memory issues
                chunk_size = 1000
                for i in tqdm(range(0, number_of_locations, chunk_size), desc="PMI calculation in chunks"):
                    end_idx = min(i + chunk_size, number_of_locations)
                    chunk = location_co_ocurrency[i:end_idx].toarray()

                    sum_of_dl = location_co_ocurrency.sum()
                    l_occurrency = np.asarray(location_co_ocurrency.sum(axis=1)).flatten()
                    c_occurrency = np.asarray(location_co_ocurrency.sum(axis=0)).flatten()

                    for j in range(chunk.shape[0]):
                        row_idx = i + j
                        line = chunk[j]
                        # PMI calculation
                        with np.errstate(divide='ignore', invalid='ignore'):
                            pmi = np.log2(
                                np.maximum(line * sum_of_dl, 1) /
                                (l_occurrency[row_idx] * c_occurrency)
                            )
                        pmi = np.maximum(pmi, 0)
                        pmi[np.isnan(pmi)] = 0
                        location_co_ocurrency[row_idx] = sparse.lil_matrix(pmi)
            else:
                # Convert to dense for PMI calculation
                co_oc_array = location_co_ocurrency.toarray().astype(np.float32)

                sum_of_dl = np.sum(co_oc_array)
                l_occurrency = np.sum(co_oc_array, axis=1).reshape(-1, 1)
                c_occurrency = np.sum(co_oc_array, axis=0).reshape(1, -1)

                # Use numba-accelerated function for PMI calculation
                result = self._numba_pmi_calculation(
                    co_oc_array, number_of_locations, sum_of_dl,
                    l_occurrency.flatten(), c_occurrency.flatten()
                )

                location_co_ocurrency = sparse.lil_matrix(result)

            # Store the matrix
            self.memory_manager.store_array("location_co_ocurrency", location_co_ocurrency)

        except Exception as e:
            logger.error(f"Error in creating location co-occurrence matrix: {e}")
            raise e

    @_time_execution
    def _create_location_time_matrix(self, users_checkins):
        """Create location-time matrix with optimized memory usage"""
        locations = users_checkins["placeid"].tolist()
        datetimes = pd.to_datetime(users_checkins["datetime"]).tolist()
        total_locations = len(users_checkins["placeid"].unique())

        # Use float32 for memory efficiency
        Dt = np.zeros((total_locations, 48), dtype=np.float32)

        # Process in batches to avoid memory issues with very large datasets
        batch_size = 100000
        for i in tqdm(range(0, len(locations), batch_size), desc="Creating location-time matrix"):
            end_idx = min(i + batch_size, len(locations))
            batch_locations = locations[i:end_idx]
            batch_datetimes = datetimes[i:end_idx]

            for j in range(len(batch_locations)):
                if batch_datetimes[j].weekday() >= 5:
                    Dt[batch_locations[j]][batch_datetimes[j].hour + 24] += 1
                else:
                    Dt[batch_locations[j]][batch_datetimes[j].hour] += 1

        # Calculate PMI for location-time matrix
        sum_of_dt = np.sum(Dt)
        l_occurrency = np.sum(Dt, axis=1).reshape(-1, 1)
        c_occurrency = np.sum(Dt, axis=0).reshape(1, -1)

        # Vectorized PMI calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            mult = l_occurrency * c_occurrency
            mult[mult == 0] = -1

            tmp = np.maximum(Dt * sum_of_dt, 1) / mult
            tmp[tmp < 0] = 0
            location_time = np.maximum(np.log2(tmp), 0)
            location_time[np.isnan(location_time)] = 0

        # Store the matrix
        self.memory_manager.store_array("location_time", location_time)

    def _objective_function(self, l2_weight):
        """Calculate objective function value with memory-efficient retrieval"""
        # Get matrices from memory manager
        user_location_frequency = self.memory_manager.get_array("user_location_frequency")
        user_time_frequency = self.memory_manager.get_array("user_time_frequency")
        location_co_ocurrency = self.memory_manager.get_array("location_co_ocurrency")
        location_time = self.memory_manager.get_array("location_time")

        # Calculate components with chunking for large matrices
        def first_component(l2_weight):
            # User-item modeling component
            if isinstance(user_location_frequency, np.ndarray) or user_location_frequency.shape[0] < 5000:
                first_eq = l2_weight * norm(
                    user_location_frequency - np.dot(self.user_activity, self.activity_location.T)
                )
            else:
                # Process in chunks for large sparse matrices
                chunk_size = 1000
                error_sum = 0
                for i in range(0, user_location_frequency.shape[0], chunk_size):
                    end_idx = min(i + chunk_size, user_location_frequency.shape[0])
                    chunk = user_location_frequency[i:end_idx].toarray()
                    pred_chunk = np.dot(self.user_activity[i:end_idx], self.activity_location.T)
                    error_sum += np.sum((chunk - pred_chunk) ** 2)
                first_eq = l2_weight * np.sqrt(error_sum)

            second_eq = (1 - l2_weight) * norm(
                user_time_frequency - np.dot(self.user_activity, self.activity_time.T)
            )
            return first_eq + second_eq

        def second_component(l2_weight):
            # Trajectory embedding component
            if isinstance(location_co_ocurrency, np.ndarray) or location_co_ocurrency.shape[0] < 5000:
                first_eq = l2_weight * norm(
                    location_co_ocurrency - np.dot(
                        self.target_location_embedding, self.context_location_embedding.T
                    )
                )
            else:
                # Process in chunks for large sparse matrices
                chunk_size = 1000
                error_sum = 0
                for i in range(0, location_co_ocurrency.shape[0], chunk_size):
                    end_idx = min(i + chunk_size, location_co_ocurrency.shape[0])
                    chunk = location_co_ocurrency[i:end_idx].toarray()
                    pred_chunk = np.dot(
                        self.target_location_embedding[i:end_idx],
                        self.context_location_embedding.T
                    )
                    error_sum += np.sum((chunk - pred_chunk) ** 2)
                first_eq = l2_weight * np.sqrt(error_sum)

            second_eq = (1 - l2_weight) * norm(
                location_time - np.dot(self.target_location_embedding, self.time_slot_embedding.T)
            )
            return first_eq + second_eq

        def third_component(l2_weight):
            # Collaborative learning component
            first_eq = l2_weight * norm(
                self.activity_location - np.dot(self.context_location_embedding, self.activity_embedding.T)
            )
            second_eq = (1 - l2_weight) * norm(
                self.activity_time - np.dot(self.time_slot_embedding, self.activity_embedding.T)
            )
            return first_eq + second_eq

        # Calculate each component
        activity_modeling_component = first_component(l2_weight)
        trajectory_embedding_component = second_component(l2_weight)
        collaborative_learning_component = third_component(l2_weight)

        # Calculate regularization terms
        regularization = (
                self._weight * norm(self.user_activity) +
                self._weight * norm(self.activity_time) +
                self._weight * norm(self.activity_embedding) +
                self._weight * norm(self.activity_location) +
                self._weight * norm(self.context_location_embedding) +
                self._weight * norm(self.target_location_embedding) +
                self._weight * norm(self.time_slot_embedding)
        )

        # Return total objective function value
        return (
                activity_modeling_component +
                trajectory_embedding_component +
                collaborative_learning_component +
                regularization
        )

    @_time_execution
    def _initialize_parameters(self, checkins, K, M):
        """Initialize model parameters with efficient memory usage"""
        total_locations = len(checkins["placeid"].unique())
        total_users = len(checkins["userid"].unique())
        time_slot = 48

        # Use float32 for memory efficiency
        self.activity_location = np.random.normal(size=(total_locations, K)).astype(np.float32)
        self.activity_time = np.random.normal(size=(time_slot, K)).astype(np.float32)
        self.user_activity = np.random.normal(size=(total_users, K)).astype(np.float32)
        self.activity_embedding = np.random.normal(size=(K, M)).astype(np.float32)
        self.target_location_embedding = np.random.normal(size=(total_locations, M)).astype(np.float32)
        self.context_location_embedding = np.random.normal(size=(total_locations, M)).astype(np.float32)
        self.time_slot_embedding = np.random.normal(size=(time_slot, M)).astype(np.float32)

        # Store matrices that will be used frequently
        for name, matrix in [
            ("activity_location", self.activity_location),
            ("activity_time", self.activity_time),
            ("user_activity", self.user_activity),
            ("activity_embedding", self.activity_embedding),
            ("target_location_embedding", self.target_location_embedding),
            ("context_location_embedding", self.context_location_embedding),
            ("time_slot_embedding", self.time_slot_embedding)
        ]:
            self.memory_manager.store_array(name, matrix)

    def _optimize_one_parameter(self, param_name, func, K=None, M=None, l2_weight=None):
        """Optimize a single parameter with memory-efficient operations"""
        # Calculate new parameter value
        if M is not None:
            new_value = func(M, l2_weight)
        else:
            new_value = func(K, l2_weight)

        # Apply non-negativity constraint for specific parameters
        if param_name in ["user_activity", "activity_location", "activity_time"]:
            new_value[new_value < 0] = 0

        # Update parameter
        setattr(self, param_name, new_value)

        # Store updated parameter
        self.memory_manager.store_array(param_name, new_value)

        return new_value

    def user_activity_embedding_function(self, K, l2_weight):
        """Calculate user activity embedding with memory-efficient operations"""
        # Get required matrices from memory manager
        user_location_frequency = self.memory_manager.get_array("user_location_frequency")
        user_time_frequency = self.memory_manager.get_array("user_time_frequency")
        activity_location = self.memory_manager.get_array("activity_location")
        activity_time = self.memory_manager.get_array("activity_time")

        # Calculate first equation with memory-efficient matrix multiplication
        if sparse.issparse(user_location_frequency):
            first_eq_part1 = l2_weight * (user_location_frequency.dot(activity_location))
        else:
            first_eq_part1 = l2_weight * np.dot(user_location_frequency, activity_location)

        first_eq_part2 = (1 - l2_weight) * np.dot(user_time_frequency, activity_time)
        first_equation = first_eq_part1 + first_eq_part2

        # Calculate second equation
        second_eq_part1 = l2_weight * np.dot(activity_location.T, activity_location)
        second_eq_part2 = (1 - l2_weight) * np.dot(activity_time.T, activity_time) + (l2_weight * np.identity(K))
        second_equation = second_eq_part1 + second_eq_part2

        # Calculate and return result
        return np.dot(first_equation, inverse(second_equation))

    def acticity_location_embedding_function(self, K, l2_weight):
        """Calculate activity location embedding with memory-efficient operations"""
        # Get required matrices from memory manager
        user_location_frequency = self.memory_manager.get_array("user_location_frequency")
        user_activity = self.memory_manager.get_array("user_activity")
        context_location_embedding = self.memory_manager.get_array("context_location_embedding")
        activity_embedding = self.memory_manager.get_array("activity_embedding")

        # Calculate first equation with memory-efficient matrix multiplication
        if sparse.issparse(user_location_frequency):
            part1 = user_location_frequency.T.dot(user_activity)
        else:
            part1 = np.dot(user_location_frequency.T, user_activity)

        first_equation = l2_weight * (
                part1 + np.dot(context_location_embedding, activity_embedding.T)
        )

        # Calculate second equation
        second_equation = (
                                  l2_weight * np.dot(user_activity.T, user_activity)
                          ) + ((self._weight + l2_weight) * np.identity(K))

        # Calculate and return result
        return np.dot(first_equation, inverse(second_equation))

    def activity_time_embedding_function(self, K, l2_weight):
        """Calculate activity time embedding with memory-efficient operations"""
        # Get required matrices from memory manager
        user_time_frequency = self.memory_manager.get_array("user_time_frequency")
        user_activity = self.memory_manager.get_array("user_activity")
        time_slot_embedding = self.memory_manager.get_array("time_slot_embedding")
        activity_embedding = self.memory_manager.get_array("activity_embedding")

        # Calculate first equation
        first_equation = (1 - l2_weight) * (
                np.dot(user_time_frequency.T, user_activity)
                + np.dot(time_slot_embedding, activity_embedding.T)
        )

        # Calculate second equation
        second_equation = (1 - l2_weight) * (
                np.dot(user_activity.T, user_activity)
                + (1 - self._weight + l2_weight) * np.identity(K)
        )

        # Calculate and return result
        return np.dot(first_equation, inverse(second_equation))

    def activity_embedding_function(self, M, l2_weight):
        """Calculate activity embedding with memory-efficient operations"""
        # Get required matrices from memory manager
        activity_location = self.memory_manager.get_array("activity_location")
        context_location_embedding = self.memory_manager.get_array("context_location_embedding")
        activity_time = self.memory_manager.get_array("activity_time")
        time_slot_embedding = self.memory_manager.get_array("time_slot_embedding")

        # Calculate first equation
        first_equation = (
                                 l2_weight * np.dot(activity_location.T, context_location_embedding)
                         ) + (
                                 (1 - l2_weight) * np.dot(activity_time.T, time_slot_embedding)
                         )

        # Calculate second equation
        second_equation = (
                (l2_weight * np.dot(context_location_embedding.T, context_location_embedding))
                + ((1 - l2_weight) * np.dot(time_slot_embedding.T, time_slot_embedding))
                + (self._weight * np.identity(M))
        )

        # Calculate and return result
        return np.dot(first_equation, inverse(second_equation))

    def target_location_embedding_function(self, M, l2_weight):
        """Calculate target location embedding with memory-efficient operations"""
        # Get required matrices from memory manager
        location_co_ocurrency = self.memory_manager.get_array("location_co_ocurrency")
        context_location_embedding = self.memory_manager.get_array("context_location_embedding")
        location_time = self.memory_manager.get_array("location_time")
        time_slot_embedding = self.memory_manager.get_array("time_slot_embedding")

        # Calculate first equation with memory-efficient matrix multiplication
        if sparse.issparse(location_co_ocurrency):
            first_eq_part1 = l2_weight * location_co_ocurrency.dot(context_location_embedding)
        else:
            first_eq_part1 = l2_weight * np.dot(location_co_ocurrency, context_location_embedding)

        first_eq_part2 = (1 - l2_weight) * np.dot(location_time, time_slot_embedding)
        first_equation = first_eq_part1 + first_eq_part2

        # Calculate second equation
        second_equation = (
                (l2_weight * np.dot(context_location_embedding.T, context_location_embedding))
                + ((1 - l2_weight) * np.dot(time_slot_embedding.T, time_slot_embedding))
                + (self._weight * np.identity(M))
        )

        # Calculate and return result
        return np.dot(first_equation, inverse(second_equation))

    def context_location_embedding_function(self, M, l2_weight):
        """Calculate context location embedding with memory-efficient operations"""
        # Get required matrices from memory manager
        location_co_ocurrency = self.memory_manager.get_array("location_co_ocurrency")
        target_location_embedding = self.memory_manager.get_array("target_location_embedding")
        activity_location = self.memory_manager.get_array("activity_location")
        activity_embedding = self.memory_manager.get_array("activity_embedding")

        # Calculate first equation with memory-efficient matrix multiplication
        if sparse.issparse(location_co_ocurrency):
            first_eq_part1 = location_co_ocurrency.T.dot(target_location_embedding)
        else:
            first_eq_part1 = np.dot(location_co_ocurrency.T, target_location_embedding)

        first_equation = l2_weight * (
                first_eq_part1 + np.dot(activity_location, activity_embedding)
        )

        # Calculate second equation
        second_equation = (
                                  l2_weight * (
                                  np.dot(target_location_embedding.T, target_location_embedding)
                                  + np.dot(activity_embedding.T, activity_embedding)
                          )
                          ) + (self._weight * np.identity(M))

        # Calculate and return result
        return np.dot(first_equation, inverse(second_equation))

    def time_slot_embedding_function(self, M, l2_weight):
        """Calculate time slot embedding with memory-efficient operations"""
        # Get required matrices from memory manager
        location_time = self.memory_manager.get_array("location_time")
        target_location_embedding = self.memory_manager.get_array("target_location_embedding")
        activity_time = self.memory_manager.get_array("activity_time")
        activity_embedding = self.memory_manager.get_array("activity_embedding")

        # Calculate first equation
        first_equation = (1 - l2_weight) * (
                np.dot(location_time.T, target_location_embedding)
                + np.dot(activity_time, activity_embedding)
        )

        # Calculate second equation
        second_equation = (
                                  (1 - l2_weight) * (
                                  np.dot(target_location_embedding.T, target_location_embedding)
                                  + np.dot(activity_embedding.T, activity_embedding)
                          )
                          ) + (self._weight * np.identity(M))

        # Calculate and return result
        return np.dot(first_equation, inverse(second_equation))

    @_time_execution
    def _optimize_parameters(self, K, M, l2_weight):
        """Optimize parameters with parallel processing for independent variables"""
        # Define parameter optimization tasks
        param_tasks = [
            ('user_activity', self.user_activity_embedding_function, K),
            ('activity_location', self.acticity_location_embedding_function, K),
            ('activity_time', self.activity_time_embedding_function, K),
            ('activity_embedding', self.activity_embedding_function, M),
            ('target_location_embedding', self.target_location_embedding_function, M),
            ('context_location_embedding', self.context_location_embedding_function, M),
            ('time_slot_embedding', self.time_slot_embedding_function, M)
        ]

        # Process parameters one by one for now (safer for optimization stability)
        # Could parallelize some of these with careful dependency management
        for param_name, func, dim in param_tasks:
            self._optimize_one_parameter(param_name, func, K if dim == K else None, M if dim == M else None, l2_weight)

            # Force garbage collection after each parameter update
            gc.collect()

    @_time_execution
    def start(self, checkins, l2_weight=0.1, K=10, M=100, max_iterations=10, convergence_threshold=0.01,
              checkpoint_dir=None):
        """Start the optimization process with improved performance and monitoring"""
        logger.info(f'Starting HMRM optimization with K={K}, M={M}, l2_weight={l2_weight}')
        start_time = datetime.now()

        # Set up checkpoint directory
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        # Preprocess checkins
        checkins["datetime"] = pd.to_datetime(checkins["datetime"])

        # Create frequency matrices in parallel
        logger.info("Creating frequency matrices...")
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Start all tasks
            future_user_location = executor.submit(self._create_user_location_frequency_matrix, checkins)
            future_location_cooc = executor.submit(self._create_location_coocurrency_matrix, checkins)
            future_user_time = executor.submit(self._create_user_time_frequency_matrix, checkins)

            # Wait for all tasks to complete
            future_user_location.result()
            future_location_cooc.result()
            future_user_time.result()

        # Create location-time matrix (depends on other matrices)
        self._create_location_time_matrix(checkins)

        logger.info('All matrices created successfully')

        # Initialize parameters
        self._initialize_parameters(checkins, K, M)

        # Optimization loop with improved convergence criteria and checkpointing
        prev_objective = float('inf')

        logger.info("Starting parameter optimization")
        pbar = tqdm(range(max_iterations), desc="Optimization iterations")
        for i in pbar:
            # Optimize parameters
            self._optimize_parameters(K, M, l2_weight)

            # Calculate objective function value
            objective_func = self._objective_function(l2_weight)
            improvement = prev_objective - objective_func
            improvement_percent = (improvement / prev_objective) * 100 if prev_objective != 0 else float('inf')

            # Update progress bar
            pbar.set_postfix({"objective": f"{objective_func:.2f}", "improvement": f"{improvement_percent:.2f}%"})

            # Save checkpoint if requested
            if checkpoint_dir:
                checkpoint_file = os.path.join(checkpoint_dir, f"hmrm_checkpoint_iter_{i}.h5")
                with h5py.File(checkpoint_file, 'w') as f:
                    for param_name in ["user_activity", "activity_location", "activity_time",
                                       "activity_embedding", "target_location_embedding",
                                       "context_location_embedding", "time_slot_embedding"]:
                        param_value = getattr(self, param_name)
                        f.create_dataset(param_name, data=param_value)

                logger.info(f"Checkpoint saved to {checkpoint_file}")

            # Check for convergence
            if improvement < convergence_threshold * prev_objective:
                logger.info(f"Converged after {i + 1} iterations (improvement: {improvement_percent:.4f}%)")
                break

            prev_objective = objective_func

        # Log optimization summary
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        logger.info(f"HMRM optimization completed in {total_time:.2f} seconds")

        # Clean up memory
        gc.collect()

        return {
            "user_activity": self.user_activity,
            "activity_location": self.activity_location,
            "activity_time": self.activity_time,
            "activity_embedding": self.activity_embedding,
            "target_location_embedding": self.target_location_embedding,
            "context_location_embedding": self.context_location_embedding,
            "time_slot_embedding": self.time_slot_embedding
        }


class HmrmBaselineNew:
    def __init__(self, file=None, weight=0.5, K=7, embedding_size=50, num_workers=None,
                 checkpoint_dir=None, convergence_threshold=0.01, max_iterations=10):
        self.optimizer = Optimizer(num_workers=num_workers)
        self.input_file = file
        self.weight = weight
        self.K = K
        self.embedding_size = embedding_size
        self.checkpoint_dir = checkpoint_dir
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations

    def start(self):
        logger.info(f"Starting HMRM with file: {self.input_file}")
        start_time = datetime.now()

        # Read input file in chunks for large files
        chunksize = 1000000  # Adjust based on available memory
        file_size = os.path.getsize(self.input_file)

        if file_size > 500 * 1024 * 1024:  # If file > 500MB
            logger.info(f"Large file detected ({file_size / 1024 / 1024:.1f} MB), reading in chunks")
            chunks = []
            for chunk in tqdm(pd.read_csv(self.input_file, chunksize=chunksize), desc="Reading file chunks"):
                chunks.append(chunk)
            users_checkin = pd.concat(chunks)
        else:
            users_checkin = pd.read_csv(self.input_file, index_col=False)

        # Clean data
        users_checkin = users_checkin.dropna(axis=1)

        # Store original user and place IDs
        usersid = users_checkin.userid
        placeid_mapping = dict(zip(range(users_checkin['placeid'].unique().size), users_checkin['placeid'].unique()))

        # Factorize IDs for more efficient processing
        logger.info("Factorizing user and place IDs")
        users_checkin.userid = pd.factorize(users_checkin.userid)[0].astype(int)
        users_checkin.placeid = pd.factorize(users_checkin.placeid)[0].astype(int)

        # Start optimization
        logger.info(f"Starting HMRM optimization with K={self.K}, embedding_size={self.embedding_size}")
        optimization_result = self.optimizer.start(
            users_checkin,
            self.weight,
            self.K,
            self.embedding_size,
            max_iterations=self.max_iterations,
            convergence_threshold=self.convergence_threshold,
            checkpoint_dir=self.checkpoint_dir
        )

        # Create result dataframe
        logger.info("Creating result dataframe")
        df = pd.DataFrame(
            data=np.concatenate(
                (
                    optimization_result["context_location_embedding"],
                    optimization_result["target_location_embedding"],
                ),
                axis=1,
            )
        )

        # Add additional information
        try:
            values = []
            for i in tqdm(range(df.shape[0]), desc="Adding categories"):
                category = users_checkin[users_checkin["placeid"] == i]["category"].unique()
                if len(category) > 0:
                    values.append(category[0])
                else:
                    values.append(None)

            df["category"] = values
            df['placeid'] = list(map(lambda x: placeid_mapping[x], range(df.shape[0])))

        except Exception as e:
            logger.error(f'Error adding categories: {e}')

        # Log execution summary
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        logger.info(f"HMRM completed in {total_time:.2f} seconds")

        return df


# Import psutil for memory monitoring
try:
    import psutil
except ImportError:
    # Define a simplified version if psutil is not available
    class PsutilMock:
        @staticmethod
        def virtual_memory():
            class MemInfo:
                def __init__(self):
                    self.available = 32 * 1024 * 1024 * 1024  # Assume 8GB

            return MemInfo()


    psutil = PsutilMock()
