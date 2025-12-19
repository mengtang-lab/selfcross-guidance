import numpy as np
import torch

from utils3d.markov_util import create_single_point_heatmap, joint_bilateral_upsampling
from numba import njit

@njit
def heap_push(heap, size, threshold, y, x):
    """
    Push a new element into the binary heap.
    """
    heap[size, 0] = threshold
    heap[size, 1] = y
    heap[size, 2] = x
    current = size
    size += 1

    # Heapify up
    while current > 0:
        parent = (current - 1) // 2
        if heap[current, 0] < heap[parent, 0]:
            # Swap current with parent
            heap[current], heap[parent] = heap[parent].copy(), heap[current].copy()
            current = parent
        else:
            break

    return size


@njit
def heap_pop(heap, size):
    """
    Pop the smallest element from the binary heap.
    """
    top = heap[0].copy()
    size -= 1
    heap[0] = heap[size]

    # Heapify down
    current = 0
    while True:
        left = 2 * current + 1
        right = 2 * current + 2
        smallest = current

        if left < size and heap[left, 0] < heap[smallest, 0]:
            smallest = left
        if right < size and heap[right, 0] < heap[smallest, 0]:
            smallest = right

        if smallest != current:
            # Swap current with smallest child
            heap[current], heap[smallest] = heap[smallest].copy(), heap[current].copy()
            current = smallest
        else:
            break

    return top, size


@njit
def flood_fill_with_min_threshold(heatmap: np.ndarray, start_point):
    """
    Flood-fill algorithm to compute the minimum required threshold to reach each pixel.

    :param heatmap: 2D array of intensity values (height, width)
    :param start_point: Tuple of x and y position in integer image coordinates (x_pos, y_pos)
    :return: Map of the minimum required threshold to reach each pixel
    """
    # Get the dimensions of the heatmap
    height, width = heatmap.shape

    # Initialize an output array to store the minimum threshold for each pixel
    min_threshold_map = np.full((height, width), np.inf)

    # Create a queued array to keep track of pixels that have already been added to the heap
    queued = np.zeros(shape=(height, width), dtype=np.uint8)

    # Priority queue (custom binary heap)
    max_heap_size = height * width
    heap = np.empty((max_heap_size, 3), dtype=np.float64)
    heap_size = 0

    # Get the intensity value of the starting pixel
    start_x, start_y = start_point
    start_intensity = heatmap[start_y, start_x]
    queued[start_y, start_x] = 1

    # Push the starting pixel to the heap with 0 threshold
    heap_size = heap_push(heap, heap_size, 0.0, start_y, start_x)  # simply set it to one, set the start threshold to zero
    # min_threshold_map[start_y, start_x] = 0.0

    # 4-connected neighbors (up, down, left, right)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Floodfill process
    repeats = 0
    while heap_size > 0:
        # Get the pixel with the current smallest threshold
        top, heap_size = heap_pop(heap, heap_size)
        current_threshold, y, x = top
        repeats += 1

        # Update the minimum threshold map
        y, x = int(y), int(x)
        min_threshold_map[y, x] = current_threshold

        # Explore 4-connected neighbors
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and not queued[ny, nx]:
                # Calculate the intensity difference between the current pixel and the neighbor and not discovered previously
                neighbor_intensity = heatmap[ny, nx]
                threshold = max(current_threshold, abs(neighbor_intensity - start_intensity))

                # Add the neighbor to the heap
                heap_size = heap_push(heap, heap_size, threshold, ny, nx)
                queued[ny, nx] = 1

    return min_threshold_map


def create_semantic_markov_map_from_start_state(A: torch.Tensor, start_state, max_iterations=50, threshold=0.3, linear_interpolation=True):
    """
    Creates the semantic markov map vector from a given start state and transition matrix A

    :param A: doubly stochastic matrix of size (r, r)
    :param start_state: probability start vector of size r.
    :param max_iterations: The maximum number of Markov chain iterations
    :param threshold: Relative probability threshold to measure the saturation to the uniform distribution
    :param linear_interpolation: Whether to perform linear interpolation between consecutive time steps, or not
    :return: The iteration count as vector
    """
    iteration_tracker = torch.zeros_like(start_state)
    not_yet_passed_mask = torch.ones_like(start_state, dtype=torch.bool)
    iteration_tracker[:] = max_iterations if linear_interpolation else max_iterations + 1
    # Initialize iteration zero
    state = start_state.clone()
    not_yet_passed_mask[state > threshold] = False  # almost all True at this moment
    iteration_tracker[state > threshold] = 0  # almost all max_iterations at this moment, record at which iter it's reached
    # for i in range(state.shape[0]):
    #     if state[i]>0: print(state[i])  # state.shape[0]
    # Iterate Markov chain
    for i in range(max_iterations):
        prev_state = state.clone()

        # Calculate new state
        state = state @ A
        state = state / torch.max(state)

        # Get mask on where to store new result
        now_passed_mask = state > threshold
        update_mask = torch.logical_and(not_yet_passed_mask, now_passed_mask)  # return the position
        not_yet_passed_mask[update_mask] = False

        # Update
        if linear_interpolation:
            state_distance_to_threshold = state - threshold
            state_delta = (1 - threshold) if i == 0 else (state - prev_state)
            state_threshold_gradient = 1 - state_distance_to_threshold / (state_delta + 0.000001)
            iteration_tracker[update_mask] = (float(i) + state_threshold_gradient)[update_mask]
        else:
            iteration_tracker[update_mask] = float(i + 1)

        if torch.sum(not_yet_passed_mask) == 0:
            break

    # Finally, we fill empty spots with the maximum iterations
    return iteration_tracker


def create_markov_map_from_point(A_tensor: torch.Tensor,
                                 point: tuple,
                                 relative_probability_threshold=0.3,
                                 max_iterations=50,
                                 jbu_spatial_sigma=1.0,
                                 jbu_range_sigma=0.1,
                                 use_flood_fill=False):
    """
    :param jbu_spatial_sigma:
    :param relative_probability_threshold:
    :param jbu_range_sigma:
    :param use_flood_fill:
    :param image: (H, W, 3), numpy array of np.uint8 datatype with values from 0 to 255
    :param A_tensor: (h, w, h, w)
    :param point: 2D point in image space
    :return:
    """
    h, w = A_tensor.shape
    
    # # Get the start state from a single point
    # start_state = create_single_point_heatmap(
    #     point=point,
    #     in_shape=(h, w),
    #     out_shape=(h, w),
    #     dtype=A_tensor.dtype,
    #     device=A_tensor.device
    # )

    start_state = torch.zeros(h, dtype=A_tensor.dtype,  device=A_tensor.device)
    index = point[0] * np.sqrt(h) + point[1]
    start_state[index.astype(int)] = 1  

    # Create semantic Markov-map
    lowres_semantic_markov_map = create_semantic_markov_map_from_start_state(
        A=A_tensor,
        start_state=start_state,
        max_iterations=max_iterations,
        threshold=relative_probability_threshold
    ).reshape(64,64)

    # Upscaling
    # img_tensor = torch.tensor(image / 255.0, dtype=torch.float, device=A_tensor.device).permute((2, 0, 1))[None]
    # highres_semantic_markov_map = joint_bilateral_upsampling(lowres_semantic_markov_map[None, None], img_tensor, jbu_spatial_sigma, jbu_range_sigma)[0, 0]

    # Flood fill
    if use_flood_fill:
        return torch.tensor(flood_fill_with_min_threshold(
            heatmap=lowres_semantic_markov_map.cpu().numpy(),
            start_point=point
        ), dtype=A_tensor.dtype, device=A_tensor.device)
    else:
        return lowres_semantic_markov_map


