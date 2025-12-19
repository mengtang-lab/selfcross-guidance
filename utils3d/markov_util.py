import numpy as np
import torch
import torch.nn.functional as F

def adjust_temperature_and_ipf(attention: torch.Tensor,
                               temperature=0.65):
    # bh, bw, h, w = attention.shape
    # Change temperature and apply IPF
    attn = change_temperature(attention, temperature=temperature)
    attn = matrix_ipf(attn, iterations=200)
    return attn

def change_temperature(x: torch.Tensor, temperature, dim=-1):
    """
    Applies the softmax along dim (default is last dimension)
    x: (..., C)
    """
    return torch.softmax(torch.log(x) / temperature, dim=dim)


def matrix_ipf(matrix: torch.Tensor, iterations=15):
    """
    matrix: (H, W)
    """
    for i in range(iterations):
        matrix = matrix / torch.sum(matrix, dim=0)[None, :]
        matrix = matrix / torch.sum(matrix, dim=1)[:, None]
    return matrix


def sobel_gradient(x: torch.Tensor):
    """
    :param x: (batch, channel, height, width)
    """
    # Based on https://discuss.pytorch.org/t/how-to-calculate-the-gradient-of-images/1407/5
    B, C, H, W = x.shape
    filter_x = torch.tensor([[[[1, 0, -1],
                               [2, 0, -2],
                               [1, 0, -1]]]], dtype=x.dtype, device=x.device)
    filter_y = torch.tensor([[[[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]]]], dtype=x.dtype, device=x.device)

    x_batched = x.reshape(B*C, 1, H, W)
    x_padded = F.pad(x_batched, (1, 1, 1, 1), 'reflect')
    G_x = F.conv2d(x_padded, filter_x).reshape(B, C, H, W)
    G_y = F.conv2d(x_padded, filter_y).reshape(B, C, H, W)
    G = torch.sqrt(torch.sum(G_x**2 + G_y**2, dim=1))
    return G


def get_integer_sobel_gradient(x: torch.Tensor, quantization_steps=1024):
    gradient = sobel_gradient(x)
    gradient = gradient / torch.max(gradient)
    gradient = (gradient * quantization_steps).to(torch.long)
    return gradient


def collect_segment_border_pixels(x: torch.Tensor, distances: torch.Tensor):
    """
    Collects for each possible threshold of distances the segment borders pixel values.
    :param x: (channel, height, width). Should be type long because it will use cumsum
    :param distances: (height, width). Same height and width as x. Should be float64 or long (without duplicate values) to give more precise borders
    :return: Each list element describes the state at which it and all its before elements belong to the foreground segment.
    """

    def _inverse_permutation(perm):
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.size(0), device=perm.device)
        return inv

    C, H, W = x.shape
    ordered_indices = torch.argsort(distances.reshape(-1), dim=0, descending=False, stable=True)
    outputs_sequential = distances.reshape(-1)[ordered_indices].clone()

    # -----------------------
    # Estimate Border Lengths
    # -----------------------
    # First we need to find which direct neighbour pixel would have the lowest value, if it is lower than
    # the pixel itself. We want to know the index of this pixel in the ordered_indices
    ord_indices_map = _inverse_permutation(ordered_indices).reshape((H, W))
    got_added_by = ord_indices_map.clone()
    d_smaller = ord_indices_map[1:, :] < got_added_by[:-1, :]
    got_added_by[:-1, :][d_smaller] = ord_indices_map[1:, :][d_smaller]
    r_smaller = ord_indices_map[:, 1:] < got_added_by[:, :-1]
    got_added_by[:, :-1][r_smaller] = ord_indices_map[:, 1:][r_smaller]
    l_smaller = ord_indices_map[:, :-1] < got_added_by[:, 1:]
    got_added_by[:, 1:][l_smaller] = ord_indices_map[:, :-1][l_smaller]
    u_smaller = ord_indices_map[:-1, :] < got_added_by[1:, :]
    got_added_by[1:, :][u_smaller] = ord_indices_map[:-1, :][u_smaller]

    # Now that we know which idx adds to the list, we want to count them for each cell to know their contribution
    border = torch.zeros((H, W), dtype=torch.long, device=distances.device)
    mask_pixel_was_on_border_stack = ord_indices_map != got_added_by
    mask_pixel_adds_right_neighbor = ord_indices_map[:, :-1] == got_added_by[:, 1:]
    mask_pixel_adds_bottom_neighbor = ord_indices_map[:-1, :] == got_added_by[1:, :]
    mask_pixel_adds_left_neighbor = ord_indices_map[:, 1:] == got_added_by[:, :-1]
    mask_pixel_adds_upper_neighbor = ord_indices_map[1:, :] == got_added_by[:-1, :]
    border[mask_pixel_was_on_border_stack] -= 1                                 # If it belonged to
    border[:, :-1][mask_pixel_adds_right_neighbor] += 1                         # If right neighbor bigger or equal (equal because of stable sort)
    border[:-1, :][mask_pixel_adds_bottom_neighbor] += 1                        # If bottom neighbor bigger or equal (equal ...)
    border[:, 1:][mask_pixel_adds_left_neighbor] += 1                           # If left neighbor is bigger
    border[1:, :][mask_pixel_adds_upper_neighbor] += 1                          # If upper neighbor is bigger
    border_lengths = torch.cumsum(border.reshape(-1)[ordered_indices], dim=0)   # How much border surrounds the segment after (at index) pixel is considered part of the foreground segment

    # Now collecting the given data from x along the borders
    x_border = torch.zeros((C, H, W), dtype=torch.long, device=distances.device)
    x_border[:, mask_pixel_was_on_border_stack] -= x[:, mask_pixel_was_on_border_stack]                             # Taking it from the stack
    x_border[:, :, :-1][:, mask_pixel_adds_right_neighbor] += x[:, :, 1:][:, mask_pixel_adds_right_neighbor]        # If right neighbor bigger or equal (equal because of stable sort)
    x_border[:, :-1, :][:, mask_pixel_adds_bottom_neighbor] += x[:, 1:, :][:, mask_pixel_adds_bottom_neighbor]      # If bottom neighbor bigger or equal (equal ...)
    x_border[:, :, 1:][:, mask_pixel_adds_left_neighbor] += x[:, :, :-1][:, mask_pixel_adds_left_neighbor]          # If left neighbor is bigger
    x_border[:, 1:, :][:, mask_pixel_adds_upper_neighbor] += x[:, :-1, :][:, mask_pixel_adds_upper_neighbor]        # If upper neighbor is bigger
    x_border_csum = torch.cumsum(x_border.reshape(C, H * W)[:, ordered_indices], dim=1)                             # At each index, what is the segment's sum of all border pixels of x values

    return {
        'values': x_border_csum,                        # At each idx: The summed up values of x of all border pixels. The border is the surrounding of the segment consisting of pixel at idx and its previous indices' pixels
        'distances': outputs_sequential,                # The distance of the current added pixel at this index
        'border_lengths': border_lengths,               # At each idx: How many pixels the border is made of (only and allways is zero at the last idx)
        'project_to_image_indices': ord_indices_map,
        'project_to_list_indices': ordered_indices,
    }


def normal_pdf_unscaled(x: torch.Tensor, std=1.0):
    return torch.exp(-0.5 * torch.square(x / std))


def normal_pdf_np(x: torch.Tensor, std=1.0):
    c = 1 / (std * np.sqrt(2 * np.pi))
    return c * np.exp(-0.5 * np.square(x / std))


def joint_bilateral_upsampling(x_lowres: torch.Tensor, reference_highres: torch.Tensor, spatial_sigma=1.0, range_sigma=0.1):
    """
    This implementation is based on this paper: https://johanneskopf.de/publications/jbu/paper/FinalPaper_0185.pdf with 2 adjustments
    1. We sample the x_lowres dense, based with bilinear interpolation
    2. We allow the reference to have multiple channels and we use an isotropic gaussian (range_sigma * I) instead of a 1D gaussian

    :param x_lowres: solution / features we want to upscale. Shape should be: (batch, channel, height, width)
    :param reference_highres: Reference image with the target resolution. Shape should be: (batch, reference_channels, targetHeight, targetWidth)
    :param spatial_sigma: scale or std of the spatial gaussian. It is in the lowres image coordinates and deciding how the lowres x pixels are weight (how big the influence)
    :param range_sigma: how big the std should be in the color range. If colors are closer they have higher weight (Default here is set for reference_highres values in the range from 0 to 1)
    :return: upscaled x_lowres
    """
    _, _, lowH, lowW = x_lowres.shape

    # Make lowres same shape as highres
    upscale_factor_x = reference_highres.shape[3] / x_lowres.shape[3]
    upscale_factor_y = reference_highres.shape[2] / x_lowres.shape[2]
    x_highres = F.interpolate(x_lowres, reference_highres.shape[2:], mode='bilinear')
    B, C, H, W = x_highres.shape

    # -----------------
    # Main spatial loop
    # -----------------
    # Because of potential memory limitations, we do this in for loops
    # We aim for 3 spatial sigmas to get 99.7 percent of spatial information (along one axis)
    # This is different to the paper where they set this to a fixed 5x5 window
    max_sigma = 3
    max_radius_x = max_sigma * upscale_factor_x * spatial_sigma
    max_radius_y = max_sigma * upscale_factor_y * spatial_sigma
    out_sum = torch.zeros((B, C, H, W), dtype=x_highres.dtype, device=x_highres.device)
    out_weight = torch.zeros((B, H, W), dtype=x_highres.dtype, device=x_highres.device)
    for dy in range(-int(max_radius_y + 1), int(max_radius_y + 2)):
        spatial_y = dy / (upscale_factor_y * spatial_sigma)
        for dx in range(-int(max_radius_x + 1), int(max_radius_x + 2)):
            spatial_x = dx / (upscale_factor_x * spatial_sigma)

            # Ignore points outside 3 sigma
            if spatial_x ** 2 + spatial_y ** 2 > max_sigma ** 2:
                continue

            # ----------------
            # Calculate shifts
            # ----------------
            in_pad_y = max(0, dy)
            in_pad_x = max(0, dx)
            in_end_y = H - max(0, -dy)
            in_end_x = W - max(0, -dx)
            out_pad_y = max(0, -dy)
            out_pad_x = max(0, -dx)
            out_end_y = H - max(0, dy)
            out_end_x = W - max(0, dx)

            # -----------
            # Get patches
            # -----------
            ref_in_patch = reference_highres[:, :, in_pad_y:in_end_y, in_pad_x:in_end_x]
            ref_out_patch = reference_highres[:, :, out_pad_y:out_end_y, out_pad_x:out_end_x]

            # ----------------
            # Calculate weight
            # ----------------
            spatial_weight = float(normal_pdf_np(spatial_x) * normal_pdf_np(spatial_y))
            range_weight = torch.prod(normal_pdf_unscaled(ref_in_patch - ref_out_patch, std=range_sigma), dim=1)
            weight = spatial_weight * range_weight

            # ------
            # Sum up
            # ------
            out_weight[:, out_pad_y:out_end_y, out_pad_x:out_end_x] += weight
            out_sum[:, :, out_pad_y:out_end_y, out_pad_x:out_end_x] += weight[:, None, :, :] * x_highres[:, :, in_pad_y:in_end_y, in_pad_x:in_end_x]

    # Calculate final output
    out = out_sum / out_weight[:, None, :, :]
    return out


def create_single_point_heatmap(point: [tuple, np.ndarray],
                                in_shape: tuple,
                                out_shape: tuple,
                                dtype,
                                device):
    """
    Draws the selected point on an empty heatmap with the dimensions of the given attention tensor.
    It also puts it on the same device and uses the same datatype as the given attention tensor.

    :param point: 2D array or tuple of image coordinates in img space
    :param in_shape: Shape of the image (inH, inW) in which the point was clicked
    :param out_shape: Resolution of the output shape (outH, outW)
    :return: Segmentation map in the shape of (lowH, lowW) which only segments the closest point
    """
    attn_coords = (np.array(out_shape)[::-1] - 1) * np.array(point) / np.array(in_shape)[::-1]
    x0 = min(max(0, int(attn_coords[0])), out_shape[1] - 1)
    y0 = min(max(0, int(attn_coords[1])), out_shape[0] - 1)
    x1 = min(x0 + 1, out_shape[1] - 1)
    y1 = min(y0 + 1, out_shape[0] - 1)
    factor_x = attn_coords[0] - x0
    factor_y = attn_coords[1] - y0
    inv_x = 1.0 - factor_x
    inv_y = 1.0 - factor_y
    # print('point',point,out_shape,in_shape,attn_coords)
    # print(x0,y0,x1,y1,factor_x,factor_y)
    seg_map = torch.zeros((out_shape[0], out_shape[1]), dtype=dtype, device=device)
    seg_map[y0, x0] += inv_x * inv_y  # near pixels share the whole value of the point
    seg_map[y0, x1] += factor_x * inv_y
    seg_map[y1, x0] += inv_x * factor_y
    seg_map[y1, x1] += factor_x * factor_y

    # We normalize to get a segmentation map with max value of 1
    seg_map /= torch.max(seg_map)
    return seg_map