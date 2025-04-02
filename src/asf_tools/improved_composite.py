"""Create a local-resolution-weighted composite from Sentinel-1 RTC products.

Create a local-resolution-weighted composite from a set of Sentinel-1 RTC
products (D. Small, 2012) with improved handling for missing area files and
reduced gaps in the final composite. The local resolution, defined as the inverse of the
local contributing (scattering) area, is used to weight each RTC products'
contributions to the composite image on a pixel-by-pixel basis. The composite image
is created as a Cloud Optimized GeoTIFF (COG). Additionally, a COG specifying
the number of rasters contributing to each composite pixel is created.

References:
    David Small, 2012: <https://doi.org/10.1109/IGARSS.2012.6350465>
"""

import argparse
import logging
import os
import sys
from statistics import multimode
from tempfile import TemporaryDirectory

import numpy as np
from osgeo import gdal
from scipy import ndimage

from asf_tools.raster import read_as_array, write_cog
from asf_tools.util import get_epsg_code


gdal.UseExceptions()
log = logging.getLogger(__name__)


def get_target_epsg_code(codes: list[int]) -> int:
    """Determine the target UTM EPSG projection for the output composite

    Args:
        codes: List of UTM EPSG codes

    Returns:
        target: UTM EPSG code
    """
    # use median east/west UTM zone of all files, regardless of hemisphere
    # UTM EPSG codes for each hemisphere will look like:
    #   North: 326XX
    #   South: 327XX
    valid_codes = list(range(32601, 32661)) + list(range(32701, 32761))
    if bad_codes := set(codes) - set(valid_codes):
        raise ValueError(f'Non UTM EPSG code encountered: {bad_codes}')

    hemispheres = [c // 100 * 100 for c in codes]
    # if even modes, choose lowest (North)
    target_hemisphere = min(multimode(hemispheres))

    zones = sorted([c % 100 for c in codes])
    # if even length, choose fist of median two
    target_zone = zones[(len(zones) - 1) // 2]

    return target_hemisphere + target_zone


def get_area_raster(raster: str) -> str:
    """Determine the path of the area raster for a given backscatter raster based on naming conventions for HyP3 RTC
    products

    Args:
        raster: path of the backscatter raster, e.g. S1A_IW_20181102T155531_DVP_RTC30_G_gpuned_5685_VV.tif

    Returns:
        area_raster: path of the area raster, e.g. S1A_IW_20181102T155531_DVP_RTC30_G_gpuned_5685_area.tif
    """
    return '_'.join(raster.split('_')[:-1] + ['area.tif'])


def get_full_extent(raster_info: dict):
    """Determine the corner coordinates and geotransform for the full extent of a set of rasters

    Args:
        raster_info: A dictionary of gdal.Info results for the set of rasters

    Returns:
        upper_left: The upper left corner of the extent as a tuple
        upper_right: The lower right corner of the extent as a tuple
        geotransform: The geotransform of the extent as a list
    """
    upper_left_corners = [info['cornerCoordinates']['upperLeft'] for info in raster_info.values()]
    lower_right_corners = [info['cornerCoordinates']['lowerRight'] for info in raster_info.values()]

    ulx = min([ul[0] for ul in upper_left_corners])
    uly = max([ul[1] for ul in upper_left_corners])
    lrx = max([lr[0] for lr in lower_right_corners])
    lry = min([lr[1] for lr in lower_right_corners])

    log.debug(f'Full extent raster upper left: ({ulx, uly}); lower right: ({lrx, lry})')

    trans = []
    for info in raster_info.values():
        # Only need info from any one raster
        trans = info['geoTransform']
        break

    trans[0] = ulx
    trans[3] = uly

    return (ulx, uly), (lrx, lry), trans


def check_area_file_exists(raster):
    """Check if an area file exists for the given raster

    Args:
        raster: Path to the raster file

    Returns:
        exists: Boolean indicating if the area file exists
    """
    area_raster = get_area_raster(raster)
    return os.path.exists(area_raster)


def estimate_area(values, resolution):
    """Estimate local area based on backscatter values when area file is missing

    This function creates an approximate area estimate based on typical SAR geometry.
    It uses a combination of value-based and distance-based weighting to approximate
    the local contributing area.

    Args:
        values: Backscatter values
        resolution: Pixel resolution in meters

    Returns:
        estimated_areas: Estimated area values
    """
    # Create a mask for valid data
    mask = values > 0
    
    # Start with a uniform area equal to pixel size squared
    # This is a baseline estimate for flat terrain
    base_area = resolution * resolution
    areas = np.ones_like(values) * base_area
    
    # Apply a gradient-based adjustment
    # Higher gradient typically means higher local area
    if np.any(mask):
        # Calculate gradient magnitude using Sobel operator
        gradient_x = ndimage.sobel(values, axis=1)
        gradient_y = ndimage.sobel(values, axis=0)
        gradient = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Normalize gradient to [0, 1] range
        if gradient.max() > 0:
            normalized_gradient = gradient / gradient.max()
            
            # Adjust area based on gradient (higher gradient = larger area)
            # Range from 0.5x to 2x base area
            area_factor = 0.5 + 1.5 * normalized_gradient
            areas[mask] = base_area * area_factor[mask]
    
    return areas


def reproject_to_target(raster_info: dict, target_epsg_code: int, target_resolution: float, directory: str) -> dict:
    """Reprojects a set of raster images to a common projection and resolution

    Args:
        raster_info: A dictionary of gdal.Info results for the set of rasters
        target_epsg_code: The integer EPSG code for the target projection
        target_resolution: The target resolution
        directory: The directory in which to create the reprojected files

    Returns:
        target_raster_info: An updated dictionary of gdal.Info results for the reprojected files
    """
    target_raster_info = {}
    for raster, info in raster_info.items():
        epsg_code = get_epsg_code(info)
        resolution = info['geoTransform'][1]
        
        # Add slightly expanded bounds to reduce gaps
        buffer_size = 2 * target_resolution  # 2-pixel buffer
        
        if epsg_code != target_epsg_code or resolution != target_resolution:
            log.info(f'Reprojecting {raster}')
            reprojected_raster = os.path.join(directory, os.path.basename(raster))
            
            # Get original bounds
            ulx, uly = info['cornerCoordinates']['upperLeft']
            lrx, lry = info['cornerCoordinates']['lowerRight']
            
            # Calculate expanded bounds
            expanded_ulx = ulx - buffer_size
            expanded_uly = uly + buffer_size
            expanded_lrx = lrx + buffer_size
            expanded_lry = lry - buffer_size
            
            gdal.Warp(
                reprojected_raster,
                raster,
                dstSRS=f'EPSG:{target_epsg_code}',
                xRes=target_resolution,
                yRes=target_resolution,
                outputBounds=[expanded_ulx, expanded_lry, expanded_lrx, expanded_uly],
                targetAlignedPixels=True,
                multithread=True,
            )

            # Check if area raster exists
            if check_area_file_exists(raster):
                area_raster = get_area_raster(raster)
                log.info(f'Reprojecting {area_raster}')
                reprojected_area_raster = os.path.join(directory, os.path.basename(area_raster))
                gdal.Warp(
                    reprojected_area_raster,
                    area_raster,
                    dstSRS=f'EPSG:{target_epsg_code}',
                    xRes=target_resolution,
                    yRes=target_resolution,
                    outputBounds=[expanded_ulx, expanded_lry, expanded_lrx, expanded_uly],
                    targetAlignedPixels=True,
                    multithread=True,
                )
            else:
                log.warning(f'Area raster not found for {raster}')

            target_raster_info[reprojected_raster] = gdal.Info(reprojected_raster, format='json')
        else:
            log.info(f'No need to reproject {raster}')
            target_raster_info[raster] = info

    return target_raster_info


def fill_small_gaps(data, max_size=3):
    """Fill small gaps in the data using interpolation
    
    Args:
        data: Numpy array of data
        max_size: Maximum size of gaps to fill (in pixels)
        
    Returns:
        filled_data: Numpy array with small gaps filled
    """
    # Create a mask of gaps (zeros in the data)
    mask = data == 0
    
    # Use a binary closing operation to identify small gaps
    structure = ndimage.generate_binary_structure(2, 2)  # Use 2x2 structure for 8-connectivity
    closed_mask = ndimage.binary_closing(~mask, structure=structure, iterations=max_size)
    
    # Identify small gaps (areas that are 1 in closed_mask but 0 in ~mask)
    small_gaps = np.logical_and(closed_mask, mask)
    
    # If there are no small gaps, return the original data
    if not np.any(small_gaps):
        return data
    
    # Create a copy of the data for filling
    filled_data = data.copy()
    
    # Use a Gaussian filter to interpolate values for small gaps
    # We apply it to the entire image but only use the results for the small gaps
    smoothed = ndimage.gaussian_filter(data, sigma=max_size/2)
    
    # Fill the small gaps with the smoothed values
    filled_data[small_gaps] = smoothed[small_gaps]
    
    return filled_data


def make_composite_improved(out_name: str, rasters: list[str], resolution: float | None = None, fill_gaps: bool = True):
    """Creates a local-resolution-weighted composite from Sentinel-1 RTC products with improved handling

    Args:
        out_name: The base name of the output GeoTIFFs
        rasters: A list of file paths of the images to composite
        resolution: The pixel size for the output GeoTIFFs
        fill_gaps: Whether to fill small gaps in the output

    Returns:
        out_raster: Path to the created composite backscatter GeoTIFF
        out_counts_raster: Path to the created GeoTIFF with counts of scenes contributing to each pixel
    """
    if not rasters:
        raise ValueError('Must specify at least one raster to composite')

    raster_info = {}
    for raster in rasters:
        raster_info[raster] = gdal.Info(raster, format='json')
        # Check area raster exists without failing if it doesn't
        if not check_area_file_exists(raster):
            log.warning(f'Area raster not found for {raster}')

    target_epsg_code = get_target_epsg_code([get_epsg_code(info) for info in raster_info.values()])
    log.debug(f'Composite projection is EPSG:{target_epsg_code}')

    if resolution is None:
        resolution = max([info['geoTransform'][1] for info in raster_info.values()])
    log.debug(f'Composite resolution is {resolution} meters')

    # resample rasters to maximum resolution & common UTM zone
    with TemporaryDirectory(prefix='reprojected_') as temp_dir:
        raster_info = reproject_to_target(
            raster_info,
            target_epsg_code=target_epsg_code,
            target_resolution=resolution,
            directory=temp_dir,
        )

        # Get extent of union of all images
        full_ul, full_lr, full_trans = get_full_extent(raster_info)

        nx = int(abs(full_ul[0] - full_lr[0]) // resolution)
        ny = int(abs(full_ul[1] - full_lr[1]) // resolution)

        outputs = np.zeros((ny, nx))
        weights = np.zeros(outputs.shape)
        counts = np.zeros(outputs.shape, dtype=np.int8)

        for raster, info in raster_info.items():
            log.info(f'Processing raster {raster}')
            log.debug(
                f'Raster upper left: {info["cornerCoordinates"]["upperLeft"]}; '
                f'lower right: {info["cornerCoordinates"]["lowerRight"]}'
            )

            values = read_as_array(raster)

            # Check if area raster exists and use it, otherwise estimate area
            area_raster = get_area_raster(raster)
            if os.path.exists(area_raster):
                log.info(f'Using area raster: {area_raster}')
                areas = read_as_array(area_raster)
            else:
                log.info(f'Estimating areas for {raster}')
                areas = estimate_area(values, resolution)

            ulx, uly = info['cornerCoordinates']['upperLeft']
            y_index_start = int((full_ul[1] - uly) // resolution)
            y_index_end = y_index_start + values.shape[0]

            x_index_start = int((ulx - full_ul[0]) // resolution)
            x_index_end = x_index_start + values.shape[1]

            # Ensure indices are within bounds
            y_index_start = max(0, y_index_start)
            y_index_end = min(ny, y_index_end)
            x_index_start = max(0, x_index_start)
            x_index_end = min(nx, x_index_end)
            
            # Adjust arrays if necessary
            values_height = y_index_end - y_index_start
            values_width = x_index_end - x_index_start
            
            if values_height <= 0 or values_width <= 0:
                log.warning(f"Raster {raster} is outside the composite bounds, skipping")
                continue
                
            if values_height < values.shape[0] or values_width < values.shape[1]:
                values = values[:values_height, :values_width]
                areas = areas[:values_height, :values_width]

            log.debug(
                f'Placing values in output grid at {y_index_start}:{y_index_end} and {x_index_start}:{x_index_end}'
            )

            # Use a more lenient mask for what's considered valid data
            mask = (values < 0.0001) | (areas <= 0)
            
            # Calculate weights with careful handling of zeros and small values
            raster_weights = np.zeros_like(areas, dtype=np.float32)
            valid_indices = ~mask
            
            if np.any(valid_indices):
                # For valid pixels, calculate 1/area but avoid division by zero
                min_valid_area = max(1e-10, np.min(areas[valid_indices]))
                safe_areas = np.maximum(areas, min_valid_area)
                raster_weights = 1.0 / safe_areas
                
                # Set weights for masked pixels to zero
                raster_weights[mask] = 0
                
                # Apply additional edge distance weighting to create smoother transitions
                # This gives higher weight to pixels further from the edge of valid data
                distance_weights = ndimage.distance_transform_edt(~mask)
                if distance_weights.max() > 0:
                    distance_weights = 0.5 + 0.5 * (distance_weights / distance_weights.max())
                    raster_weights *= distance_weights

            outputs[y_index_start:y_index_end, x_index_start:x_index_end] += values * raster_weights
            weights[y_index_start:y_index_end, x_index_start:x_index_end] += raster_weights
            counts[y_index_start:y_index_end, x_index_start:x_index_end] += ~mask

    # Divide by the total weight applied
    mask = weights > 0
    outputs[mask] /= weights[mask]
    del weights
    
    # Fill small gaps if requested
    if fill_gaps:
        log.info("Filling small gaps in the composite")
        outputs = fill_small_gaps(outputs, max_size=3)

    out_raster = write_cog(
        f'{out_name}.tif', 
        outputs, 
        full_trans, 
        target_epsg_code, 
        nodata_value=0,
        options=[
            'COMPRESS=DEFLATE', 
            'PREDICTOR=2', 
            'TILED=YES',
            'BLOCKXSIZE=512', 
            'BLOCKYSIZE=512',
            'BIGTIFF=YES'
        ]
    )
    del outputs

    out_counts_raster = write_cog(
        f'{out_name}_counts.tif',
        counts,
        full_trans,
        target_epsg_code,
        dtype=gdal.GDT_Int16,
        options=[
            'COMPRESS=DEFLATE', 
            'PREDICTOR=2', 
            'TILED=YES',
            'BLOCKXSIZE=512', 
            'BLOCKYSIZE=512',
            'BIGTIFF=YES'
        ]
    )
    del counts

    return out_raster, out_counts_raster


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('out_name', help='Base name of output composite GeoTIFF (without extension)')
    parser.add_argument('rasters', nargs='+', help='Sentinel-1 GeoTIFF rasters to composite')
    parser.add_argument(
        '-r',
        '--resolution',
        type=float,
        help='Desired output resolution in meters (default is the max resolution of all the input files)',
    )
    parser.add_argument(
        '--no-fill-gaps',
        action='store_true',
        help='Disable small gap filling (default is to fill small gaps)',
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='Turn on verbose logging')
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        stream=sys.stdout,
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=level,
    )
    log.debug(' '.join(sys.argv))
    log.info(f'Creating a composite of {len(args.rasters)} rasters')

    raster, counts = make_composite_improved(
        args.out_name, 
        args.rasters, 
        args.resolution,
        fill_gaps=not args.no_fill_gaps
    )

    log.info(f'Composite created successfully: {raster}')
    log.info(f'Number of rasters contributing to each pixel: {counts}')


if __name__ == "__main__":
    main()
