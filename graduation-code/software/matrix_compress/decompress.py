from __future__ import annotations

import struct
from typing import Dict, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp


class DecompressionError(Exception):
    """Exception raised for errors during decompression."""
    pass


def decompress_matrix(input_path: str, format: Optional[str] = None) -> sp.spmatrix:
    """
    Decompress a matrix from the specified format.
    
    Args:
        input_path: Path to the compressed file
        format: The compression format ("coo", "csr", "bscr"). If None, auto-detect.
    
    Returns:
        A scipy sparse matrix
    
    Raises:
        DecompressionError: If decompression fails
    """
    try:
        with open(input_path, 'rb') as f:
            # Read magic number to detect format if not specified
            magic_number = struct.unpack('<I', f.read(4))[0]
            
            if format is None:
                if magic_number == 0x434F4F:
                    format = "coo"
                elif magic_number == 0x435352:
                    format = "csr"
                elif magic_number == 0x42534352:
                    format = "bscr"
                else:
                    raise DecompressionError(f"Unknown format: magic number {magic_number}")
            else:
                # Verify magic number matches specified format
                expected_magic = {
                    "coo": 0x434F4F,
                    "csr": 0x435352,
                    "bscr": 0x42534352
                }.get(format.lower())
                
                if expected_magic is None:
                    raise DecompressionError(f"Unsupported format: {format}")
                
                if magic_number != expected_magic:
                    raise DecompressionError(f"Magic number mismatch for format {format}")
            
            # Reset file pointer to start
            f.seek(0)
            
            # Decompress based on format
            if format.lower() == "coo":
                return _decompress_coo(f)
            elif format.lower() == "csr":
                return _decompress_csr(f)
            elif format.lower() == "bscr":
                return _decompress_bscr(f)
            else:
                raise DecompressionError(f"Unsupported format: {format}")
                
    except Exception as e:
        if isinstance(e, DecompressionError):
            raise
        raise DecompressionError(f"Error decompressing matrix: {str(e)}")


def _decompress_coo(file_handle) -> sp.spmatrix:
    """
    Decompress a matrix from COO format.
    
    Args:
        file_handle: Open file handle to the compressed file
    
    Returns:
        A scipy COO matrix
    """
    # Read header
    magic_number = struct.unpack('<I', file_handle.read(4))[0]
    n = struct.unpack('<I', file_handle.read(4))[0]
    nnz = struct.unpack('<I', file_handle.read(4))[0]
    dtype = struct.unpack('<I', file_handle.read(4))[0]
    
    # Read data
    row_data = np.frombuffer(file_handle.read(nnz * 4), dtype=np.int32)
    col_data = np.frombuffer(file_handle.read(nnz * 4), dtype=np.int32)
    val_data = np.frombuffer(file_handle.read(nnz * 4), dtype=np.float32)
    
    # Create COO matrix
    matrix = sp.coo_matrix((val_data, (row_data, col_data)), shape=(n, n))
    return matrix


def _decompress_csr(file_handle) -> sp.spmatrix:
    """
    Decompress a matrix from CSR format.
    
    Args:
        file_handle: Open file handle to the compressed file
    
    Returns:
        A scipy CSR matrix
    """
    # Read header
    magic_number = struct.unpack('<I', file_handle.read(4))[0]
    n = struct.unpack('<I', file_handle.read(4))[0]
    nnz = struct.unpack('<I', file_handle.read(4))[0]
    dtype = struct.unpack('<I', file_handle.read(4))[0]
    
    # Read data
    indptr = np.frombuffer(file_handle.read((n + 1) * 4), dtype=np.int32)
    indices = np.frombuffer(file_handle.read(nnz * 4), dtype=np.int32)
    data = np.frombuffer(file_handle.read(nnz * 4), dtype=np.float32)
    
    # Create CSR matrix
    matrix = sp.csr_matrix((data, indices, indptr), shape=(n, n))
    return matrix


def _decompress_bscr(file_handle) -> sp.spmatrix:
    """
    Decompress a matrix from BSCR format.
    
    Args:
        file_handle: Open file handle to the compressed file
    
    Returns:
        A scipy CSR matrix
    """
    # Read header
    magic_number = struct.unpack('<I', file_handle.read(4))[0]
    n = struct.unpack('<I', file_handle.read(4))[0]
    num_blocks = struct.unpack('<I', file_handle.read(4))[0]
    dtype = struct.unpack('<I', file_handle.read(4))[0]
    block_size = struct.unpack('<I', file_handle.read(4))[0]
    
    num_block_rows = (n + block_size - 1) // block_size
    
    # Read data
    block_indptr = np.frombuffer(file_handle.read((num_block_rows + 1) * 4), dtype=np.int32)
    block_indices = np.frombuffer(file_handle.read(num_blocks * 4), dtype=np.int32)
    
    # Read block data
    block_data = []
    for _ in range(num_blocks):
        block = np.frombuffer(file_handle.read(block_size * block_size * 4), dtype=np.float32)
        block = block.reshape((block_size, block_size))
        block_data.append(block)
    
    # Create matrix
    matrix = sp.lil_matrix((n, n), dtype=np.float32)
    
    for block_row in range(num_block_rows):
        row_start = block_row * block_size
        row_end = min((block_row + 1) * block_size, n)
        
        start_idx = block_indptr[block_row]
        end_idx = block_indptr[block_row + 1]
        
        for i in range(start_idx, end_idx):
            block_col = block_indices[i]
            col_start = block_col * block_size
            col_end = min((block_col + 1) * block_size, n)
            
            block = block_data[i - start_idx]
            
            # Copy block data to matrix
            matrix[row_start:row_end, col_start:col_end] = block[:row_end-row_start, :col_end-col_start]
    
    return matrix.tocsr()


def validate_compression(original: sp.spmatrix, decompressed: sp.spmatrix) -> Dict[str, any]:
    """
    Validate that decompressed matrix matches original.
    
    Args:
        original: The original matrix
        decompressed: The decompressed matrix
    
    Returns:
        A dictionary with validation results
    """
    # Check shape
    shape_match = original.shape == decompressed.shape
    
    # Check nnz
    nnz_match = original.nnz == decompressed.nnz
    
    # Check values
    if shape_match:
        # Convert to CSR for comparison
        original_csr = original.tocsr()
        decompressed_csr = decompressed.tocsr()
        
        # Check indices and data
        indices_match = np.array_equal(original_csr.indices, decompressed_csr.indices)
        indptr_match = np.array_equal(original_csr.indptr, decompressed_csr.indptr)
        data_match = np.allclose(original_csr.data, decompressed_csr.data)
        
        values_match = indices_match and indptr_match and data_match
    else:
        values_match = False
    
    # Calculate error statistics if values don't match exactly
    if not values_match and shape_match:
        # Compute maximum absolute error
        diff = original - decompressed
        max_error = np.max(np.abs(diff.data)) if diff.nnz > 0 else 0.0
        
        # Compute mean absolute error
        mean_error = np.mean(np.abs(diff.data)) if diff.nnz > 0 else 0.0
    else:
        max_error = 0.0
        mean_error = 0.0
    
    return {
        "shape_match": shape_match,
        "nnz_match": nnz_match,
        "values_match": values_match,
        "max_error": max_error,
        "mean_error": mean_error,
        "is_valid": shape_match and nnz_match and values_match
    }


def decompress_to_file(
    input_path: str,
    output_path: str,
    format: Optional[str] = None,
    mat_var_name: str = "A"
) -> Dict[str, any]:
    """
    Decompress a matrix and save it to a .mat file.
    
    Args:
        input_path: Path to the compressed file
        output_path: Path to save the .mat file
        format: The compression format ("coo", "csr", "bscr"). If None, auto-detect.
        mat_var_name: Variable name for the matrix in the .mat file
    
    Returns:
        A dictionary with decompression statistics
    """
    import scipy.io as sio
    
    matrix = decompress_matrix(input_path, format)
    
    # Save to .mat file
    sio.savemat(output_path, {mat_var_name: matrix.toarray()})
    
    # Return statistics
    return {
        "matrix_shape": matrix.shape,
        "nnz": matrix.nnz,
        "density": matrix.nnz / (matrix.shape[0] * matrix.shape[1])
    }
