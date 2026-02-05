from __future__ import annotations

import os
import struct
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.io as sio
import scipy.sparse as sp


class CompressionError(Exception):
    """Exception raised for errors during compression."""
    pass


def read_mat_file(file_path: str) -> sp.spmatrix:
    """
    Read a matrix from a .mat file.
    
    Args:
        file_path: Path to the .mat file
    
    Returns:
        A scipy sparse matrix
    
    Raises:
        CompressionError: If the file cannot be read or does not contain a matrix
    """
    try:
        mat_contents = sio.loadmat(file_path)
        # Find the first matrix in the file
        for key, value in mat_contents.items():
            if isinstance(value, np.ndarray) and value.ndim == 2:
                # Convert to sparse matrix if dense
                if isinstance(value, np.ndarray):
                    return sp.csr_matrix(value)
                return value
        raise CompressionError("No matrix found in .mat file")
    except Exception as e:
        raise CompressionError(f"Error reading .mat file: {str(e)}")


def compress_matrix(
    matrix: sp.spmatrix,
    format: str,
    output_path: str,
    block_size: Optional[int] = 16
) -> Dict[str, any]:
    """
    Compress a sparse matrix to the specified format.
    
    Args:
        matrix: The sparse matrix to compress
        format: The compression format ("coo", "csr", "bscr")
        output_path: Path to save the compressed file
        block_size: Block size for BSCR format (default: 16)
    
    Returns:
        A dictionary with compression statistics
    
    Raises:
        CompressionError: If compression fails
    """
    try:
        # Convert to CSR format for processing
        if not isinstance(matrix, sp.csr_matrix):
            matrix = matrix.tocsr()
        
        # Validate matrix
        if matrix.shape[0] != matrix.shape[1]:
            raise CompressionError("Only square matrices are supported")
        
        # Get matrix properties
        n = matrix.shape[0]
        nnz = matrix.nnz
        density = nnz / (n * n)
        
        # Compress based on format
        if format.lower() == "coo":
            compression_stats = _compress_coo(matrix, output_path)
        elif format.lower() == "csr":
            compression_stats = _compress_csr(matrix, output_path)
        elif format.lower() == "bscr":
            compression_stats = _compress_bscr(matrix, output_path, block_size)
        else:
            raise CompressionError(f"Unsupported format: {format}")
        
        # Add general statistics
        compression_stats.update({
            "original_size": n * n * 4,  # Assuming float32
            "original_nnz": nnz,
            "original_density": density,
            "matrix_size": n
        })
        
        return compression_stats
        
    except Exception as e:
        if isinstance(e, CompressionError):
            raise
        raise CompressionError(f"Error compressing matrix: {str(e)}")


def _compress_coo(matrix: sp.csr_matrix, output_path: str) -> Dict[str, any]:
    """
    Compress matrix to COO format.
    
    Args:
        matrix: The CSR matrix to compress
        output_path: Path to save the compressed file
    
    Returns:
        Compression statistics
    """
    # Convert to COO format
    coo_matrix = matrix.tocoo()
    
    # Prepare data
    n = matrix.shape[0]
    nnz = coo_matrix.nnz
    
    # Calculate file size
    header_size = 16  # 4 bytes for format, 4 for n, 4 for nnz, 4 for dtype
    data_size = nnz * (4 + 4 + 4)  # row, col, value (each 4 bytes)
    total_size = header_size + data_size
    
    # Write to file
    with open(output_path, 'wb') as f:
        # Write header
        f.write(struct.pack('<I', 0x434F4F))  # 'COO' magic number
        f.write(struct.pack('<I', n))
        f.write(struct.pack('<I', nnz))
        f.write(struct.pack('<I', 0))  # dtype: 0 for float32
        
        # Write data
        f.write(coo_matrix.row.astype(np.int32).tobytes())
        f.write(coo_matrix.col.astype(np.int32).tobytes())
        f.write(coo_matrix.data.astype(np.float32).tobytes())
    
    # Calculate compression ratio
    original_size = n * n * 4
    compression_ratio = original_size / total_size
    
    return {
        "format": "COO",
        "compressed_size": total_size,
        "compression_ratio": compression_ratio,
        "block_size": None
    }


def _compress_csr(matrix: sp.csr_matrix, output_path: str) -> Dict[str, any]:
    """
    Compress matrix to CSR format.
    
    Args:
        matrix: The CSR matrix to compress
        output_path: Path to save the compressed file
    
    Returns:
        Compression statistics
    """
    # Prepare data
    n = matrix.shape[0]
    nnz = matrix.nnz
    
    # Calculate file size
    header_size = 16  # 4 bytes for format, 4 for n, 4 for nnz, 4 for dtype
    data_size = (n + 1) * 4 + nnz * 4 + nnz * 4  # indptr, indices, data
    total_size = header_size + data_size
    
    # Write to file
    with open(output_path, 'wb') as f:
        # Write header
        f.write(struct.pack('<I', 0x435352))  # 'CSR' magic number
        f.write(struct.pack('<I', n))
        f.write(struct.pack('<I', nnz))
        f.write(struct.pack('<I', 0))  # dtype: 0 for float32
        
        # Write data
        f.write(matrix.indptr.astype(np.int32).tobytes())
        f.write(matrix.indices.astype(np.int32).tobytes())
        f.write(matrix.data.astype(np.float32).tobytes())
    
    # Calculate compression ratio
    original_size = n * n * 4
    compression_ratio = original_size / total_size
    
    return {
        "format": "CSR",
        "compressed_size": total_size,
        "compression_ratio": compression_ratio,
        "block_size": None
    }


def _compress_bscr(
    matrix: sp.csr_matrix,
    output_path: str,
    block_size: int
) -> Dict[str, any]:
    """
    Compress matrix to BSCR (Block Compressed Sparse Row) format.
    
    Args:
        matrix: The CSR matrix to compress
        output_path: Path to save the compressed file
        block_size: Size of each block (block_size x block_size)
    
    Returns:
        Compression statistics
    """
    n = matrix.shape[0]
    num_blocks = (n + block_size - 1) // block_size
    
    # Build block structure
    block_indptr = [0]
    block_indices = []
    block_data = []
    
    current_block = 0
    
    for i in range(num_blocks):
        row_start = i * block_size
        row_end = min((i + 1) * block_size, n)
        
        row_has_data = False
        
        for j in range(num_blocks):
            col_start = j * block_size
            col_end = min((j + 1) * block_size, n)
            
            # Extract block
            block = matrix[row_start:row_end, col_start:col_end]
            
            if block.nnz > 0:
                # Store block column index
                block_indices.append(j)
                
                # Store block data as dense array
                dense_block = np.zeros((block_size, block_size), dtype=np.float32)
                dense_block[:row_end-row_start, :col_end-col_start] = block.toarray()
                block_data.append(dense_block.flatten())
                
                row_has_data = True
        
        if row_has_data:
            current_block += len(block_indices) - block_indptr[-1]
        
        block_indptr.append(current_block)
    
    # Calculate file size
    header_size = 20  # 4 bytes for format, 4 for n, 4 for nnz, 4 for dtype, 4 for block_size
    data_size = (
        len(block_indptr) * 4 +  # block_indptr
        len(block_indices) * 4 +  # block_indices
        len(block_data) * block_size * block_size * 4  # block_data
    )
    total_size = header_size + data_size
    
    # Write to file
    with open(output_path, 'wb') as f:
        # Write header
        f.write(struct.pack('<I', 0x42534352))  # 'BSCR' magic number
        f.write(struct.pack('<I', n))
        f.write(struct.pack('<I', len(block_data)))
        f.write(struct.pack('<I', 0))  # dtype: 0 for float32
        f.write(struct.pack('<I', block_size))
        
        # Write data
        f.write(np.array(block_indptr, dtype=np.int32).tobytes())
        f.write(np.array(block_indices, dtype=np.int32).tobytes())
        for block in block_data:
            f.write(block.tobytes())
    
    # Calculate compression ratio
    original_size = n * n * 4
    compression_ratio = original_size / total_size
    
    return {
        "format": "BSCR",
        "compressed_size": total_size,
        "compression_ratio": compression_ratio,
        "block_size": block_size
    }


def compress_from_file(
    input_path: str,
    output_path: str,
    format: str,
    block_size: Optional[int] = 16
) -> Dict[str, any]:
    """
    Compress a matrix from a .mat file.
    
    Args:
        input_path: Path to the input .mat file
        output_path: Path to save the compressed file
        format: The compression format ("coo", "csr", "bscr")
        block_size: Block size for BSCR format (default: 16)
    
    Returns:
        A dictionary with compression statistics
    """
    matrix = read_mat_file(input_path)
    return compress_matrix(matrix, format, output_path, block_size)
