from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import scipy.sparse as sp
import scipy.io as sio

from matrix_compress.compress import compress_matrix, read_mat_file
from matrix_compress.decompress import decompress_matrix, validate_compression


class TestMatrixCompression(unittest.TestCase):
    """
    Test cases for matrix compression and decompression.
    """
    
    def setUp(self) -> None:
        """
        Set up test fixtures.
        """
        # Create a small test matrix
        self.n = 32
        self.density = 0.1
        
        # Create a sparse matrix
        rng = np.random.default_rng(42)
        self.matrix = sp.random(
            self.n, self.n, density=self.density, format="csr", random_state=rng
        )
        self.matrix = self.matrix + self.matrix.T  # Make symmetric
        self.matrix = self.matrix + sp.eye(self.n, format="csr") * self.n  # Make diagonally dominant
        
        # Create temporary files for testing
        self.temp_dir = tempfile.mkdtemp()
        self.mat_file = os.path.join(self.temp_dir, "test_matrix.mat")
        
        # Save matrix to .mat file
        sio.savemat(self.mat_file, {"A": self.matrix.toarray()})
    
    def tearDown(self) -> None:
        """
        Clean up test fixtures.
        """
        # Remove temporary files
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_read_mat_file(self) -> None:
        """
        Test reading a matrix from a .mat file.
        """
        matrix = read_mat_file(self.mat_file)
        self.assertIsInstance(matrix, sp.spmatrix)
        self.assertEqual(matrix.shape, (self.n, self.n))
    
    def test_compress_decompress_coo(self) -> None:
        """
        Test compressing and decompressing a matrix using COO format.
        """
        compressed_file = os.path.join(self.temp_dir, "compressed.coo")
        
        # Compress matrix
        stats = compress_matrix(self.matrix, "coo", compressed_file)
        self.assertGreater(stats["compression_ratio"], 1.0)
        
        # Decompress matrix
        decompressed = decompress_matrix(compressed_file)
        self.assertIsInstance(decompressed, sp.spmatrix)
        self.assertEqual(decompressed.shape, (self.n, self.n))
        
        # Validate decompression
        validation = validate_compression(self.matrix, decompressed)
        self.assertTrue(validation["is_valid"])
    
    def test_compress_decompress_csr(self) -> None:
        """
        Test compressing and decompressing a matrix using CSR format.
        """
        compressed_file = os.path.join(self.temp_dir, "compressed.csr")
        
        # Compress matrix
        stats = compress_matrix(self.matrix, "csr", compressed_file)
        self.assertGreater(stats["compression_ratio"], 1.0)
        
        # Decompress matrix
        decompressed = decompress_matrix(compressed_file)
        self.assertIsInstance(decompressed, sp.spmatrix)
        self.assertEqual(decompressed.shape, (self.n, self.n))
        
        # Validate decompression
        validation = validate_compression(self.matrix, decompressed)
        self.assertTrue(validation["is_valid"])
    
    def test_compress_decompress_bscr(self) -> None:
        """
        Test compressing and decompressing a matrix using BSCR format.
        """
        compressed_file = os.path.join(self.temp_dir, "compressed.bscr")
        
        # Compress matrix with different block sizes
        for block_size in [8, 16, 32]:
            stats = compress_matrix(self.matrix, "bscr", compressed_file, block_size)
            self.assertGreater(stats["compression_ratio"], 1.0)
            
            # Decompress matrix
            decompressed = decompress_matrix(compressed_file)
            self.assertIsInstance(decompressed, sp.spmatrix)
            self.assertEqual(decompressed.shape, (self.n, self.n))
            
            # Validate decompression
            validation = validate_compression(self.matrix, decompressed)
            self.assertTrue(validation["is_valid"])
    
    def test_format_auto_detection(self) -> None:
        """
        Test automatic format detection during decompression.
        """
        formats = ["coo", "csr", "bscr"]
        
        for fmt in formats:
            compressed_file = os.path.join(self.temp_dir, f"compressed.{fmt}")
            
            # Compress matrix
            compress_matrix(self.matrix, fmt, compressed_file)
            
            # Decompress without specifying format (auto-detect)
            decompressed = decompress_matrix(compressed_file)
            self.assertIsInstance(decompressed, sp.spmatrix)
            self.assertEqual(decompressed.shape, (self.n, self.n))
            
            # Validate decompression
            validation = validate_compression(self.matrix, decompressed)
            self.assertTrue(validation["is_valid"])
    
    def test_compression_ratio(self) -> None:
        """
        Test that compression ratios are reasonable.
        """
        formats = ["coo", "csr", "bscr"]
        
        for fmt in formats:
            compressed_file = os.path.join(self.temp_dir, f"compressed.{fmt}")
            
            # Compress matrix
            stats = compress_matrix(self.matrix, fmt, compressed_file)
            
            # Check that compression ratio is greater than 1
            self.assertGreater(stats["compression_ratio"], 1.0)
            
            # Check that compressed size is smaller than original
            self.assertLess(stats["compressed_size"], stats["original_size"])
    
    def test_large_matrix(self) -> None:
        """
        Test compression and decompression with a larger matrix.
        """
        # Create a larger test matrix
        n_large = 128
        density = 0.05
        
        rng = np.random.default_rng(42)
        large_matrix = sp.random(
            n_large, n_large, density=density, format="csr", random_state=rng
        )
        large_matrix = large_matrix + large_matrix.T
        large_matrix = large_matrix + sp.eye(n_large, format="csr") * n_large
        
        compressed_file = os.path.join(self.temp_dir, "large_compressed.csr")
        
        # Compress matrix
        stats = compress_matrix(large_matrix, "csr", compressed_file)
        self.assertGreater(stats["compression_ratio"], 1.0)
        
        # Decompress matrix
        decompressed = decompress_matrix(compressed_file)
        self.assertIsInstance(decompressed, sp.spmatrix)
        self.assertEqual(decompressed.shape, (n_large, n_large))
        
        # Validate decompression
        validation = validate_compression(large_matrix, decompressed)
        self.assertTrue(validation["is_valid"])


if __name__ == "__main__":
    unittest.main()
