from __future__ import annotations

import argparse
import json
import os
from typing import Optional

from matrix_compress.compress import compress_from_file
from matrix_compress.decompress import decompress_matrix, validate_compression
import scipy.sparse as sp
import scipy.io as sio


def main() -> None:
    """
    Main entry point for the command-line interface.
    """
    parser = argparse.ArgumentParser(
        description="Matrix compression and decompression tool"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Compress command
    compress_parser = subparsers.add_parser(
        "compress",
        help="Compress a matrix from a .mat file"
    )
    compress_parser.add_argument(
        "input",
        help="Path to the input .mat file"
    )
    compress_parser.add_argument(
        "output",
        help="Path to save the compressed file"
    )
    compress_parser.add_argument(
        "--format",
        choices=["coo", "csr", "bscr"],
        default="csr",
        help="Compression format (default: csr)"
    )
    compress_parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Block size for BSCR format (default: 16)"
    )
    compress_parser.add_argument(
        "--stats",
        action="store_true",
        help="Print compression statistics"
    )
    
    # Decompress command
    decompress_parser = subparsers.add_parser(
        "decompress",
        help="Decompress a matrix from a compressed file"
    )
    decompress_parser.add_argument(
        "input",
        help="Path to the compressed file"
    )
    decompress_parser.add_argument(
        "output",
        help="Path to save the decompressed .mat file"
    )
    decompress_parser.add_argument(
        "--format",
        choices=["coo", "csr", "bscr"],
        default=None,
        help="Compression format (default: auto-detect)"
    )
    decompress_parser.add_argument(
        "--validate",
        type=str,
        default=None,
        help="Path to the original .mat file for validation"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == "compress":
        handle_compress(args)
    elif args.command == "decompress":
        handle_decompress(args)
    else:
        parser.print_help()


def handle_compress(args) -> None:
    """
    Handle the compress command.
    
    Args:
        args: Command-line arguments
    """
    try:
        print(f"Compressing {args.input} to {args.output} using {args.format} format...")
        
        stats = compress_from_file(
            args.input,
            args.output,
            args.format,
            args.block_size
        )
        
        print(f"Compression completed successfully!")
        
        if args.stats:
            print("\nCompression Statistics:")
            print(json.dumps(stats, indent=2))
        else:
            print(f"Original size: {stats['original_size']:,} bytes")
            print(f"Compressed size: {stats['compressed_size']:,} bytes")
            print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
            print(f"Matrix size: {stats['matrix_size']}x{stats['matrix_size']}")
            print(f"Original density: {stats['original_density']:.4f}")
            
    except Exception as e:
        print(f"Error during compression: {str(e)}")


def handle_decompress(args) -> None:
    """
    Handle the decompress command.
    
    Args:
        args: Command-line arguments
    """
    try:
        print(f"Decompressing {args.input} to {args.output}...")
        
        # Decompress the matrix
        matrix = decompress_matrix(args.input, args.format)
        
        # Save to .mat file
        sio.savemat(args.output, {"A": matrix.toarray()})
        
        print(f"Decompression completed successfully!")
        print(f"Matrix size: {matrix.shape[0]}x{matrix.shape[1]}")
        print(f"Number of non-zero elements: {matrix.nnz}")
        print(f"Density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.4f}")
        
        # Validate if original is provided
        if args.validate:
            print("\nValidating decompression...")
            original = sio.loadmat(args.validate)["A"]
            original_sparse = sp.csr_matrix(original)
            
            validation = validate_compression(original_sparse, matrix)
            
            if validation["is_valid"]:
                print("Validation PASSED: Decompressed matrix matches original!")
            else:
                print("Validation FAILED: Decompressed matrix differs from original!")
                print(f"Shape match: {validation['shape_match']}")
                print(f"NNZ match: {validation['nnz_match']}")
                print(f"Values match: {validation['values_match']}")
                print(f"Max error: {validation['max_error']:.6e}")
                print(f"Mean error: {validation['mean_error']:.6e}")
        
    except Exception as e:
        print(f"Error during decompression: {str(e)}")


if __name__ == "__main__":
    main()
