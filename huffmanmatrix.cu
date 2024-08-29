#include <torch/all.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>


// helper struct to store the huffman tree
// this should have 3 values
//a pointer to the left child
//a pointer to the right child
//a value

struct huffnode {
    huffnode* left;
    huffnode* right;
    int value;
};



// helper function to create the huffman tree from 
// two arrays, one of the values, and the other of the encoded byte string as a uint8 (we can assume that there are less than 256 unique values)

__device__ huffnode* create_huffman_tree(int* values, uint8_t* encoded_bytes, int num_values) {
    // create a list of huffnodes
    huffnode* parent = new huffnode;

    // for each value
    for (int i = 0; i < num_values; i++) {
        huffnode* current = parent;
        uint8_t byte_sequence = encoded_bytes[i];
        // for each bit in the byte sequence
        for (int j = 0; j < 8; j++) {
            // if the bit is 1, go right
            if (byte_sequence & (1 << j)) {
                if (current->right == NULL) {
                    current->right = new huffnode;
                }
                current = current->right;
            } else {
                // if the bit is 0, go left
                if (current->left == NULL) {
                    current->left = new huffnode;
                }
                current = current->left;
            }
        }
        // set the value of the leaf node to the value
        current->value = values[i];
    }

    return parent;
}

// define the global variables
// we assume that the matrix multiplication that we need to perform is
// XA^T where X is a matrix of size K x N, where K is the 
// flattened batch*etc size and N is the hidden dimension
// A is a matrix of size N x N, we assume N is a power of 2
// to aid vectorization, we will directly pass the untransposed A matrix
// therefore the multiplication will be Y_{ij} = \sum_{k=0}^{K-1} X_{ik} A_{jk}
const int B_k = 32;
const int B_Nx = 32;

//furthermore, A will be a string of bits, so to aid decryption,
//we will pass in also a matrix A_indexs, which are the indexs of the starting 
//locations corresponding to every B_NA values
//furthemore, we will assume that the max bits in every B_NA values is governed by 
// MAX_BLOCKROW_BITS
const int B_NA = 32;
const int MAX_BLOCKROW_BITS = 2; // in bytes so 16 bits (May delete)
const int B_NA2 =  128; // we can assume that we can push this larger because 


//tile parameters
const int TILE_X = 2; // number of rows of X included as one tile for one thread to process
const int TILE_A = 2; // number of rows of A included as one tile for one thread to process


//actual kernel
__global__ void __launch_bounds__((B_k*B_Nx)/(TILE_A * TILE_X),1)
    HuffMatrixMult(
        const half2* X,
        const uint8_t* A,
        const int32_t* A_indexs,
        const float* Y, // output
        //huffman tree parameters
        //we assume each row of A has its own huffman tree
        const int* values,
        const uint8_t* encoded_bytes,
        const int* num_values, // number of values in each row
        int K, // number of rows in X
        int N, // number of columns in X and rows in A
    ) {

        
        
    }