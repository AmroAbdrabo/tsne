#pragma once

// Input and output dimensions
constexpr int in_dim  = 16; // input dimension
constexpr int out_dim = 2; // map dimension
constexpr int N  = 1024; // number of input data

// Tunable parameters
constexpr double perp = 30.0; // perplexity = 2^H(P)
constexpr int max_iter =  1000; // training iterations
constexpr double eta = 4.0*100.0; // learning rate (the 4.0 comes from the gradient constant in front of the sum) (Bowen eta = 200.0)
constexpr int first_phase_iter = 250;
constexpr int second_phase_iter = max_iter - first_phase_iter;
constexpr double momentum = .5, final_momentum = .8;

constexpr int random_seed = 42; // seed for random initialization
