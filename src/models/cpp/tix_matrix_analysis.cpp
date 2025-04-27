/* TIX Analysis for Multi-TS Matrix Anomalies */
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <chrono>
#include "aida_class.h"
using namespace std;
using namespace std::chrono;

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <input_csv_file> <output_file>" << endl;
        return 1;
    }
    
    string input_file = argv[1];
    string output_file = argv[2];
    
    // Read CSV file
    ifstream file(input_file);
    if (!file.is_open()) {
        cerr << "Error: Could not open input file " << input_file << endl;
        return 1;
    }
    
    // Parse file structure, handling multi-stock matrix format
    vector<vector<double>> matrix_data;
    string line;
    
    // Read data line by line
    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        vector<double> row;
        
        while (getline(ss, cell, ',')) {
            try {
                row.push_back(stod(cell));
            } catch (const std::invalid_argument& e) {
                row.push_back(0.0);  // Handle missing/invalid values
            }
        }
        
        if (!row.empty()) {
            matrix_data.push_back(row);
        }
    }
    file.close();
    
    if (matrix_data.empty()) {
        cerr << "Error: No data found in input file" << endl;
        return 1;
    }
    
    // Get matrix dimensions
    int n_stocks = matrix_data.size();
    int n_features = matrix_data[0].size();
    
    cout << "Matrix data loaded: " << n_stocks << " stocks with " << n_features << " features each" << endl;
    
    // For each stock (row in the matrix), apply TIX analysis
    ofstream fout(output_file);
    if (!fout.is_open()) {
        cerr << "Error: Could not open output file " << output_file << endl;
        return 1;
    }
    
    // Write header
    fout << "stock_idx,feature_idx,importance_score" << endl;
    
    // AIDA parameters
    int N = 100;
    string aggregate_type = "aom";
    string score_function = "variance";
    double alpha_min = 1.0;
    double alpha_max = 1.0;
    string distance_metric = "euclidean";
    
    // For each stock, treat its row as a separate time series for TIX analysis
    for (int stock_idx = 0; stock_idx < n_stocks; ++stock_idx) {
        cout << "Analyzing stock " << stock_idx + 1 << " of " << n_stocks << endl;
        
        try {
            // Create a synthetic dataset with this stock's data and some random noise samples
            // This allows AIDA to better distinguish what makes this stock's pattern unusual
            int synthetic_samples = 100;
            double* Xnum = new double[synthetic_samples * n_features];
            int* Xnom = new int[synthetic_samples];
            
            // First sample is the actual stock data
            for (int j = 0; j < n_features; ++j) {
                Xnum[j] = matrix_data[stock_idx][j];
            }
            Xnom[0] = 0;
            
            // Generate synthetic samples by adding random noise
            srand(stock_idx); // For reproducibility
            for (int i = 1; i < synthetic_samples; ++i) {
                for (int j = 0; j < n_features; ++j) {
                    // Add noise proportional to the original value
                    double noise_scale = 0.1; // 10% noise
                    double noise = (((double)rand() / RAND_MAX) * 2 - 1) * noise_scale * abs(matrix_data[stock_idx][j]);
                    Xnum[j + i * n_features] = matrix_data[stock_idx][j] + noise;
                }
                Xnom[i] = 0;
            }
            
            // Initialize AIDA
            AIDA aida(N, aggregate_type, score_function, alpha_min, alpha_max, distance_metric);
            aida.fit(synthetic_samples, n_features, Xnum, 1, Xnom, 50, min(512, synthetic_samples), n_features, n_features, 1, 1);
            
            // TIX Parameters
            double ref_rate = 2.0;
            int dim_ref = 10;
            int niter_ref = 1;
            int niter_tix = 10;
            int maxlen = 50 * n_features;
            double Tmin = 0.01;
            double Tmax = 0.015;
            
            // Allocate memory for importance scores
            double* feature_importance = new double[n_features];
            
            // Run TIX analysis for this stock's data point
            aida.tix(
                feature_importance,
                &Xnum[0],  
                &Xnom[0],
                ref_rate,
                dim_ref,
                niter_ref,
                niter_tix,
                maxlen,
                Tmin,
                Tmax
            );
            
            // Write importance scores to output file
            for (int j = 0; j < n_features; ++j) {
                fout << stock_idx << "," << j << "," << feature_importance[j] << endl;
            }
            
            // Clean up
            delete[] feature_importance;
            delete[] Xnum;
            delete[] Xnom;
            
        } catch (const std::exception& e) {
            cerr << "Error analyzing stock " << stock_idx << ": " << e.what() << endl;
        }
    }
    
    fout.close();
    cout << "TIX matrix analysis complete. Results saved to " << output_file << endl;
    
    return 0;
}