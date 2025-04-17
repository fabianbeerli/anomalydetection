#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "aida_class.h"
#include "io_module.h"

using namespace std;

// Parse command line arguments
void parse_args(int argc, char** argv, 
                string& data_file, string& nominal_file, string& output_file,
                int& n_subsamples, string& score_type, double& alpha_min, 
                double& alpha_max, string& metric) {
    
    // Default values
    data_file = "../synthetic_data/example_cross_1000_50_num.dat";
    nominal_file = "";
    output_file = "../results/scores.txt";
    n_subsamples = 100;
    score_type = "variance";
    alpha_min = 1.0;
    alpha_max = 1.0;
    metric = "manhattan";
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        
        if (arg == "--data" && i + 1 < argc) {
            data_file = argv[++i];
        } else if (arg == "--nominal" && i + 1 < argc) {
            nominal_file = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--n_subsamples" && i + 1 < argc) {
            n_subsamples = stoi(argv[++i]);
        } else if (arg == "--score_type" && i + 1 < argc) {
            score_type = argv[++i];
        } else if (arg == "--alpha_min" && i + 1 < argc) {
            alpha_min = stod(argv[++i]);
        } else if (arg == "--alpha_max" && i + 1 < argc) {
            alpha_max = stod(argv[++i]);
        } else if (arg == "--metric" && i + 1 < argc) {
            metric = argv[++i];
        }
    }
}

int main(int argc, char** argv) {
    // Parse command line arguments
    string data_file, nominal_file, output_file;
    int n_subsamples;
    string score_type, metric;
    double alpha_min, alpha_max;
    
    parse_args(argc, argv, data_file, nominal_file, output_file, 
               n_subsamples, score_type, alpha_min, alpha_max, metric);
    
    // Load numerical features
    int n, nFnum;
    double* Xnum = NULL;
    read_data(Xnum, n, nFnum, data_file);
    
    // Load or create nominal features
    int nFnom = 1;
    int* Xnom;
    
    if (!nominal_file.empty()) {
        read_data(Xnom, n, nFnom, nominal_file);
    } else {
        // Create a dummy array with all 0s
        Xnom = new int[n * nFnom];
        for (int i = 0; i < n; i++) {
            Xnom[i] = 0;
        }
    }
    
    // Initialize AIDA
    AIDA aida(n_subsamples, "aom", score_type, alpha_min, alpha_max, metric);
    
    // Fit AIDA on the data
    aida.fit(n, nFnum, Xnum, nFnom, Xnom, 50, 512, nFnum, nFnum, nFnom, nFnom);
    
    // Compute anomaly scores
    double* scores = new double[n];
    aida.score_samples(n, scores, Xnum, Xnom);
    
    // Write scores to output file
    ofstream outfile(output_file);
    outfile << n << endl;
    for (int i = 0; i < n; i++) {
        outfile << scores[i] << endl;
    }
    outfile.close();
    
    // Clean up
    delete[] Xnum;
    delete[] Xnom;
    delete[] scores;
    
    return 0;
}