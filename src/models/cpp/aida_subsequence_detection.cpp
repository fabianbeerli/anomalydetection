/* AIDA Anomaly Detection for Subsequences */

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
    // Check if input file is provided
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input_csv_file>" << endl;
        return 1;
    }

    // Input CSV file path
    string input_file = argv[1];
    
    // Output file paths
    string output_scores_file = input_file + "_AIDA_scores.dat";
    string output_anomalies_file = input_file + "_AIDA_anomalies.csv";

    // Start timing
    auto start_time = high_resolution_clock::now();

    // Read the CSV file
    ifstream file(input_file);
    if (!file.is_open()) {
        cerr << "Error: Could not open input file " << input_file << endl;
        return 1;
    }

    // Parse CSV header
    string header_line;
    getline(file, header_line);
    
    // Count the number of features
    stringstream header_stream(header_line);
    string feature_name;
    int nFnum = 0;
    while (getline(header_stream, feature_name, ',')) {
        nFnum++;
    }
    
    // Vectors to store data
    vector<vector<double>> numerical_data;
    string line;
    
    // Read numerical data
    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        vector<double> row;
        
        while (getline(ss, cell, ',')) {
            try {
                // Convert string to double
                row.push_back(stod(cell));
            } catch (const std::invalid_argument& e) {
                // Skip non-numeric cells or handle as needed
                row.push_back(0.0);
            }
        }
        
        if (!row.empty()) {
            numerical_data.push_back(row);
        }
    }
    file.close();

    // Prepare data for AIDA
    int n = numerical_data.size();
    int nFnom = 1;  // Nominal features (set to 1 with all zeros)

    cout << "Data loaded: " << n << " subsequences, " << nFnum << " features per subsequence" << endl;

    // Allocate memory for numerical and nominal features
    double* Xnum = new double[n * nFnum];
    int* Xnom = new int[n * nFnom];

    // Fill numerical features
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < nFnum; ++j) {
            Xnum[j + i * nFnum] = numerical_data[i][j];
        }
        // Fill nominal features with zeros
        Xnom[i] = 0;
    }

    // AIDA Parameters
    int N = 100;  // Number of subsamples
    string aggregate_type = "aom";
    string score_function = "variance";
    double alpha_min = 1.0;
    double alpha_max = 1.0;
    string distance_metric = "manhattan";

    // Anomaly detection parameters
    int subsample_min = 50;
    int subsample_max = min(512, n);  // Limit to dataset size
    int dmin = min(nFnum, max(2, nFnum / 2));  // At least 2 features
    int dmax = nFnum;

    // Allocate memory for scores
    double* scoresAIDA = new double[n];

    try {
        cout << "Training AIDA..." << endl;
        
        // Train AIDA
        AIDA aida(N, aggregate_type, score_function, alpha_min, alpha_max, distance_metric);
        aida.fit(n, nFnum, Xnum, nFnom, Xnom, subsample_min, subsample_max, dmin, dmax, nFnom, nFnom);
        
        cout << "Computing anomaly scores..." << endl;
        
        // Compute anomaly scores
        aida.score_samples(n, scoresAIDA, Xnum, Xnom);

        // Write scores to file
        ofstream fres(output_scores_file);
        fres << n << endl;
        for (int i = 0; i < n; ++i) {
            fres << scoresAIDA[i] << endl;
        }
        fres.close();

        // Detect anomalies (simple threshold-based approach)
        double mean_score = 0.0;
        double std_score = 0.0;
        
        // Compute mean and standard deviation
        for (int i = 0; i < n; ++i) {
            mean_score += scoresAIDA[i];
        }
        mean_score /= n;
        
        for (int i = 0; i < n; ++i) {
            std_score += (scoresAIDA[i] - mean_score) * (scoresAIDA[i] - mean_score);
        }
        std_score = sqrt(std_score / n);

        // Threshold: 2 standard deviations
        double threshold = mean_score + 2 * std_score;

        // Write anomalies to file
        ofstream fanom(output_anomalies_file);
        int anomaly_count = 0;
        
        fanom << "index,subsequence_idx,score" << endl;
        for (int i = 0; i < n; ++i) {
            if (scoresAIDA[i] > threshold) {
                fanom << i << "," << i << "," << scoresAIDA[i] << endl;
                anomaly_count++;
            }
        }
        fanom.close();

        // Stop timing
        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end_time - start_time);
        
        cout << "AIDA Subsequence Analysis Complete:" << endl;
        cout << "Total subsequences: " << n << endl;
        cout << "Anomalies detected: " << anomaly_count << endl;
        cout << "Execution time: " << duration.count() / 1000.0 << " seconds" << endl;
        cout << "Scores saved to: " << output_scores_file << endl;
        cout << "Anomalies saved to: " << output_anomalies_file << endl;
        
        // Save execution time to file
        ofstream ftime(input_file + "_AIDA_time.txt");
        ftime << duration.count() / 1000.0 << endl;
        ftime.close();
    }
    catch (const std::exception& e) {
        cerr << "Error during AIDA processing: " << e.what() << endl;
        
        // Clean up
        delete[] Xnum;
        delete[] Xnom;
        delete[] scoresAIDA;
        
        return 1;
    }

    // Clean up
    delete[] Xnum;
    delete[] Xnom;
    delete[] scoresAIDA;

    return 0;
}
