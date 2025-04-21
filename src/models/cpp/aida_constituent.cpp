/* AIDA Anomaly Detection for Constituent Stocks */
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
        cerr << "Usage: " << argv[0] << " <input_csv_file> <ticker>" << endl;
        return 1;
    }
    
    string input_file = argv[1];
    string ticker = argv[2];
    string output_scores_file = input_file + "_AIDA_scores.dat";
    string output_anomalies_file = input_file + "_AIDA_anomalies.csv";
    
    // Read CSV file
    ifstream file(input_file);
    if (!file.is_open()) {
        cerr << "Error: Could not open input file " << input_file << endl;
        return 1;
    }
    
    // Parse header
    string header_line;
    getline(file, header_line);
    stringstream header_stream(header_line);
    string field_name;
    vector<string> field_names;
    
    // Skip first column (date/index)
    getline(header_stream, field_name, ',');
    
    int nFnum = 0;
    while (getline(header_stream, field_name, ',')) {
        field_names.push_back(field_name);
        nFnum++;
    }
    
    // Read numerical data and dates
    vector<vector<double>> numerical_data;
    vector<string> dates;
    string line;
    
    while (getline(file, line)) {
        stringstream ss(line);
        string date;
        getline(ss, date, ',');  // Read date
        dates.push_back(date);
        
        vector<double> row;
        string cell;
        while (getline(ss, cell, ',')) {
            try {
                row.push_back(stod(cell));
            } catch (const std::invalid_argument& e) {
                row.push_back(0.0);  // Handle missing/invalid values
            }
        }
        
        if (!row.empty()) {
            numerical_data.push_back(row);
        }
    }
    file.close();
    
    int n = numerical_data.size();
    int nFnom = 1;  // Single nominal feature (all zeros)
    
    cout << "Data loaded for " << ticker << ": " << n << " samples, " << nFnum << " features" << endl;
    
    // Prepare data for AIDA
    double* Xnum = new double[n * nFnum];
    int* Xnom = new int[n * nFnom];
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < nFnum; ++j) {
            if (j < numerical_data[i].size()) {
                Xnum[j + i * nFnum] = numerical_data[i][j];
            } else {
                Xnum[j + i * nFnum] = 0.0;  // Handle missing columns
            }
        }
        Xnom[i] = 0;  // Single nominal feature (all zeros)
    }
    
    // AIDA Parameters
    int N = 100;  // Number of random subsamples
    string aggregate_type = "aom";
    string score_function = "variance";
    double alpha_min = 1.0;
    double alpha_max = 1.0;
    string distance_metric = "euclidean";
    int subsample_min = 50;
    int subsample_max = min(512, n);
    int dmin = max(2, nFnum / 2);
    int dmax = nFnum;
    
    double* scoresAIDA = new double[n];
    
    try {
        cout << "Training AIDA for " << ticker << "..." << endl;
        
        // Start timing
        auto start_time = high_resolution_clock::now();
        
        // Initialize and run AIDA
        AIDA aida(N, aggregate_type, score_function, alpha_min, alpha_max, distance_metric);
        aida.fit(n, nFnum, Xnum, nFnom, Xnom, subsample_min, subsample_max, dmin, dmax, nFnom, nFnom);
        aida.score_samples(n, scoresAIDA, Xnum, Xnom);
        
        // Stop timing
        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end_time - start_time);
        
        // Write scores
        ofstream fres(output_scores_file);
        fres << n << endl;
        for (int i = 0; i < n; ++i) {
            fres << scoresAIDA[i] << endl;
        }
        fres.close();
        
        // Calculate mean and std for anomaly threshold
        double mean_score = 0.0;
        double std_score = 0.0;
        
        for (int i = 0; i < n; ++i) {
            mean_score += scoresAIDA[i];
        }
        mean_score /= n;
        
        for (int i = 0; i < n; ++i) {
            std_score += (scoresAIDA[i] - mean_score) * (scoresAIDA[i] - mean_score);
        }
        std_score = sqrt(std_score / n);
        
        double threshold = mean_score + 2 * std_score;
        
        // Write anomalies
        ofstream fanom(output_anomalies_file);
        int anomaly_count = 0;
        
        fanom << "index,date,score,start_date,end_date" << endl;
        for (int i = 0; i < n; ++i) {
            if (scoresAIDA[i] > threshold) {
                fanom << i << "," << dates[i] << "," << scoresAIDA[i] << "," 
                      << dates[i] << "," << dates[i] << endl;
                anomaly_count++;
            }
        }
        fanom.close();
        
        // Save execution time
        ofstream ftime(input_file + "_AIDA_time.txt");
        ftime << duration.count() / 1000.0 << endl;
        ftime.close();
        
        // Report results
        cout << "AIDA analysis complete for " << ticker << ":" << endl;
        cout << "Total samples: " << n << endl;
        cout << "Anomalies detected: " << anomaly_count << endl;
        cout << "Execution time: " << duration.count() / 1000.0 << " seconds" << endl;
        
        // Clean up
        delete[] Xnum;
        delete[] Xnom;
        delete[] scoresAIDA;
        
        return 0;
    } catch (const std::exception& e) {
        cerr << "Error during AIDA processing: " << e.what() << endl;
        
        // Clean up
        delete[] Xnum;
        delete[] Xnom;
        delete[] scoresAIDA;
        
        return 1;
    }
}
