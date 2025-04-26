/* TIX Analysis for AIDA Anomalies */
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
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <input_csv_file> <anomaly_index> <output_file>" << endl;
        return 1;
    }
    
    string input_file = argv[1];
    int anomaly_index = stoi(argv[2]);
    string output_file = argv[3];
    
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
    
    // Read numerical data
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
    
    cout << "Data loaded: " << n << " samples, " << nFnum << " features" << endl;
    cout << "Analyzing anomaly index: " << anomaly_index << endl;
    
    if (anomaly_index < 0 || anomaly_index >= n) {
        cerr << "Error: Anomaly index out of range" << endl;
        return 1;
    }
    
    // Prepare data for AIDA and TIX
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
    
    try {
        cout << "Initializing AIDA for TIX analysis..." << endl;
        
        // Start timing
        auto start_time = high_resolution_clock::now();
        
        // Initialize AIDA
        AIDA aida(N, aggregate_type, score_function, alpha_min, alpha_max, distance_metric);
        aida.fit(n, nFnum, Xnum, nFnom, Xnom, subsample_min, subsample_max, dmin, dmax, nFnom, nFnom);
        
        // TIX Parameters
        double ref_rate = 2.0;
        int dim_ref = 10;
        int niter_ref = 1;
        int niter_tix = 10;
        int maxlen = 50 * nFnum;
        double Tmin = 0.01;
        double Tmax = 0.015;
        
        // Allocate memory for importance scores
        double* feature_importance = new double[nFnum];
        
        cout << "Running TIX analysis for point " << anomaly_index << "..." << endl;
        
        // Run TIX analysis for the anomaly
        aida.tix(
            feature_importance,
            &Xnum[anomaly_index * nFnum],
            &Xnom[anomaly_index * nFnom],
            ref_rate,
            dim_ref,
            niter_ref,
            niter_tix,
            maxlen,
            Tmin,
            Tmax
        );
        
        // End timing
        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end_time - start_time);
        
        // Write feature importance scores to output file
        ofstream fout(output_file);
        if (!fout.is_open()) {
            cerr << "Error: Could not open output file " << output_file << endl;
            delete[] Xnum;
            delete[] Xnom;
            delete[] feature_importance;
            return 1;
        }
        
        // Write header with feature names
        fout << "feature_index,feature_name,importance_score" << endl;
        
        for (int i = 0; i < nFnum; ++i) {
            string feature_name = (i < field_names.size()) ? field_names[i] : "feature_" + to_string(i);
            fout << i << "," << feature_name << "," << feature_importance[i] << endl;
        }
        
        fout.close();
        
        // Report results
        cout << "TIX analysis complete for point " << anomaly_index << endl;
        cout << "Execution time: " << duration.count() / 1000.0 << " seconds" << endl;
        cout << "Feature importance scores saved to " << output_file << endl;
        
        // Clean up
        delete[] Xnum;
        delete[] Xnom;
        delete[] feature_importance;
        
        return 0;
    } catch (const std::exception& e) {
        cerr << "Error during TIX analysis: " << e.what() << endl;
        
        // Clean up
        delete[] Xnum;
        delete[] Xnom;
        
        return 1;
    }
}
