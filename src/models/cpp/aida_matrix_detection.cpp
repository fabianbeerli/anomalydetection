/* AIDA Anomaly Detection for Matrix Data */
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
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input_csv_file>" << endl;
        return 1;
    }
    string input_file = argv[1];
    string output_scores_file = input_file + "_AIDA_scores.dat";
    string output_anomalies_file = input_file + "_AIDA_anomalies.csv";
    ifstream file(input_file);
    if (!file.is_open()) {
        cerr << "Error: Could not open input file " << input_file << endl;
        return 1;
    }
    string header_line;
    getline(file, header_line);
    stringstream header_stream(header_line);
    string feature_name;
    int nFnum = 0;
    while (getline(header_stream, feature_name, ',')) {
        nFnum++;
    }
    vector<vector<double>> numerical_data;
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        vector<double> row;
        while (getline(ss, cell, ',')) {
            try {
                row.push_back(stod(cell));
            } catch (const std::invalid_argument& e) {
                row.push_back(0.0);
            }
        }
        if (!row.empty()) {
            numerical_data.push_back(row);
        }
    }
    file.close();
    int n = numerical_data.size();
    int nFnom = 1;
    cout << "Data loaded: " << n << " rows, " << nFnum << " features per row" << endl;
    double* Xnum = new double[n * nFnum];
    int* Xnom = new int[n * nFnom];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < nFnum; ++j) {
            Xnum[j + i * nFnum] = numerical_data[i][j];
        }
        Xnom[i] = 0;
    }
    int N = 100;
    string aggregate_type = "aom";
    string score_function = "variance";
    double alpha_min = 1.0;
    double alpha_max = 1.0;
    string distance_metric = "euclidean";
    int subsample_min = 50;
    int subsample_max = min(256, n);
    int dmin = min(nFnum, max(2, nFnum / 2));
    int dmax = nFnum;
    double* scoresAIDA = new double[n];
    try {
        cout << "Training AIDA..." << endl;
        auto start_time = high_resolution_clock::now();
        AIDA aida(N, aggregate_type, score_function, alpha_min, alpha_max, distance_metric);
        aida.fit(n, nFnum, Xnum, nFnom, Xnom, subsample_min, subsample_max, dmin, dmax, nFnom, nFnom);
        cout << "Computing anomaly scores..." << endl;
        aida.score_samples(n, scoresAIDA, Xnum, Xnom);
        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end_time - start_time);
        ofstream fres(output_scores_file);
        fres << n << endl;
        for (int i = 0; i < n; ++i) {
            fres << scoresAIDA[i] << endl;
        }
        fres.close();
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
        ofstream fanom(output_anomalies_file);
        int anomaly_count = 0;
        fanom << "index,stock_idx,score" << endl;
        for (int i = 0; i < n; ++i) {
            if (scoresAIDA[i] > threshold) {
                fanom << i << "," << i % 30 << "," << scoresAIDA[i] << endl;
                anomaly_count++;
            }
        }
        fanom.close();
        cout << "AIDA Matrix Analysis Complete:" << endl;
        cout << "Total rows: " << n << endl;
        cout << "Anomalies detected: " << anomaly_count << endl;
        cout << "Model execution time: " << duration.count() / 1000.0 << " seconds" << endl;
        cout << "Scores saved to: " << output_scores_file << endl;
        cout << "Anomalies saved to: " << output_anomalies_file << endl;
        ofstream ftime(input_file + "_AIDA_time.txt");
        ftime << duration.count() / 1000.0 << endl;
        ftime.close();
    }
    catch (const std::exception& e) {
        cerr << "Error during AIDA processing: " << e.what() << endl;
        delete[] Xnum;
        delete[] Xnom;
        delete[] scoresAIDA;
        return 1;
    }
    delete[] Xnum;
    delete[] Xnom;
    delete[] scoresAIDA;
    return 0;
}
