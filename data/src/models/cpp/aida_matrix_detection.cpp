/* AIDA Matrix Detection for Multi-TS Analysis */
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <string>
using namespace std;

// Set these to your actual values!
const int n_rows = 30; // e.g., number of stocks
const int n_cols = 15; // e.g., window_size * n_features

double frobenius_distance(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    double sum = 0.0;
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t j = 0; j < A[0].size(); ++j)
            sum += (A[i][j] - B[i][j]) * (A[i][j] - B[i][j]);
    return sqrt(sum);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input_csv_file>" << endl;
        return 1;
    }
    string input_file = argv[1];
    ifstream file(input_file);
    if (!file.is_open()) {
        cerr << "Error: Could not open input file " << input_file << endl;
        return 1;
    }
    vector<vector<vector<double>>> matrices;
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        vector<double> flat_row;
        while (getline(ss, cell, ',')) {
            flat_row.push_back(stod(cell));
        }
        // Each row is a stock, with window_size * n_features columns
        vector<vector<double>> mat(n_rows, vector<double>(n_cols));
        for (int i = 0; i < n_rows; ++i)
            for (int j = 0; j < n_cols; ++j)
                mat[i][j] = flat_row[i * n_cols + j];
        matrices.push_back(mat);
    }
    file.close();

    // Compute anomaly scores (average Frobenius distance to all others)
    vector<double> scores(matrices.size(), 0.0);
    for (size_t i = 0; i < matrices.size(); ++i) {
        double sum_dist = 0.0;
        for (size_t j = 0; j < matrices.size(); ++j) {
            if (i == j) continue;
            sum_dist += frobenius_distance(matrices[i], matrices[j]);
        }
        scores[i] = sum_dist / (matrices.size() - 1);
    }

    // Output scores
    ofstream fres(string(input_file) + "_AIDA_scores.dat");
    fres << matrices.size() << endl;
    for (size_t i = 0; i < scores.size(); ++i)
        fres << scores[i] << endl;
    fres.close();

    // Threshold and output anomalies
    double mean = 0.0, stddev = 0.0;
    for (double s : scores) mean += s;
    mean /= scores.size();
    for (double s : scores) stddev += (s - mean) * (s - mean);
    stddev = sqrt(stddev / scores.size());
    double threshold = mean + 2 * stddev;

    ofstream fanom(string(input_file) + "_AIDA_anomalies.csv");
    fanom << "index,score" << endl;
    for (size_t i = 0; i < scores.size(); ++i) {
        if (scores[i] > threshold)
            fanom << i << "," << scores[i] << endl;
    }
    fanom.close();

    cout << "Done. Scores and anomalies written." << endl;
    return 0;
}